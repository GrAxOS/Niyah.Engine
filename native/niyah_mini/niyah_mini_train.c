#include "niyah_mini_train.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* NiyahMini native training. Standard per-head causal attention + split-half
 * RoPE + GQA + RMSNorm pre-norm + SwiGLU + tied LM head. Mirrors the
 * numerically-verified Python/TS reference; gradients checked by
 * niyah_mini_grad_check. Weight layout: W[j*in + k], out = W @ x. */

static void rmsnorm_fwd(const float* x,const float* w,int32_t n,float eps,float* out,float* r_out){
    double ss=0; for(int32_t i=0;i<n;i++) ss+=(double)x[i]*(double)x[i];
    float r=(float)(1.0/sqrt(ss/(double)n+(double)eps));
    for(int32_t i=0;i<n;i++) out[i]=x[i]*r*(w?w[i]:1.0f); if(r_out)*r_out=r;
}
static void rmsnorm_bwd(const float* dox,const float* x,const float* w,int32_t n,float r,float* dx_out,float* dw_out){
    double S=0; for(int32_t i=0;i<n;i++) S+=(double)dox[i]*(double)x[i]*(double)w[i];
    float coeff=-(r*r*r)/(float)n;
    for(int32_t i=0;i<n;i++){ dx_out[i]=dox[i]*w[i]*r+coeff*x[i]*(float)S; if(dw_out) dw_out[i]=dox[i]*x[i]*r; }
}
static void rope_cs(int32_t pos,int32_t hd,float theta,float* co,float* si){
    int32_t half=hd/2;
    for(int32_t j=0;j<half;j++){float freq=1.0f/powf(theta,(float)(2*j)/(float)hd); float ang=(float)pos*freq; float c=cosf(ang),s=sinf(ang); co[j]=c;co[j+half]=c;si[j]=s;si[j+half]=s;}
}
static void rope_fwd_v(float* x,int32_t hd,const float* cv,const float* sv){int32_t half=hd/2; for(int32_t j=0;j<half;j++){float x1=x[j],x2=x[j+half]; x[j]=x1*cv[j]-x2*sv[j]; x[j+half]=x2*cv[j]+x1*sv[j];}}
static void rope_bwd_v(float* dy,int32_t hd,const float* cv,const float* sv){int32_t half=hd/2; for(int32_t j=0;j<half;j++){float y1=dy[j],y2=dy[j+half]; dy[j]=y1*cv[j]+y2*sv[j]; dy[j+half]=y2*cv[j]-y1*sv[j];}}
static void matvec(const float* W,const float* x,int32_t on,int32_t in_n,float* out){
    for(int32_t j=0;j<on;j++){float s=0;const float* r=W+(size_t)j*(size_t)in_n; for(int32_t k=0;k<in_n;k++) s+=r[k]*x[k]; out[j]=s;}
}

NiyahStatus niyah_mini_grads_allocate(NiyahMiniGrads* g,const NiyahMiniConfig* cfg){
    if(!g||!cfg) return NIYAH_ERR_INVALID_ARG;
    memset(g,0,sizeof(*g));
    size_t dim=cfg->n_dim,vocab=cfg->n_vocab,L=cfg->n_layers,hd=dim/cfg->n_heads,kv=(size_t)cfg->n_kv_heads*hd,ff=cfg->n_ff;
    size_t per=dim+dim*dim+kv*dim+kv*dim+dim*dim+dim+ff*dim+ff*dim+dim*ff;
    size_t total=vocab*dim+L*per+dim;
    void* mem=malloc(total*sizeof(float)); if(!mem) return NIYAH_ERR_OUT_OF_MEMORY;
    g->memory_block=mem; g->memory_size=total*sizeof(float);
    float* p=(float*)mem; g->embedding=p; p+=vocab*dim;
    g->layers=(NiyahMiniLayerGrads*)malloc(L*sizeof(NiyahMiniLayerGrads));
    if(!g->layers){free(mem);memset(g,0,sizeof(*g));return NIYAH_ERR_OUT_OF_MEMORY;}
    for(size_t l=0;l<L;l++){NiyahMiniLayerGrads* gl=&g->layers[l];
        gl->attn_norm=p;p+=dim; gl->wq=p;p+=dim*dim; gl->wk=p;p+=kv*dim; gl->wv=p;p+=kv*dim;
        gl->wo=p;p+=dim*dim; gl->ffn_norm=p;p+=dim; gl->ffn_gate=p;p+=ff*dim; gl->ffn_up=p;p+=ff*dim; gl->ffn_down=p;p+=dim*ff;}
    g->final_norm=p; return NIYAH_OK;
}
void niyah_mini_grads_free(NiyahMiniGrads* g){if(!g)return; if(g->layers)free(g->layers); if(g->memory_block)free(g->memory_block); memset(g,0,sizeof(*g));}
void niyah_mini_grads_zero(NiyahMiniGrads* g,const NiyahMiniConfig* cfg){(void)cfg; if(!g)return; if(g->memory_block)memset(g->memory_block,0,g->memory_size);}

static size_t weights_n_floats(const NiyahMiniConfig* cfg){
    size_t dim=cfg->n_dim,vocab=cfg->n_vocab,L=cfg->n_layers,hd=dim/cfg->n_heads,kv=(size_t)cfg->n_kv_heads*hd,ff=cfg->n_ff;
    size_t per=dim+dim*dim+kv*dim+kv*dim+dim*dim+dim+ff*dim+ff*dim+dim*ff;
    return vocab*dim+L*per+dim;
}
NiyahStatus niyah_mini_optim_init(NiyahMiniOptimizerState* opt,const NiyahMiniConfig* cfg){
    if(!opt||!cfg) return NIYAH_ERR_INVALID_ARG; size_t n=weights_n_floats(cfg);
    opt->m=(float*)calloc(n,sizeof(float)); opt->v=(float*)calloc(n,sizeof(float));
    if(!opt->m||!opt->v){free(opt->m);free(opt->v);opt->m=opt->v=NULL;return NIYAH_ERR_OUT_OF_MEMORY;}
    opt->t=0; opt->n=n; return NIYAH_OK;
}
void niyah_mini_optim_free(NiyahMiniOptimizerState* opt){if(!opt)return;free(opt->m);free(opt->v);opt->m=opt->v=NULL;opt->n=0;}

static int tensor_list(NiyahMiniModel* m,NiyahMiniGrads* g,float** wp,float** gp,size_t* sz,int cap){
    NiyahMiniWeights* W=&m->weights; const NiyahMiniConfig* C=&m->config;
    size_t dim=C->n_dim,vocab=C->n_vocab,hd=dim/C->n_heads,kv=(size_t)C->n_kv_heads*hd,ff=C->n_ff; int i=0;
    wp[i]=W->embedding;gp[i]=g->embedding;sz[i]=vocab*dim;i++;
    for(int32_t l=0;l<C->n_layers;l++){NiyahMiniLayerWeights* lw=&W->layers[l];NiyahMiniLayerGrads* lg=&g->layers[l];
        wp[i]=lw->attn_norm;gp[i]=lg->attn_norm;sz[i]=dim;i++;
        wp[i]=lw->wq;gp[i]=lg->wq;sz[i]=dim*dim;i++;
        wp[i]=lw->wk;gp[i]=lg->wk;sz[i]=kv*dim;i++;
        wp[i]=lw->wv;gp[i]=lg->wv;sz[i]=kv*dim;i++;
        wp[i]=lw->wo;gp[i]=lg->wo;sz[i]=dim*dim;i++;
        wp[i]=lw->ffn_norm;gp[i]=lg->ffn_norm;sz[i]=dim;i++;
        wp[i]=lw->ffn_gate;gp[i]=lg->ffn_gate;sz[i]=ff*dim;i++;
        wp[i]=lw->ffn_up;gp[i]=lg->ffn_up;sz[i]=ff*dim;i++;
        wp[i]=lw->ffn_down;gp[i]=lg->ffn_down;sz[i]=dim*ff;i++;}
    wp[i]=W->final_norm;gp[i]=g->final_norm;sz[i]=dim;i++;
    (void)cap; return i;
}

NiyahStatus niyah_mini_cache_allocate(NiyahMiniTrainCache* cache,const NiyahMiniConfig* cfg,int32_t seq_len){
    int i;
    if(!cache||!cfg||seq_len<=0) return NIYAH_ERR_INVALID_ARG;
    memset(cache,0,sizeof(*cache));
    size_t T=seq_len,D=cfg->n_dim,L=cfg->n_layers,H=cfg->n_heads,hd=D/H,kv=(size_t)cfg->n_kv_heads*hd,ff=cfg->n_ff;
    size_t per=T*D+T*D+T+T*D+T*kv+T*kv+H*T*T+T*D+T*D+T*D+T*D+T+T*ff+T*ff+T*ff+T*D;
    size_t total=L*per+T*D+T*D+T;
    void* mem=malloc(total*sizeof(float)); if(!mem) return NIYAH_ERR_OUT_OF_MEMORY;
    cache->memory_block=mem; cache->memory_size=total*sizeof(float); cache->seq_len=seq_len;
    float** a[16];
    for(i=0;i<16;i++){
        a[i]=(float**)malloc(L*sizeof(float*));
        if(!a[i]){int j; for(j=0;j<i;j++) free(a[j]); free(mem); memset(cache,0,sizeof(*cache)); return NIYAH_ERR_OUT_OF_MEMORY;}
    }
    cache->x=a[0];cache->h1=a[1];cache->r1=a[2];cache->q=a[3];cache->k=a[4];cache->v=a[5];cache->probs=a[6];
    cache->attn_in=a[7];cache->ao=a[8];cache->res=a[9];cache->h2=a[10];cache->r2=a[11];cache->gate=a[12];
    cache->up=a[13];cache->ff=a[14];cache->fo=a[15];
    float* p=(float*)mem;
    for(size_t l=0;l<L;l++){
        cache->x[l]=p;p+=T*D;cache->h1[l]=p;p+=T*D;cache->r1[l]=p;p+=T;
        cache->q[l]=p;p+=T*D;cache->k[l]=p;p+=T*kv;cache->v[l]=p;p+=T*kv;
        cache->probs[l]=p;p+=H*T*T;cache->attn_in[l]=p;p+=T*D;cache->ao[l]=p;p+=T*D;
        cache->res[l]=p;p+=T*D;cache->h2[l]=p;p+=T*D;cache->r2[l]=p;p+=T;
        cache->gate[l]=p;p+=T*ff;cache->up[l]=p;p+=T*ff;cache->ff[l]=p;p+=T*ff;cache->fo[l]=p;p+=T*D;}
    cache->final_pre=p;p+=T*D;cache->final_h=p;p+=T*D;cache->r_final=p;p+=T;
    return NIYAH_OK;
}
void niyah_mini_cache_free(NiyahMiniTrainCache* cache){
    if(!cache) return;
    float** a[16]={cache->x,cache->h1,cache->r1,cache->q,cache->k,cache->v,cache->probs,cache->attn_in,
                   cache->ao,cache->res,cache->h2,cache->r2,cache->gate,cache->up,cache->ff,cache->fo};
    for(int i=0;i<16;i++) free(a[i]);
    free(cache->memory_block); memset(cache,0,sizeof(*cache));
}

NiyahStatus niyah_mini_train_forward(NiyahMiniModel* model,NiyahMiniTrainCache* cache,const int32_t* input_ids,int32_t seq_len,float* logits_out){
    if(!model||!cache||!input_ids||!logits_out||seq_len<=0) return NIYAH_ERR_INVALID_ARG;
    if(seq_len>model->config.n_ctx) return NIYAH_ERR_INVALID_ARG;
    const NiyahMiniConfig* C=&model->config;
    const int32_t D=C->n_dim,H=C->n_heads,hd=D/H,KV=C->n_kv_heads,kvd=KV*hd,FF=C->n_ff,V=C->n_vocab,T=seq_len,L=C->n_layers;
    const float eps=C->norm_eps,theta=C->rope_theta,scale=1.0f/sqrtf((float)hd);
    NiyahMiniWeights* W=&model->weights; const float* emb=W->embedding;
    for(int32_t t=0;t<T;t++) memcpy(cache->x[0]+(size_t)t*D, emb+(size_t)input_ids[t]*D, sizeof(float)*D);
    for(int32_t l=0;l<L;l++){
        NiyahMiniLayerWeights* lw=&W->layers[l];
        float* xL=(l==0)?cache->x[0]:cache->fo[l-1]; if(l>0) cache->x[l]=cache->fo[l-1];
        float* h1=cache->h1[l]; float* r1=cache->r1[l]; float* qL=cache->q[l]; float* kL=cache->k[l]; float* vL=cache->v[l];
        for(int32_t t=0;t<T;t++){
            const float* xt=xL+(size_t)t*D; float* h1t=h1+(size_t)t*D;
            rmsnorm_fwd(xt,lw->attn_norm,D,eps,h1t,&r1[t]);
            matvec(lw->wq,h1t,D,D,qL+(size_t)t*D); matvec(lw->wk,h1t,kvd,D,kL+(size_t)t*kvd); matvec(lw->wv,h1t,kvd,D,vL+(size_t)t*kvd);
            float* cv=(float*)malloc((size_t)hd*sizeof(float));
            float* sv=(float*)malloc((size_t)hd*sizeof(float));
            if(!cv||!sv){free(cv);free(sv);return NIYAH_ERR_OUT_OF_MEMORY;}
            rope_cs(t,hd,theta,cv,sv);
            for(int32_t h=0;h<H;h++) rope_fwd_v(qL+(size_t)t*D+h*hd,hd,cv,sv);
            for(int32_t h=0;h<KV;h++) rope_fwd_v(kL+(size_t)t*kvd+h*hd,hd,cv,sv);
            free(cv); free(sv);
        }
        float* probs=cache->probs[l]; float* ain=cache->attn_in[l];
        for(int32_t h=0;h<H;h++){int32_t kvh=h%KV;
            for(int32_t t=0;t<T;t++){
                const float* qt=qL+(size_t)t*D+h*hd; float* pr=probs+((size_t)h*T+t)*T; float mx=-1e30f;
                for(int32_t s=0;s<=t;s++){float dot=0;const float* ks=kL+(size_t)s*kvd+kvh*hd; for(int32_t d=0;d<hd;d++) dot+=qt[d]*ks[d]; float sc=dot*scale; pr[s]=sc; if(sc>mx)mx=sc;}
                float sm=0; for(int32_t s=0;s<=t;s++){float e=expf(pr[s]-mx);pr[s]=e;sm+=e;}
                float inv=(sm>0)?1.0f/sm:0; for(int32_t s=0;s<=t;s++) pr[s]*=inv; for(int32_t s=t+1;s<T;s++) pr[s]=0;
                float* oh=ain+(size_t)t*D+h*hd; for(int32_t d=0;d<hd;d++) oh[d]=0;
                for(int32_t s=0;s<=t;s++){float pp=pr[s]; if(pp==0)continue; const float* vs=vL+(size_t)s*kvd+kvh*hd; for(int32_t d=0;d<hd;d++) oh[d]+=pp*vs[d];}
            }
        }
        float* ao=cache->ao[l]; float* res=cache->res[l];
        for(int32_t t=0;t<T;t++){matvec(lw->wo,ain+(size_t)t*D,D,D,ao+(size_t)t*D); const float* xt=xL+(size_t)t*D; float* rt=res+(size_t)t*D; for(int32_t d=0;d<D;d++) rt[d]=xt[d]+ao[(size_t)t*D+d];}
        float* h2=cache->h2[l]; float* r2=cache->r2[l]; float* gate=cache->gate[l]; float* up=cache->up[l]; float* ff=cache->ff[l]; float* fo=cache->fo[l];
        for(int32_t t=0;t<T;t++){
            const float* rt=res+(size_t)t*D; float* h2t=h2+(size_t)t*D; rmsnorm_fwd(rt,lw->ffn_norm,D,eps,h2t,&r2[t]);
            matvec(lw->ffn_gate,h2t,FF,D,gate+(size_t)t*FF); matvec(lw->ffn_up,h2t,FF,D,up+(size_t)t*FF);
            float* fft=ff+(size_t)t*FF; for(int32_t i=0;i<FF;i++){float g=gate[(size_t)t*FF+i];float sig=1.0f/(1.0f+expf(-g));fft[i]=sig*g*up[(size_t)t*FF+i];}
            matvec(lw->ffn_down,fft,D,FF,fo+(size_t)t*D); float* fot=fo+(size_t)t*D; for(int32_t d=0;d<D;d++) fot[d]+=rt[d];
        }
    }
    float* fh=cache->final_h; float* rf=cache->r_final; float* last_out=(L>0)?cache->fo[L-1]:cache->x[0];
    for(int32_t t=0;t<T;t++){rmsnorm_fwd(last_out+(size_t)t*D,W->final_norm,D,eps,fh+(size_t)t*D,&rf[t]); memcpy(cache->final_pre+(size_t)t*D,last_out+(size_t)t*D,sizeof(float)*D);}
    for(int32_t t=0;t<T;t++) matvec(emb,fh+(size_t)t*D,V,D,logits_out+(size_t)t*V);
    return NIYAH_OK;
}

float niyah_mini_loss_and_dlogits(const float* logits,const int32_t* targets,int32_t seq_len,int32_t vocab,float* dl){
    float loss=0; int32_t n=seq_len;
    for(int32_t t=0;t<seq_len;t++){
        const float* lg=logits+(size_t)t*vocab; float* d=dl+(size_t)t*vocab;
        float mx=lg[0]; for(int32_t i=1;i<vocab;i++) if(lg[i]>mx)mx=lg[i];
        float sm=0; for(int32_t i=0;i<vocab;i++){d[i]=expf(lg[i]-mx);sm+=d[i];}
        float inv=(sm>0)?1.0f/sm:0; for(int32_t i=0;i<vocab;i++) d[i]*=inv;
        loss-=logf(d[targets[t]]+1e-30f);
        for(int32_t i=0;i<vocab;i++) d[i]=(d[i]-(i==targets[t]?1.0f:0.0f))/(float)n;
    }
    return loss/(float)n;
}

NiyahStatus niyah_mini_train_backward(NiyahMiniModel* model,NiyahMiniGrads* grads,NiyahMiniTrainCache* cache,const int32_t* input_ids,const float* dlogits){
    if(!model||!grads||!cache||!input_ids||!dlogits) return NIYAH_ERR_INVALID_ARG;
    const NiyahMiniConfig* C=&model->config;
    const int32_t D=C->n_dim,H=C->n_heads,hd=D/H,KV=C->n_kv_heads,kvd=KV*hd,FF=C->n_ff,V=C->n_vocab,T=cache->seq_len,L=C->n_layers;
    const float theta=C->rope_theta,scale=1.0f/sqrtf((float)hd);
    NiyahMiniWeights* W=&model->weights; const float* emb=W->embedding;
    niyah_mini_grads_zero(grads,C);
    float* dx=malloc((size_t)T*D*sizeof(float));
    float* dpre=malloc((size_t)T*D*sizeof(float));
    float* dwr=malloc((size_t)D*sizeof(float));
    float* d_res=malloc((size_t)T*D*sizeof(float));
    float* d_res2=malloc((size_t)T*D*sizeof(float));
    float* d_ao=malloc((size_t)T*D*sizeof(float));
    float* d_ain=malloc((size_t)T*D*sizeof(float));
    float* d_q=malloc((size_t)T*D*sizeof(float));
    float* d_k=malloc((size_t)T*kvd*sizeof(float));
    float* d_v=malloc((size_t)T*kvd*sizeof(float));
    float* d_h1=malloc((size_t)T*D*sizeof(float));
    float* d_h2=malloc((size_t)T*D*sizeof(float));
    float* d_xa=malloc((size_t)T*D*sizeof(float));
    float* d_ff=malloc((size_t)T*FF*sizeof(float));
    float* fo_raw=malloc((size_t)T*D*sizeof(float));
    float* gprobs=malloc((size_t)T*sizeof(float));
    if(!dx||!dpre||!dwr||!d_res||!d_res2||!d_ao||!d_ain||!d_q||!d_k||!d_v||!d_h1||!d_h2||!d_xa||!d_ff||!fo_raw||!gprobs){
        free(dx);free(dpre);free(dwr);free(d_res);free(d_res2);free(d_ao);free(d_ain);free(d_q);free(d_k);free(d_v);
        free(d_h1);free(d_h2);free(d_xa);free(d_ff);free(fo_raw);free(gprobs); return NIYAH_ERR_OUT_OF_MEMORY;
    }
    for(int32_t t=0;t<T;t++){
        const float* dl=dlogits+(size_t)t*V; const float* fh=cache->final_h+(size_t)t*D; float* dxt=dx+(size_t)t*D;
        for(int32_t v=0;v<V;v++){float dlv=dl[v]; if(dlv==0)continue; float* ge=grads->embedding+(size_t)v*D; for(int32_t d=0;d<D;d++) ge[d]+=dlv*fh[d];}
        for(int32_t d=0;d<D;d++){float s=0; for(int32_t v=0;v<V;v++) s+=dl[v]*emb[(size_t)v*D+d]; dxt[d]=s;}
    }
    for(int32_t d=0;d<D;d++) grads->final_norm[d]=0;
    for(int32_t t=0;t<T;t++){ rmsnorm_bwd(dx+(size_t)t*D,cache->final_pre+(size_t)t*D,W->final_norm,D,cache->r_final[t],dpre+(size_t)t*D,dwr); for(int32_t d=0;d<D;d++) grads->final_norm[d]+=dwr[d]; }
    memcpy(dx,dpre,(size_t)T*D*sizeof(float));

    for(int32_t l=L-1;l>=0;l--){
        NiyahMiniLayerWeights* lw=&W->layers[l]; NiyahMiniLayerGrads* gl=&grads->layers[l];
        float* xL=cache->x[l]; float* h1=cache->h1[l]; float* res=cache->res[l]; float* h2=cache->h2[l];
        float* gate=cache->gate[l]; float* up=cache->up[l]; float* ff=cache->ff[l]; float* ain=cache->attn_in[l];
        float* probs=cache->probs[l]; float* qL=cache->q[l]; float* kL=cache->k[l]; float* vL=cache->v[l];
        for(int32_t t=0;t<T;t++){ for(int32_t d=0;d<D;d++) d_res[(size_t)t*D+d]=dx[(size_t)t*D+d]; matvec(lw->ffn_down,ff+(size_t)t*FF,D,FF,fo_raw+(size_t)t*D); }
        for(int32_t t=0;t<T;t++){
            const float* dfo=dx+(size_t)t*D; const float* fft=ff+(size_t)t*FF; float* dfft=d_ff+(size_t)t*FF;
            for(int32_t j=0;j<D;j++){float dj=dfo[j]; float* gw=gl->ffn_down+(size_t)j*FF; for(int32_t k=0;k<FF;k++) gw[k]+=dj*fft[k];}
            for(int32_t k=0;k<FF;k++){float s=0; for(int32_t j=0;j<D;j++) s+=dfo[j]*lw->ffn_down[(size_t)j*FF+k]; dfft[k]=s;}
        }
        for(int32_t t=0;t<T;t++){
            const float* dfft=d_ff+(size_t)t*FF; const float* gt=gate+(size_t)t*FF; const float* upt=up+(size_t)t*FF; const float* h2t=h2+(size_t)t*D;
            float* dh2t=d_h2+(size_t)t*D;
            for(int32_t i=0;i<FF;i++){float g=gt[i];float sig=1.0f/(1.0f+expf(-g));
                float dgt=dfft[i]*upt[i]*sig*(1.0f+g*(1.0f-sig)); float dupt=dfft[i]*sig*g;
                float* gg=gl->ffn_gate+(size_t)i*D; float* gu=gl->ffn_up+(size_t)i*D;
                for(int32_t k=0;k<D;k++){gg[k]+=dgt*h2t[k]; gu[k]+=dupt*h2t[k];}}
            for(int32_t k=0;k<D;k++){float s=0;
                for(int32_t i=0;i<FF;i++){float g=gt[i];float sig=1.0f/(1.0f+expf(-g)); float dgt=dfft[i]*upt[i]*sig*(1.0f+g*(1.0f-sig)); float dupt=dfft[i]*sig*g;
                    s+=dgt*lw->ffn_gate[(size_t)i*D+k]+dupt*lw->ffn_up[(size_t)i*D+k];}
                dh2t[k]=s;}
        }
        for(int32_t d=0;d<D;d++) gl->ffn_norm[d]=0;
        for(int32_t t=0;t<T;t++){ rmsnorm_bwd(d_h2+(size_t)t*D,res+(size_t)t*D,lw->ffn_norm,D,cache->r2[t],d_res2+(size_t)t*D,dwr); for(int32_t d=0;d<D;d++) gl->ffn_norm[d]+=dwr[d]; for(int32_t d=0;d<D;d++) d_res[(size_t)t*D+d]+=d_res2[(size_t)t*D+d]; }
        for(int32_t t=0;t<T;t++) for(int32_t d=0;d<D;d++) d_ao[(size_t)t*D+d]=d_res[(size_t)t*D+d];
        for(int32_t t=0;t<T;t++){
            const float* dao=d_ao+(size_t)t*D; const float* aint=ain+(size_t)t*D; float* daint=d_ain+(size_t)t*D;
            for(int32_t j=0;j<D;j++){float dj=dao[j]; float* gw=gl->wo+(size_t)j*D; for(int32_t k=0;k<D;k++) gw[k]+=dj*aint[k];}
            for(int32_t k=0;k<D;k++){float s=0; for(int32_t j=0;j<D;j++) s+=dao[j]*lw->wo[(size_t)j*D+k]; daint[k]=s;}
        }
        memset(d_q,0,(size_t)T*D*sizeof(float)); memset(d_k,0,(size_t)T*kvd*sizeof(float)); memset(d_v,0,(size_t)T*kvd*sizeof(float));
        for(int32_t h=0;h<H;h++){int32_t kvh=h%KV;
            for(int32_t t=0;t<T;t++){
                const float* dout=d_ain+(size_t)t*D+h*hd; const float* pr=probs+((size_t)h*T+t)*T; float sum_pg=0;
                for(int32_t s=0;s<=t;s++){float dot=0; const float* vs=vL+(size_t)s*kvd+kvh*hd; for(int32_t d=0;d<hd;d++) dot+=dout[d]*vs[d]; gprobs[s]=dot; sum_pg+=pr[s]*dot;}
                for(int32_t s=0;s<=t;s++){ float dA=pr[s]*(gprobs[s]-sum_pg);
                    const float* ks=kL+(size_t)s*kvd+kvh*hd; const float* qt=qL+(size_t)t*D+h*hd;
                    float* dqt=d_q+(size_t)t*D+h*hd; float* dkst=d_k+(size_t)s*kvd+kvh*hd; float* dvst=d_v+(size_t)s*kvd+kvh*hd;
                    for(int32_t d=0;d<hd;d++){dqt[d]+=dA*ks[d]*scale; dkst[d]+=dA*qt[d]*scale;}
                    for(int32_t d=0;d<hd;d++) dvst[d]+=pr[s]*dout[d]; }
            }
        }
        for(int32_t t=0;t<T;t++){float* cv=(float*)malloc((size_t)hd*sizeof(float));float* sv=(float*)malloc((size_t)hd*sizeof(float));if(!cv||!sv){free(cv);free(sv);return NIYAH_ERR_OUT_OF_MEMORY;} rope_cs(t,hd,theta,cv,sv); for(int32_t h=0;h<H;h++) rope_bwd_v(d_q+(size_t)t*D+h*hd,hd,cv,sv); for(int32_t h=0;h<KV;h++) rope_bwd_v(d_k+(size_t)t*kvd+h*hd,hd,cv,sv); free(cv);free(sv);}
        for(int32_t t=0;t<T;t++){
            const float* h1t=h1+(size_t)t*D; const float* dqt=d_q+(size_t)t*D; const float* dkt=d_k+(size_t)t*kvd; const float* dvt=d_v+(size_t)t*kvd; float* dh1t=d_h1+(size_t)t*D;
            for(int32_t j=0;j<D;j++){float dj=dqt[j]; float* gw=gl->wq+(size_t)j*D; for(int32_t k=0;k<D;k++) gw[k]+=dj*h1t[k];}
            for(int32_t j=0;j<kvd;j++){float dj=dkt[j]; float* gw=gl->wk+(size_t)j*D; for(int32_t k=0;k<D;k++) gw[k]+=dj*h1t[k];}
            for(int32_t j=0;j<kvd;j++){float dj=dvt[j]; float* gw=gl->wv+(size_t)j*D; for(int32_t k=0;k<D;k++) gw[k]+=dj*h1t[k];}
            for(int32_t k=0;k<D;k++){float s=0; for(int32_t j=0;j<D;j++) s+=dqt[j]*lw->wq[(size_t)j*D+k]; for(int32_t j=0;j<kvd;j++) s+=dkt[j]*lw->wk[(size_t)j*D+k]+dvt[j]*lw->wv[(size_t)j*D+k]; dh1t[k]=s;}
        }
        for(int32_t d=0;d<D;d++) gl->attn_norm[d]=0;
        for(int32_t t=0;t<T;t++){ rmsnorm_bwd(d_h1+(size_t)t*D,xL+(size_t)t*D,lw->attn_norm,D,cache->r1[t],d_xa+(size_t)t*D,dwr); for(int32_t d=0;d<D;d++) gl->attn_norm[d]+=dwr[d]; for(int32_t d=0;d<D;d++) dx[(size_t)t*D+d]=d_res[(size_t)t*D+d]+d_xa[(size_t)t*D+d]; }
    }
    for(int32_t t=0;t<T;t++){int32_t id=input_ids[t]; float* ge=grads->embedding+(size_t)id*D; for(int32_t d=0;d<D;d++) ge[d]+=dx[(size_t)t*D+d];}
    free(dx);free(dpre);free(dwr);free(d_res);free(d_res2);free(d_ao);free(d_ain);free(d_q);free(d_k);free(d_v);
    free(d_h1);free(d_h2);free(d_xa);free(d_ff);free(fo_raw);free(gprobs);
    return NIYAH_OK;
}

float niyah_mini_clip_grads(NiyahMiniGrads* grads,const NiyahMiniConfig* cfg,float max_norm){
    size_t dim=cfg->n_dim,vocab=cfg->n_vocab,L=cfg->n_layers,hd=dim/cfg->n_heads,kv=(size_t)cfg->n_kv_heads*hd,ff=cfg->n_ff;
    float* ptrs[4096]; size_t sizes[4096]; int i=0;
    ptrs[i]=grads->embedding; sizes[i]=vocab*dim; i++;
    for(size_t l=0;l<L;l++){NiyahMiniLayerGrads* gl=&grads->layers[l];
        ptrs[i]=gl->attn_norm;sizes[i]=dim;i++; ptrs[i]=gl->wq;sizes[i]=dim*dim;i++; ptrs[i]=gl->wk;sizes[i]=kv*dim;i++;
        ptrs[i]=gl->wv;sizes[i]=kv*dim;i++; ptrs[i]=gl->wo;sizes[i]=dim*dim;i++; ptrs[i]=gl->ffn_norm;sizes[i]=dim;i++;
        ptrs[i]=gl->ffn_gate;sizes[i]=ff*dim;i++; ptrs[i]=gl->ffn_up;sizes[i]=ff*dim;i++; ptrs[i]=gl->ffn_down;sizes[i]=dim*ff;i++;}
    ptrs[i]=grads->final_norm; sizes[i]=dim; i++;
    double total=0;
    for(int t=0;t<i;t++){float* g=ptrs[t]; for(size_t k=0;k<sizes[t];k++) total+=(double)g[k]*(double)g[k];}
    float norm=(float)sqrt(total);
    if(max_norm>0 && norm>max_norm && norm>0){float s=max_norm/norm; for(int t=0;t<i;t++){float* g=ptrs[t]; for(size_t k=0;k<sizes[t];k++) g[k]*=s;}}
    return norm;
}

NiyahStatus niyah_mini_step_adamw(NiyahMiniModel* model,NiyahMiniGrads* grads,NiyahMiniOptimizerState* opt,
                                  const NiyahMiniConfig* cfg,float lr,float beta1,float beta2,float eps,float weight_decay){
    if(!model||!grads||!opt||!cfg) return NIYAH_ERR_INVALID_ARG;
    int cap=2+9*cfg->n_layers;
    float** wp=malloc(cap*sizeof(float*)); float** gp=malloc(cap*sizeof(float*)); size_t* sz=malloc(cap*sizeof(size_t));
    if(!wp||!gp||!sz){free(wp);free(gp);free(sz);return NIYAH_ERR_OUT_OF_MEMORY;}
    int n=tensor_list(model,grads,wp,gp,sz,cap);
    opt->t++; int64_t tt=opt->t;
    float bc1=1.0f-(float)pow(beta1,(double)tt), bc2=1.0f-(float)pow(beta2,(double)tt);
    size_t off=0;
    for(int t=0;t<n;t++){
        float* w=wp[t]; float* g=gp[t]; size_t S=sz[t]; float* m=opt->m+off; float* v=opt->v+off;
        for(size_t k=0;k<S;k++){
            float gk=g[k];
            m[k]=beta1*m[k]+(1.0f-beta1)*gk;
            v[k]=beta2*v[k]+(1.0f-beta2)*gk*gk;
            float mhat=m[k]/bc1, vhat=v[k]/bc2;
            w[k]-=lr*(mhat/(sqrtf(vhat)+eps)+weight_decay*w[k]);
        }
        off+=S;
    }
    free(wp);free(gp);free(sz);
    return NIYAH_OK;
}

NiyahStatus niyah_mini_step_sgd(NiyahMiniModel* model,NiyahMiniGrads* grads,NiyahMiniOptimizerState* opt,
                                const NiyahMiniConfig* cfg,float lr,float momentum,float weight_decay){
    if(!model||!grads||!opt||!cfg) return NIYAH_ERR_INVALID_ARG;
    int cap=2+9*cfg->n_layers;
    float** wp=malloc(cap*sizeof(float*)); float** gp=malloc(cap*sizeof(float*)); size_t* sz=malloc(cap*sizeof(size_t));
    if(!wp||!gp||!sz){free(wp);free(gp);free(sz);return NIYAH_ERR_OUT_OF_MEMORY;}
    int n=tensor_list(model,grads,wp,gp,sz,cap);
    size_t off=0;
    for(int t=0;t<n;t++){
        float* w=wp[t]; float* g=gp[t]; size_t S=sz[t]; float* m=opt->m+off;
        for(size_t k=0;k<S;k++){ m[k]=momentum*m[k]+g[k]; w[k]-=lr*(m[k]+weight_decay*w[k]); }
        off+=S;
    }
    free(wp);free(gp);free(sz);
    return NIYAH_OK;
}

NiyahStatus niyah_mini_grad_check(NiyahMiniModel* model,const NiyahMiniConfig* cfg,
                                  const int32_t* input_ids,int32_t seq_len,
                                  float* max_rel_out,int32_t* n_checked_out){
    if(!model||!cfg||!input_ids||seq_len<=0) return NIYAH_ERR_INVALID_ARG;
    NiyahMiniGrads grads; NiyahStatus st=niyah_mini_grads_allocate(&grads,cfg); if(st!=NIYAH_OK) return st;
    NiyahMiniTrainCache cache; st=niyah_mini_cache_allocate(&cache,cfg,seq_len); if(st!=NIYAH_OK){niyah_mini_grads_free(&grads);return st;}
    size_t ll=(size_t)seq_len*cfg->n_vocab;
    float* logits=malloc(ll*sizeof(float)); float* dlog=malloc(ll*sizeof(float)); int32_t* targets=malloc((size_t)seq_len*sizeof(int32_t));
    if(!logits||!dlog||!targets){free(logits);free(dlog);free(targets);niyah_mini_cache_free(&cache);niyah_mini_grads_free(&grads);return NIYAH_ERR_OUT_OF_MEMORY;}
    for(int32_t t=0;t<seq_len;t++) targets[t]=input_ids[t];
    st=niyah_mini_train_forward(model,&cache,input_ids,seq_len,logits); if(st!=NIYAH_OK) goto done;
    niyah_mini_loss_and_dlogits(logits,targets,seq_len,cfg->n_vocab,dlog);
    st=niyah_mini_train_backward(model,&grads,&cache,input_ids,dlog); if(st!=NIYAH_OK) goto done;
    {
        int cap=2+9*cfg->n_layers;
        float** wp=malloc(cap*sizeof(float*)); float** gp=malloc(cap*sizeof(float*)); size_t* sz=malloc(cap*sizeof(size_t));
        if(!wp||!gp||!sz){free(wp);free(gp);free(sz);st=NIYAH_ERR_OUT_OF_MEMORY;goto done;}
        int n=tensor_list(model,&grads,wp,gp,sz,cap);
        float max_rel=0; int32_t checked=0; unsigned int rng=0x1234567u; float epsfd=1e-3f;
        for(int t=0;t<n;t++){
            size_t S=sz[t]; if(S==0) continue;
            int per=(S<6)?(int)S:6;
            for(int c=0;c<per;c++){
                rng=rng*1664525u+1013904223u; size_t idx=((size_t)(rng>>8))%S;
                float* wptr=wp[t]; float* gptr=gp[t]; float orig=wptr[idx];
                wptr[idx]=orig+epsfd; st=niyah_mini_train_forward(model,&cache,input_ids,seq_len,logits); if(st!=NIYAH_OK){free(wp);free(gp);free(sz);goto done;}
                float lp=niyah_mini_loss_and_dlogits(logits,targets,seq_len,cfg->n_vocab,dlog);
                wptr[idx]=orig-epsfd; st=niyah_mini_train_forward(model,&cache,input_ids,seq_len,logits); if(st!=NIYAH_OK){free(wp);free(gp);free(sz);goto done;}
                float lm=niyah_mini_loss_and_dlogits(logits,targets,seq_len,cfg->n_vocab,dlog);
                wptr[idx]=orig;
                float num=(lp-lm)/(2.0f*epsfd), ana=gptr[idx];
                float rel=fabsf(num-ana)/(fabsf(num)+fabsf(ana)+1e-12f);
                if(rel>max_rel) max_rel=rel; checked++;
            }
        }
        free(wp);free(gp);free(sz);
        if(max_rel_out)*max_rel_out=max_rel; if(n_checked_out)*n_checked_out=checked;
    }
done:
    free(logits);free(dlog);free(targets);
    niyah_mini_cache_free(&cache); niyah_mini_grads_free(&grads);
    return st;
}
