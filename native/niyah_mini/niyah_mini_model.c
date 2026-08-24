#include "niyah_mini_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ==========================================================================
 * NiyahMini Model Implementation
 * 
 * Original transformer implementation with:
 * - Multi-head self-attention with RoPE
 * - Feed-forward network with SwiGLU
 * - Pre-norm architecture
 * - Grouped-Query Attention
 * - Residual connections
 * 
 * NO borrowed code from any existing model implementation.
 * ========================================================================== */

/* ==========================================================================
 * Weight Initialization
 * ========================================================================== */

/* Xavier/Glorot initialization */
static void init_xavier(float* weights, size_t n_in, size_t n_out) {
    const float scale = sqrtf(2.0f / (float)(n_in + n_out));
    for (size_t i = 0; i < n_in * n_out; i++) {
        /* Simple deterministic initialization for reproducibility */
        float val = ((float)(i % 1000) / 1000.0f - 0.5f) * 2.0f * scale;
        weights[i] = val;
    }
}

/* Small random initialization */
static void init_small(float* weights, size_t count, float scale) {
    for (size_t i = 0; i < count; i++) {
        float val = ((float)(i % 1000) / 1000.0f - 0.5f) * 2.0f * scale;
        weights[i] = val;
    }
}

/* ==========================================================================
 * Model Initialization
 * ========================================================================== */

NiyahStatus niyah_mini_model_init(
    NiyahMiniModel* model,
    const NiyahMiniConfig* config
) {
    if (!model || !config) {
        return NIYAH_ERR_INVALID_ARG;
    }
    
    /* Validate config */
    NiyahStatus status = niyah_mini_config_validate(config);
    if (status != NIYAH_OK) {
        return status;
    }
    
    /* Copy config */
    model->config = *config;
    
    /* Initialize weights */
    memset(&model->weights, 0, sizeof(model->weights));
    
    /* Allocate weights */
    status = niyah_mini_weights_allocate(&model->weights, config);
    if (status != NIYAH_OK) {
        return status;
    }
    
    /* Initialize weights with small values */
    niyah_mini_weights_init_small(&model->weights, config, 0.02f);
    
    /* Initialize runtime */
    model->runtime = NULL;
    
    /* Initialize KV cache */
    model->kv_cache_k = NULL;
    model->kv_cache_v = NULL;
    model->kv_cache_seq_len = 0;
    
    /* Initialize scratch */
    model->scratch = NULL;
    model->scratch_size = 0;
    
    return NIYAH_OK;
}

/* ==========================================================================
 * Weight Allocation
 * ========================================================================== */

size_t niyah_mini_weights_memory_size(const NiyahMiniConfig* config) {
    if (!config) return 0;
    
    const size_t dim = (size_t)config->n_dim;
    const size_t vocab = (size_t)config->n_vocab;
    const size_t n_layers = (size_t)config->n_layers;
    const size_t n_heads = (size_t)config->n_heads;
    const size_t n_kv_heads = (size_t)config->n_kv_heads;
    const size_t head_dim = dim / n_heads;
    const size_t kv_dim = n_kv_heads * head_dim;
    const size_t n_ff = (size_t)config->n_ff;
    
    size_t total = 0;
    
    /* Embedding: vocab * dim */
    total += vocab * dim;
    
    /* Per layer: */
    /* - attn_norm: dim */
    /* - wq: dim * dim */
    /* - wk: kv_dim * dim */
    /* - wv: kv_dim * dim */
    /* - wo: dim * dim */
    /* - ffn_norm: dim */
    /* - ffn_gate: n_ff * dim */
    /* - ffn_up: n_ff * dim */
    /* - ffn_down: dim * n_ff */
    const size_t per_layer = dim + (dim * dim) + (kv_dim * dim) + (kv_dim * dim) + (dim * dim) + dim + (n_ff * dim) + (n_ff * dim) + (dim * n_ff);
    total += n_layers * per_layer;
    
    /* Final norm: dim */
    total += dim;
    
    /* LM head: vocab * dim (unless tied) */
    if (!config->tie_word_embeddings) {
        total += vocab * dim;
    }
    
    return total * sizeof(float);
}

NiyahStatus niyah_mini_weights_allocate(
    NiyahMiniWeights* weights,
    const NiyahMiniConfig* config
) {
    if (!weights || !config) {
        return NIYAH_ERR_INVALID_ARG;
    }
    
    /* Free existing memory */
    niyah_mini_weights_free(weights);
    
    /* Calculate total memory needed */
    size_t total_size = niyah_mini_weights_memory_size(config);
    
    /* Allocate single block */
    void* memory_block = malloc(total_size);
    if (!memory_block) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    
    /* Setup weights structure */
    weights->memory_block = memory_block;
    weights->memory_size = total_size;
    weights->owns_memory = true;
    
    /* Assign pointers */
    float* ptr = (float*)memory_block;
    
    const size_t dim = (size_t)config->n_dim;
    const size_t vocab = (size_t)config->n_vocab;
    const size_t n_layers = (size_t)config->n_layers;
    const size_t n_heads = (size_t)config->n_heads;
    const size_t n_kv_heads = (size_t)config->n_kv_heads;
    const size_t head_dim = dim / n_heads;
    const size_t kv_dim = n_kv_heads * head_dim;
    const size_t n_ff = (size_t)config->n_ff;
    
    /* Embedding */
    weights->embedding = ptr;
    ptr += vocab * dim;
    
    /* Layers */
    weights->layers = (NiyahMiniLayerWeights*)malloc(n_layers * sizeof(NiyahMiniLayerWeights));
    if (!weights->layers) {
        free(memory_block);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    
    for (size_t l = 0; l < n_layers; l++) {
        /* Attention norm */
        weights->layers[l].attn_norm = ptr;
        ptr += dim;
        
        /* Query projection */
        weights->layers[l].wq = ptr;
        ptr += dim * dim;
        
        /* Key projection */
        weights->layers[l].wk = ptr;
        ptr += kv_dim * dim;
        
        /* Value projection */
        weights->layers[l].wv = ptr;
        ptr += kv_dim * dim;
        
        /* Output projection */
        weights->layers[l].wo = ptr;
        ptr += dim * dim;
        
        /* FFN norm */
        weights->layers[l].ffn_norm = ptr;
        ptr += dim;
        
        /* FFN gate */
        weights->layers[l].ffn_gate = ptr;
        ptr += n_ff * dim;
        
        /* FFN up */
        weights->layers[l].ffn_up = ptr;
        ptr += n_ff * dim;
        
        /* FFN down */
        weights->layers[l].ffn_down = ptr;
        ptr += dim * n_ff;
    }
    
    weights->n_layers = (int32_t)n_layers;
    
    /* Final norm */
    weights->final_norm = ptr;
    ptr += dim;
    
    /* LM head (optional) */
    if (!config->tie_word_embeddings) {
        weights->lm_head = ptr;
        ptr += vocab * dim;
    } else {
        weights->lm_head = weights->embedding;  /* Tie to embedding */
    }
    
    return NIYAH_OK;
}

void niyah_mini_weights_free(NiyahMiniWeights* weights) {
    if (!weights) return;
    
    if (weights->owns_memory && weights->memory_block) {
        free(weights->memory_block);
    }
    
    if (weights->layers) {
        free(weights->layers);
    }
    
    weights->memory_block = NULL;
    weights->memory_size = 0;
    weights->owns_memory = false;
    weights->embedding = NULL;
    weights->layers = NULL;
    weights->n_layers = 0;
    weights->final_norm = NULL;
    weights->lm_head = NULL;
}

/* ==========================================================================
 * Weight Initialization
 * ========================================================================== */

void niyah_mini_weights_init_xavier(
    NiyahMiniWeights* weights,
    const NiyahMiniConfig* config
) {
    if (!weights || !config) return;
    
    const size_t dim = (size_t)config->n_dim;
    const size_t vocab = (size_t)config->n_vocab;
    const size_t n_layers = (size_t)config->n_layers;
    const size_t n_heads = (size_t)config->n_heads;
    const size_t n_kv_heads = (size_t)config->n_kv_heads;
    const size_t head_dim = dim / n_heads;
    const size_t kv_dim = n_kv_heads * head_dim;
    const size_t n_ff = (size_t)config->n_ff;
    
    /* Embedding */
    init_xavier(weights->embedding, vocab, dim);
    
    /* Layers */
    for (size_t l = 0; l < n_layers; l++) {
        /* Attention norm */
        init_xavier(weights->layers[l].attn_norm, dim, 1);
        
        /* Query projection */
        init_xavier(weights->layers[l].wq, dim, dim);
        
        /* Key projection */
        init_xavier(weights->layers[l].wk, kv_dim, dim);
        
        /* Value projection */
        init_xavier(weights->layers[l].wv, kv_dim, dim);
        
        /* Output projection */
        init_xavier(weights->layers[l].wo, dim, dim);
        
        /* FFN norm */
        init_xavier(weights->layers[l].ffn_norm, dim, 1);
        
        /* FFN gate */
        init_xavier(weights->layers[l].ffn_gate, n_ff, dim);
        
        /* FFN up */
        init_xavier(weights->layers[l].ffn_up, n_ff, dim);
        
        /* FFN down */
        init_xavier(weights->layers[l].ffn_down, dim, n_ff);
    }
    
    /* Final norm */
    init_xavier(weights->final_norm, dim, 1);
    
    /* LM head (if not tied) */
    if (!config->tie_word_embeddings && weights->lm_head) {
        init_xavier(weights->lm_head, vocab, dim);
    }
}

void niyah_mini_weights_init_small(
    NiyahMiniWeights* weights,
    const NiyahMiniConfig* config,
    float scale
) {
    if (!weights || !config) return;
    
    size_t total_floats = niyah_mini_weights_memory_size(config) / sizeof(float);
    init_small((float*)weights->memory_block, total_floats, scale);
}

/* ==========================================================================
 * Model Loading/Saving
 * ========================================================================== */

NiyahStatus niyah_mini_model_load_weights(
    NiyahMiniModel* model,
    const char* weights_path
) {
    if (!model || !weights_path) {
        return NIYAH_ERR_INVALID_ARG;
    }
    
    /* Open file */
    FILE* f = fopen(weights_path, "rb");
    if (!f) {
        return NIYAH_ERR_IO;
    }
    
    /* Get file size */
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    /* Check size */
    size_t expected_size = niyah_mini_weights_memory_size(&model->config);
    if ((size_t)file_size != expected_size) {
        fclose(f);
        return NIYAH_ERR_SHAPE;
    }
    
    /* Read weights */
    size_t bytes_read = fread(model->weights.memory_block, 1, file_size, f);
    fclose(f);
    
    if (bytes_read != (size_t)file_size) {
        return NIYAH_ERR_IO;
    }
    
    return NIYAH_OK;
}

NiyahStatus niyah_mini_model_save_weights(
    const NiyahMiniModel* model,
    const char* weights_path
) {
    if (!model || !weights_path) {
        return NIYAH_ERR_INVALID_ARG;
    }
    
    /* Open file */
    FILE* f = fopen(weights_path, "wb");
    if (!f) {
        return NIYAH_ERR_IO;
    }
    
    /* Write weights */
    size_t bytes_written = fwrite(
        model->weights.memory_block,
        1,
        model->weights.memory_size,
        f
    );
    fclose(f);
    
    if (bytes_written != model->weights.memory_size) {
        return NIYAH_ERR_IO;
    }
    
    return NIYAH_OK;
}

/* ==========================================================================
 * Forward State Management
 * ========================================================================== */

size_t niyah_mini_forward_state_memory_size(
    const NiyahMiniConfig* config,
    int32_t max_seq_len
) {
    if (!config || max_seq_len <= 0) return 0;
    
    const size_t dim = (size_t)config->n_dim;
    const size_t n_ff = (size_t)config->n_ff;
    const size_t seq_len = (size_t)max_seq_len;
    
    size_t total = 0;
    
    /* Hidden states */
    total += seq_len * dim;  /* hidden */
    
    /* Norm outputs */
    total += seq_len * dim;  /* norm1 */
    total += seq_len * dim;  /* norm2 */
    
    /* Attention outputs */
    total += seq_len * dim;  /* attn_out */
    
    /* FFN outputs */
    total += seq_len * dim;  /* ffn_out */
    
    /* Attention intermediate */
    total += seq_len * dim;  /* q */
    total += seq_len * dim;  /* k (max dim) */
    total += seq_len * dim;  /* v (max dim) */
    total += seq_len * seq_len;  /* attn_scores */
    total += seq_len * seq_len;  /* attn_probs */
    
    /* FFN intermediate */
    total += seq_len * n_ff;  /* ffn_gate_out */
    total += seq_len * n_ff;  /* ffn_up_out */
    
    /* Layer output */
    total += seq_len * dim;  /* layer_out */
    
    return total * sizeof(float);
}

NiyahStatus niyah_mini_forward_state_init(
    NiyahMiniForwardState* state,
    const NiyahMiniConfig* config,
    int32_t max_seq_len
) {
    if (!state || !config || max_seq_len <= 0) {
        return NIYAH_ERR_INVALID_ARG;
    }
    
    memset(state, 0, sizeof(*state));
    
    size_t total_size = niyah_mini_forward_state_memory_size(config, max_seq_len);
    
    /* Allocate single block */
    void* memory_block = malloc(total_size);
    if (!memory_block) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    
    /* Assign pointers */
    float* ptr = (float*)memory_block;
    
    const size_t dim = (size_t)config->n_dim;
    const size_t n_ff = (size_t)config->n_ff;
    const size_t seq_len = (size_t)max_seq_len;
    
    state->hidden = ptr;
    ptr += seq_len * dim;
    
    state->norm1 = ptr;
    ptr += seq_len * dim;
    
    state->attn_out = ptr;
    ptr += seq_len * dim;
    
    state->norm2 = ptr;
    ptr += seq_len * dim;
    
    state->ffn_out = ptr;
    ptr += seq_len * dim;
    
    state->q = ptr;
    ptr += seq_len * dim;
    
    state->k = ptr;
    ptr += seq_len * dim;
    
    state->v = ptr;
    ptr += seq_len * dim;
    
    state->attn_scores = ptr;
    ptr += seq_len * seq_len;
    
    state->attn_probs = ptr;
    ptr += seq_len * seq_len;
    
    state->ffn_gate_out = ptr;
    ptr += seq_len * n_ff;
    
    state->ffn_up_out = ptr;
    ptr += seq_len * n_ff;
    
    state->layer_out = ptr;
    ptr += seq_len * dim;
    
    state->memory_block = memory_block;
    state->memory_size = total_size;
    
    return NIYAH_OK;
}

void niyah_mini_forward_state_free(NiyahMiniForwardState* state) {
    if (!state) return;
    
    if (state->memory_block) {
        free(state->memory_block);
    }
    
    memset(state, 0, sizeof(*state));
}

/* ==========================================================================
 * Core Transformers: RMSNorm
 * ========================================================================== */

static void rmsnorm(
    float* out,
    const float* x,
    const float* weight,
    int32_t n,
    float eps
) {
    /* Compute sum of squares */
    double sum_sq = 0.0;
    for (int32_t i = 0; i < n; i++) {
        sum_sq += (double)x[i] * (double)x[i];
    }
    
    /* Compute normalization factor */
    double mean_sq = sum_sq / (double)n;
    float scale = (float)(1.0 / sqrt(mean_sq + (double)eps));
    
    /* Apply normalization and scale */
    for (int32_t i = 0; i < n; i++) {
        out[i] = x[i] * scale * (weight ? weight[i] : 1.0f);
    }
}

/* ==========================================================================
 * Core Transformers: RoPE (Rotary Position Embedding)
 * ========================================================================== */

static void rope_apply(
    float* x,
    int32_t seq_len,
    int32_t dim,
    int32_t n_heads,
    int32_t pos_offset,
    float theta
) {
    const int32_t head_dim = dim / n_heads;
    
    for (int32_t pos = 0; pos < seq_len; pos++) {
        for (int32_t h = 0; h < n_heads; h++) {
            float* x_head = x + (pos * dim) + (h * head_dim);
            
            for (int32_t i = 0; i < head_dim; i += 2) {
                if (i + 1 >= head_dim) break;
                
                float angle = (float)(pos + pos_offset) / 
                    powf(theta, (float)i / (float)head_dim);
                
                float cos_val = cosf(angle);
                float sin_val = sinf(angle);
                
                float x0 = x_head[i];
                float x1 = x_head[i + 1];
                
                x_head[i] = x0 * cos_val - x1 * sin_val;
                x_head[i + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

/* ==========================================================================
 * Core Transformers: SwiGLU Activation
 * ========================================================================== */

static void swiglu(
    float* out,
    const float* gate,
    const float* up,
    int32_t n
) {
    for (int32_t i = 0; i < n; i++) {
        float g = gate[i];
        /* SILU activation for gate */
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        out[i] = sigmoid_g * up[i];
    }
}

/* ==========================================================================
 * Core Transformers: Attention
 * ========================================================================== */

static void attention_forward(
    float* out,
    float* q,
    float* k,
    float* v,
    int32_t seq_len,
    int32_t dim,
    int32_t n_heads,
    int32_t n_kv_heads,
    float* scratch
) {
    const int32_t head_dim = dim / n_heads;
    const int32_t kv_dim = n_kv_heads * head_dim;
    const float scale = 1.0f / sqrtf((float)head_dim);
    
    /* Compute attention scores */
    float* scores = scratch;
    
    for (int32_t i = 0; i < seq_len; i++) {
        for (int32_t j = 0; j < seq_len; j++) {
            /* Compute dot product for this position pair */
            float dot = 0.0f;
            for (int32_t h = 0; h < n_heads; h++) {
                const float* q_head = q + (i * dim) + (h * head_dim);
                const float* k_head = k + (j * dim) + ((h % n_kv_heads) * head_dim);
                
                for (int32_t d = 0; d < head_dim; d++) {
                    dot += q_head[d] * k_head[d];
                }
            }
            scores[i * seq_len + j] = dot * scale;
        }
    }
    
    /* Apply causal mask (upper triangular) */
    for (int32_t i = 0; i < seq_len; i++) {
        for (int32_t j = 0; j < seq_len; j++) {
            if (j > i) {
                scores[i * seq_len + j] = -INFINITY;
            }
        }
    }
    
    /* Softmax */
    float* probs = scratch + seq_len * seq_len;
    
    for (int32_t i = 0; i < seq_len; i++) {
        float* scores_row = scores + i * seq_len;
        float* probs_row = probs + i * seq_len;
        
        /* Find max */
        float max_val = scores_row[0];
        for (int32_t j = 1; j <= i; j++) {
            if (scores_row[j] > max_val) {
                max_val = scores_row[j];
            }
        }
        
        /* Compute exp and sum */
        float sum = 0.0f;
        for (int32_t j = 0; j <= i; j++) {
            float exp_val = expf(scores_row[j] - max_val);
            probs_row[j] = exp_val;
            sum += exp_val;
        }
        
        /* Normalize */
        float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (int32_t j = 0; j <= i; j++) {
            probs_row[j] *= inv_sum;
        }
        
        /* Zero out masked positions */
        for (int32_t j = i + 1; j < seq_len; j++) {
            probs_row[j] = 0.0f;
        }
    }
    
    /* Compute weighted sum */
    for (int32_t i = 0; i < seq_len; i++) {
        float* out_row = out + i * dim;
        memset(out_row, 0, dim * sizeof(float));
        
        for (int32_t j = 0; j < seq_len; j++) {
            float prob = probs[i * seq_len + j];
            if (prob == 0.0f) continue;
            
            const float* v_row = v + j * dim;
            
            /* Distribute to appropriate heads */
            for (int32_t h = 0; h < n_heads; h++) {
                int32_t kv_head = h % n_kv_heads;
                const float* v_head = v_row + (kv_head * head_dim);
                float* out_head = out_row + (h * head_dim);
                
                for (int32_t d = 0; d < head_dim; d++) {
                    out_head[d] += prob * v_head[d];
                }
            }
        }
    }
}

/* ==========================================================================
 * Core Transformers: Feed-Forward Network
 * ========================================================================== */

static void ffn_forward(
    float* out,
    const float* x,
    const NiyahMiniLayerWeights* w,
    int32_t seq_len,
    int32_t dim,
    int32_t n_ff,
    float* scratch
) {
    /* Gate and up projections */
    float* gate = scratch;
    float* up = scratch + seq_len * n_ff;
    
    /* Compute gate = x @ w->ffn_gate */
    for (int32_t i = 0; i < seq_len; i++) {
        const float* x_row = x + i * dim;
        float* gate_row = gate + i * n_ff;
        
        for (int32_t j = 0; j < n_ff; j++) {
            float sum = 0.0f;
            for (int32_t k = 0; k < dim; k++) {
                sum += x_row[k] * w->ffn_gate[j * dim + k];
            }
            gate_row[j] = sum;
        }
    }
    
    /* Compute up = x @ w->ffn_up */
    for (int32_t i = 0; i < seq_len; i++) {
        const float* x_row = x + i * dim;
        float* up_row = up + i * n_ff;
        
        for (int32_t j = 0; j < n_ff; j++) {
            float sum = 0.0f;
            for (int32_t k = 0; k < dim; k++) {
                sum += x_row[k] * w->ffn_up[j * dim + k];
            }
            up_row[j] = sum;
        }
    }
    
    /* Apply SwiGLU */
    for (int32_t i = 0; i < seq_len; i++) {
        swiglu(scratch + i * n_ff, gate + i * n_ff, up + i * n_ff, n_ff);
    }
    
    /* Down projection: out = swiglu_output @ w->ffn_down */
    for (int32_t i = 0; i < seq_len; i++) {
        const float* swiglu_row = scratch + i * n_ff;
        float* out_row = out + i * dim;
        
        for (int32_t j = 0; j < dim; j++) {
            float sum = 0.0f;
            for (int32_t k = 0; k < n_ff; k++) {
                sum += swiglu_row[k] * w->ffn_down[j * n_ff + k];
            }
            out_row[j] = sum;
        }
    }
}

/* ==========================================================================
 * Forward Pass: Single Layer
 * ========================================================================== */

static void layer_forward(
    float* out,
    const float* x,
    const NiyahMiniLayerWeights* w,
    const NiyahMiniConfig* config,
    NiyahMiniForwardState* state,
    int32_t seq_len,
    int32_t position_offset
) {
    const int32_t dim = config->n_dim;
    const int32_t n_heads = config->n_heads;
    const int32_t n_kv_heads = config->n_kv_heads;
    const int32_t n_ff = config->n_ff;
    const float norm_eps = config->norm_eps;
    const float rope_theta = config->rope_theta;
    
    /* Pre-norm 1 */
    rmsnorm(state->norm1, x, w->attn_norm, dim, norm_eps);
    
    /* Project to Q, K, V */
    for (int32_t i = 0; i < seq_len; i++) {
        const float* x_norm = state->norm1 + i * dim;
        
        /* Query */
        float* q_row = state->q + i * dim;
        for (int32_t j = 0; j < dim; j++) {
            float sum = 0.0f;
            for (int32_t k = 0; k < dim; k++) {
                sum += x_norm[k] * w->wq[j * dim + k];
            }
            q_row[j] = sum;
        }
        
        /* Key and Value (grouped) */
        const int32_t head_dim = dim / n_heads;
        const int32_t kv_dim = n_kv_heads * head_dim;
        
        float* k_row = state->k + i * dim;
        float* v_row = state->v + i * dim;
        
        for (int32_t j = 0; j < kv_dim; j++) {
            float sum_k = 0.0f;
            float sum_v = 0.0f;
            for (int32_t k = 0; k < dim; k++) {
                sum_k += x_norm[k] * w->wk[j * dim + k];
                sum_v += x_norm[k] * w->wv[j * dim + k];
            }
            if (j < dim) {
                k_row[j] = sum_k;
                v_row[j] = sum_v;
            }
        }
    }
    
    /* Apply RoPE to Q and K */
    rope_apply(state->q, seq_len, dim, n_heads, position_offset, rope_theta);
    rope_apply(state->k, seq_len, dim, n_kv_heads, position_offset, rope_theta);
    
    /* Attention */
    attention_forward(
        state->attn_out, state->q, state->k, state->v,
        seq_len, dim, n_heads, n_kv_heads, state->attn_scores
    );
    
    /* Output projection */
    for (int32_t i = 0; i < seq_len; i++) {
        const float* attn_row = state->attn_out + i * dim;
        float* out_row = state->layer_out + i * dim;
        
        for (int32_t j = 0; j < dim; j++) {
            float sum = 0.0f;
            for (int32_t k = 0; k < dim; k++) {
                sum += attn_row[k] * w->wo[j * dim + k];
            }
            out_row[j] = sum;
        }
    }
    
    /* Residual connection */
    for (int32_t i = 0; i < seq_len * dim; i++) {
        state->layer_out[i] += x[i];
    }
    
    /* Pre-norm 2 */
    rmsnorm(state->norm2, state->layer_out, w->ffn_norm, dim, norm_eps);
    
    /* FFN */
    ffn_forward(
        state->ffn_out, state->norm2, w, seq_len, dim, n_ff,
        state->ffn_gate_out
    );
    
    /* Residual connection */
    for (int32_t i = 0; i < seq_len * dim; i++) {
        out[i] = state->ffn_out[i] + state->layer_out[i];
    }
}

/* ==========================================================================
 * Forward Pass: Full Model
 * ========================================================================== */

NiyahStatus niyah_mini_forward_sequence(
    NiyahMiniModel* model,
    NiyahMiniForwardState* state,
    const int32_t* input_ids,
    int32_t seq_len,
    float* logits_out
) {
    if (!model || !state || !input_ids || !logits_out || seq_len <= 0) {
        return NIYAH_ERR_INVALID_ARG;
    }
    
    const NiyahMiniConfig* config = &model->config;
    const NiyahMiniWeights* w = &model->weights;
    const int32_t dim = config->n_dim;
    
    /* Token embeddings */
    for (int32_t i = 0; i < seq_len; i++) {
        int32_t token_id = input_ids[i];
        const float* embedding = w->embedding + token_id * dim;
        float* hidden_row = state->hidden + i * dim;
        
        for (int32_t j = 0; j < dim; j++) {
            hidden_row[j] = embedding[j];
        }
    }
    
    /* Process through layers */
    for (int32_t l = 0; l < config->n_layers; l++) {
        layer_forward(
            state->hidden, state->hidden,
            &w->layers[l], config, state,
            seq_len, 0  /* position offset */
        );
    }
    
    /* Final normalization */
    rmsnorm(state->hidden, state->hidden, w->final_norm, dim, config->norm_eps);
    
    /* Output logits */
    for (int32_t i = 0; i < seq_len; i++) {
        const float* hidden_row = state->hidden + i * dim;
        float* logits_row = logits_out + i * config->n_vocab;
        
        for (int32_t j = 0; j < config->n_vocab; j++) {
            float sum = 0.0f;
            for (int32_t k = 0; k < dim; k++) {
                sum += hidden_row[k] * w->lm_head[j * dim + k];
            }
            logits_row[j] = sum;
        }
    }
    
    return NIYAH_OK;
}

NiyahStatus niyah_mini_forward_token(
    NiyahMiniModel* model,
    NiyahMiniForwardState* state,
    int32_t token_id,
    int32_t position,
    float* logits_out
) {
    if (!model || !state || !logits_out || token_id < 0 || position < 0) {
        return NIYAH_ERR_INVALID_ARG;
    }
    
    const NiyahMiniConfig* config = &model->config;
    const int32_t dim = config->n_dim;
    
    /* Token embedding */
    const float* embedding = model->weights.embedding + token_id * dim;
    
    /* Copy to hidden state at position */
    float* hidden_row = state->hidden + position * dim;
    for (int32_t i = 0; i < dim; i++) {
        hidden_row[i] = embedding[i];
    }
    
    /* Process through layers */
    for (int32_t l = 0; l < config->n_layers; l++) {
        layer_forward(
            state->hidden, state->hidden,
            &model->weights.layers[l], config, state,
            position + 1, position  /* seq_len, position_offset */
        );
    }
    
    /* Final normalization */
    rmsnorm(state->hidden, state->hidden, model->weights.final_norm, dim, config->norm_eps);
    
    /* Output logits (only for last position) */
    const float* hidden_row_out = state->hidden + position * dim;
    for (int32_t j = 0; j < config->n_vocab; j++) {
        float sum = 0.0f;
        for (int32_t k = 0; k < dim; k++) {
            sum += hidden_row_out[k] * model->weights.lm_head[j * dim + k];
        }
        logits_out[j] = sum;
    }
    
    return NIYAH_OK;
}

/* ==========================================================================
 * Model Cleanup
 * ========================================================================== */

void niyah_mini_model_free(NiyahMiniModel* model) {
    if (!model) return;
    
    niyah_mini_weights_free(&model->weights);
    
    if (model->kv_cache_k) {
        free(model->kv_cache_k);
    }
    if (model->kv_cache_v) {
        free(model->kv_cache_v);
    }
    if (model->scratch) {
        free(model->scratch);
    }
    
    memset(model, 0, sizeof(*model));
}

/* ==========================================================================
 * Inference: Generate Text
 * ========================================================================== */

NiyahStatus niyah_mini_generate(
    NiyahMiniModel* model,
    const int32_t* prompt_ids,
    int32_t prompt_len,
    int32_t max_tokens,
    float temperature,
    int32_t* output_ids,
    int32_t* output_len
) {
    if (!model || !prompt_ids || prompt_len <= 0 || !output_ids || !output_len) {
        return NIYAH_ERR_INVALID_ARG;
    }
    
    /* Copy prompt to output */
    for (int32_t i = 0; i < prompt_len; i++) {
        output_ids[i] = prompt_ids[i];
    }
    *output_len = prompt_len;
    
    /* TODO: Implement sampling with temperature */
    /* For now, just return the prompt */
    
    return NIYAH_OK;
}

NiyahStatus niyah_mini_get_logits(
    NiyahMiniModel* model,
    const int32_t* input_ids,
    int32_t seq_len,
    float* logits
) {
    if (!model || !input_ids || seq_len <= 0 || !logits) {
        return NIYAH_ERR_INVALID_ARG;
    }
    
    /* Allocate forward state */
    NiyahMiniForwardState state;
    NiyahStatus status = niyah_mini_forward_state_init(&state, &model->config, seq_len);
    if (status != NIYAH_OK) {
        return status;
    }
    
    /* Forward pass */
    status = niyah_mini_forward_sequence(model, &state, input_ids, seq_len, logits);
    
    /* Free state */
    niyah_mini_forward_state_free(&state);
    
    return status;
}

void niyah_mini_reset_kv_cache(NiyahMiniModel* model) {
    if (!model) return;
    
    if (model->kv_cache_k) {
        memset(model->kv_cache_k, 0, model->kv_cache_seq_len * sizeof(float));
    }
    if (model->kv_cache_v) {
        memset(model->kv_cache_v, 0, model->kv_cache_seq_len * sizeof(float));
    }
    model->kv_cache_seq_len = 0;
}
