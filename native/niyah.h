#ifndef NIYAH_H
#define NIYAH_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * Version
 * ========================================================================== */

#define NIYAH_VERSION_MAJOR 0
#define NIYAH_VERSION_MINOR 2
#define NIYAH_VERSION_PATCH 0

#define NIYAH_STRINGIFY_(x) #x
#define NIYAH_STRINGIFY(x) NIYAH_STRINGIFY_(x)

/* NIYAH_VERSION was referenced by niyah_core.c but never defined. */
#define NIYAH_VERSION                        \
    NIYAH_STRINGIFY(NIYAH_VERSION_MAJOR) "." \
    NIYAH_STRINGIFY(NIYAH_VERSION_MINOR) "." \
    NIYAH_STRINGIFY(NIYAH_VERSION_PATCH)

/* ==========================================================================
 * Export / linkage
 * ========================================================================== */

#if defined(_WIN32)
#  if defined(NIYAH_BUILD_SHARED)
#    define NIYAH_API __declspec(dllexport)
#  elif defined(NIYAH_USE_SHARED)
#    define NIYAH_API __declspec(dllimport)
#  else
#    define NIYAH_API
#  endif
#else
#  define NIYAH_API __attribute__((visibility("default")))
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define NIYAH_RESTRICT __restrict__
#elif defined(_MSC_VER)
#  define NIYAH_RESTRICT __restrict
#else
#  define NIYAH_RESTRICT
#endif

/* ==========================================================================
 * Limits
 * ========================================================================== */

#define NIYAH_MAX_TOKENS   4096
#define NIYAH_MAX_DIM      8192
#define NIYAH_MAX_LAYERS     64
#define NIYAH_MAX_HEADS      64
#define NIYAH_MAX_VOCAB   262144
#define NIYAH_MAX_SEQ_LEN  8192

/* ==========================================================================
 * Status codes
 * ========================================================================== */

typedef enum {
    NIYAH_OK                = 0,
    NIYAH_ERR_INVALID_ARG   = -1,
    NIYAH_ERR_OUT_OF_MEMORY = -2,
    NIYAH_ERR_IO            = -3,
    NIYAH_ERR_UNSUPPORTED   = -4,
    NIYAH_ERR_NO_WEIGHTS    = -5,
    NIYAH_ERR_SHAPE         = -6,
    NIYAH_ERR_NOT_FOUND     = -7,
    NIYAH_ERR_OVERFLOW      = -8
} NiyahStatus;

NIYAH_API const char* niyah_status_to_string(NiyahStatus status);

/* ==========================================================================
 * Truth (three-valued / Kleene logic)
 * ========================================================================== */

typedef enum {
    NIYAH_FALSE   = 0,
    NIYAH_TRUE    = 1,
    NIYAH_UNKNOWN = 2
} NiyahTruth;

NIYAH_API const char* niyah_version(void);
NIYAH_API int32_t     niyah_version_major(void);
NIYAH_API int32_t     niyah_version_minor(void);
NIYAH_API int32_t     niyah_version_patch(void);

NIYAH_API const char* niyah_truth_to_string(NiyahTruth truth);
NIYAH_API NiyahTruth  niyah_truth_from_string(const char* text);
NIYAH_API NiyahTruth  niyah_truth_not(NiyahTruth a);
NIYAH_API NiyahTruth  niyah_truth_and(NiyahTruth a, NiyahTruth b);
NIYAH_API NiyahTruth  niyah_truth_or(NiyahTruth a, NiyahTruth b);
NIYAH_API NiyahTruth  niyah_truth_implies(NiyahTruth a, NiyahTruth b);

/* ==========================================================================
 * Model
 * ========================================================================== */

typedef struct {
    int32_t n_vocab;
    int32_t n_embd;
    int32_t n_head;
    int32_t n_layer;
    int32_t n_ctx;
    int32_t type;

    /* Added in 0.2.0. Zero means "derive a sane default". */
    int32_t n_kv_head;          /* grouped-query attention; 0 => n_head   */
    int32_t n_ff;               /* FFN hidden dim; 0 => 4 * n_embd        */
    int32_t eos_token_id;
    int32_t bos_token_id;
    float   rope_theta;         /* 0 => 10000.0f                          */
    float   norm_eps;           /* 0 => 1e-5f                             */
    bool    tie_word_embeddings;
} NiyahModelConfig;

typedef struct {
    NiyahModelConfig config;
    void*  weights;
    size_t weights_size;
} NiyahModel;

/* Resolve zero-valued optional fields to their documented defaults. */
NIYAH_API void   niyah_model_config_normalize(NiyahModelConfig* config);
NIYAH_API size_t niyah_model_expected_floats(const NiyahModelConfig* config);

/* Read the flat float32 blob produced by tools/gguf_to_niyah.py. */
NIYAH_API NiyahStatus niyah_model_load(NiyahModel* model,
                                       const NiyahModelConfig* config,
                                       const char* weights_path);
NIYAH_API NiyahStatus niyah_model_load_config_json(NiyahModelConfig* config,
                                                   const char* json_path);
NIYAH_API void        niyah_model_free(NiyahModel* model);

/*
 * Typed view over NiyahModel.weights. Matches the on-disk order emitted by
 * tools/gguf_to_niyah.py:
 *   embedding, then per layer { attn_norm, wq, wk, wv, wo, ffn_norm,
 *   ffn_gate, ffn_up, ffn_down }, then final_norm, lm_head.
 * All projection matrices are row-major [out_features][in_features].
 */
typedef struct {
    const float* attn_norm;
    const float* wq;
    const float* wk;
    const float* wv;
    const float* wo;
    const float* ffn_norm;
    const float* ffn_gate;
    const float* ffn_up;
    const float* ffn_down;
} NiyahLayerWeights;

typedef struct {
    const float* embedding;
    const float* final_norm;
    const float* lm_head;
    NiyahLayerWeights* layers;
    int32_t n_layer;
} NiyahModelWeights;

NIYAH_API NiyahStatus niyah_model_weights_map(NiyahModelWeights* out,
                                              const NiyahModel* model);
NIYAH_API void        niyah_model_weights_unmap(NiyahModelWeights* weights);

/* ==========================================================================
 * Source / storage / web
 * ========================================================================== */

typedef enum {
    NIYAH_SOURCE_LOCAL = 0,
    NIYAH_SOURCE_WEB   = 1,
    NIYAH_SOURCE_DB    = 2,
    NIYAH_SOURCE_API   = 3
} NiyahSourceType;

typedef struct {
    NiyahSourceType type;
    char* uri;
    char* metadata;
} NiyahSource;

NIYAH_API NiyahStatus niyah_source_init(NiyahSource* source,
                                        NiyahSourceType type,
                                        const char* uri,
                                        const char* metadata);
NIYAH_API void        niyah_source_free(NiyahSource* source);
NIYAH_API const char* niyah_source_type_name(NiyahSourceType type);

typedef struct {
    char*  path;
    void*  data;
    size_t size;
} NiyahStorage;

NIYAH_API NiyahStatus niyah_storage_read(NiyahStorage* storage, const char* path);
NIYAH_API NiyahStatus niyah_storage_write(const char* path,
                                          const void* data,
                                          size_t size);
NIYAH_API void        niyah_storage_free(NiyahStorage* storage);

typedef struct {
    char*   url;
    char*   content;
    char*   title;
    int64_t timestamp;
} NiyahWebPage;

NIYAH_API void niyah_webpage_free(NiyahWebPage* page);

/* ==========================================================================
 * Crawler
 * ========================================================================== */

typedef struct {
    int32_t max_depth;
    int32_t max_pages;
    bool    follow_links;
} NiyahCrawlerConfig;

typedef struct {
    NiyahCrawlerConfig config;
    NiyahWebPage**     pages;
    int32_t            n_pages;
    int32_t            page_capacity;
} NiyahCrawler;

NIYAH_API NiyahStatus niyah_crawler_init(NiyahCrawler* crawler,
                                         const NiyahCrawlerConfig* config);
NIYAH_API NiyahStatus niyah_crawler_add_page(NiyahCrawler* crawler,
                                             NiyahWebPage* page);
NIYAH_API void        niyah_crawler_free(NiyahCrawler* crawler);

/* ==========================================================================
 * Search (in-process result carrier; the BM25 index lives in search/)
 * ========================================================================== */

typedef struct {
    char*   query;
    int32_t max_results;
    int64_t timeout_ms;
} NiyahSearchQuery;

typedef struct {
    char* title;
    char* snippet;
    char* url;
    float score;
} NiyahSearchResult;

typedef struct {
    NiyahSearchResult* results;
    int32_t            n_results;
    int32_t            capacity;
} NiyahSearchResponse;

NIYAH_API NiyahStatus niyah_search_response_init(NiyahSearchResponse* response,
                                                 int32_t capacity);
NIYAH_API NiyahStatus niyah_search_response_push(NiyahSearchResponse* response,
                                                 const char* title,
                                                 const char* snippet,
                                                 const char* url,
                                                 float score);
NIYAH_API void        niyah_search_response_sort(NiyahSearchResponse* response);
NIYAH_API void        niyah_search_response_free(NiyahSearchResponse* response);

/* ==========================================================================
 * Graph
 *
 * NOTE: the previous header declared `struct NiyahGraphEdge** edges` inside
 * NiyahGraphNode while NiyahGraphEdge was a typedef of an *anonymous* struct.
 * Those are two distinct types, so no conforming code could populate `edges`.
 * Both types now carry real struct tags.
 * ========================================================================== */

typedef struct NiyahGraphNode NiyahGraphNode;
typedef struct NiyahGraphEdge NiyahGraphEdge;

struct NiyahGraphNode {
    char*            id;
    char*            label;
    void*            data;
    NiyahGraphEdge** edges;
    int32_t          n_edges;
    int32_t          edge_capacity;
};

struct NiyahGraphEdge {
    NiyahGraphNode* from;
    NiyahGraphNode* to;
    char*           relation;
    float           weight;
};

typedef struct {
    NiyahGraphNode** nodes;
    int32_t          n_nodes;
    int32_t          node_capacity;
    NiyahGraphEdge** edges;
    int32_t          n_edges;
    int32_t          edge_capacity;
} NiyahGraph;

NIYAH_API NiyahGraph*     niyah_graph_create(void);
NIYAH_API void            niyah_graph_destroy(NiyahGraph* graph);
NIYAH_API NiyahGraphNode* niyah_graph_add_node(NiyahGraph* graph,
                                               const char* id,
                                               const char* label,
                                               void* data);
NIYAH_API NiyahGraphNode* niyah_graph_find_node(const NiyahGraph* graph,
                                                const char* id);
NIYAH_API NiyahGraphEdge* niyah_graph_add_edge(NiyahGraph* graph,
                                               const char* from_id,
                                               const char* to_id,
                                               const char* relation,
                                               float weight);
NIYAH_API int32_t         niyah_graph_neighbors(const NiyahGraph* graph,
                                                const char* id,
                                                NiyahGraphNode** out,
                                                int32_t max_out);

/* ==========================================================================
 * Embedding
 * ========================================================================== */

typedef struct {
    float*  vector;
    int32_t dim;
    char*   doc_id;
} NiyahEmbedding;

NIYAH_API NiyahStatus niyah_embedding_init(NiyahEmbedding* embedding,
                                           int32_t dim,
                                           const char* doc_id);
NIYAH_API void        niyah_embedding_free(NiyahEmbedding* embedding);
NIYAH_API float       niyah_dot(const float* a, const float* b, int32_t n);
NIYAH_API float       niyah_l2_norm(const float* v, int32_t n);
NIYAH_API void        niyah_normalize(float* v, int32_t n);
NIYAH_API float       niyah_cosine_similarity(const float* a,
                                              const float* b,
                                              int32_t n);
NIYAH_API int32_t     niyah_embedding_top_k(const NiyahEmbedding* query,
                                            const NiyahEmbedding* corpus,
                                            int32_t corpus_size,
                                            int32_t k,
                                            int32_t* out_indices,
                                            float* out_scores);

/* ==========================================================================
 * Linear algebra kernels
 * ========================================================================== */

/* out[m][n] = a[m][k] * b[k][n]   (all row-major) */
NIYAH_API void niyah_matmul(float* out,
                            const float* a,
                            const float* b,
                            int32_t m, int32_t k, int32_t n);

/* out[m][n] = a[m][k] * b[n][k]^T -- b stored [out_features][in_features],
 * which is the layout every projection matrix uses on disk. */
NIYAH_API void niyah_matmul_bt(float* out,
                               const float* a,
                               const float* b,
                               int32_t m, int32_t k, int32_t n);

/* out[n_out] = w[n_out][n_in] * x[n_in] */
NIYAH_API void niyah_matvec(float* out,
                            const float* w,
                            const float* x,
                            int32_t n_out, int32_t n_in);

NIYAH_API void niyah_add_inplace(float* dst, const float* src, int32_t n);
NIYAH_API void niyah_scale_inplace(float* dst, float scale, int32_t n);

NIYAH_API void  niyah_softmax(float* x, int32_t n);
NIYAH_API void  niyah_softmax_temperature(float* x, int32_t n, float temperature);
NIYAH_API void  niyah_log_softmax(float* x, int32_t n);
NIYAH_API int32_t niyah_argmax(const float* x, int32_t n);

NIYAH_API void niyah_rmsnorm(float* x, const float* weight, int32_t n, float eps);
NIYAH_API void niyah_rmsnorm_to(float* out,
                                const float* x,
                                const float* weight,
                                int32_t n,
                                float eps);
NIYAH_API void niyah_layernorm(float* x,
                               const float* weight,
                               const float* bias,
                               int32_t n,
                               float eps);

NIYAH_API float niyah_silu(float x);
NIYAH_API float niyah_gelu(float x);
/* x[i] = silu(gate[i]) * x[i]  (SwiGLU: gate branch times up branch) */
NIYAH_API void  niyah_swiglu_forward(float* x, const float* gate, int32_t n);
NIYAH_API void  niyah_swiglu_to(float* out,
                                const float* up,
                                const float* gate,
                                int32_t n);

/* Rotary embeddings, NeoX/GGUF half-split convention. */
NIYAH_API void niyah_rope_forward(float* x, int32_t seq, int32_t dim, int32_t n_head);
NIYAH_API void niyah_rope_forward_ex(float* x,
                                     int32_t seq,
                                     int32_t dim,
                                     int32_t n_head,
                                     int32_t pos_offset,
                                     float theta);

/* ==========================================================================
 * Attention
 * ========================================================================== */

typedef struct {
    float*  qkv;   /* scratch, >= 3 * batch * seq * dim floats */
    float*  out;
    int32_t batch;
    int32_t seq;
    int32_t dim;
    int32_t n_head;
} NiyahAttentionState;

typedef struct {
    float*  q;
    float*  k;
    float*  v;
    float*  out;
    int32_t batch;
    int32_t seq;
    int32_t dim;
    int32_t n_head;
} NiyahMultiHeadAttentionState;

NIYAH_API void niyah_attention_forward(NiyahAttentionState* state,
                                       const float* x,
                                       float* out);
NIYAH_API void niyah_multihead_attention_forward(NiyahMultiHeadAttentionState* state);

/* Incremental decoding cache: [layer][kv_head][pos][head_dim]. */
typedef struct {
    float*  k;
    float*  v;
    int32_t n_layer;
    int32_t n_kv_head;
    int32_t head_dim;
    int32_t max_seq;
    int32_t length;
} NiyahKVCache;

NIYAH_API NiyahStatus niyah_kv_cache_init(NiyahKVCache* cache,
                                          int32_t n_layer,
                                          int32_t n_kv_head,
                                          int32_t head_dim,
                                          int32_t max_seq);
NIYAH_API void        niyah_kv_cache_reset(NiyahKVCache* cache);
NIYAH_API void        niyah_kv_cache_free(NiyahKVCache* cache);

/* Single-token causal attention against the cache. */
NIYAH_API NiyahStatus niyah_attention_decode(float* out,
                                             const float* q,
                                             NiyahKVCache* cache,
                                             int32_t layer,
                                             int32_t n_head,
                                             int32_t position,
                                             float* scratch);

/* ==========================================================================
 * Transformer layer
 * ========================================================================== */

typedef struct {
    float*  attn_out;
    float*  ffn_out;
    float*  norm1_out;
    float*  norm2_out;
    int32_t batch;
    int32_t seq;
    int32_t dim;
    int32_t n_head;
} NiyahTransformerLayerState;

/* Pre-norm residual block with parameter-free attention. */
NIYAH_API void niyah_transformer_layer_forward(NiyahTransformerLayerState* state,
                                               const float* x);

/* Full weighted block: RMSNorm -> GQA+RoPE -> residual -> RMSNorm ->
 * SwiGLU FFN -> residual. Writes in place into `hidden`. */
NIYAH_API NiyahStatus niyah_transformer_layer_forward_weighted(
    float* hidden,
    const NiyahLayerWeights* w,
    const NiyahModelConfig* config,
    NiyahKVCache* cache,
    int32_t layer,
    int32_t position,
    float* scratch);

/* Floats of scratch required by niyah_transformer_layer_forward_weighted. */
NIYAH_API size_t niyah_transformer_scratch_floats(const NiyahModelConfig* config);

/* ==========================================================================
 * Runtime (arena allocator)
 * ========================================================================== */

typedef struct {
    void*   memory_pool;
    size_t  memory_size;
    int32_t device_id;
    bool    use_gpu;
} NiyahRuntimeConfig;

typedef struct {
    NiyahRuntimeConfig config;
    void*              context;
} NiyahRuntime;

NIYAH_API NiyahRuntime* niyah_runtime_create(const NiyahRuntimeConfig* config);
NIYAH_API void          niyah_runtime_destroy(NiyahRuntime* runtime);
NIYAH_API void*         niyah_runtime_alloc(NiyahRuntime* runtime, size_t bytes);
NIYAH_API float*        niyah_runtime_alloc_floats(NiyahRuntime* runtime, size_t count);
NIYAH_API void          niyah_runtime_reset(NiyahRuntime* runtime);
NIYAH_API size_t        niyah_runtime_used(const NiyahRuntime* runtime);
NIYAH_API size_t        niyah_runtime_capacity(const NiyahRuntime* runtime);

/* ==========================================================================
 * Sampler
 * ========================================================================== */

typedef enum {
    NIYAH_SAMPLE_GREEDY      = 0,
    NIYAH_SAMPLE_TOP_K       = 1,
    NIYAH_SAMPLE_TOP_P       = 2,
    NIYAH_SAMPLE_TEMPERATURE = 3
} NiyahSampleStrategy;

typedef struct {
    NiyahSampleStrategy strategy;
    float               temperature;
    int32_t             top_k;
    float               top_p;
} NiyahSamplerConfig;

NIYAH_API int32_t niyah_sample(const float* logits,
                               int32_t n_vocab,
                               const NiyahSamplerConfig* config);
NIYAH_API void    niyah_sampler_seed(uint64_t seed);
NIYAH_API void    niyah_sampler_apply_repetition_penalty(float* logits,
                                                         int32_t n_vocab,
                                                         const int32_t* history,
                                                         int32_t history_len,
                                                         float penalty);

/* ==========================================================================
 * Telemetry
 * ========================================================================== */

typedef struct {
    int64_t start_time;   /* nanoseconds, monotonic */
    int64_t end_time;
    int64_t memory_used;
    int32_t tokens_processed;
} NiyahTelemetry;

NIYAH_API void    niyah_telemetry_start(NiyahTelemetry* telemetry);
NIYAH_API void    niyah_telemetry_end(NiyahTelemetry* telemetry);
NIYAH_API int64_t niyah_telemetry_now_ns(void);
NIYAH_API double  niyah_telemetry_elapsed_ms(const NiyahTelemetry* telemetry);
NIYAH_API double  niyah_telemetry_tokens_per_second(const NiyahTelemetry* telemetry);

/* ==========================================================================
 * Tokenizer
 * ========================================================================== */

typedef struct {
    char**  vocab;
    int32_t* ids;
    int32_t n_vocab;
} NiyahTokenizerVocab;

typedef struct {
    NiyahTokenizerVocab vocab;
    int32_t*            tokens;
    int32_t             n_tokens;
    bool                owns_vocab;
} NiyahTokenizer;

NIYAH_API int32_t niyah_tokenize(NiyahTokenizer* tokenizer,
                                 const char* text,
                                 int32_t* tokens,
                                 int32_t max_tokens);
NIYAH_API char*   niyah_detokenize(NiyahTokenizer* tokenizer,
                                   const int32_t* tokens,
                                   int32_t n_tokens);
/* Newline-delimited vocab file; line number is the token id. */
NIYAH_API NiyahStatus niyah_tokenizer_load(NiyahTokenizer* tokenizer,
                                           const char* vocab_path);
NIYAH_API void        niyah_tokenizer_free(NiyahTokenizer* tokenizer);
NIYAH_API int32_t     niyah_tokenizer_lookup(const NiyahTokenizer* tokenizer,
                                             const char* piece);

/* ==========================================================================
 * LLM
 * ========================================================================== */

typedef struct {
    NiyahModel         model;
    NiyahTokenizer     tokenizer;
    NiyahRuntime       runtime;
    NiyahSamplerConfig sampler;
} NiyahLLM;

typedef struct {
    char*          text;
    int32_t        n_tokens;
    float*         logits;      /* last-step logits, n_vocab floats */
    NiyahTelemetry telemetry;
    NiyahStatus    status;
} NiyahLLMOutput;

/*
 * Runs a real decode loop when llm->model.weights is populated.
 * With no weights it returns status = NIYAH_ERR_NO_WEIGHTS and text = NULL.
 * It never fabricates output.
 */
NIYAH_API NiyahLLMOutput niyah_llm_generate(NiyahLLM* llm,
                                            const char* prompt,
                                            int32_t max_tokens);
NIYAH_API NiyahStatus niyah_llm_forward(NiyahLLM* llm,
                                        int32_t token,
                                        int32_t position,
                                        NiyahKVCache* cache,
                                        float* logits,
                                        float* scratch);
NIYAH_API void niyah_llm_output_free(NiyahLLMOutput* output);

/* ==========================================================================
 * Bridge (stable C ABI for the C# UI in ui/Niyah.App)
 * ========================================================================== */

typedef struct NiyahBridgeContext NiyahBridgeContext;

NIYAH_API NiyahBridgeContext* niyah_bridge_create(NiyahLLM* llm);
NIYAH_API void                niyah_bridge_destroy(NiyahBridgeContext* ctx);
NIYAH_API NiyahGraph*         niyah_bridge_graph(NiyahBridgeContext* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NIYAH_H */
