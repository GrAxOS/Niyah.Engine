#ifndef NIYAH_H
#define NIYAH_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// ============================================================================
// NIYAH CORE
// ============================================================================

#define NIYAH_VERSION_MAJOR 0
#define NIYAH_VERSION_MINOR 1
#define NIYAH_VERSION_PATCH 0

#define NIYAH_MAX_TOKENS 4096
#define NIYAH_MAX_DIM 4096
#define NIYAH_MAX_LAYERS 32
#define NIYAH_MAX_HEADS 32
#define NIYAH_MAX_VOCAB 50257
#define NIYAH_MAX_SEQ_LEN 1024

// ============================================================================
// NIYAH TRUTH
// ============================================================================

typedef enum {
    NIYAH_FALSE = 0,
    NIYAH_TRUE = 1,
    NIYAH_UNKNOWN = 2
} NiyahTruth;

// ============================================================================
// NIYAH MODEL
// ============================================================================

typedef struct {
    int32_t n_vocab;
    int32_t n_embd;
    int32_t n_head;
    int32_t n_layer;
    int32_t n_ctx;
    int32_t type;
} NiyahModelConfig;

typedef struct {
    NiyahModelConfig config;
    void* weights;
    size_t weights_size;
} NiyahModel;

// ============================================================================
// NIYAH SOURCE
// ============================================================================

typedef enum {
    NIYAH_SOURCE_LOCAL = 0,
    NIYAH_SOURCE_WEB = 1,
    NIYAH_SOURCE_DB = 2,
    NIYAH_SOURCE_API = 3
} NiyahSourceType;

typedef struct {
    NiyahSourceType type;
    char* uri;
    char* metadata;
} NiyahSource;

// ============================================================================
// NIYAH STORAGE
// ============================================================================

typedef struct {
    char* path;
    void* data;
    size_t size;
} NiyahStorage;

// ============================================================================
// NIYAH WEB
// ============================================================================

typedef struct {
    char* url;
    char* content;
    char* title;
    int64_t timestamp;
} NiyahWebPage;

// ============================================================================
// NIYAH CRAWLER
// ============================================================================

typedef struct {
    int32_t max_depth;
    int32_t max_pages;
    bool follow_links;
} NiyahCrawlerConfig;

typedef struct {
    NiyahCrawlerConfig config;
    NiyahWebPage** pages;
    int32_t n_pages;
} NiyahCrawler;

// ============================================================================
// NIYAH SEARCH
// ============================================================================

typedef struct {
    char* query;
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
    int32_t n_results;
} NiyahSearchResponse;

// ============================================================================
// NIYAH DOCUMENT
// ============================================================================

typedef struct {
    char* id;
    char* title;
    char* content;
    char** tokens;
    int32_t n_tokens;
    int64_t created_at;
} NiyahDocument;

// ============================================================================
// NIYAH GRAPH
// ============================================================================

typedef struct NiyahGraphNode {
    char* id;
    char* label;
    void* data;
    struct NiyahGraphEdge** edges;
    int32_t n_edges;
} NiyahGraphNode;

typedef struct {
    NiyahGraphNode* from;
    NiyahGraphNode* to;
    char* relation;
    float weight;
} NiyahGraphEdge;

typedef struct {
    NiyahGraphNode** nodes;
    int32_t n_nodes;
    NiyahGraphEdge** edges;
    int32_t n_edges;
} NiyahGraph;

// ============================================================================
// NIYAH EMBEDDING
// ============================================================================

typedef struct {
    float* vector;
    int32_t dim;
    char* doc_id;
} NiyahEmbedding;

// ============================================================================
// NIYAH ATTENTION
// ============================================================================

typedef struct {
    float* qkv;
    float* out;
    int32_t batch;
    int32_t seq;
    int32_t dim;
    int32_t n_head;
} NiyahAttentionState;

void niyah_attention_forward(NiyahAttentionState* state, const float* x, float* out);

// ============================================================================
// NIYAH ATTENTION MULTIHEAD
// ============================================================================

typedef struct {
    float* q;
    float* k;
    float* v;
    float* out;
    int32_t batch;
    int32_t seq;
    int32_t dim;
    int32_t n_head;
} NiyahMultiHeadAttentionState;

void niyah_multihead_attention_forward(NiyahMultiHeadAttentionState* state);

// ============================================================================
// NIYAH TRANSFORMER LAYER
// ============================================================================

typedef struct {
    float* attn_out;
    float* ffn_out;
    float* norm1_out;
    float* norm2_out;
    int32_t batch;
    int32_t seq;
    int32_t dim;
    int32_t n_head;
} NiyahTransformerLayerState;

void niyah_transformer_layer_forward(NiyahTransformerLayerState* state, const float* x);

// ============================================================================
// NIYAH RUNTIME
// ============================================================================

typedef struct {
    void* memory_pool;
    size_t memory_size;
    int32_t device_id;
    bool use_gpu;
} NiyahRuntimeConfig;

typedef struct {
    NiyahRuntimeConfig config;
    void* context;
} NiyahRuntime;

// ============================================================================
// NIYAH SAMPLER
// ============================================================================

typedef enum {
    NIYAH_SAMPLE_GREEDY = 0,
    NIYAH_SAMPLE_TOP_K = 1,
    NIYAH_SAMPLE_TOP_P = 2,
    NIYAH_SAMPLE_TEMPERATURE = 3
} NiyahSampleStrategy;

typedef struct {
    NiyahSampleStrategy strategy;
    float temperature;
    int32_t top_k;
    float top_p;
} NiyahSamplerConfig;

int32_t niyah_sample(const float* logits, int32_t n_vocab, const NiyahSamplerConfig* config);

// ============================================================================
// NIYAH SOFTMAX
// ============================================================================

void niyah_softmax(float* x, int32_t n);

// ============================================================================
// NIYAH RMSNORM
// ============================================================================

void niyah_rmsnorm(float* x, const float* weight, int32_t n, float eps);

// ============================================================================
// NIYAH SWIGLU
// ============================================================================

void niyah_swiglu_forward(float* x, const float* gate, int32_t n);

// ============================================================================
// NIYAH MATMUL
// ============================================================================

void niyah_matmul(float* out, const float* a, const float* b, int32_t m, int32_t k, int32_t n);

// ============================================================================
// NIYAH ROPE
// ============================================================================

void niyah_rope_forward(float* x, int32_t seq, int32_t dim, int32_t n_head);

// ============================================================================
// NIYAH TELEMETRY
// ============================================================================

typedef struct {
    int64_t start_time;
    int64_t end_time;
    int64_t memory_used;
    int32_t tokens_processed;
} NiyahTelemetry;

void niyah_telemetry_start(NiyahTelemetry* telemetry);
void niyah_telemetry_end(NiyahTelemetry* telemetry);

// ============================================================================
// NIYAH TOKENIZER
// ============================================================================

typedef struct {
    char** vocab;
    int32_t* ids;
    int32_t n_vocab;
} NiyahTokenizerVocab;

typedef struct {
    NiyahTokenizerVocab vocab;
    int32_t* tokens;
    int32_t n_tokens;
} NiyahTokenizer;

int32_t niyah_tokenize(NiyahTokenizer* tokenizer, const char* text, int32_t* tokens, int32_t max_tokens);
char* niyah_detokenize(NiyahTokenizer* tokenizer, const int32_t* tokens, int32_t n_tokens);

// ============================================================================
// NIYAH LLM
// ============================================================================

typedef struct {
    NiyahModel model;
    NiyahTokenizer tokenizer;
    NiyahRuntime runtime;
    NiyahSamplerConfig sampler;
} NiyahLLM;

typedef struct {
    char* text;
    int32_t n_tokens;
    float* logits;
    NiyahTelemetry telemetry;
} NiyahLLMOutput;

NiyahLLMOutput niyah_llm_generate(NiyahLLM* llm, const char* prompt, int32_t max_tokens);

// ============================================================================
// NIYAH BRIDGE
// ============================================================================

typedef struct {
    NiyahLLM* llm;
    NiyahGraph* knowledge_graph;
    NiyahSearchResponse* search_results;
} NiyahBridgeContext;

NiyahBridgeContext* niyah_bridge_create(NiyahLLM* llm);
void niyah_bridge_destroy(NiyahBridgeContext* ctx);

// ============================================================================
// HELPERS
// ============================================================================

static inline const char* niyah_version(void) {
    return "0.1.0";
}

static inline const char* niyah_truth_to_string(NiyahTruth truth) {
    switch (truth) {
        case NIYAH_FALSE: return "false";
        case NIYAH_TRUE: return "true";
        case NIYAH_UNKNOWN: return "unknown";
        default: return "invalid";
    }
}

#endif // NIYAH_H
