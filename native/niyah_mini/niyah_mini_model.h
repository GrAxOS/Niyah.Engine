#ifndef NIYAH_MINI_MODEL_H
#define NIYAH_MINI_MODEL_H

#include "../niyah.h"
#include "niyah_mini_config.h"

/* ==========================================================================
 * NiyahMini Model - Original Transformer Implementation
 * 
 * A completely original transformer model designed for:
 * - Small, trainable locally (millions of parameters)
 * - Arabic/English/code multilingual support
 * - Evidence-aware inference
 * - Deterministic, reproducible behavior
 * - Memory-efficient C11 implementation
 * 
 * NO Llama, Qwen, Mistral, or other model code/weights used.
 * ========================================================================== */

/* Weight tensor structure for a single layer */
typedef struct {
    /* Attention weights */
    float* attn_norm;      /* [dim] - Attention norm weights */
    float* wq;             /* [dim][dim] - Query projection */
    float* wk;             /* [kv_dim][dim] - Key projection */
    float* wv;             /* [kv_dim][dim] - Value projection */
    float* wo;             /* [dim][dim] - Output projection */
    
    /* Feed-forward weights */
    float* ffn_norm;        /* [dim] - FFN norm weights */
    float* ffn_gate;        /* [n_ff][dim] - Gate projection */
    float* ffn_up;          /* [n_ff][dim] - Up projection */
    float* ffn_down;        /* [dim][n_ff] - Down projection */
} NiyahMiniLayerWeights;

/* Complete model weights */
typedef struct {
    /* Token embedding */
    float* embedding;      /* [vocab_size][dim] */
    
    /* Layer weights */
    NiyahMiniLayerWeights* layers;
    int32_t n_layers;
    
    /* Final normalization */
    float* final_norm;      /* [dim] */
    
    /* Output head (optional, can be tied to embedding) */
    float* lm_head;        /* [vocab_size][dim] or NULL if tied */
    
    /* Memory management */
    void* memory_block;    /* Single allocation for all weights */
    size_t memory_size;
    bool owns_memory;
} NiyahMiniWeights;

/* Model state for inference */
typedef struct {
    NiyahMiniConfig config;
    NiyahMiniWeights weights;
    
    /* Runtime state */
    void* runtime;
    
    /* KV cache for decoding */
    float* kv_cache_k;     /* [n_layers][n_kv_heads][max_seq][head_dim] */
    float* kv_cache_v;     /* [n_layers][n_kv_heads][max_seq][head_dim] */
    int32_t kv_cache_seq_len;
    
    /* Scratch buffers */
    float* scratch;        /* General scratch space */
    size_t scratch_size;
} NiyahMiniModel;

/* Forward pass state */
typedef struct {
    float* hidden;         /* [seq_len][dim] - Current hidden states */
    float* norm1;          /* [seq_len][dim] - Norm 1 output */
    float* attn_out;       /* [seq_len][dim] - Attention output */
    float* norm2;          /* [seq_len][dim] - Norm 2 output */
    float* ffn_out;        /* [seq_len][dim] - FFN output */
    
    /* Attention intermediate */
    float* q;             /* [seq_len][dim] - Query */
    float* k;             /* [seq_len][dim] - Key (max dim) */
    float* v;             /* [seq_len][dim] - Value (max dim) */
    float* attn_scores;    /* [seq_len][seq_len] - Attention scores */
    float* attn_probs;     /* [seq_len][seq_len] - Attention probabilities */
    
    /* FFN intermediate */
    float* ffn_gate_out;   /* [seq_len][n_ff] - Gate output */
    float* ffn_up_out;     /* [seq_len][n_ff] - Up output */
    
    /* Layer outputs */
    float* layer_out;      /* [seq_len][dim] - Layer output */
    
    /* Memory management */
    void* memory_block;    /* Single allocation for all state */
    size_t memory_size;
} NiyahMiniForwardState;

/* ==========================================================================
 * Model Initialization and Management
 * ========================================================================== */

/* Initialize model with configuration */
NIYAH_API NiyahStatus niyah_mini_model_init(
    NiyahMiniModel* model,
    const NiyahMiniConfig* config
);

/* Load model weights from file */
NIYAH_API NiyahStatus niyah_mini_model_load_weights(
    NiyahMiniModel* model,
    const char* weights_path
);

/* Save model weights to file */
NIYAH_API NiyahStatus niyah_mini_model_save_weights(
    const NiyahMiniModel* model,
    const char* weights_path
);

/* Load model from JSON config and binary weights */
NIYAH_API NiyahStatus niyah_mini_model_load(
    NiyahMiniModel* model,
    const char* config_path,
    const char* weights_path
);

/* Save model to JSON config and binary weights */
NIYAH_API NiyahStatus niyah_mini_model_save(
    const NiyahMiniModel* model,
    const char* config_path,
    const char* weights_path
);

/* Free model resources */
NIYAH_API void niyah_mini_model_free(NiyahMiniModel* model);

/* ==========================================================================
 * Forward Pass
 * ========================================================================== */

/* Initialize forward state */
NIYAH_API NiyahStatus niyah_mini_forward_state_init(
    NiyahMiniForwardState* state,
    const NiyahMiniConfig* config,
    int32_t max_seq_len
);

/* Free forward state */
NIYAH_API void niyah_mini_forward_state_free(NiyahMiniForwardState* state);

/* Run forward pass for a single token (decode step) */
NIYAH_API NiyahStatus niyah_mini_forward_token(
    NiyahMiniModel* model,
    NiyahMiniForwardState* state,
    int32_t token_id,
    int32_t position,
    float* logits_out
);

/* Run forward pass for a sequence (prefill step) */
NIYAH_API NiyahStatus niyah_mini_forward_sequence(
    NiyahMiniModel* model,
    NiyahMiniForwardState* state,
    const int32_t* input_ids,
    int32_t seq_len,
    float* logits_out
);

/* ==========================================================================
 * Training Utilities
 * ========================================================================== */

/* Initialize weights with Xavier/Glorot initialization */
NIYAH_API void niyah_mini_weights_init_xavier(
    NiyahMiniWeights* weights,
    const NiyahMiniConfig* config
);

/* Initialize weights with small random values */
NIYAH_API void niyah_mini_weights_init_small(
    NiyahMiniWeights* weights,
    const NiyahMiniConfig* config,
    float scale
);

/* Copy weights from another model */
NIYAH_API NiyahStatus niyah_mini_weights_copy(
    NiyahMiniWeights* dst,
    const NiyahMiniWeights* src,
    const NiyahMiniConfig* config
);

/* Scale weights (for learning rate scheduling) */
NIYAH_API void niyah_mini_weights_scale(
    NiyahMiniWeights* weights,
    float scale
);

/* ==========================================================================
 * Inference Utilities
 * ========================================================================== */

/* Generate text from prompt */
NIYAH_API NiyahStatus niyah_mini_generate(
    NiyahMiniModel* model,
    const int32_t* prompt_ids,
    int32_t prompt_len,
    int32_t max_tokens,
    float temperature,
    int32_t* output_ids,
    int32_t* output_len
);

/* Get logits for next token */
NIYAH_API NiyahStatus niyah_mini_get_logits(
    NiyahMiniModel* model,
    const int32_t* input_ids,
    int32_t seq_len,
    float* logits
);

/* Reset KV cache */
NIYAH_API void niyah_mini_reset_kv_cache(NiyahMiniModel* model);

/* ==========================================================================
 * Memory Management
 * ========================================================================== */

/* Allocate weights memory */
NIYAH_API NiyahStatus niyah_mini_weights_allocate(
    NiyahMiniWeights* weights,
    const NiyahMiniConfig* config
);

/* Free weights memory */
NIYAH_API void niyah_mini_weights_free(NiyahMiniWeights* weights);

/* Calculate required memory for weights */
NIYAH_API size_t niyah_mini_weights_memory_size(const NiyahMiniConfig* config);

/* Calculate required memory for forward state */
NIYAH_API size_t niyah_mini_forward_state_memory_size(
    const NiyahMiniConfig* config,
    int32_t max_seq_len
);

#endif /* NIYAH_MINI_MODEL_H */
