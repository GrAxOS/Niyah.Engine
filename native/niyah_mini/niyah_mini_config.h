#ifndef NIYAH_MINI_CONFIG_H
#define NIYAH_MINI_CONFIG_H

#include "../niyah.h"

/* ==========================================================================
 * NiyahMini - Original Lightweight Transformer Architecture
 * 
 * A from-scratch transformer designed for:
 * - Small, trainable locally (millions of parameters)
 * - Arabic/English/code multilingual support
 * - Evidence-aware inference
 * - Deterministic, reproducible behavior
 * - Memory-efficient C11 implementation
 * 
 * NO Llama, Qwen, Mistral, or other model code/weights used.
 * ========================================================================== */

/* Model hyperparameters for NiyahMini variants */
#define NIYAH_MINI_TINY   0  /* ~2.0M params: 4 layers, 128 dim, 4 heads */
#define NIYAH_MINI_SMALL  1  /* ~12.1M params: 8 layers, 256 dim, 8 heads */
#define NIYAH_MINI_BASE   2  /* ~64.0M params: 12 layers, 512 dim, 8 heads */
#define NIYAH_MINI_MEDIUM 3  /* ~191.9M params: 16 layers, 768 dim, 12 heads */

/* Default configuration for NiyahMini-Base */
#define NIYAH_MINI_DEFAULT_N_LAYERS     12
#define NIYAH_MINI_DEFAULT_N_DIM        512
#define NIYAH_MINI_DEFAULT_N_HEADS      8
#define NIYAH_MINI_DEFAULT_N_KV_HEADS   4  /* Grouped-Query Attention */
#define NIYAH_MINI_DEFAULT_N_FF         2048  /* Feed-forward dimension */
#define NIYAH_MINI_DEFAULT_N_VOCAB     16384  /* Vocabulary size */
#define NIYAH_MINI_DEFAULT_N_CTX        2048   /* Context length */
#define NIYAH_MINI_DEFAULT_ROPE_THETA  10000.0f
#define NIYAH_MINI_DEFAULT_NORM_EPS    1e-5f

/* Activation types */
typedef enum {
    NIYAH_MINI_ACT_SILU = 0,
    NIYAH_MINI_ACT_GELU = 1,
    NIYAH_MINI_ACT_RELU = 2
} NiyahMiniActivation;

/* Position encoding types */
typedef enum {
    NIYAH_MINI_POS_ROPE = 0,
    NIYAH_MINI_POS_ABSOLUTE = 1,
    NIYAH_MINI_POS_RELATIVE = 2
} NiyahMiniPositionEncoding;

/* NiyahMini model configuration */
typedef struct {
    int32_t n_layers;
    int32_t n_dim;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t n_ff;
    int32_t n_vocab;
    int32_t n_ctx;
    float rope_theta;
    float norm_eps;
    NiyahMiniActivation activation;
    NiyahMiniPositionEncoding pos_encoding;
    bool tie_word_embeddings;
    int32_t variant;  /* NIYAH_MINI_TINY, SMALL, BASE, MEDIUM */
} NiyahMiniConfig;

/* Initialize with default values */
NIYAH_API void niyah_mini_config_init(NiyahMiniConfig* config, int32_t variant);

/* Validate configuration */
NIYAH_API NiyahStatus niyah_mini_config_validate(const NiyahMiniConfig* config);

/* Calculate expected number of parameters */
NIYAH_API uint64_t niyah_mini_config_n_params(const NiyahMiniConfig* config);

/* Calculate expected number of float32 weights */
NIYAH_API size_t niyah_mini_config_n_floats(const NiyahMiniConfig* config);

#endif /* NIYAH_MINI_CONFIG_H */
