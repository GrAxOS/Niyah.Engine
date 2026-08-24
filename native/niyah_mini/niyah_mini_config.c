#include "niyah_mini_config.h"

#include <stdbool.h>

/* Preset configurations for different model sizes */
static const NiyahMiniConfig PRESETS[] = {
    /* TINY: ~1M params */
    {
        .n_layers = 4,
        .n_dim = 128,
        .n_heads = 4,
        .n_kv_heads = 2,
        .n_ff = 512,
        .n_vocab = 8192,
        .n_ctx = 512,
        .rope_theta = 10000.0f,
        .norm_eps = 1e-5f,
        .activation = NIYAH_MINI_ACT_SILU,
        .pos_encoding = NIYAH_MINI_POS_ROPE,
        .tie_word_embeddings = true,
        .variant = NIYAH_MINI_TINY
    },
    /* SMALL: ~4M params */
    {
        .n_layers = 8,
        .n_dim = 256,
        .n_heads = 8,
        .n_kv_heads = 4,
        .n_ff = 1024,
        .n_vocab = 16384,
        .n_ctx = 1024,
        .rope_theta = 10000.0f,
        .norm_eps = 1e-5f,
        .activation = NIYAH_MINI_ACT_SILU,
        .pos_encoding = NIYAH_MINI_POS_ROPE,
        .tie_word_embeddings = true,
        .variant = NIYAH_MINI_SMALL
    },
    /* BASE: ~12M params */
    {
        .n_layers = 12,
        .n_dim = 512,
        .n_heads = 8,
        .n_kv_heads = 4,
        .n_ff = 2048,
        .n_vocab = 32768,
        .n_ctx = 2048,
        .rope_theta = 10000.0f,
        .norm_eps = 1e-5f,
        .activation = NIYAH_MINI_ACT_SILU,
        .pos_encoding = NIYAH_MINI_POS_ROPE,
        .tie_word_embeddings = true,
        .variant = NIYAH_MINI_BASE
    },
    /* MEDIUM: ~36M params */
    {
        .n_layers = 16,
        .n_dim = 768,
        .n_heads = 12,
        .n_kv_heads = 6,
        .n_ff = 3072,
        .n_vocab = 65536,
        .n_ctx = 4096,
        .rope_theta = 10000.0f,
        .norm_eps = 1e-5f,
        .activation = NIYAH_MINI_ACT_SILU,
        .pos_encoding = NIYAH_MINI_POS_ROPE,
        .tie_word_embeddings = true,
        .variant = NIYAH_MINI_MEDIUM
    }
};

void niyah_mini_config_init(NiyahMiniConfig* config, int32_t variant)
{
    if (!config) return;

    if (variant >= 0 && variant < (int32_t)(sizeof(PRESETS) / sizeof(PRESETS[0]))) {
        *config = PRESETS[variant];
    } else {
        /* Default to BASE */
        *config = PRESETS[NIYAH_MINI_BASE];
        config->variant = variant;
    }
}

NiyahStatus niyah_mini_config_validate(const NiyahMiniConfig* config)
{
    if (!config) {
        return NIYAH_ERR_INVALID_ARG;
    }

    if (config->n_layers <= 0 || config->n_layers > NIYAH_MAX_LAYERS) {
        return NIYAH_ERR_INVALID_ARG;
    }

    if (config->n_dim <= 0 || config->n_dim > NIYAH_MAX_DIM) {
        return NIYAH_ERR_INVALID_ARG;
    }

    if (config->n_heads <= 0 || config->n_heads > NIYAH_MAX_HEADS) {
        return NIYAH_ERR_INVALID_ARG;
    }

    if (config->n_kv_heads <= 0 || config->n_kv_heads > config->n_heads) {
        return NIYAH_ERR_INVALID_ARG;
    }

    if (config->n_ff <= 0) {
        return NIYAH_ERR_INVALID_ARG;
    }

    if (config->n_vocab <= 0 || config->n_vocab > NIYAH_MAX_VOCAB) {
        return NIYAH_ERR_INVALID_ARG;
    }

    if (config->n_ctx <= 0 || config->n_ctx > NIYAH_MAX_SEQ_LEN) {
        return NIYAH_ERR_INVALID_ARG;
    }

    if (config->n_dim % config->n_heads != 0) {
        return NIYAH_ERR_SHAPE;
    }

    if (config->n_heads % config->n_kv_heads != 0) {
        return NIYAH_ERR_SHAPE;
    }

    if (config->rope_theta <= 0.0f) {
        return NIYAH_ERR_INVALID_ARG;
    }

    if (config->norm_eps <= 0.0f) {
        return NIYAH_ERR_INVALID_ARG;
    }

    return NIYAH_OK;
}

uint64_t niyah_mini_config_n_params(const NiyahMiniConfig* config)
{
    if (!config) return 0;

    const uint64_t dim = (uint64_t)config->n_dim;
    const uint64_t vocab = (uint64_t)config->n_vocab;
    const uint64_t n_layers = (uint64_t)config->n_layers;
    const uint64_t n_heads = (uint64_t)config->n_heads;
    const uint64_t n_kv_heads = (uint64_t)config->n_kv_heads;
    const uint64_t head_dim = dim / n_heads;
    const uint64_t kv_dim = n_kv_heads * head_dim;
    const uint64_t n_ff = (uint64_t)config->n_ff;

    /* Embedding: vocab * dim */
    uint64_t total = vocab * dim;

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
    const uint64_t per_layer = dim + (dim * dim) + (kv_dim * dim) + (kv_dim * dim) + (dim * dim) + dim + (n_ff * dim) + (n_ff * dim) + (dim * n_ff);

    total += n_layers * per_layer;

    /* Final norm: dim */
    total += dim;

    /* LM head: vocab * dim (unless tied) */
    if (!config->tie_word_embeddings) {
        total += vocab * dim;
    }

    return total;
}

size_t niyah_mini_config_n_floats(const NiyahMiniConfig* config)
{
    return (size_t)niyah_mini_config_n_params(config);
}
