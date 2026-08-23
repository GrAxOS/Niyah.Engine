#include "niyah_llm.h"
#include "niyah_matmul.h"
#include "niyah_rmsnorm.h"
#include "niyah_rope.h"
#include "niyah_attention.h"
#include "niyah_swiglu.h"
#include "niyah_softmax.h"
#include "niyah_sampler.h"
#include "niyah_embedding.h"
#include "niyah_tokenizer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * Weight loading implementation (previously missing)
 * ============================================================================ */

bool niyah_llm_weights_load_from_buffer(
    NiyahLlmModelWeights *weights,
    const NiyahLlmConfig *config,
    const uint8_t *buffer,
    size_t buffer_size)
{
    if (!weights || !config || !buffer) {
        return false;
    }

    const uint8_t *ptr = buffer;
    const uint8_t *end = buffer + buffer_size;

    const uint32_t vocab_size = config->vocab_size;
    const uint32_t dim = config->dim;
    const uint32_t hidden_dim = config->hidden_dim;
    const uint32_t layer_count = config->layer_count;
    const uint32_t heads = config->heads;
    const uint32_t kv_heads = config->kv_heads;
    const uint32_t head_dim = dim / heads;

    /* Load embedding weights */
    size_t embedding_size = (size_t)vocab_size * dim * sizeof(float);
    if (ptr + embedding_size > end) return false;
    memcpy(weights->embedding, ptr, embedding_size);
    ptr += embedding_size;

    /* Load layer weights */
    for (uint32_t layer = 0; layer < layer_count; ++layer) {
        NiyahLlmLayerWeights *lw = &weights->layers[layer];

        /* RMSNorm weights (attention input) */
        size_t norm_size = dim * sizeof(float);
        if (ptr + norm_size > end) return false;
        memcpy(lw->attn_norm, ptr, norm_size);
        ptr += norm_size;

        /* Q projection: [heads][head_dim][dim] */
        size_t q_size = (size_t)heads * head_dim * dim * sizeof(float);
        if (ptr + q_size > end) return false;
        memcpy(lw->q, ptr, q_size);
        ptr += q_size;

        /* K projection: [kv_heads][head_dim][dim] */
        size_t k_size = (size_t)kv_heads * head_dim * dim * sizeof(float);
        if (ptr + k_size > end) return false;
        memcpy(lw->k, ptr, k_size);
        ptr += k_size;

        /* V projection: [kv_heads][head_dim][dim] */
        size_t v_size = (size_t)kv_heads * head_dim * dim * sizeof(float);
        if (ptr + v_size > end) return false;
        memcpy(lw->v, ptr, v_size);
        ptr += v_size;

        /* O projection: [dim][heads][head_dim] */
        size_t o_size = (size_t)dim * heads * head_dim * sizeof(float);
        if (ptr + o_size > end) return false;
        memcpy(lw->o, ptr, o_size);
        ptr += o_size;

        /* RMSNorm weights (FFN input) */
        if (ptr + norm_size > end) return false;
        memcpy(lw->ffn_norm, ptr, norm_size);
        ptr += norm_size;

        /* FFN gate: [hidden_dim][dim] */
        size_t ffn_gate_size = (size_t)hidden_dim * dim * sizeof(float);
        if (ptr + ffn_gate_size > end) return false;
        memcpy(lw->ffn_gate, ptr, ffn_gate_size);
        ptr += ffn_gate_size;

        /* FFN up: [hidden_dim][dim] */
        size_t ffn_up_size = (size_t)hidden_dim * dim * sizeof(float);
        if (ptr + ffn_up_size > end) return false;
        memcpy(lw->ffn_up, ptr, ffn_up_size);
        ptr += ffn_up_size;

        /* FFN down: [dim][hidden_dim] */
        size_t ffn_down_size = (size_t)dim * hidden_dim * sizeof(float);
        if (ptr + ffn_down_size > end) return false;
        memcpy(lw->ffn_down, ptr, ffn_down_size);
        ptr += ffn_down_size;
    }

    /* Load final RMSNorm weights */
    if (ptr + norm_size > end) return false;
    memcpy(weights->final_norm, ptr, norm_size);
    ptr += norm_size;

    /* Load LM head (usually tied to embedding) */
    size_t lm_head_size = (size_t)vocab_size * dim * sizeof(float);
    if (ptr + lm_head_size > end) return false;
    memcpy(weights->lm_head, ptr, lm_head_size);
    ptr += lm_head_size;

    return (ptr <= end);
}

void niyah_llm_weights_unload(NiyahLlmModelWeights *weights) {
    /* Nothing to free - weights point to external buffer */
    (void)weights;
}

const NiyahLlmModelWeights *niyah_llm_weights_view(const NiyahLlmGenerationState *state) {
    return &state->weights;
}

/* ============================================================================
 * Layer normalization (with optional weights)
 * ============================================================================ */

static void layer_norm(
    const float *input,
    float *output,
    const float *weights,
    uint32_t dim)
{
    float mean = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        mean += input[i];
    }
    mean /= (float)dim;

    float variance = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= (float)dim;

    const float eps = 1e-5f;
    const float std = 1.0f / sqrtf(variance + eps);

    if (weights) {
        for (uint32_t i = 0; i < dim; ++i) {
            output[i] = (input[i] - mean) * std * weights[i];
        }
    } else {
        for (uint32_t i = 0; i < dim; ++i) {
            output[i] = (input[i] - mean) * std;
        }
    }
}

/* ============================================================================
 * QKV projection (FIXED: now uses weight matrices)
 * ============================================================================ */

static void project_qkv(
    const float *input,           /* [dim] */
    const float *q_weights,       /* [heads][head_dim][dim] */
    const float *k_weights,       /* [kv_heads][head_dim][dim] */
    const float *v_weights,       /* [kv_heads][head_dim][dim] */
    float *q_output,              /* [heads][head_dim] */
    float *k_output,              /* [kv_heads][head_dim] */
    float *v_output,              /* [kv_heads][head_dim] */
    uint32_t dim,
    uint32_t heads,
    uint32_t kv_heads,
    uint32_t head_dim)
{
    /* Q projection: q = input @ Q^T */
    for (uint32_t h = 0; h < heads; ++h) {
        const float *q_w = q_weights + (size_t)h * head_dim * dim;
        float *q_h = q_output + (size_t)h * head_dim;
        
        for (uint32_t i = 0; i < head_dim; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < dim; ++j) {
                sum += input[j] * q_w[i * dim + j];
            }
            q_h[i] = sum;
        }
    }

    /* K projection: k = input @ K^T */
    for (uint32_t h = 0; h < kv_heads; ++h) {
        const float *k_w = k_weights + (size_t)h * head_dim * dim;
        float *k_h = k_output + (size_t)h * head_dim;
        
        for (uint32_t i = 0; i < head_dim; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < dim; ++j) {
                sum += input[j] * k_w[i * dim + j];
            }
            k_h[i] = sum;
        }
    }

    /* V projection: v = input @ V^T */
    for (uint32_t h = 0; h < kv_heads; ++h) {
        const float *v_w = v_weights + (size_t)h * head_dim * dim;
        float *v_h = v_output + (size_t)h * head_dim;
        
        for (uint32_t i = 0; i < head_dim; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < dim; ++j) {
                sum += input[j] * v_w[i * dim + j];
            }
            v_h[i] = sum;
        }
    }
}

/* ============================================================================
 * Output projection (FIXED: now uses weight matrix)
 * ============================================================================ */

static void project_o(
    const float *input,           /* [heads][head_dim] */
    const float *o_weights,       /* [dim][heads][head_dim] */
    float *output,                /* [dim] */
    uint32_t dim,
    uint32_t heads,
    uint32_t head_dim)
{
    memset(output, 0, dim * sizeof(float));
    
    for (uint32_t d = 0; d < dim; ++d) {
        float sum = 0.0f;
        for (uint32_t h = 0; h < heads; ++h) {
            const float *o_h = o_weights + (size_t)d * heads * head_dim + (size_t)h * head_dim;
            const float *input_h = input + (size_t)h * head_dim;
            
            for (uint32_t i = 0; i < head_dim; ++i) {
                sum += input_h[i] * o_h[i];
            }
        }
        output[d] = sum;
    }
}

/* ============================================================================
 * Generation step (FIXED: complete forward pass)
 * ============================================================================ */

bool niyah_llm_generation_step(
    NiyahLlmGenerationState *state,
    uint32_t *next_token,
    float *probability)
{
    if (!state || !next_token || !probability) {
        return false;
    }

    if (state->finished) {
        return false;
    }

    const uint32_t dim = state->config.dim;
    const uint32_t hidden_dim = state->config.hidden_dim;
    const uint32_t layer_count = state->config.layer_count;
    const uint32_t heads = state->config.heads;
    const uint32_t kv_heads = state->config.kv_heads;
    const uint32_t head_dim = dim / heads;
    const float temperature = state->sampler_config.temperature;
    const float top_p = state->sampler_config.top_p;
    const uint32_t top_k = state->sampler_config.top_k;

    /* Get embedding for current token */
    const float *embedding = state->weights.embedding + (size_t)state->current_token * dim;
    memcpy(state->hidden, embedding, dim * sizeof(float));

    /* Forward pass through all layers */
    for (uint32_t layer = 0; layer < layer_count; ++layer) {
        const NiyahLlmLayerWeights *lw = &state->weights.layers[layer];

        /* === Attention block === */
        
        /* RMSNorm (attention input) */
        layer_norm(state->hidden, state->normalized, lw->attn_norm, dim);

        /* QKV projection (FIXED: now uses weights) */
        project_qkv(
            state->normalized,
            lw->q, lw->k, lw->v,
            state->q, state->k, state->v,
            dim, heads, kv_heads, head_dim
        );

        /* Apply RoPE to Q and K */
        niyah_llm_apply_rope(
            state->q,
            state->k,
            state->position,
            heads,
            head_dim,
            10000.0f
        );

        /* Append K and V to cache */
        niyah_llm_kv_cache_append(
            &state->kv_cache,
            layer,
            state->k,
            state->v,
            kv_heads,
            head_dim
        );

        /* Multi-head attention with GQA */
        niyah_llm_attention(
            state->q,
            &state->kv_cache,
            layer,
            state->position,
            state->attn,
            dim,
            heads,
            kv_heads,
            head_dim
        );

        /* Output projection (FIXED: now uses weights) */
        project_o(
            state->attn,
            lw->o,
            state->attn_output,
            dim,
            heads,
            head_dim
        );

        /* Residual connection */
        for (uint32_t i = 0; i < dim; ++i) {
            state->hidden[i] += state->attn_output[i];
        }

        /* === FFN block === */
        
        /* RMSNorm (FFN input) */
        layer_norm(state->hidden, state->normalized, lw->ffn_norm, dim);

        /* SwiGLU FFN (FIXED: now executed) */
        niyah_llm_ffn_swiglu(
            state->normalized,
            lw->ffn_gate,
            lw->ffn_up,
            lw->ffn_down,
            state->ffn,
            dim,
            hidden_dim
        );

        /* Residual connection */
        for (uint32_t i = 0; i < dim; ++i) {
            state->hidden[i] += state->ffn[i];
        }

        state->position++;
    }

    /* Final RMSNorm */
    layer_norm(state->hidden, state->normalized, state->weights.final_norm, dim);

    /* LM head: logits = normalized @ lm_head^T */
    niyah_llm_matmul_f32(
        state->normalized,
        state->weights.lm_head,
        state->logits,
        1,
        dim,
        state->config.vocab_size
    );

    /* Apply temperature and sample */
    float *logits_scaled = malloc(state->config.vocab_size * sizeof(float));
    if (!logits_scaled) {
        return false;
    }

    for (uint32_t i = 0; i < state->config.vocab_size; ++i) {
        logits_scaled[i] = state->logits[i] / temperature;
    }

    *next_token = niyah_sampler_sample(
        logits_scaled,
        state->config.vocab_size,
        top_k,
        top_p,
        &state->rng,
        probability
    );

    free(logits_scaled);

    state->current_token = *next_token;
    state->generated_tokens++;

    if (state->generated_tokens >= state->max_tokens || *next_token == state->config.eos_token) {
        state->finished = true;
    }

    return true;
}
