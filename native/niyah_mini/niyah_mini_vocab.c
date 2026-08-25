#include "niyah_mini_vocab.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

static uint32_t niyah_mini_hash_string(const char *str)
{
    uint32_t hash = 5381U;
    while (*str != '\0') {
        hash = ((hash << 5U) + hash) + (uint32_t)(unsigned char)*str;
        ++str;
    }
    return hash;
}

static int size_mul_ok(size_t a, size_t b, size_t *out)
{
    if (a != 0U && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int size_add_ok(size_t a, size_t b, size_t *out)
{
    if (b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}

void niyah_mini_vocab_config_init(NiyahMiniVocabConfig *config)
{
    if (!config) return;
    config->vocab_size = 32768;
    config->min_frequency = 5;
    config->n_merges = 10000;
    config->coverage_threshold = 0.99f;
    config->include_arabic = true;
    config->include_english = true;
    config->include_code = true;
    config->include_numbers = true;
    config->preserve_provenance = true;
}

void niyah_mini_vocab_free(NiyahMiniVocab *vocab)
{
    int32_t i;
    if (!vocab) return;
    if (vocab->owns_entries && vocab->entries) {
        for (i = 0; i < vocab->n_vocab; ++i) free(vocab->entries[i].token);
        free(vocab->entries);
    }
    vocab->entries = NULL;
    vocab->n_vocab = 0;
    vocab->capacity = 0;
    vocab->owns_entries = false;
}

static NiyahStatus vocab_reserve(NiyahMiniVocab *vocab, int32_t required)
{
    NiyahMiniVocabEntry *new_entries;
    size_t bytes;
    int32_t new_capacity;
    if (!vocab || required < 0) return NIYAH_ERR_INVALID_ARG;
    if (required <= vocab->capacity) return NIYAH_OK;
    if (required > NIYAH_MAX_VOCAB) return NIYAH_ERR_SHAPE;
    new_capacity = vocab->capacity > 0 ? vocab->capacity : 256;
    while (new_capacity < required) {
        if (new_capacity > NIYAH_MAX_VOCAB / 2) {
            new_capacity = NIYAH_MAX_VOCAB;
        } else {
            new_capacity *= 2;
        }
        if (new_capacity < 0 || new_capacity > NIYAH_MAX_VOCAB) return NIYAH_ERR_OVERFLOW;
    }
    if (!size_mul_ok((size_t)new_capacity, sizeof(*new_entries), &bytes)) return NIYAH_ERR_OVERFLOW;
    new_entries = (NiyahMiniVocabEntry *)realloc(vocab->entries, bytes);
    if (!new_entries) return NIYAH_ERR_OUT_OF_MEMORY;
    if (new_capacity > vocab->capacity) {
        memset(new_entries + vocab->capacity, 0, (size_t)(new_capacity - vocab->capacity) * sizeof(*new_entries));
    }
    vocab->entries = new_entries;
    vocab->capacity = new_capacity;
    vocab->owns_entries = true;
    return NIYAH_OK;
}

static NiyahStatus vocab_append_token(NiyahMiniVocab *vocab, const char *token, float score)
{
    size_t len;
    char *copy;
    NiyahStatus status;
    if (!vocab || !token) return NIYAH_ERR_INVALID_ARG;
    len = strlen(token);
    if (len == 0U || len > NIYAH_MINI_MAX_TOKEN_LEN) return NIYAH_ERR_INVALID_ARG;
    if (niyah_mini_vocab_lookup(vocab, token) >= 0) return NIYAH_OK;
    status = vocab_reserve(vocab, vocab->n_vocab + 1);
    if (status != NIYAH_OK) return status;
    if (len == SIZE_MAX) return NIYAH_ERR_OVERFLOW;
    copy = (char *)malloc(len + 1U);
    if (!copy) return NIYAH_ERR_OUT_OF_MEMORY;
    memcpy(copy, token, len + 1U);
    vocab->entries[vocab->n_vocab].token = copy;
    vocab->entries[vocab->n_vocab].score = score;
    vocab->entries[vocab->n_vocab].hash = niyah_mini_hash_string(token);
    ++vocab->n_vocab;
    return NIYAH_OK;
}

NiyahStatus niyah_mini_vocab_add(NiyahMiniVocab *vocab, const char *token, float score)
{
    return vocab_append_token(vocab, token, score);
}

NiyahStatus niyah_mini_vocab_load(NiyahMiniVocab *vocab, const char *vocab_path, const char *merges_path)
{
    FILE *f;
    char line[NIYAH_MINI_MAX_TOKEN_LEN + 2];
    NiyahStatus status;
    size_t len;
    (void)merges_path;
    if (!vocab || !vocab_path) return NIYAH_ERR_INVALID_ARG;
    niyah_mini_vocab_free(vocab);
    f = fopen(vocab_path, "rb");
    if (!f) return NIYAH_ERR_IO;
    while (fgets(line, sizeof(line), f)) {
        len = strlen(line);
        while (len > 0U && (line[len - 1U] == '\n' || line[len - 1U] == '\r')) line[--len] = '\0';
        if (len == 0U) continue;
        status = vocab_append_token(vocab, line, 0.0f);
        if (status != NIYAH_OK) {
            fclose(f);
            niyah_mini_vocab_free(vocab);
            return status;
        }
    }
    if (ferror(f)) {
        fclose(f);
        niyah_mini_vocab_free(vocab);
        return NIYAH_ERR_IO;
    }
    fclose(f);
    return NIYAH_OK;
}

NiyahStatus niyah_mini_vocab_save(const NiyahMiniVocab *vocab, const char *vocab_path, const char *merges_path)
{
    FILE *f;
    int32_t i;
    (void)merges_path;
    if (!vocab || !vocab_path || vocab->n_vocab < 0) return NIYAH_ERR_INVALID_ARG;
    if (vocab->n_vocab > 0 && !vocab->entries) return NIYAH_ERR_INVALID_ARG;
    f = fopen(vocab_path, "wb");
    if (!f) return NIYAH_ERR_IO;
    for (i = 0; i < vocab->n_vocab; ++i) {
        if (!vocab->entries[i].token || fprintf(f, "%s\n", vocab->entries[i].token) < 0) {
            fclose(f);
            return NIYAH_ERR_IO;
        }
    }
    if (fclose(f) != 0) return NIYAH_ERR_IO;
    return NIYAH_OK;
}

int32_t niyah_mini_vocab_lookup(const NiyahMiniVocab *vocab, const char *token)
{
    uint32_t hash;
    int32_t i;
    if (!vocab || !token || vocab->n_vocab < 0 || (vocab->n_vocab > 0 && !vocab->entries)) return -1;
    hash = niyah_mini_hash_string(token);
    for (i = 0; i < vocab->n_vocab; ++i) {
        if (vocab->entries[i].token && vocab->entries[i].hash == hash && strcmp(vocab->entries[i].token, token) == 0) return i;
    }
    return -1;
}

const char *niyah_mini_vocab_id_to_token(const NiyahMiniVocab *vocab, int32_t id)
{
    if (!vocab || id < 0 || id >= vocab->n_vocab || !vocab->entries) return NULL;
    return vocab->entries[id].token;
}

static void bpe_zero(NiyahMiniBPE *bpe)
{
    if (bpe) memset(bpe, 0, sizeof(*bpe));
}

NiyahStatus niyah_mini_bpe_init(NiyahMiniBPE *bpe, const NiyahMiniVocab *vocab, const NiyahMiniBPEMerge *merges, int32_t n_merges)
{
    int32_t i;
    size_t bytes;
    if (!bpe) return NIYAH_ERR_INVALID_ARG;
    if (n_merges < 0 || n_merges > NIYAH_MAX_VOCAB) return NIYAH_ERR_INVALID_ARG;
    bpe_zero(bpe);
    if (vocab) {
        bpe->vocab = *vocab;
        bpe->vocab.owns_entries = false;
    }
    if (n_merges == 0) return NIYAH_OK;
    if (!merges) return NIYAH_ERR_INVALID_ARG;
    if (!size_mul_ok((size_t)n_merges, sizeof(*bpe->merges), &bytes)) return NIYAH_ERR_OVERFLOW;
    bpe->merges = (NiyahMiniBPEMerge *)calloc(1U, bytes);
    if (!bpe->merges) return NIYAH_ERR_OUT_OF_MEMORY;
    for (i = 0; i < n_merges; ++i) {
        size_t len;
        if (merges[i].first >= (uint32_t)vocab->n_vocab || merges[i].second >= (uint32_t)vocab->n_vocab || !merges[i].pair) {
            niyah_mini_bpe_free(bpe);
            return NIYAH_ERR_SHAPE;
        }
        len = strlen(merges[i].pair);
        if (len == 0U || len > NIYAH_MINI_MAX_TOKEN_LEN) {
            niyah_mini_bpe_free(bpe);
            return NIYAH_ERR_INVALID_ARG;
        }
        bpe->merges[i] = merges[i];
        bpe->merges[i].pair = (char *)malloc(len + 1U);
        if (!bpe->merges[i].pair) {
            niyah_mini_bpe_free(bpe);
            return NIYAH_ERR_OUT_OF_MEMORY;
        }
        memcpy(bpe->merges[i].pair, merges[i].pair, len + 1U);
    }
    bpe->n_merges = n_merges;
    bpe->merge_capacity = n_merges;
    bpe->owns_merges = true;
    return NIYAH_OK;
}

void niyah_mini_bpe_free(NiyahMiniBPE *bpe)
{
    int32_t i;
    if (!bpe) return;
    if (bpe->owns_merges && bpe->merges) {
        for (i = 0; i < bpe->n_merges; ++i) free(bpe->merges[i].pair);
        free(bpe->merges);
    }
    free(bpe->token_to_id);
    bpe->token_to_id = NULL;
    bpe->cache_size = 0;
    bpe->merges = NULL;
    bpe->n_merges = 0;
    bpe->merge_capacity = 0;
    bpe->owns_merges = false;
    bpe->vocab.entries = NULL;
    bpe->vocab.n_vocab = 0;
    bpe->vocab.capacity = 0;
    bpe->vocab.owns_entries = false;
}

static int32_t byte_fallback_id(const NiyahMiniVocab *vocab, unsigned char c)
{
    char token[2];
    token[0] = (char)c;
    token[1] = '\0';
    return niyah_mini_vocab_lookup(vocab, token);
}

int32_t niyah_mini_bpe_tokenize(NiyahMiniBPE *bpe, const char *text, int32_t *tokens, int32_t max_tokens)
{
    size_t text_len;
    size_t pos;
    int32_t count;
    if (!bpe || !text || !tokens || max_tokens <= 0) return 0;
    text_len = strlen(text);
    pos = 0U;
    count = 0;
    while (pos < text_len && count < max_tokens) {
        size_t best_len = 0U;
        int32_t best_id = -1;
        size_t len;
        for (len = 1U; len <= NIYAH_MINI_MAX_TOKEN_LEN && len <= text_len - pos; ++len) {
            char buf[NIYAH_MINI_MAX_TOKEN_LEN + 1U];
            int32_t id;
            memcpy(buf, text + pos, len);
            buf[len] = '\0';
            id = niyah_mini_vocab_lookup(&bpe->vocab, buf);
            if (id >= 0) {
                best_len = len;
                best_id = id;
            }
        }
        if (best_id < 0) {
            best_id = byte_fallback_id(&bpe->vocab, (unsigned char)text[pos]);
            if (best_id < 0) best_id = NIYAH_MINI_UNK_TOKEN_ID;
            best_len = 1U;
        }
        tokens[count++] = best_id;
        pos += best_len;
    }
    return count;
}

char *niyah_mini_bpe_detokenize(NiyahMiniBPE *bpe, const int32_t *tokens, int32_t n_tokens)
{
    size_t capacity = 1U;
    size_t used = 0U;
    char *out;
    int32_t i;
    if (!bpe || n_tokens < 0 || (n_tokens > 0 && !tokens)) return NULL;
    for (i = 0; i < n_tokens; ++i) {
        const char *tok = niyah_mini_vocab_id_to_token(&bpe->vocab, tokens[i]);
        size_t len;
        if (!tok) {
            if (tokens[i] >= 0 && tokens[i] <= 255) len = 1U;
            else continue;
        } else {
            len = strlen(tok);
        }
        if (len > SIZE_MAX - capacity) return NULL;
        capacity += len;
    }
    out = (char *)malloc(capacity);
    if (!out) return NULL;
    for (i = 0; i < n_tokens; ++i) {
        const char *tok = niyah_mini_vocab_id_to_token(&bpe->vocab, tokens[i]);
        if (tok) {
            size_t len = strlen(tok);
            memcpy(out + used, tok, len);
            used += len;
        } else if (tokens[i] >= 0 && tokens[i] <= 255) {
            out[used++] = (char)(unsigned char)tokens[i];
        }
    }
    out[used] = '\0';
    return out;
}

NiyahStatus niyah_mini_vocab_build(NiyahMiniVocab *vocab, const char **texts, const size_t *text_lengths, int32_t n_texts, const NiyahMiniVocabConfig *config)
{
    static const char *special[] = {"<pad>", "<bos>", "<eos>", "<unk>", "<fact>", "<inference>", "<unknown>", "<conflicted>", "<ar>", "<en>", "<code>", "<source>", "<cite>"};
    uint32_t counts[256];
    int32_t i;
    size_t j;
    NiyahStatus status;
    if (!vocab || !texts || !text_lengths || !config || n_texts <= 0) return NIYAH_ERR_INVALID_ARG;
    if (config->vocab_size <= 0 || config->vocab_size > NIYAH_MAX_VOCAB || config->min_frequency < 1 || config->n_merges < 0 || config->coverage_threshold <= 0.0f || config->coverage_threshold > 1.0f) return NIYAH_ERR_INVALID_ARG;
    niyah_mini_vocab_free(vocab);
    memset(counts, 0, sizeof(counts));
    for (i = 0; i < (int32_t)(sizeof(special) / sizeof(special[0])); ++i) {
        status = vocab_append_token(vocab, special[i], 0.0f);
        if (status != NIYAH_OK) goto fail;
    }
    for (i = 0; i < n_texts; ++i) {
        if (!texts[i]) { status = NIYAH_ERR_INVALID_ARG; goto fail; }
        if (text_lengths[i] > SIZE_MAX - 1U) { status = NIYAH_ERR_OVERFLOW; goto fail; }
        for (j = 0; j < text_lengths[i]; ++j) {
            unsigned char c = (unsigned char)texts[i][j];
            if (counts[c] != UINT32_MAX) ++counts[c];
        }
    }
    for (i = 0; i < 256 && vocab->n_vocab < config->vocab_size; ++i) {
        if (counts[i] >= (uint32_t)config->min_frequency) {
            char token[2];
            token[0] = (char)i;
            token[1] = '\0';
            status = vocab_append_token(vocab, token, (float)counts[i]);
            if (status != NIYAH_OK) goto fail;
        }
    }
    return NIYAH_OK;
fail:
    niyah_mini_vocab_free(vocab);
    return status;
}
