#include "niyah_tokenizer.h"

#include <string.h>

bool niyah_tokenizer_vocab_find(const NiyahVocabEntry *vocab,
                                 size_t vocab_size, const char *token,
                                 size_t length, uint32_t *out_id) {
    if (vocab == NULL || token == NULL || out_id == NULL) {
        return false;
    }
    for (size_t i = 0u; i < vocab_size; ++i) {
        if (vocab[i].length == length &&
            memcmp(vocab[i].token, token, length) == 0) {
            *out_id = (uint32_t)i;
            return true;
        }
    }
    return false;
}

static bool find_merge(const NiyahMergeRule *merges, size_t merge_count,
                        uint32_t left, uint32_t right,
                        const NiyahMergeRule **out_rule) {
    for (size_t i = 0u; i < merge_count; ++i) {
        if (merges[i].left == left && merges[i].right == right) {
            *out_rule = &merges[i];
            return true;
        }
    }
    return false;
}

bool niyah_tokenizer_encode_bytes(const char *text, size_t text_len,
                                   const NiyahVocabEntry *vocab,
                                   size_t vocab_size,
                                   const NiyahMergeRule *merges,
                                   size_t merge_count,
                                   uint32_t *token_ids_scratch,
                                   size_t max_tokens, size_t *out_count) {
    if (text == NULL || vocab == NULL || token_ids_scratch == NULL ||
        out_count == NULL) {
        return false;
    }
    if (text_len == 0u || max_tokens < text_len) {
        return false;
    }
    if (merge_count > 0u && merges == NULL) {
        return false;
    }

    size_t count = text_len;
    for (size_t i = 0u; i < text_len; ++i) {
        uint32_t id = 0u;
        if (!niyah_tokenizer_vocab_find(vocab, vocab_size, text + i, 1u,
                                         &id)) {
            return false;
        }
        token_ids_scratch[i] = id;
    }

    for (;;) {
        size_t best_index = 0u;
        uint32_t best_merged = 0u;
        bool found = false;
        uint32_t best_rank = 0u;

        for (size_t i = 0u; i + 1u < count; ++i) {
            const NiyahMergeRule *rule = NULL;
            if (find_merge(merges, merge_count, token_ids_scratch[i],
                            token_ids_scratch[i + 1u], &rule)) {
                if (!found || rule->rank < best_rank) {
                    found = true;
                    best_rank = rule->rank;
                    best_merged = rule->merged;
                    best_index = i;
                }
            }
        }

        if (!found) {
            break;
        }

        token_ids_scratch[best_index] = best_merged;
        for (size_t i = best_index + 1u; i + 1u < count; ++i) {
            token_ids_scratch[i] = token_ids_scratch[i + 1u];
        }
        count -= 1u;
    }

    *out_count = count;
    return true;
}

bool niyah_tokenizer_decode(const uint32_t *token_ids, size_t count,
                             const NiyahVocabEntry *vocab, size_t vocab_size,
                             char *out_buffer, size_t out_buffer_size,
                             size_t *out_length) {
    if (token_ids == NULL || vocab == NULL || out_buffer == NULL ||
        out_length == NULL) {
        return false;
    }

    size_t written = 0u;
    for (size_t i = 0u; i < count; ++i) {
        if (token_ids[i] >= vocab_size) {
            return false;
        }
        const NiyahVocabEntry *entry = &vocab[token_ids[i]];
        if (written + entry->length > out_buffer_size) {
            return false;
        }
        memcpy(out_buffer + written, entry->token, entry->length);
        written += entry->length;
    }

    *out_length = written;
    return true;
}
