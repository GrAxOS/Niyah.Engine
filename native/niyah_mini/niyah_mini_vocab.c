#include "niyah_mini_vocab.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Simple hash function for strings */
static uint32_t niyah_mini_hash_string(const char* str) {
    uint32_t hash = 5381;
    while (*str) {
        hash = ((hash << 5) + hash) + (uint32_t)(unsigned char)(*str++);
    }
    return hash;
}

/* Initialize vocabulary builder config */
void niyah_mini_vocab_config_init(NiyahMiniVocabConfig* config) {
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

/* Initialize vocabulary */
static NiyahStatus niyah_mini_vocab_init(NiyahMiniVocab* vocab) {
    if (!vocab) return NIYAH_ERR_INVALID_ARG;
    
    vocab->entries = NULL;
    vocab->n_vocab = 0;
    vocab->capacity = 0;
    vocab->owns_entries = false;
    
    return NIYAH_OK;
}

/* Free vocabulary */
void niyah_mini_vocab_free(NiyahMiniVocab* vocab) {
    if (!vocab) return;
    
    if (vocab->owns_entries && vocab->entries) {
        for (int32_t i = 0; i < vocab->n_vocab; i++) {
            free(vocab->entries[i].token);
        }
        free(vocab->entries);
    }
    
    vocab->entries = NULL;
    vocab->n_vocab = 0;
    vocab->capacity = 0;
    vocab->owns_entries = false;
}

/* Add a token to vocabulary */
NIYAH_API NiyahStatus niyah_mini_vocab_add(NiyahMiniVocab* vocab, const char* token, float score) {
    if (!vocab || !token) return NIYAH_ERR_INVALID_ARG;
    
    /* Check if we need to grow */
    if (vocab->n_vocab >= vocab->capacity) {
        int32_t new_capacity = vocab->capacity > 0 ? vocab->capacity * 2 : 256;
        NiyahMiniVocabEntry* new_entries = (NiyahMiniVocabEntry*)realloc(
            vocab->entries, (size_t)new_capacity * sizeof(NiyahMiniVocabEntry));
        if (!new_entries) return NIYAH_ERR_OUT_OF_MEMORY;
        
        vocab->entries = new_entries;
        vocab->capacity = new_capacity;
        vocab->owns_entries = true;
    }
    
    /* Allocate token string */
    size_t len = strlen(token);
    char* token_copy = (char*)malloc(len + 1);
    if (!token_copy) return NIYAH_ERR_OUT_OF_MEMORY;
    
    memcpy(token_copy, token, len + 1);
    
    /* Add entry */
    vocab->entries[vocab->n_vocab].token = token_copy;
    vocab->entries[vocab->n_vocab].score = score;
    vocab->entries[vocab->n_vocab].hash = niyah_mini_hash_string(token);
    vocab->n_vocab++;
    
    return NIYAH_OK;
}

/* Load vocabulary from file */
NiyahStatus niyah_mini_vocab_load(
    NiyahMiniVocab* vocab,
    const char* vocab_path,
    const char* merges_path
) {
    if (!vocab || !vocab_path) return NIYAH_ERR_INVALID_ARG;
    
    niyah_mini_vocab_init(vocab);
    vocab->owns_entries = true;
    
    /* Open vocabulary file */
    FILE* f = fopen(vocab_path, "r");
    if (!f) return NIYAH_ERR_IO;
    
    /* Read vocabulary entries */
    char line[NIYAH_MINI_MAX_TOKEN_LEN + 2];
    while (fgets(line, sizeof(line), f)) {
        /* Remove newline */
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        
        if (len > 0) {
            NiyahStatus status = niyah_mini_vocab_add(vocab, line, 0.0f);
            if (status != NIYAH_OK) {
                fclose(f);
                return status;
            }
        }
    }
    
    fclose(f);
    return NIYAH_OK;
}

/* Save vocabulary to file */
NiyahStatus niyah_mini_vocab_save(
    const NiyahMiniVocab* vocab,
    const char* vocab_path,
    const char* merges_path
) {
    if (!vocab || !vocab_path) return NIYAH_ERR_INVALID_ARG;
    
    /* Open vocabulary file */
    FILE* f = fopen(vocab_path, "w");
    if (!f) return NIYAH_ERR_IO;
    
    /* Write vocabulary entries */
    for (int32_t i = 0; i < vocab->n_vocab; i++) {
        fprintf(f, "%s\n", vocab->entries[i].token);
    }
    
    fclose(f);
    return NIYAH_OK;
}

/* Lookup token ID */
int32_t niyah_mini_vocab_lookup(const NiyahMiniVocab* vocab, const char* token) {
    if (!vocab || !token || !vocab->entries) return -1;
    
    uint32_t target_hash = niyah_mini_hash_string(token);
    
    for (int32_t i = 0; i < vocab->n_vocab; i++) {
        if (vocab->entries[i].hash == target_hash && 
            strcmp(vocab->entries[i].token, token) == 0) {
            return i;
        }
    }
    
    return -1;
}

/* Get token string by ID */
const char* niyah_mini_vocab_id_to_token(const NiyahMiniVocab* vocab, int32_t id) {
    if (!vocab || !vocab->entries || id < 0 || id >= vocab->n_vocab) {
        return NULL;
    }
    return vocab->entries[id].token;
}

/* Initialize BPE tokenizer */
NiyahStatus niyah_mini_bpe_init(
    NiyahMiniBPE* bpe,
    const NiyahMiniVocab* vocab,
    const NiyahMiniBPEMerge* merges,
    int32_t n_merges
) {
    if (!bpe) return NIYAH_ERR_INVALID_ARG;
    
    memset(bpe, 0, sizeof(*bpe));
    
    /* Copy vocabulary */
    if (vocab) {
        bpe->vocab = *vocab;
        bpe->vocab.owns_entries = false;  /* We don't own the entries */
    }
    
    /* Copy merges */
    if (merges && n_merges > 0) {
        bpe->merges = (NiyahMiniBPEMerge*)malloc((size_t)n_merges * sizeof(NiyahMiniBPEMerge));
        if (!bpe->merges) return NIYAH_ERR_OUT_OF_MEMORY;
        
        memcpy(bpe->merges, merges, (size_t)n_merges * sizeof(NiyahMiniBPEMerge));
        bpe->n_merges = n_merges;
        bpe->merge_capacity = n_merges;
        bpe->owns_merges = true;
    }
    
    return NIYAH_OK;
}

/* Free BPE tokenizer */
void niyah_mini_bpe_free(NiyahMiniBPE* bpe) {
    if (!bpe) return;
    
    niyah_mini_vocab_free(&bpe->vocab);
    
    if (bpe->owns_merges && bpe->merges) {
        free(bpe->merges);
    }
    
    if (bpe->token_to_id) {
        free(bpe->token_to_id);
    }
    
    memset(bpe, 0, sizeof(*bpe));
}

/* Simple BPE tokenization (greedy longest match) */
int32_t niyah_mini_bpe_tokenize(
    NiyahMiniBPE* bpe,
    const char* text,
    int32_t* tokens,
    int32_t max_tokens
) {
    if (!bpe || !text || !tokens || max_tokens <= 0) {
        return 0;
    }
    
    const size_t text_len = strlen(text);
    int32_t count = 0;
    size_t pos = 0;
    
    while (pos < text_len && count < max_tokens) {
        size_t best_len = 0;
        int32_t best_id = -1;
        
        /* Try to match longest token first */
        for (int32_t len = 1; len <= NIYAH_MINI_MAX_TOKEN_LEN && pos + len <= text_len; len++) {
            /* Extract substring */
            char token_buf[NIYAH_MINI_MAX_TOKEN_LEN + 1];
            memcpy(token_buf, text + pos, len);
            token_buf[len] = '\0';
            
            /* Lookup in vocabulary */
            int32_t tid = niyah_mini_vocab_lookup(&bpe->vocab, token_buf);
            if (tid >= 0) {
                best_len = len;
                best_id = tid;
            }
        }
        
        if (best_id >= 0) {
            tokens[count++] = best_id;
            pos += best_len;
        } else {
            /* Byte fallback */
            tokens[count++] = (int32_t)(unsigned char)text[pos];
            pos++;
        }
    }
    
    return count;
}

/* Detokenize tokens back to text */
char* niyah_mini_bpe_detokenize(
    NiyahMiniBPE* bpe,
    const int32_t* tokens,
    int32_t n_tokens
) {
    if (!bpe || !tokens || n_tokens <= 0) {
        char* empty = (char*)malloc(1);
        if (empty) empty[0] = '\0';
        return empty;
    }
    
    /* Estimate output size */
    size_t capacity = (size_t)n_tokens * NIYAH_MINI_MAX_TOKEN_LEN + 1;
    char* out = (char*)malloc(capacity);
    if (!out) return NULL;
    
    size_t used = 0;
    
    for (int32_t i = 0; i < n_tokens; i++) {
        int32_t tid = tokens[i];
        const char* token = niyah_mini_vocab_id_to_token(&bpe->vocab, tid);
        
        if (token) {
            size_t len = strlen(token);
            if (used + len >= capacity) {
                size_t next_cap = capacity * 2;
                char* grown = (char*)realloc(out, next_cap);
                if (!grown) {
                    free(out);
                    return NULL;
                }
                out = grown;
                capacity = next_cap;
            }
            memcpy(out + used, token, len);
            used += len;
        } else {
            /* Handle byte tokens */
            if (tid >= 0 && tid <= 255) {
                if (used + 1 >= capacity) {
                    size_t next_cap = capacity * 2;
                    char* grown = (char*)realloc(out, next_cap);
                    if (!grown) {
                        free(out);
                        return NULL;
                    }
                    out = grown;
                    capacity = next_cap;
                }
                out[used++] = (char)(unsigned char)tid;
            }
        }
    }
    
    out[used] = '\0';
    return out;
}

/* Build vocabulary from text (simplified version for C) */
NiyahStatus niyah_mini_vocab_build(
    NiyahMiniVocab* vocab,
    const char** texts,
    const size_t* text_lengths,
    int32_t n_texts,
    const NiyahMiniVocabConfig* config
) {
    if (!vocab || !texts || !text_lengths || n_texts <= 0 || !config) {
        return NIYAH_ERR_INVALID_ARG;
    }
    
    niyah_mini_vocab_init(vocab);
    vocab->owns_entries = true;
    
    /* Add special tokens first */
    const char* special_tokens[] = {
        "<pad>", "<bos>", "<eos>", "<unk>",
        "<fact>", "<inference>", "<unknown>", "<conflicted>",
        "<ar>", "<en>", "<code>",
        "<source>", "<cite>"
    };
    
    for (size_t i = 0; i < sizeof(special_tokens) / sizeof(special_tokens[0]); i++) {
        NiyahStatus status = niyah_mini_vocab_add(vocab, special_tokens[i], 0.0f);
        if (status != NIYAH_OK) {
            niyah_mini_vocab_free(vocab);
            return status;
        }
    }
    
    /* Count character frequencies */
    int32_t* char_counts = (int32_t*)calloc(256, sizeof(int32_t));
    if (!char_counts) {
        niyah_mini_vocab_free(vocab);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    
    /* Count frequencies and add characters */
    for (int32_t i = 0; i < n_texts; i++) {
        const char* text = texts[i];
        size_t len = text_lengths[i];
        
        for (size_t j = 0; j < len; j++) {
            unsigned char c = (unsigned char)text[j];
            char_counts[c]++;
        }
    }
    
    /* Add frequent characters */
    for (int32_t c = 0; c < 256; c++) {
        if (char_counts[c] >= config->min_frequency) {
            char buf[2] = {(char)c, '\0'};
            NiyahStatus status = niyah_mini_vocab_add(vocab, buf, (float)char_counts[c]);
            if (status != NIYAH_OK) {
                free(char_counts);
                niyah_mini_vocab_free(vocab);
                return status;
            }
        }
    }
    
    free(char_counts);
    return NIYAH_OK;
}
