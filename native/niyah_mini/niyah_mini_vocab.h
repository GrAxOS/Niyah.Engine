#ifndef NIYAH_MINI_VOCAB_H
#define NIYAH_MINI_VOCAB_H

#include "../niyah.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NIYAH_MINI_PAD_TOKEN_ID 0
#define NIYAH_MINI_BOS_TOKEN_ID 1
#define NIYAH_MINI_EOS_TOKEN_ID 2
#define NIYAH_MINI_UNK_TOKEN_ID 3
#define NIYAH_MINI_FACT_TOKEN_ID 4
#define NIYAH_MINI_INFERENCE_TOKEN_ID 5
#define NIYAH_MINI_UNKNOWN_TOKEN_ID 6
#define NIYAH_MINI_CONFLICTED_TOKEN_ID 7
#define NIYAH_MINI_AR_TOKEN_ID 8
#define NIYAH_MINI_EN_TOKEN_ID 9
#define NIYAH_MINI_CODE_TOKEN_ID 10
#define NIYAH_MINI_SOURCE_TOKEN_ID 11
#define NIYAH_MINI_CITE_TOKEN_ID 12
#define NIYAH_MINI_SPECIAL_TOKENS 128
#define NIYAH_MINI_MAX_TOKEN_LEN 256

typedef struct {
    char *token;
    float score;
    uint32_t hash;
} NiyahMiniVocabEntry;

typedef struct {
    NiyahMiniVocabEntry *entries;
    int32_t n_vocab;
    int32_t capacity;
    bool owns_entries;
} NiyahMiniVocab;

typedef struct {
    char *pair;
    uint32_t first;
    uint32_t second;
    float score;
} NiyahMiniBPEMerge;

typedef struct {
    NiyahMiniVocab vocab;
    NiyahMiniBPEMerge *merges;
    int32_t n_merges;
    int32_t merge_capacity;
    bool owns_merges;
    uint32_t *token_to_id;
    int32_t cache_size;
} NiyahMiniBPE;

typedef enum {
    NIYAH_MINI_TOKENIZE_BPE = 0,
    NIYAH_MINI_TOKENIZE_WORD = 1,
    NIYAH_MINI_TOKENIZE_BYTE = 2
} NiyahMiniTokenizeMode;

typedef struct {
    int32_t vocab_size;
    int32_t min_frequency;
    int32_t n_merges;
    float coverage_threshold;
    bool include_arabic;
    bool include_english;
    bool include_code;
    bool include_numbers;
    bool preserve_provenance;
} NiyahMiniVocabConfig;

NIYAH_API void niyah_mini_vocab_config_init(NiyahMiniVocabConfig *config);
NIYAH_API NiyahStatus niyah_mini_vocab_build(NiyahMiniVocab *vocab, const char **texts, const size_t *text_lengths, int32_t n_texts, const NiyahMiniVocabConfig *config);
NIYAH_API NiyahStatus niyah_mini_vocab_load(NiyahMiniVocab *vocab, const char *vocab_path, const char *merges_path);
NIYAH_API NiyahStatus niyah_mini_vocab_save(const NiyahMiniVocab *vocab, const char *vocab_path, const char *merges_path);
NIYAH_API void niyah_mini_vocab_free(NiyahMiniVocab *vocab);
NIYAH_API NiyahStatus niyah_mini_vocab_add(NiyahMiniVocab *vocab, const char *token, float score);
NIYAH_API NiyahStatus niyah_mini_bpe_init(NiyahMiniBPE *bpe, const NiyahMiniVocab *vocab, const NiyahMiniBPEMerge *merges, int32_t n_merges);
NIYAH_API void niyah_mini_bpe_free(NiyahMiniBPE *bpe);
NIYAH_API int32_t niyah_mini_bpe_tokenize(NiyahMiniBPE *bpe, const char *text, int32_t *tokens, int32_t max_tokens);
NIYAH_API char *niyah_mini_bpe_detokenize(NiyahMiniBPE *bpe, const int32_t *tokens, int32_t n_tokens);
NIYAH_API int32_t niyah_mini_vocab_lookup(const NiyahMiniVocab *vocab, const char *token);
NIYAH_API const char *niyah_mini_vocab_id_to_token(const NiyahMiniVocab *vocab, int32_t id);

#ifdef __cplusplus
}
#endif

#endif
