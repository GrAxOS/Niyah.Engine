#ifndef NIYAH_MINI_VOCAB_H
#define NIYAH_MINI_VOCAB_H

#include "../niyah.h"

/* ==========================================================================
 * Original Arabic-Centric Vocabulary for NiyahMini
 * 
 * Features:
 * - Subword tokenization with BPE (Byte Pair Encoding)
 * - Special tokens for Arabic, English, code, and evidence
 * - No borrowed vocabularies from Llama, Qwen, Mistral, etc.
 * - Built from scratch for Niyah's needs
 * ========================================================================== */

/* Special token IDs (reserved) */
#define NIYAH_MINI_PAD_TOKEN_ID      0
#define NIYAH_MINI_BOS_TOKEN_ID      1
#define NIYAH_MINI_EOS_TOKEN_ID      2
#define NIYAH_MINI_UNK_TOKEN_ID      3

/* Evidence-related special tokens */
#define NIYAH_MINI_FACT_TOKEN_ID     4
#define NIYAH_MINI_INFERENCE_TOKEN_ID 5
#define NIYAH_MINI_UNKNOWN_TOKEN_ID   6
#define NIYAH_MINI_CONFLICTED_TOKEN_ID 7

/* Language markers */
#define NIYAH_MINI_AR_TOKEN_ID       8   /* Arabic context */
#define NIYAH_MINI_EN_TOKEN_ID       9   /* English context */
#define NIYAH_MINI_CODE_TOKEN_ID     10  /* Code context */

/* Provenance tokens */
#define NIYAH_MINI_SOURCE_TOKEN_ID   11  /* Source citation marker */
#define NIYAH_MINI_CITE_TOKEN_ID     12  /* Citation marker */

/* Reserved range for special tokens */
#define NIYAH_MINI_SPECIAL_TOKENS    128

/* Maximum token length in characters */
#define NIYAH_MINI_MAX_TOKEN_LEN     256

/* Vocabulary entry */
typedef struct {
    char* token;           /* The token string (UTF-8) */
    float score;           /* BPE merge score */
    uint32_t hash;         /* Hash for quick lookup */
} NiyahMiniVocabEntry;

/* Vocabulary structure */
typedef struct {
    NiyahMiniVocabEntry* entries;  /* Array of vocabulary entries */
    int32_t n_vocab;              /* Number of entries */
    int32_t capacity;             /* Allocated capacity */
    bool owns_entries;            /* Whether we own the memory */
} NiyahMiniVocab;

/* BPE merge rule */
typedef struct {
    char* pair;             /* String pair to merge */
    uint32_t first;         /* First token ID */
    uint32_t second;        /* Second token ID */
    float score;           /* Merge score */
} NiyahMiniBPEMerge;

/* BPE tokenizer state */
typedef struct {
    NiyahMiniVocab vocab;
    NiyahMiniBPEMerge* merges;   /* BPE merge rules */
    int32_t n_merges;
    int32_t merge_capacity;
    bool owns_merges;
    
    /* Caching for performance */
    uint32_t* token_to_id;    /* Direct lookup table for common tokens */
    int32_t cache_size;
} NiyahMiniBPE;

/* Tokenization modes */
typedef enum {
    NIYAH_MINI_TOKENIZE_BPE = 0,      /* Standard BPE */
    NIYAH_MINI_TOKENIZE_WORD = 1,     /* Word-level (fallback) */
    NIYAH_MINI_TOKENIZE_BYTE = 2      /* Byte-level (always works) */
} NiyahMiniTokenizeMode;

/* ==========================================================================
 * Vocabulary Builder Configuration
 * ========================================================================== */

typedef struct {
    int32_t vocab_size;              /* Target vocabulary size */
    int32_t min_frequency;           /* Minimum frequency for inclusion */
    int32_t n_merges;                /* Number of BPE merges */
    float coverage_threshold;       /* Target coverage of training data */
    bool include_arabic;             /* Include Arabic text */
    bool include_english;            /* Include English text */
    bool include_code;               /* Include code text */
    bool include_numbers;            /* Include numeric tokens */
    bool preserve_provenance;        /* Keep track of source for each token */
} NiyahMiniVocabConfig;

/* Initialize vocabulary builder config */
NIYAH_API void niyah_mini_vocab_config_init(NiyahMiniVocabConfig* config);

/* Build vocabulary from training text */
NIYAH_API NiyahStatus niyah_mini_vocab_build(
    NiyahMiniVocab* vocab,
    const char** texts,
    const size_t* text_lengths,
    int32_t n_texts,
    const NiyahMiniVocabConfig* config
);

/* Load vocabulary from file */
NIYAH_API NiyahStatus niyah_mini_vocab_load(
    NiyahMiniVocab* vocab,
    const char* vocab_path,
    const char* merges_path
);

/* Save vocabulary to file */
NIYAH_API NiyahStatus niyah_mini_vocab_save(
    const NiyahMiniVocab* vocab,
    const char* vocab_path,
    const char* merges_path
);

/* Free vocabulary */
NIYAH_API void niyah_mini_vocab_free(NiyahMiniVocab* vocab);

/* Add a token to vocabulary */
NIYAH_API NiyahStatus niyah_mini_vocab_add(
    NiyahMiniVocab* vocab,
    const char* token,
    float score
);

/* Initialize BPE tokenizer */
NIYAH_API NiyahStatus niyah_mini_bpe_init(
    NiyahMiniBPE* bpe,
    const NiyahMiniVocab* vocab,
    const NiyahMiniBPEMerge* merges,
    int32_t n_merges
);

/* Free BPE tokenizer */
NIYAH_API void niyah_mini_bpe_free(NiyahMiniBPE* bpe);

/* Tokenize text using BPE */
NIYAH_API int32_t niyah_mini_bpe_tokenize(
    NiyahMiniBPE* bpe,
    const char* text,
    int32_t* tokens,
    int32_t max_tokens
);

/* Detokenize tokens back to text */
NIYAH_API char* niyah_mini_bpe_detokenize(
    NiyahMiniBPE* bpe,
    const int32_t* tokens,
    int32_t n_tokens
);

/* Get token ID for a string (exact match) */
NIYAH_API int32_t niyah_mini_vocab_lookup(
    const NiyahMiniVocab* vocab,
    const char* token
);

/* Get token string for an ID */
NIYAH_API const char* niyah_mini_vocab_id_to_token(
    const NiyahMiniVocab* vocab,
    int32_t id
);

#endif /* NIYAH_MINI_VOCAB_H */
