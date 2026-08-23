#ifndef NIYAH_TOKENIZER_H
#define NIYAH_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * niyah_tokenizer: generic byte-level BPE (Byte Pair Encoding) engine, the
 * same algorithm family used by GPT-2/Llama-style tokenizers. This module
 * implements the ALGORITHM only -- it consumes a caller-supplied vocab
 * table and merge-rule table (e.g. loaded from a real tokenizer.json /
 * vocab.json + merges.txt pair) rather than embedding any specific
 * model's vocabulary. That data is external to the engine by design,
 * consistent with this project's "local-first, external data is loaded,
 * not hardcoded" convention.
 *
 * Caller owns all buffers; no allocation is performed here.
 */

typedef struct {
    const char *token;  /* pointer to the token's UTF-8 bytes, not owned */
    size_t length;       /* byte length of the token */
} NiyahVocabEntry;

typedef struct {
    uint32_t left;    /* left token id of the pair */
    uint32_t right;   /* right token id of the pair */
    uint32_t merged;  /* resulting token id after merging left+right */
    uint32_t rank;     /* merge priority; lower rank merges first */
} NiyahMergeRule;

/* Linear search for an exact-match vocab entry. Returns false if not
 * found. O(vocab_size); a real deployment would use a hash map, but this
 * project's storage layer has none yet -- see niyah_storage.c. */
bool niyah_tokenizer_vocab_find(const NiyahVocabEntry *vocab,
                                 size_t vocab_size, const char *token,
                                 size_t length, uint32_t *out_id);

/*
 * Encodes raw text into token ids using byte-level BPE:
 *   1. Each byte of `text` becomes an initial single-byte token, looked up
 *      in `vocab` (the vocab must contain an entry for every byte value
 *      that appears in the text, as is standard for byte-level BPE).
 *   2. Repeatedly finds the adjacent token pair with the lowest-rank
 *      matching merge rule and merges it, until no adjacent pair matches
 *      any rule.
 *
 * token_ids_scratch: caller-owned buffer, capacity max_tokens. Must be at
 * least as large as text_len (the worst case, zero merges).
 * out_count: final number of tokens written.
 * Returns false on NULL pointers, zero-length input, an unknown byte, an
 * undersized scratch buffer, or size_t overflow.
 */
bool niyah_tokenizer_encode_bytes(const char *text, size_t text_len,
                                   const NiyahVocabEntry *vocab,
                                   size_t vocab_size,
                                   const NiyahMergeRule *merges,
                                   size_t merge_count,
                                   uint32_t *token_ids_scratch,
                                   size_t max_tokens, size_t *out_count);

/* Concatenates vocab[token_ids[i]].token for each token, into out_buffer
 * (size out_buffer_size). Returns false on NULL pointers, an out-of-range
 * token id, or insufficient buffer space. *out_length is always set to
 * the number of bytes written (or that would be needed) on success. */
bool niyah_tokenizer_decode(const uint32_t *token_ids, size_t count,
                             const NiyahVocabEntry *vocab, size_t vocab_size,
                             char *out_buffer, size_t out_buffer_size,
                             size_t *out_length);

#endif
