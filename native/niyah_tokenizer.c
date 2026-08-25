#include "niyah.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
<<<<<<< HEAD
#include <ctype.h>

/* ── Simple whitespace tokenizer with vocab lookup ─────────────────────── */

/* Split text on whitespace; return token count */
static int32_t split_words(const char* text,
                            char words[][64], int32_t max_words) {
    int32_t count = 0;
    int32_t wi    = 0;
    int32_t in_word = 0;

    for (const char* p = text; *p && count < max_words; p++) {
        if (isspace((unsigned char)*p)) {
            if (in_word) {
                words[count][wi] = '\0';
                count++;
                wi     = 0;
                in_word = 0;
            }
        } else {
            if (wi < 63) words[count][wi++] = (char)tolower((unsigned char)*p);
            in_word = 1;
        }
    }
    if (in_word && count < max_words) {
        words[count][wi] = '\0';
        count++;
    }
    return count;
=======

/*
 * Greedy longest-match tokenisation against the loaded vocab, with a byte
 * fallback.
 *
 * Correctness note (2026-08):
 *   The previous fallback emitted the raw byte value as a token id and the
 *   comment claimed this kept the mapping "total and reversible". It did not.
 *   niyah_tokenizer_load() assigns ids sequentially from 0, so byte 0xD8 (216)
 *   and vocab entry #216 are the same id. niyah_detokenize() resolves vocab
 *   entries before byte values, so any fallback byte below n_vocab decoded as
 *   an unrelated vocab piece.
 *
 *   This is not an edge case for Arabic: every letter in U+0600..U+06FF is
 *   encoded with a 0xD8 or 0xD9 lead byte, so essentially all Arabic input
 *   was corrupted on the way back out.
 *
 *   Fallback now resolves a real "<0xXX>" vocab entry, which is the GGUF /
 *   llama.cpp byte-token convention, so fallback ids are genuine vocab ids and
 *   cannot collide.
 */

int32_t niyah_tokenizer_lookup(const NiyahTokenizer* tokenizer,
                               const char* piece)
{
    if (!tokenizer || !piece || !tokenizer->vocab.vocab) {
        return -1;
    }
    for (int32_t i = 0; i < tokenizer->vocab.n_vocab; ++i) {
        const char* entry = tokenizer->vocab.vocab[i];
        if (entry && strcmp(entry, piece) == 0) {
            return tokenizer->vocab.ids ? tokenizer->vocab.ids[i] : i;
        }
    }
    return -1;
}

/*
 * "<0x41>" -> 0x41. Returns -1 for anything that is not a byte token.
 * Accepts either hex case; both spellings occur in published vocabs.
 */
static int byte_token_value(const char* piece)
{
    if (!piece) {
        return -1;
    }
    if (piece[0] != '<' || piece[1] != '0' ||
        (piece[2] != 'x' && piece[2] != 'X')) {
        return -1;
    }

    int value = 0;
    int digits = 0;
    const char* p = piece + 3;

    for (; *p && *p != '>'; ++p, ++digits) {
        int d;
        if (*p >= '0' && *p <= '9') {
            d = *p - '0';
        } else if (*p >= 'a' && *p <= 'f') {
            d = *p - 'a' + 10;
        } else if (*p >= 'A' && *p <= 'F') {
            d = *p - 'A' + 10;
        } else {
            return -1;
        }
        value = value * 16 + d;
    }

    if (*p != '>' || p[1] != '\0' || digits == 0 || digits > 2) {
        return -1;
    }
    return value;   /* 0..255 */
}

static int32_t byte_token_id(const NiyahTokenizer* tokenizer, unsigned char b)
{
    char piece[8];

    snprintf(piece, sizeof(piece), "<0x%02X>", (unsigned)b);
    const int32_t upper = niyah_tokenizer_lookup(tokenizer, piece);
    if (upper >= 0) {
        return upper;
    }

    snprintf(piece, sizeof(piece), "<0x%02x>", (unsigned)b);
    return niyah_tokenizer_lookup(tokenizer, piece);
>>>>>>> origin/main
}

int32_t niyah_tokenize(NiyahTokenizer* tokenizer,
                       const char* text,
<<<<<<< HEAD
                       int32_t* tokens, int32_t max_tokens) {
    if (!text || !tokens || max_tokens <= 0) return 0;

    /* Fast path: no vocab → character-level encoding */
    if (!tokenizer || tokenizer->vocab.n_vocab == 0) {
        int32_t n = 0;
        for (const char* p = text; *p && n < max_tokens; p++, n++)
            tokens[n] = (int32_t)(unsigned char)*p;
        return n;
    }

    char words[NIYAH_MAX_TOKENS][64];
    int32_t wcount = split_words(text, words, NIYAH_MAX_TOKENS);
    int32_t n      = 0;

    for (int32_t w = 0; w < wcount && n < max_tokens; w++) {
        /* Linear scan of vocab */
        int32_t found = -1;
        for (int32_t v = 0; v < tokenizer->vocab.n_vocab; v++) {
            if (tokenizer->vocab.vocab[v] &&
                strcmp(tokenizer->vocab.vocab[v], words[w]) == 0) {
                found = tokenizer->vocab.ids[v];
                break;
            }
        }
        tokens[n++] = (found >= 0) ? found : 1; /* 1 = <unk> */
    }
    return n;
}

char* niyah_detokenize(NiyahTokenizer* tokenizer,
                       const int32_t* tokens, int32_t n_tokens) {
    if (!tokens || n_tokens <= 0) return NULL;

    /* Estimate output size: max 32 chars per token + spaces */
    size_t buf_size = (size_t)n_tokens * 33 + 1;
    char*  buf      = (char*)malloc(buf_size);
    if (!buf) return NULL;
    buf[0] = '\0';

    size_t pos = 0;
    for (int32_t i = 0; i < n_tokens && pos < buf_size - 2; i++) {
        int32_t id = tokens[i];

        if (!tokenizer || tokenizer->vocab.n_vocab == 0) {
            /* Character-level decode */
            if (id > 0 && id < 256) buf[pos++] = (char)id;
        } else {
            /* Vocab decode */
            for (int32_t v = 0; v < tokenizer->vocab.n_vocab; v++) {
                if (tokenizer->vocab.ids[v] == id && tokenizer->vocab.vocab[v]) {
                    size_t len = strlen(tokenizer->vocab.vocab[v]);
                    if (pos + len + 2 < buf_size) {
                        if (pos > 0) buf[pos++] = ' ';
                        memcpy(buf + pos, tokenizer->vocab.vocab[v], len);
                        pos += len;
                    }
                    break;
                }
            }
        }
    }
    buf[pos] = '\0';
    return buf;
=======
                       int32_t* tokens,
                       int32_t max_tokens)
{
    if (!tokenizer || !text || !tokens || max_tokens <= 0) {
        return 0;
    }

    const size_t  len     = strlen(text);
    const int32_t n_vocab = tokenizer->vocab.vocab
        ? tokenizer->vocab.n_vocab : 0;

    int32_t count = 0;
    size_t  pos   = 0;
    bool    warned = false;

    while (pos < len && count < max_tokens) {
        int32_t best_id  = -1;
        size_t  best_len = 0;

        for (int32_t i = 0; i < n_vocab; ++i) {
            const char* piece = tokenizer->vocab.vocab[i];
            if (!piece || !piece[0]) {
                continue;
            }
            const size_t plen = strlen(piece);
            if (plen <= best_len || plen > len - pos) {
                continue;
            }
            if (memcmp(text + pos, piece, plen) == 0) {
                best_len = plen;
                best_id  = tokenizer->vocab.ids
                    ? tokenizer->vocab.ids[i] : i;
            }
        }

        if (best_id >= 0 && best_len > 0) {
            tokens[count++] = best_id;
            pos += best_len;
            continue;
        }

        /*
         * Byte fallback. Prefer a genuine "<0xXX>" vocab entry so the id is a
         * real vocab id and the mapping is reversible.
         */
        const unsigned char raw = (unsigned char)text[pos];
        int32_t fallback = byte_token_id(tokenizer, raw);

        if (fallback < 0 && (int32_t)raw >= n_vocab) {
            /* No byte tokens in this vocab, but this id cannot collide. */
            fallback = (int32_t)raw;
        }
        if (fallback < 0) {
            fallback = niyah_tokenizer_lookup(tokenizer, "<unk>");
        }

        if (fallback < 0) {
            /*
             * Unrepresentable. Dropping the byte is lossy, but it is at least
             * declared: the caller can compare the token count against the
             * input length. Emitting the raw value here is what used to
             * corrupt Arabic.
             */
            if (!warned) {
                fprintf(stderr,
                        "niyah: vocab has no <0x%02X> byte token and no "
                        "<unk>; byte 0x%02X dropped. UTF-8 round trip is "
                        "lossy with this vocab.\n",
                        (unsigned)raw, (unsigned)raw);
                warned = true;
            }
            pos += 1u;
            continue;
        }

        tokens[count++] = fallback;
        pos += 1u;
    }

    /*
     * Do NOT store `tokens` or `count` back into the tokenizer struct.
     * The buffer belongs to the caller; aliasing it here creates a
     * use-after-free hazard when the caller reuses or frees the buffer.
     * Callers must use the return value to obtain the token count.
     */
    return count;
}

char* niyah_detokenize(NiyahTokenizer* tokenizer,
                       const int32_t* tokens,
                       int32_t n_tokens)
{
    if (!tokens || n_tokens <= 0) {
        char* empty = (char*)malloc(1);
        if (empty) {
            empty[0] = '\0';
        }
        return empty;
    }

    const int32_t n_vocab = (tokenizer && tokenizer->vocab.vocab)
        ? tokenizer->vocab.n_vocab : 0;

    size_t capacity = (size_t)n_tokens * 8u + 1u;
    char*  out      = (char*)malloc(capacity);
    if (!out) {
        return NULL;
    }
    size_t used = 0;

    for (int32_t t = 0; t < n_tokens; ++t) {
        const int32_t id    = tokens[t];
        const char*   piece = NULL;
        char          byte_buf[2];

        for (int32_t i = 0; i < n_vocab; ++i) {
            const int32_t vid = tokenizer->vocab.ids
                ? tokenizer->vocab.ids[i] : i;
            if (vid == id) {
                piece = tokenizer->vocab.vocab[i];
                break;
            }
        }

        if (piece) {
            /* A "<0xXX>" entry represents one raw byte, not that literal. */
            const int bv = byte_token_value(piece);
            if (bv >= 0) {
                byte_buf[0] = (char)(unsigned char)bv;
                byte_buf[1] = '\0';
                piece = byte_buf;
            }
        } else if (id >= 0 && id <= 255 && id >= n_vocab) {
            /*
             * Mirrors the tokenizer: a bare byte id is only trusted when it
             * lies outside the vocab id space and therefore cannot be a
             * mis-resolved vocab entry.
             */
            byte_buf[0] = (char)(unsigned char)id;
            byte_buf[1] = '\0';
            piece = byte_buf;
        } else {
            continue;   /* unknown id: skip rather than invent a character */
        }

        const size_t plen = strlen(piece);
        if (used + plen + 1u > capacity) {
            size_t next = capacity * 2u;
            while (next < used + plen + 1u) {
                next *= 2u;
            }
            char* grown = (char*)realloc(out, next);
            if (!grown) {
                free(out);
                return NULL;
            }
            out      = grown;
            capacity = next;
        }

        memcpy(out + used, piece, plen);
        used += plen;
    }

    out[used] = '\0';
    return out;
}

NiyahStatus niyah_tokenizer_load(NiyahTokenizer* tokenizer,
                                 const char* vocab_path)
{
    if (!tokenizer || !vocab_path) {
        return NIYAH_ERR_INVALID_ARG;
    }

    FILE* f = fopen(vocab_path, "rb");
    if (!f) {
        return NIYAH_ERR_IO;
    }

    memset(tokenizer, 0, sizeof(*tokenizer));

    int32_t  capacity    = 1024;
    char**   vocab       = (char**)calloc((size_t)capacity, sizeof(char*));
    int32_t* ids         = (int32_t*)calloc((size_t)capacity, sizeof(int32_t));
    if (!vocab || !ids) {
        free(vocab);
        free(ids);
        fclose(f);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    int32_t     n           = 0;
    NiyahStatus load_status = NIYAH_OK;
    char        line[1024];

    while (fgets(line, sizeof(line), f)) {
        size_t l = strlen(line);

        /*
         * Detect fgets truncation: if the buffer is full and the last byte
         * is not a newline, the vocab entry was longer than sizeof(line)-1.
         * Storing a truncated piece would silently corrupt the vocabulary;
         * drain the rest of the line and skip it.
         */
        if (l == sizeof(line) - 1u && line[l - 1u] != '\n') {
            int ch;
            while ((ch = fgetc(f)) != EOF && ch != '\n') {
                /* drain the oversized line */
            }
            load_status = NIYAH_ERR_IO;
            continue;
        }

        while (l > 0 && (line[l - 1] == '\n' || line[l - 1] == '\r')) {
            line[--l] = '\0';
        }

        if (n >= capacity) {
            const int32_t next = capacity * 2;
            char**   gv = (char**)realloc(vocab, (size_t)next * sizeof(char*));
            int32_t* gi = (int32_t*)realloc(ids,  (size_t)next * sizeof(int32_t));
            if (!gv || !gi) {
                if (gv) vocab = gv;
                if (gi) ids   = gi;
                for (int32_t i = 0; i < n; ++i) free(vocab[i]);
                free(vocab);
                free(ids);
                fclose(f);
                return NIYAH_ERR_OUT_OF_MEMORY;
            }
            vocab    = gv;
            ids      = gi;
            capacity = next;
        }

        char* copy = (char*)malloc(l + 1u);
        if (!copy) {
            for (int32_t i = 0; i < n; ++i) free(vocab[i]);
            free(vocab);
            free(ids);
            fclose(f);
            return NIYAH_ERR_OUT_OF_MEMORY;
        }
        memcpy(copy, line, l + 1u);

        vocab[n] = copy;
        ids[n]   = n;
        ++n;
    }

    fclose(f);

    tokenizer->vocab.vocab   = vocab;
    tokenizer->vocab.ids     = ids;
    tokenizer->vocab.n_vocab = n;
    tokenizer->owns_vocab    = true;

    return load_status;
}

void niyah_tokenizer_free(NiyahTokenizer* tokenizer)
{
    if (!tokenizer) {
        return;
    }
    if (tokenizer->owns_vocab && tokenizer->vocab.vocab) {
        for (int32_t i = 0; i < tokenizer->vocab.n_vocab; ++i) {
            free(tokenizer->vocab.vocab[i]);
        }
        free(tokenizer->vocab.vocab);
        free(tokenizer->vocab.ids);
    }
    memset(tokenizer, 0, sizeof(*tokenizer));
>>>>>>> origin/main
}
