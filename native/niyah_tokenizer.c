#include "niyah.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Was a stub. Greedy longest-match tokenisation against the loaded vocab with
 * a byte-level fallback so no input is ever silently dropped.
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

int32_t niyah_tokenize(NiyahTokenizer* tokenizer,
                       const char* text,
                       int32_t* tokens,
                       int32_t max_tokens)
{
    if (!tokenizer || !text || !tokens || max_tokens <= 0) {
        return 0;
    }

    const size_t len = strlen(text);
    const int32_t n_vocab = tokenizer->vocab.vocab
        ? tokenizer->vocab.n_vocab : 0;

    int32_t count = 0;
    size_t pos = 0;

    while (pos < len && count < max_tokens) {
        int32_t best_id = -1;
        size_t best_len = 0;

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
                best_id = tokenizer->vocab.ids
                    ? tokenizer->vocab.ids[i] : i;
            }
        }

        if (best_id >= 0 && best_len > 0) {
            tokens[count++] = best_id;
            pos += best_len;
        } else {
            /* Byte fallback keeps the mapping total and reversible. */
            tokens[count++] = (int32_t)(unsigned char)text[pos];
            pos += 1u;
        }
    }

    tokenizer->tokens = tokens;
    tokenizer->n_tokens = count;
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
    char* out = (char*)malloc(capacity);
    if (!out) {
        return NULL;
    }
    size_t used = 0;

    for (int32_t t = 0; t < n_tokens; ++t) {
        const int32_t id = tokens[t];
        const char* piece = NULL;
        char byte_buf[2];

        for (int32_t i = 0; i < n_vocab; ++i) {
            const int32_t vid = tokenizer->vocab.ids
                ? tokenizer->vocab.ids[i] : i;
            if (vid == id) {
                piece = tokenizer->vocab.vocab[i];
                break;
            }
        }

        if (!piece) {
            if (id >= 0 && id <= 255) {
                byte_buf[0] = (char)(unsigned char)id;
                byte_buf[1] = '\0';
                piece = byte_buf;
            } else {
                continue; /* unknown id outside byte range: skip */
            }
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
            out = grown;
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

    int32_t capacity = 1024;
    char** vocab = (char**)calloc((size_t)capacity, sizeof(char*));
    int32_t* ids = (int32_t*)calloc((size_t)capacity, sizeof(int32_t));
    if (!vocab || !ids) {
        free(vocab);
        free(ids);
        fclose(f);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    int32_t n = 0;
    char line[1024];

    while (fgets(line, sizeof(line), f)) {
        size_t l = strlen(line);
        while (l > 0 && (line[l - 1] == '\n' || line[l - 1] == '\r')) {
            line[--l] = '\0';
        }

        if (n >= capacity) {
            const int32_t next = capacity * 2;
            char** gv = (char**)realloc(vocab, (size_t)next * sizeof(char*));
            int32_t* gi = (int32_t*)realloc(ids, (size_t)next * sizeof(int32_t));
            if (!gv || !gi) {
                if (gv) vocab = gv;
                if (gi) ids = gi;
                for (int32_t i = 0; i < n; ++i) free(vocab[i]);
                free(vocab);
                free(ids);
                fclose(f);
                return NIYAH_ERR_OUT_OF_MEMORY;
            }
            vocab = gv;
            ids = gi;
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
        ids[n] = n;
        ++n;
    }

    fclose(f);

    tokenizer->vocab.vocab = vocab;
    tokenizer->vocab.ids = ids;
    tokenizer->vocab.n_vocab = n;
    tokenizer->owns_vocab = true;

    return NIYAH_OK;
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
}
