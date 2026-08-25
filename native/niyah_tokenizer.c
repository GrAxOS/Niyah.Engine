#include "niyah.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
}

int32_t niyah_tokenize(NiyahTokenizer* tokenizer,
                       const char* text,
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
}
