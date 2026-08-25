#include "niyah_document.h"

#include <stdlib.h>
#include <string.h>

/* ── Document lifecycle ─────────────────────────────────────────────────── */

NiyahDocument* niyah_document_create(const char* id,
                                      const char* title,
                                      const char* content) {
    NiyahDocument* doc = (NiyahDocument*)calloc(1, sizeof(NiyahDocument));
    if (!doc) return NULL;

    doc->id      = id      ? _strdup(id)      : NULL;
    doc->title   = title   ? _strdup(title)   : NULL;
    doc->content = content ? _strdup(content) : NULL;
    doc->created_at = 0; /* caller sets if desired */
    doc->n_tokens   = 0;
    doc->tokens     = NULL;

    return doc;
}

void niyah_document_free(NiyahDocument* doc) {
    if (!doc) return;
    free(doc->id);
    free(doc->title);
    free(doc->content);
    if (doc->tokens) {
        for (int32_t i = 0; i < doc->n_tokens; i++) free(doc->tokens[i]);
        free(doc->tokens);
    }
    free(doc);
}

/* Tokenise document content into word array */
void niyah_document_tokenize(NiyahDocument* doc) {
    if (!doc || !doc->content) return;

    /* Free previous tokens */
    if (doc->tokens) {
        for (int32_t i = 0; i < doc->n_tokens; i++) free(doc->tokens[i]);
        free(doc->tokens);
        doc->tokens   = NULL;
        doc->n_tokens = 0;
    }

    /* Count words */
    const char* p     = doc->content;
    int32_t     count = 0;
    int32_t     in_w  = 0;
    for (; *p; p++) {
        int c = (unsigned char)*p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (in_w) { count++; in_w = 0; }
        } else {
            in_w = 1;
        }
    }
    if (in_w) count++;
    if (count == 0) return;

    doc->tokens = (char**)calloc((size_t)count, sizeof(char*));
    if (!doc->tokens) return;

    p    = doc->content;
    in_w = 0;
    int32_t wi  = 0;
    int32_t idx = 0;
    char word[256];

    for (; *p; p++) {
        int c = (unsigned char)*p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (in_w) {
                word[wi] = '\0';
                doc->tokens[idx++] = _strdup(word);
                wi = 0; in_w = 0;
            }
        } else if (wi < 254) {
            word[wi++] = (char)c;
            in_w = 1;
        }
    }
    if (in_w) {
        word[wi] = '\0';
        doc->tokens[idx++] = _strdup(word);
    }
    doc->n_tokens = idx;
}
