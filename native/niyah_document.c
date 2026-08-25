#include "niyah_document.h"

#include <stdlib.h>
#include <string.h>

<<<<<<< HEAD
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
=======
/* Was: `// Document stubs`. */

#define NIYAH_FENCE_LEN 3u
#define NIYAH_MATH_LEN  2u

void niyah_document_init(NiyahDocument* document)
{
    if (!document) {
        return;
    }
    document->source = NULL;
    document->source_length = 0;
    document->segments = NULL;
    document->segment_count = 0;
    document->segment_capacity = 0;
}

void niyah_document_free(NiyahDocument* document)
{
    if (!document) {
        return;
    }
    free(document->segments);
    document->segments = NULL;
    document->segment_count = 0;
    document->segment_capacity = 0;
    /* `source` is borrowed; nothing to release. */
}

const char* niyah_segment_kind_name(NiyahSegmentKind kind)
{
    switch (kind) {
        case NIYAH_SEGMENT_TEXT:  return "text";
        case NIYAH_SEGMENT_CODE:  return "code";
        case NIYAH_SEGMENT_LATEX: return "latex";
        default:                  return "unknown";
    }
}

static int push_segment(NiyahDocument* document,
                        NiyahSegmentKind kind,
                        size_t offset,
                        size_t length,
                        size_t content_offset,
                        size_t content_length,
                        size_t info_offset,
                        size_t info_length)
{
    if (length == 0) {
        return 0; /* never emit empty runs */
    }

    if (document->segment_count == document->segment_capacity) {
        const size_t next = document->segment_capacity
            ? document->segment_capacity * 2u : 8u;
        NiyahSegment* grown = (NiyahSegment*)realloc(
            document->segments, next * sizeof(NiyahSegment));
        if (!grown) {
            return -2;
        }
        document->segments = grown;
        document->segment_capacity = next;
    }

    NiyahSegment* seg = &document->segments[document->segment_count];
    seg->kind = kind;
    seg->offset = offset;
    seg->length = length;
    seg->content_offset = content_offset;
    seg->content_length = content_length;
    seg->info_offset = info_offset;
    seg->info_length = info_length;

    ++document->segment_count;
    return 0;
}

/* Index of `needle` in source[from..len), or SIZE_MAX. */
static size_t find_from(const char* source,
                        size_t len,
                        size_t from,
                        const char* needle,
                        size_t needle_len)
{
    if (needle_len == 0 || from >= len || len - from < needle_len) {
        return (size_t)-1;
    }
    for (size_t i = from; i + needle_len <= len; ++i) {
        if (memcmp(source + i, needle, needle_len) == 0) {
            return i;
        }
    }
    return (size_t)-1;
}

int niyah_document_parse(NiyahDocument* document, const char* source)
{
    if (!document || !source) {
        return -1;
    }

    /* Reset any previous parse but keep borrowing semantics. */
    free(document->segments);
    document->segments = NULL;
    document->segment_count = 0;
    document->segment_capacity = 0;

    document->source = source;
    document->source_length = strlen(source);

    const char* s = source;
    const size_t len = document->source_length;

    size_t i = 0;
    size_t text_start = 0;
    int rc = 0;

    while (i < len) {
        /* Fenced code block: ```info\n ... ``` */
        if (len - i >= NIYAH_FENCE_LEN && memcmp(s + i, "```", NIYAH_FENCE_LEN) == 0) {
            const size_t close = find_from(s, len, i + NIYAH_FENCE_LEN,
                                           "```", NIYAH_FENCE_LEN);
            if (close != (size_t)-1) {
                rc = push_segment(document, NIYAH_SEGMENT_TEXT,
                                  text_start, i - text_start,
                                  text_start, i - text_start, 0, 0);
                if (rc != 0) {
                    goto fail;
                }

                /* Info string runs from after the fence to the first newline. */
                size_t info_start = i + NIYAH_FENCE_LEN;
                size_t info_end = info_start;
                while (info_end < close && s[info_end] != '\n') {
                    ++info_end;
                }

                const size_t body_start = (info_end < close) ? info_end + 1u : close;
                const size_t end = close + NIYAH_FENCE_LEN;

                rc = push_segment(document, NIYAH_SEGMENT_CODE,
                                  i, end - i,
                                  body_start,
                                  close > body_start ? close - body_start : 0,
                                  info_start, info_end - info_start);
                if (rc != 0) {
                    goto fail;
                }

                i = end;
                text_start = i;
                continue;
            }
            /* Unterminated fence: fall through and treat as plain text. */
            i += NIYAH_FENCE_LEN;
            continue;
        }

        /* Display math: $$ ... $$ */
        if (len - i >= NIYAH_MATH_LEN && memcmp(s + i, "$$", NIYAH_MATH_LEN) == 0) {
            const size_t close = find_from(s, len, i + NIYAH_MATH_LEN,
                                           "$$", NIYAH_MATH_LEN);
            if (close != (size_t)-1) {
                rc = push_segment(document, NIYAH_SEGMENT_TEXT,
                                  text_start, i - text_start,
                                  text_start, i - text_start, 0, 0);
                if (rc != 0) {
                    goto fail;
                }

                const size_t end = close + NIYAH_MATH_LEN;
                rc = push_segment(document, NIYAH_SEGMENT_LATEX,
                                  i, end - i,
                                  i + NIYAH_MATH_LEN,
                                  close - (i + NIYAH_MATH_LEN), 0, 0);
                if (rc != 0) {
                    goto fail;
                }

                i = end;
                text_start = i;
                continue;
            }
            i += NIYAH_MATH_LEN;
            continue;
        }

        ++i;
    }

    rc = push_segment(document, NIYAH_SEGMENT_TEXT,
                      text_start, len - text_start,
                      text_start, len - text_start, 0, 0);
    if (rc != 0) {
        goto fail;
    }

    return 0;

fail:
    niyah_document_free(document);
    document->source = source;
    document->source_length = len;
    return rc;
}

size_t niyah_document_count_kind(const NiyahDocument* document,
                                 NiyahSegmentKind kind)
{
    if (!document || !document->segments) {
        return 0;
    }
    size_t n = 0;
    for (size_t i = 0; i < document->segment_count; ++i) {
        if (document->segments[i].kind == kind) {
            ++n;
        }
    }
    return n;
}

const char* niyah_document_segment_text(const NiyahDocument* document,
                                        size_t index,
                                        size_t* out_length)
{
    if (!document || !document->source || !document->segments ||
        index >= document->segment_count) {
        if (out_length) {
            *out_length = 0;
        }
        return NULL;
    }

    const NiyahSegment* seg = &document->segments[index];
    if (out_length) {
        *out_length = seg->content_length;
    }
    return document->source + seg->content_offset;
>>>>>>> origin/main
}
