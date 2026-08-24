#ifndef NIYAH_DOCUMENT_H
#define NIYAH_DOCUMENT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Segment-level document model.
 *
 * Deliberately self-contained: it does NOT include niyah.h. Three different
 * `NiyahDocument` shapes used to exist in this repo (one in native/niyah.h, one
 * in search/niyah_index.h, and the one this test suite expects). Keeping this
 * header dependency-free is what makes the three able to coexist.
 *
 * Segments are zero-copy views into the caller's buffer. The document borrows
 * `source`; it must outlive the document.
 */

typedef enum {
    NIYAH_SEGMENT_TEXT  = 0,
    NIYAH_SEGMENT_CODE  = 1,
    NIYAH_SEGMENT_LATEX = 2
} NiyahSegmentKind;

typedef struct {
    NiyahSegmentKind kind;
    /* Span including delimiters (``` fences, $$ markers). */
    size_t offset;
    size_t length;
    /* Span of the inner payload, delimiters stripped. */
    size_t content_offset;
    size_t content_length;
    /* Info string of a fenced block, e.g. "c" in ```c. Borrowed, may be 0-len. */
    size_t info_offset;
    size_t info_length;
} NiyahSegment;

typedef struct {
    const char*   source;
    size_t        source_length;
    NiyahSegment* segments;
    size_t        segment_count;
    size_t        segment_capacity;
} NiyahDocument;

/* Zeroes the document. Always call before parse. */
void niyah_document_init(NiyahDocument* document);

/* Splits `source` into text / code / latex segments.
 * Returns 0 on success, -1 on invalid arguments, -2 on allocation failure. */
int niyah_document_parse(NiyahDocument* document, const char* source);

/* Releases the segment array. Does not touch `source`. */
void niyah_document_free(NiyahDocument* document);

const char* niyah_segment_kind_name(NiyahSegmentKind kind);

/* Convenience accessors. */
size_t      niyah_document_count_kind(const NiyahDocument* document,
                                      NiyahSegmentKind kind);
const char* niyah_document_segment_text(const NiyahDocument* document,
                                        size_t index,
                                        size_t* out_length);

#ifdef __cplusplus
}
#endif

#endif /* NIYAH_DOCUMENT_H */
