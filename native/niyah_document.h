#ifndef NIYAH_DOCUMENT_H
#define NIYAH_DOCUMENT_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NIYAH_SEGMENT_TEXT = 1,
    NIYAH_SEGMENT_CODE = 2,
    NIYAH_SEGMENT_LATEX = 3
} NiyahSegmentKind;

typedef struct {
    uint32_t index;
    NiyahSegmentKind kind;
    size_t start;
    size_t length;
} NiyahSegment;

typedef struct {
    const char *source;
    size_t source_length;
    NiyahSegment *segments;
    size_t segment_count;
    size_t segment_capacity;
} NiyahDocument;

void niyah_document_init(NiyahDocument *document);
void niyah_document_free(NiyahDocument *document);
int niyah_document_parse(NiyahDocument *document, const char *source);
const char *niyah_segment_kind_name(NiyahSegmentKind kind);

#ifdef __cplusplus
}
#endif

#endif
