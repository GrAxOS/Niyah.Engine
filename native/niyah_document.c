#include "niyah_document.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int reserve_segments(NiyahDocument *document, size_t extra)
{
    if (extra > SIZE_MAX - document->segment_count)
        return -1;

    size_t required = document->segment_count + extra;
    if (required <= document->segment_capacity)
        return 0;

    size_t capacity = document->segment_capacity == 0 ? 8u : document->segment_capacity;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2u)
            return -1;
        capacity *= 2u;
    }

    if (capacity > SIZE_MAX / sizeof(*document->segments))
        return -1;

    NiyahSegment *next = realloc(document->segments, capacity * sizeof(*next));
    if (next == NULL)
        return -1;

    document->segments = next;
    document->segment_capacity = capacity;
    return 0;
}

static int add_segment(NiyahDocument *document,
                       NiyahSegmentKind kind,
                       size_t start,
                       size_t length)
{
    if (length == 0)
        return 0;

    if (document->segment_count > UINT32_MAX)
        return -1;

    if (reserve_segments(document, 1u) != 0)
        return -1;

    NiyahSegment *segment = &document->segments[document->segment_count];
    segment->index = (uint32_t)document->segment_count;
    segment->kind = kind;
    segment->start = start;
    segment->length = length;
    document->segment_count++;
    return 0;
}

static bool starts_with_at(const char *source, size_t length, size_t pos,
                           const char *marker, size_t marker_length)
{
    return pos <= length && marker_length <= length - pos &&
           memcmp(source + pos, marker, marker_length) == 0;
}

static int marker_kind(const char *source, size_t length, size_t pos,
                       const char **open, size_t *open_length,
                       const char **close, size_t *close_length,
                       NiyahSegmentKind *kind)
{
    if (starts_with_at(source, length, pos, "```", 3u)) {
        *open = "```"; *open_length = 3u; *close = "```"; *close_length = 3u;
        *kind = NIYAH_SEGMENT_CODE; return 1;
    }
    if (starts_with_at(source, length, pos, "$$", 2u)) {
        *open = "$$"; *open_length = 2u; *close = "$$"; *close_length = 2u;
        *kind = NIYAH_SEGMENT_LATEX; return 1;
    }
    if (starts_with_at(source, length, pos, "\\[", 2u)) {
        *open = "\\["; *open_length = 2u; *close = "\\]"; *close_length = 2u;
        *kind = NIYAH_SEGMENT_LATEX; return 1;
    }
    if (starts_with_at(source, length, pos, "\\(", 2u)) {
        *open = "\\("; *open_length = 2u; *close = "\\)"; *close_length = 2u;
        *kind = NIYAH_SEGMENT_LATEX; return 1;
    }
    return 0;
}

void niyah_document_init(NiyahDocument *document)
{
    if (document == NULL)
        return;
    document->source = NULL;
    document->source_length = 0;
    document->segments = NULL;
    document->segment_count = 0;
    document->segment_capacity = 0;
}

void niyah_document_free(NiyahDocument *document)
{
    if (document == NULL)
        return;
    free(document->segments);
    niyah_document_init(document);
}

int niyah_document_parse(NiyahDocument *document, const char *source)
{
    if (document == NULL || source == NULL)
        return -1;

    NiyahDocument next;
    niyah_document_init(&next);
    next.source = source;
    next.source_length = strlen(source);

    size_t cursor = 0;
    size_t text_start = 0;

    while (cursor < next.source_length) {
        const char *open = NULL, *close = NULL;
        size_t open_length = 0, close_length = 0;
        NiyahSegmentKind kind = NIYAH_SEGMENT_TEXT;

        if (!marker_kind(source, next.source_length, cursor,
                         &open, &open_length, &close, &close_length, &kind)) {
            cursor++;
            continue;
        }

        if (add_segment(&next, NIYAH_SEGMENT_TEXT,
                        text_start, cursor - text_start) != 0)
            goto fail;

        size_t content_start = cursor;
        size_t close_pos = cursor + open_length;
        while (close_pos < next.source_length &&
               !starts_with_at(source, next.source_length, close_pos,
                               close, close_length))
            close_pos++;

        size_t end = close_pos < next.source_length
            ? close_pos + close_length
            : next.source_length;

        if (add_segment(&next, kind, content_start, end - content_start) != 0)
            goto fail;

        cursor = end;
        text_start = end;
    }

    if (add_segment(&next, NIYAH_SEGMENT_TEXT,
                    text_start, next.source_length - text_start) != 0)
        goto fail;

    niyah_document_free(document);
    *document = next;
    return 0;

fail:
    niyah_document_free(&next);
    return -1;
}

const char *niyah_segment_kind_name(NiyahSegmentKind kind)
{
    switch (kind) {
    case NIYAH_SEGMENT_TEXT: return "text";
    case NIYAH_SEGMENT_CODE: return "code";
    case NIYAH_SEGMENT_LATEX: return "latex";
    default: return "unknown";
    }
}
