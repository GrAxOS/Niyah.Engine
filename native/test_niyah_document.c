#include "niyah_document.h"

#include <assert.h>
#include <string.h>

int main(void)
{
    const char *source =
        "hello ```c\nint x = 1;\n``` world $$x^2$$ done";

    NiyahDocument document;
    niyah_document_init(&document);

    assert(niyah_document_parse(&document, source) == 0);
    assert(document.source == source);
    assert(document.source_length == strlen(source));
    assert(document.segment_count == 5u);
    assert(document.segments[0].kind == NIYAH_SEGMENT_TEXT);
    assert(document.segments[1].kind == NIYAH_SEGMENT_CODE);
    assert(document.segments[2].kind == NIYAH_SEGMENT_TEXT);
    assert(document.segments[3].kind == NIYAH_SEGMENT_LATEX);
    assert(document.segments[4].kind == NIYAH_SEGMENT_TEXT);

    assert(strcmp(niyah_segment_kind_name(NIYAH_SEGMENT_CODE), "code") == 0);
    assert(strcmp(niyah_segment_kind_name(NIYAH_SEGMENT_LATEX), "latex") == 0);

    niyah_document_free(&document);
    return 0;
}
