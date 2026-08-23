#include "niyah.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

NiyahContext* niyah_create(void) {
    return niyah_context_create();
}

void niyah_destroy(NiyahContext* ctx) {
    niyah_context_destroy(ctx);
}

NiyahSearchResult* niyah_search(NiyahContext* ctx, const char* query) {
    return niyah_context_search(ctx, query);
}

void niyah_search_free(NiyahSearchResult* result) {
    niyah_search_result_free(result);
}

NiyahDocument* niyah_add_document(NiyahContext* ctx, const char* content) {
    return niyah_context_add_document(ctx, content);
}

const char* niyah_get_version(void) {
    return NIYAH_VERSION;
}
