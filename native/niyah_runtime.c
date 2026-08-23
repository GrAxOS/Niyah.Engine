#include "niyah.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

NiyahContext* niyah_context_create(void) {
    NiyahContext* ctx = calloc(1, sizeof(NiyahContext));
    if (!ctx) return NULL;
    ctx->graph = calloc(1, sizeof(NiyahGraph));
    if (!ctx->graph) { free(ctx); return NULL; }
    return ctx;
}

void niyah_context_destroy(NiyahContext* ctx) {
    if (!ctx) return;
    if (ctx->graph) niyah_graph_free(ctx->graph);
    free(ctx);
}

NiyahSearchResult* niyah_context_search(NiyahContext* ctx, const char* query) {
    (void)ctx; (void)query;
    return NULL;
}

NiyahDocument* niyah_context_add_document(NiyahContext* ctx, const char* content) {
    (void)ctx; (void)content;
    return NULL;
}
