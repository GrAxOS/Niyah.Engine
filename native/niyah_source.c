#include "niyah.h"
#include <stdlib.h>
#include <string.h>

/* ── Source management ──────────────────────────────────────────────────── */

NiyahSource* niyah_source_create(NiyahSourceType type,
                                  const char* uri,
                                  const char* metadata) {
    NiyahSource* s = (NiyahSource*)calloc(1, sizeof(NiyahSource));
    if (!s) return NULL;
    s->type     = type;
    s->uri      = uri      ? _strdup(uri)      : NULL;
    s->metadata = metadata ? _strdup(metadata) : NULL;
    return s;
}

void niyah_source_free(NiyahSource* s) {
    if (!s) return;
    free(s->uri);
    free(s->metadata);
    free(s);
}
