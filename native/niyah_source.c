#include "niyah.h"

#include <stdlib.h>
#include <string.h>

<<<<<<< HEAD
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
=======
/* Was: `// Source stubs`. */

static char* dup_str(const char* s)
{
    if (!s) {
        return NULL;
    }
    const size_t n = strlen(s) + 1u;
    char* out = (char*)malloc(n);
    if (out) {
        memcpy(out, s, n);
    }
    return out;
}

const char* niyah_source_type_name(NiyahSourceType type)
{
    switch (type) {
        case NIYAH_SOURCE_LOCAL: return "local";
        case NIYAH_SOURCE_WEB:   return "web";
        case NIYAH_SOURCE_DB:    return "db";
        case NIYAH_SOURCE_API:   return "api";
        default:                 return "unknown";
    }
}

NiyahStatus niyah_source_init(NiyahSource* source,
                              NiyahSourceType type,
                              const char* uri,
                              const char* metadata)
{
    if (!source || !uri) {
        return NIYAH_ERR_INVALID_ARG;
    }

    memset(source, 0, sizeof(*source));
    source->type = type;

    source->uri = dup_str(uri);
    if (!source->uri) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    if (metadata) {
        source->metadata = dup_str(metadata);
        if (!source->metadata) {
            free(source->uri);
            source->uri = NULL;
            return NIYAH_ERR_OUT_OF_MEMORY;
        }
    }

    return NIYAH_OK;
}

void niyah_source_free(NiyahSource* source)
{
    if (!source) {
        return;
    }
    free(source->uri);
    free(source->metadata);
    memset(source, 0, sizeof(*source));
}

void niyah_webpage_free(NiyahWebPage* page)
{
    if (!page) {
        return;
    }
    free(page->url);
    free(page->content);
    free(page->title);
    memset(page, 0, sizeof(*page));
>>>>>>> origin/main
}
