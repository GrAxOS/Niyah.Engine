#include "niyah_url.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

static bool prefix_ci(const char *s, const char *prefix)
{
    while (*prefix) {
        unsigned char a = (unsigned char)*s++;
        unsigned char b = (unsigned char)*prefix++;
        if (a == 0u || tolower(a) != tolower(b)) return false;
    }
    return true;
}

static bool append_char(char *out, size_t cap, size_t *len, char c)
{
    if (*len >= cap - 1u) return false;
    out[(*len)++] = c;
    out[*len] = '\0';
    return true;
}

static bool append_range(char *out, size_t cap, size_t *len,
                         const char *begin, const char *end)
{
    for (const char *p = begin; p < end; ++p)
        if (!append_char(out, cap, len, *p)) return false;
    return true;
}

bool niyah_url_canonicalize(const char *input, char *output, size_t output_size)
{
    if (!input || !output || output_size < 8u) return false;
    output[0] = '\0';

    const char *p = input;
    const char *scheme = NULL;
    size_t scheme_len = 0u;
    if (prefix_ci(p, "https://")) { scheme = "https://"; scheme_len = 8u; }
    else if (prefix_ci(p, "http://")) { scheme = "http://"; scheme_len = 7u; }
    else return false;

    size_t len = 0u;
    if (!append_range(output, output_size, &len, scheme, scheme + scheme_len)) return false;
    p += scheme_len;

    const char *host_start = p;
    while (*p && *p != '/' && *p != '?' && *p != '#') ++p;
    if (p == host_start) return false;
    const char *host_end = p;

    size_t host_len = (size_t)(host_end - host_start);
    bool ipv6 = host_len >= 2u && host_start[0] == '[';
    size_t host_limit = host_len;
    if (!ipv6) {
        const char *colon = NULL;
        for (const char *q = host_start; q < host_end; ++q)
            if (*q == ':') colon = q;
        if (colon) {
            const char *port = colon + 1;
            size_t port_len = (size_t)(host_end - port);
            if ((strcmp(scheme, "http://") == 0 && port_len == 2u &&
                 strncmp(port, "80", 2u) == 0) ||
                (strcmp(scheme, "https://") == 0 && port_len == 3u &&
                 strncmp(port, "443", 3u) == 0))
                host_limit = (size_t)(colon - host_start);
        }
    }

    for (size_t i = 0u; i < host_limit; ++i) {
        unsigned char c = (unsigned char)host_start[i];
        if (!append_char(output, output_size, &len, (char)tolower(c))) return false;
    }

    if (*p == '/') {
        const char *path = p;
        const char *fragment = strchr(path, '#');
        const char *end = fragment ? fragment : path + strlen(path);
        if (!append_range(output, output_size, &len, path, end)) return false;
    } else if (*p == '?') {
        const char *query = p;
        const char *fragment = strchr(query, '#');
        const char *end = fragment ? fragment : query + strlen(query);
        if (!append_range(output, output_size, &len, query, end)) return false;
    } else {
        if (!append_char(output, output_size, &len, '/')) return false;
    }

    return true;
}

static bool split_base(const char *base,
                       char *authority, size_t authority_size,
                       char *base_path, size_t base_path_size,
                       char *scheme, size_t scheme_size)
{
    if (!base || !authority || !base_path || !scheme) return false;
    if (prefix_ci(base, "https://")) {
        if (scheme_size < 6u) return false;
        memcpy(scheme, "https", 6u);
    } else if (prefix_ci(base, "http://")) {
        if (scheme_size < 5u) return false;
        memcpy(scheme, "http", 5u);
    } else return false;

    const char *authority_start = strchr(base, ':');
    if (!authority_start || authority_start[1] != '/' || authority_start[2] != '/') return false;
    authority_start += 3;
    const char *path_start = strpbrk(authority_start, "/?#");
    if (!path_start) path_start = base + strlen(base);

    size_t authority_len = (size_t)(path_start - authority_start);
    if (authority_len + 1u > authority_size) return false;
    memcpy(authority, authority_start, authority_len);
    authority[authority_len] = '\0';

    const char *fragment = strchr(path_start, '#');
    const char *path_end = fragment ? fragment : base + strlen(base);
    if ((size_t)(path_end - path_start) + 1u > base_path_size) return false;
    memcpy(base_path, path_start, (size_t)(path_end - path_start));
    base_path[path_end - path_start] = '\0';
    return true;
}

static bool normalize_path(const char *path, char *out, size_t cap)
{
    char work[4096];
    size_t path_len = strlen(path);
    if (path_len >= sizeof(work) || cap < 2u) return false;
    memcpy(work, path, path_len + 1u);

    const bool absolute = work[0] == '/';
    size_t out_len = 0u;
    if (absolute && !append_char(out, cap, &out_len, '/')) return false;

    char *p = work;
    while (*p) {
        while (*p == '/') ++p;
        if (*p == '\0') break;
        char *segment = p;
        while (*p && *p != '/') ++p;
        char saved = *p;
        *p = '\0';

        if (strcmp(segment, ".") == 0) {
            /* no-op */
        } else if (strcmp(segment, "..") == 0) {
            if (out_len > (absolute ? 1u : 0u)) {
                if (out_len > 0u && out[out_len - 1u] == '/') --out_len;
                while (out_len > (absolute ? 1u : 0u) && out[out_len - 1u] != '/') --out_len;
                out[out_len] = '\0';
            }
        } else {
            if (out_len > 0u && out[out_len - 1u] != '/' &&
                !append_char(out, cap, &out_len, '/')) return false;
            if (!append_range(out, cap, &out_len, segment, segment + strlen(segment))) return false;
        }
        *p = saved;
    }

    if (out_len == 0u && absolute) {
        if (!append_char(out, cap, &out_len, '/')) return false;
    }
    return true;
}

bool niyah_url_resolve(const char *base,
                       const char *reference,
                       char *output,
                       size_t output_size)
{
    if (!base || !reference || !output || output_size < 8u || reference[0] == '\0') return false;

    if (prefix_ci(reference, "http://") || prefix_ci(reference, "https://"))
        return niyah_url_canonicalize(reference, output, output_size);

    char authority[2048];
    char base_path[4096];
    char scheme[8] = {0};
    if (!split_base(base, authority, sizeof(authority), base_path, sizeof(base_path),
                    scheme, sizeof(scheme))) return false;

    const char *ref_end = strchr(reference, '#');
    if (!ref_end) ref_end = reference + strlen(reference);

    char target_path[4096] = {0};
    if (reference[0] == '/' && reference[1] == '/') {
        char absolute[6144];
        int written = snprintf(absolute, sizeof(absolute), "%s:%s", scheme, reference);
        if (written < 0 || (size_t)written >= sizeof(absolute)) return false;
        return niyah_url_canonicalize(absolute, output, output_size);
    }

    if (reference[0] == '#') {
        const char *base_query = strchr(base_path, '?');
        size_t base_path_len = base_query ? (size_t)(base_query - base_path) : strlen(base_path);
        if (base_path_len + 1u > sizeof(target_path)) return false;
        memcpy(target_path, base_path, base_path_len);
        target_path[base_path_len] = '\0';
    } else if (reference[0] == '?') {
        const char *base_query = strchr(base_path, '?');
        size_t base_path_len = base_query ? (size_t)(base_query - base_path) : strlen(base_path);
        size_t ref_len = (size_t)(ref_end - reference);
        if (base_path_len + ref_len + 1u > sizeof(target_path)) return false;
        memcpy(target_path, base_path, base_path_len);
        memcpy(target_path + base_path_len, reference, ref_len);
        target_path[base_path_len + ref_len] = '\0';
    } else if (reference[0] == '/') {
        size_t ref_len = (size_t)(ref_end - reference);
        if (ref_len + 1u > sizeof(target_path)) return false;
        memcpy(target_path, reference, ref_len);
        target_path[ref_len] = '\0';
    } else {
        const char *base_query = strchr(base_path, '?');
        size_t path_len = base_query ? (size_t)(base_query - base_path) : strlen(base_path);
        size_t dir_len = path_len;
        while (dir_len > 0u && base_path[dir_len - 1u] != '/') --dir_len;
        size_t ref_len = (size_t)(ref_end - reference);
        if (dir_len + ref_len + 1u > sizeof(target_path)) return false;
        memcpy(target_path, base_path, dir_len);
        memcpy(target_path + dir_len, reference, ref_len);
        target_path[dir_len + ref_len] = '\0';
    }

    char normalized[4096] = {0};
    if (!normalize_path(target_path, normalized, sizeof(normalized))) return false;

    char absolute[6144];
    int written = snprintf(absolute, sizeof(absolute), "%s://%s%s", scheme, authority,
                           normalized[0] ? normalized : "/");
    if (written < 0 || (size_t)written >= sizeof(absolute)) return false;
    return niyah_url_canonicalize(absolute, output, output_size);
}
