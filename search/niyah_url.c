#include "niyah_url.h"

#include <ctype.h>
#include <string.h>

static bool prefix_ci(const char *s, const char *prefix) {
    while (*prefix) {
        unsigned char a = (unsigned char)*s++;
        unsigned char b = (unsigned char)*prefix++;
        if (tolower(a) != tolower(b)) return false;
    }
    return true;
}

static bool append_char(char *out, size_t cap, size_t *len, char c) {
    if (*len + 1 >= cap) return false;
    out[(*len)++] = c;
    out[*len] = '\0';
    return true;
}

bool niyah_url_canonicalize(const char *input, char *output, size_t output_size) {
    if (!input || !output || output_size < 8) return false;
    output[0] = '\0';

    const char *p = input;
    const char *scheme = NULL;
    if (prefix_ci(p, "https://")) scheme = "https://";
    else if (prefix_ci(p, "http://")) scheme = "http://";
    else return false;

    size_t len = 0;
    for (size_t i = 0; scheme[i] != '\0'; ++i) {
        if (!append_char(output, output_size, &len, scheme[i])) return false;
    }
    p += strlen(scheme);

    /* Host is required. Lowercase it and strip an explicit default port. */
    const char *host_start = p;
    while (*p && *p != '/' && *p != '?' && *p != '#') ++p;
    if (p == host_start) return false;
    const char *host_end = p;

    size_t host_len = (size_t)(host_end - host_start);
    bool ipv6 = host_len >= 2 && host_start[0] == '[';
    size_t host_limit = host_len;
    if (!ipv6) {
        const char *colon = NULL;
        for (const char *q = host_start; q < host_end; ++q) {
            if (*q == ':') colon = q;
        }
        if (colon) {
            const char *port = colon + 1;
            if (strcmp(scheme, "http://") == 0 && strcmp(port, "80") == 0) host_limit = (size_t)(colon - host_start);
            if (strcmp(scheme, "https://") == 0 && strcmp(port, "443") == 0) host_limit = (size_t)(colon - host_start);
        }
    }

    for (size_t i = 0; i < host_limit; ++i) {
        unsigned char c = (unsigned char)host_start[i];
        char out_c = (char)tolower(c);
        if (!append_char(output, output_size, &len, out_c)) return false;
    }

    /* Preserve path/query, drop fragment; collapse a trailing slash for root only. */
    if (*p == '/') {
        const char *path = p;
        const char *fragment = strchr(path, '#');
        const char *end = fragment ? fragment : path + strlen(path);
        if ((size_t)(end - path) == 1 && path[0] == '/') {
            if (!append_char(output, output_size, &len, '/')) return false;
        } else {
            for (const char *q = path; q < end; ++q) {
                if (!append_char(output, output_size, &len, *q)) return false;
            }
        }
    } else if (*p == '?') {
        const char *query = p;
        const char *fragment = strchr(query, '#');
        const char *end = fragment ? fragment : query + strlen(query);
        for (const char *q = query; q < end; ++q) {
            if (!append_char(output, output_size, &len, *q)) return false;
        }
    }

    return len > strlen(scheme);
}
