#include "niyah_url.h"

#include <assert.h>
#include <string.h>

int main(void)
{
    char out[2048];

    assert(niyah_url_canonicalize("HTTPS://Example.COM:443/a#frag", out, sizeof(out)));
    assert(strcmp(out, "https://example.com/a") == 0);

    assert(niyah_url_resolve("https://example.com/docs/page.html", "../api?q=1#top",
                             out, sizeof(out)));
    assert(strcmp(out, "https://example.com/api?q=1") == 0);

    assert(niyah_url_resolve("https://example.com/docs/page.html", "/root", out, sizeof(out)));
    assert(strcmp(out, "https://example.com/root") == 0);

    assert(niyah_url_resolve("https://example.com/docs/page.html", "//cdn.example/x",
                             out, sizeof(out)));
    assert(strcmp(out, "https://cdn.example/x") == 0);

    return 0;
}
