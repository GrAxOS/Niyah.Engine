#include "html_extract.hpp"

#include <cassert>
#include <string>

int main() {
    const std::string html =
        "<html><head><title>Niyah</title></head><body>"
        "<script>ignore()</script><h1>Search</h1><a href=\"/docs\">Docs</a>"
        "</body></html>";

    const auto doc = niyah::search::extract_html(html, "https://example.test/");
    assert(doc.title == "Niyah");
    assert(doc.text.find("Search") != std::string::npos);
    assert(doc.text.find("ignore") == std::string::npos);
    assert(doc.links.size() == 1);
    assert(doc.links[0] == "/docs");
    return 0;
}
