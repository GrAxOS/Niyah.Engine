#include "html_extract.hpp"

#include <algorithm>
#include <cctype>
#include <regex>
#include <string_view>

namespace niyah::search {
namespace {

std::string decode_basic_entities(std::string value) {
    const std::pair<std::string_view, std::string_view> entities[] = {
        {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"},
        {"&quot;", "\""}, {"&#39;", "'"}
    };
    for (const auto& [from, to] : entities) {
        std::size_t pos = 0;
        while ((pos = value.find(from, pos)) != std::string::npos) {
            value.replace(pos, from.size(), to);
            pos += to.size();
        }
    }
    return value;
}

std::string strip_tags(const std::string& html) {
    std::string out;
    out.reserve(html.size());
    bool in_tag = false;
    for (char c : html) {
        if (c == '<') { in_tag = true; out.push_back(' '); continue; }
        if (c == '>') { in_tag = false; out.push_back(' '); continue; }
        if (!in_tag) out.push_back(c);
    }
    return out;
}

void collapse_space(std::string& value) {
    std::string out;
    out.reserve(value.size());
    bool space = false;
    for (unsigned char c : value) {
        if (std::isspace(c)) {
            if (!space) out.push_back(' ');
            space = true;
        } else {
            out.push_back(static_cast<char>(c));
            space = false;
        }
    }
    while (!out.empty() && out.front() == ' ') out.erase(out.begin());
    while (!out.empty() && out.back() == ' ') out.pop_back();
    value.swap(out);
}

}  // namespace

HtmlDocument extract_html(const std::string& html, const std::string&) {
    HtmlDocument result;
    std::smatch match;
    const std::regex title_re(R"(<title\b[^>]*>([\s\S]*?)</title>)", std::regex::icase);
    if (std::regex_search(html, match, title_re) && match.size() > 1)
        result.title = match[1].str();

    std::string body = html;
    body = std::regex_replace(body, std::regex(R"(<script\b[^>]*>[\s\S]*?</script>)", std::regex::icase), " ");
    body = std::regex_replace(body, std::regex(R"(<style\b[^>]*>[\s\S]*?</style>)", std::regex::icase), " ");
    result.text = decode_basic_entities(strip_tags(body));
    collapse_space(result.title);
    collapse_space(result.text);

    const std::regex href_re(R"(href\s*=\s*[\"']([^\"']+)[\"'])", std::regex::icase);
    for (std::sregex_iterator it(html.begin(), html.end(), href_re), end; it != end; ++it) {
        const std::string href = (*it)[1].str();
        if (!href.empty() && href[0] != '#') result.links.push_back(href);
        if (result.links.size() >= 2048) break;
    }
    return result;
}

}  // namespace niyah::search
