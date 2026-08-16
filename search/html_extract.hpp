#pragma once

#include <string>
#include <vector>

namespace niyah::search {

struct HtmlDocument {
    std::string title;
    std::string text;
    std::vector<std::string> links;
};

HtmlDocument extract_html(const std::string& html,
                          const std::string& base_url);

}  // namespace niyah::search
