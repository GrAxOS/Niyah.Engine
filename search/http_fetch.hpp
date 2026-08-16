#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace niyah::search {

struct FetchLimits {
    long connect_timeout_seconds = 5;
    long total_timeout_seconds = 15;
    std::size_t max_body_bytes = 4 * 1024 * 1024;
    long max_redirects = 5;
};

struct FetchResult {
    long status = 0;
    std::string effective_url;
    std::string content_type;
    std::string body;
    std::string error;
};

FetchResult http_get(const std::string& url,
                     const std::string& user_agent,
                     const FetchLimits& limits = {});

}  // namespace niyah::search
