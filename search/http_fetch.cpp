#include "http_fetch.hpp"

#include <curl/curl.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <cctype>
#include <limits>

namespace niyah::search {
namespace {

struct Context {
    std::string* body = nullptr;
    std::size_t limit = 0;
    bool exceeded = false;
};

size_t write_body(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* ctx = static_cast<Context*>(userdata);
    if (!ctx || !ctx->body) return 0;
    if (size != 0 && nmemb > std::numeric_limits<std::size_t>::max() / size) {
        ctx->exceeded = true;
        return 0;
    }

    const std::size_t bytes = size * nmemb;
    const std::size_t current = ctx->body->size();
    if (current > ctx->limit || bytes > ctx->limit - current) {
        ctx->exceeded = true;
        return 0;
    }

    ctx->body->append(ptr, bytes);
    return bytes;
}

size_t write_headers(char* ptr, size_t size, size_t nmemb, void* userdata) {
    if (size != 0 && nmemb > std::numeric_limits<std::size_t>::max() / size)
        return 0;

    const std::size_t bytes = size * nmemb;
    auto* content_type = static_cast<std::string*>(userdata);
    if (!content_type) return bytes;

    const std::string header(ptr, bytes);
    constexpr char prefix[] = "Content-Type:";
    if (header.size() >= sizeof(prefix) - 1 &&
        std::equal(prefix, prefix + sizeof(prefix) - 1, header.data(),
                   [](char a, char b) {
                       return std::tolower(static_cast<unsigned char>(a)) ==
                              std::tolower(static_cast<unsigned char>(b));
                   })) {
        *content_type = header.substr(sizeof(prefix) - 1);
        while (!content_type->empty() &&
               (content_type->front() == ' ' || content_type->front() == '\t'))
            content_type->erase(content_type->begin());
        while (!content_type->empty() &&
               (content_type->back() == '\r' || content_type->back() == '\n'))
            content_type->pop_back();
    }
    return bytes;
}

bool basic_http_url_ok(const std::string& url) {
    const bool http = url.rfind("http://", 0) == 0;
    const bool https = url.rfind("https://", 0) == 0;
    if (!http && !https) return false;

    const std::size_t authority_start = url.find("://") + 3u;
    const std::size_t authority_end = url.find_first_of("/?#", authority_start);
    const std::string authority = url.substr(
        authority_start,
        authority_end == std::string::npos ? std::string::npos : authority_end - authority_start);

    if (authority.empty()) return false;
    if (authority.find('@') != std::string::npos) return false;
    if (authority == "localhost" || authority.rfind("localhost:", 0) == 0) return false;
    if (authority == "127.0.0.1" || authority.rfind("127.0.0.1:", 0) == 0) return false;
    if (authority == "0.0.0.0" || authority.rfind("0.0.0.0:", 0) == 0) return false;
    if (authority == "[::1]" || authority.rfind("[::1]:", 0) == 0) return false;
    if (authority == "169.254.169.254" || authority.rfind("169.254.169.254:", 0) == 0) return false;
    return true;
}

}  // namespace

FetchResult http_get(const std::string& url,
                     const std::string& user_agent,
                     const FetchLimits& limits) {
    FetchResult result;
    if (!basic_http_url_ok(url)) {
        result.error = "URL rejected by fetch policy";
        return result;
    }
    if (limits.max_body_bytes == 0 || limits.connect_timeout_seconds <= 0 ||
        limits.total_timeout_seconds <= 0 || limits.max_redirects < 0) {
        result.error = "invalid fetch limits";
        return result;
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        result.error = "curl initialization failed";
        return result;
    }

    std::string body;
    std::string content_type;
    Context context{&body, limits.max_body_bytes, false};

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_USERAGENT,
                     user_agent.empty() ? "Niyah.Engine/0.1" : user_agent.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, limits.max_redirects);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, limits.connect_timeout_seconds);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, limits.total_timeout_seconds);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_body);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &context);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, write_headers);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &content_type);
    curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "http,https");
    curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS_STR, "http,https");
    curl_easy_setopt(curl, CURLOPT_DISALLOW_USERNAME_IN_URL, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);

    const CURLcode code = curl_easy_perform(curl);
    if (code != CURLE_OK) {
        result.error = context.exceeded ? "response body limit exceeded" : curl_easy_strerror(code);
        curl_easy_cleanup(curl);
        return result;
    }

    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &result.status);
    char* effective = nullptr;
    curl_easy_getinfo(curl, CURLINFO_EFFECTIVE_URL, &effective);
    if (effective) result.effective_url = effective;
    result.content_type = std::move(content_type);
    result.body = std::move(body);
    curl_easy_cleanup(curl);
    return result;
}

}  // namespace niyah::search
