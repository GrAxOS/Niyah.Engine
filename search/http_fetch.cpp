#include "http_fetch.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <curl/curl.h>

#include <algorithm>
#include <cctype>
#include <limits>
#include <string>
#include <utility>

namespace niyah::search {
namespace {

struct Context {
    std::string* body = nullptr;
    std::size_t limit = 0;
    bool exceeded = false;
};

size_t write_body(
    char* ptr,
    size_t size,
    size_t nmemb,
    void* userdata
) {
    auto* ctx = static_cast<Context*>(userdata);

    if (!ctx || !ctx->body) {
        return 0;
    }

    if (size != 0 &&
        nmemb > (std::numeric_limits<std::size_t>::max)() / size) {
        ctx->exceeded = true;
        return 0;
    }

    const std::size_t bytes = size * nmemb;
    const std::size_t current = ctx->body->size();

    if (current > ctx->limit ||
        bytes > ctx->limit - current) {
        ctx->exceeded = true;
        return 0;
    }

    try {
        ctx->body->append(ptr, bytes);
    } catch (...) {
        ctx->exceeded = true;
        return 0;
    }

    return bytes;
}

size_t write_headers(
    char* ptr,
    size_t size,
    size_t nmemb,
    void* userdata
) {
    if (size != 0 &&
        nmemb > (std::numeric_limits<std::size_t>::max)() / size) {
        return 0;
    }

    const std::size_t bytes = size * nmemb;
    auto* content_type = static_cast<std::string*>(userdata);

    if (!content_type) {
        return bytes;
    }

    const std::string header(ptr, bytes);
    constexpr char prefix[] = "Content-Type:";
    constexpr std::size_t prefix_size = sizeof(prefix) - 1;

    if (header.size() < prefix_size) {
        return bytes;
    }

    const bool matches = std::equal(
        prefix,
        prefix + prefix_size,
        header.data(),
        [](char left, char right) {
            return std::tolower(
                static_cast<unsigned char>(left)
            ) == std::tolower(
                static_cast<unsigned char>(right)
            );
        }
    );

    if (!matches) {
        return bytes;
    }

    *content_type = header.substr(prefix_size);

    while (!content_type->empty() &&
           (content_type->front() == ' ' ||
            content_type->front() == '\t')) {
        content_type->erase(content_type->begin());
    }

    while (!content_type->empty() &&
           (content_type->back() == '\r' ||
            content_type->back() == '\n')) {
        content_type->pop_back();
    }

    return bytes;
}

bool basic_http_url_ok(const std::string& url) {
    const bool http = url.rfind("http://", 0) == 0;
    const bool https = url.rfind("https://", 0) == 0;

    if (!http && !https) {
        return false;
    }

    const std::size_t scheme_end = url.find("://");

    if (scheme_end == std::string::npos) {
        return false;
    }

    const std::size_t authority_start = scheme_end + 3u;
    const std::size_t authority_end =
        url.find_first_of("/?#", authority_start);

    const std::string authority = url.substr(
        authority_start,
        authority_end == std::string::npos
            ? std::string::npos
            : authority_end - authority_start
    );

    if (authority.empty()) {
        return false;
    }

    if (authority.find('@') != std::string::npos) {
        return false;
    }

    if (authority == "localhost" ||
        authority.rfind("localhost:", 0) == 0) {
        return false;
    }

    if (authority == "127.0.0.1" ||
        authority.rfind("127.0.0.1:", 0) == 0) {
        return false;
    }

    if (authority == "0.0.0.0" ||
        authority.rfind("0.0.0.0:", 0) == 0) {
        return false;
    }

    if (authority == "[::1]" ||
        authority.rfind("[::1]:", 0) == 0) {
        return false;
    }

    if (authority == "169.254.169.254" ||
        authority.rfind("169.254.169.254:", 0) == 0) {
        return false;
    }

    return true;
}

#define NIYAH_SETOPT(handle, option, value)                    \
    do {                                                        \
        const CURLcode setopt_code =                           \
            curl_easy_setopt((handle), (option), (value));     \
        if (setopt_code != CURLE_OK) {                         \
            result.error = curl_easy_strerror(setopt_code);    \
            curl_easy_cleanup(curl);                           \
            return result;                                     \
        }                                                       \
    } while (0)

}  // namespace

FetchResult http_get(
    const std::string& url,
    const std::string& user_agent,
    const FetchLimits& limits
) {
    FetchResult result;

    if (!basic_http_url_ok(url)) {
        result.error = "URL rejected by fetch policy";
        return result;
    }

    if (limits.max_body_bytes == 0 ||
        limits.connect_timeout_seconds <= 0 ||
        limits.total_timeout_seconds <= 0 ||
        limits.max_redirects < 0) {
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

    Context context{
        &body,
        limits.max_body_bytes,
        false
    };

    const char* effective_user_agent =
        user_agent.empty()
            ? "Niyah.Engine/0.1"
            : user_agent.c_str();

    NIYAH_SETOPT(curl, CURLOPT_URL, url.c_str());
    NIYAH_SETOPT(curl, CURLOPT_USERAGENT, effective_user_agent);

    /*
     * Redirects are disabled until every Location target
     * is validated by the URL/IP policy.
     */
    NIYAH_SETOPT(curl, CURLOPT_FOLLOWLOCATION, 0L);
    NIYAH_SETOPT(curl, CURLOPT_MAXREDIRS, limits.max_redirects);

    NIYAH_SETOPT(
        curl,
        CURLOPT_CONNECTTIMEOUT,
        limits.connect_timeout_seconds
    );

    NIYAH_SETOPT(
        curl,
        CURLOPT_TIMEOUT,
        limits.total_timeout_seconds
    );

    NIYAH_SETOPT(curl, CURLOPT_WRITEFUNCTION, write_body);
    NIYAH_SETOPT(curl, CURLOPT_WRITEDATA, &context);

    NIYAH_SETOPT(curl, CURLOPT_HEADERFUNCTION, write_headers);
    NIYAH_SETOPT(curl, CURLOPT_HEADERDATA, &content_type);

#if LIBCURL_VERSION_NUM >= 0x075500
    NIYAH_SETOPT(curl, CURLOPT_PROTOCOLS_STR, "http,https");
    NIYAH_SETOPT(curl, CURLOPT_REDIR_PROTOCOLS_STR, "http,https");
#else
    NIYAH_SETOPT(curl, CURLOPT_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS);
    NIYAH_SETOPT(
        curl,
        CURLOPT_REDIR_PROTOCOLS,
        CURLPROTO_HTTP | CURLPROTO_HTTPS
    );
#endif

#if LIBCURL_VERSION_NUM >= 0x075500
    NIYAH_SETOPT(curl, CURLOPT_DISALLOW_USERNAME_IN_URL, 1L);
#endif

    NIYAH_SETOPT(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    NIYAH_SETOPT(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    NIYAH_SETOPT(curl, CURLOPT_ACCEPT_ENCODING, "");
    NIYAH_SETOPT(curl, CURLOPT_HTTPGET, 1L);

    const CURLcode perform_code = curl_easy_perform(curl);

    if (perform_code != CURLE_OK) {
        result.error =
            context.exceeded
                ? "response body limit exceeded"
                : curl_easy_strerror(perform_code);

        curl_easy_cleanup(curl);
        return result;
    }

    long response_code = 0;

    const CURLcode info_code =
        curl_easy_getinfo(
            curl,
            CURLINFO_RESPONSE_CODE,
            &response_code
        );

    if (info_code != CURLE_OK) {
        result.error = curl_easy_strerror(info_code);
        curl_easy_cleanup(curl);
        return result;
    }

    result.status = response_code;

    char* effective_url = nullptr;

    const CURLcode effective_info_code =
        curl_easy_getinfo(
            curl,
            CURLINFO_EFFECTIVE_URL,
            &effective_url
        );

    if (effective_info_code != CURLE_OK) {
        result.error = curl_easy_strerror(effective_info_code);
        curl_easy_cleanup(curl);
        return result;
    }

    if (effective_url) {
        result.effective_url = effective_url;
    }

    result.content_type = std::move(content_type);
    result.body = std::move(body);

    curl_easy_cleanup(curl);
    return result;
}

#undef NIYAH_SETOPT

}  // namespace niyah::search
