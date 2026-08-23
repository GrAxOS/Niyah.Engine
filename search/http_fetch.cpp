#include "http_fetch.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <curl/curl.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#endif

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
    if (size != 0 && nmemb > (std::numeric_limits<std::size_t>::max)() / size) {
        ctx->exceeded = true;
        return 0;
    }
    const std::size_t bytes = size * nmemb;
    const std::size_t current = ctx->body->size();
    if (current > ctx->limit || bytes > ctx->limit - current) {
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

size_t write_headers(char* ptr, size_t size, size_t nmemb, void* userdata) {
    if (size != 0 && nmemb > (std::numeric_limits<std::size_t>::max)() / size) return 0;
    const std::size_t bytes = size * nmemb;
    auto* content_type = static_cast<std::string*>(userdata);
    if (!content_type) return bytes;
    const std::string header(ptr, bytes);
    constexpr char prefix[] = "Content-Type:";
    constexpr std::size_t prefix_size = sizeof(prefix) - 1u;
    if (header.size() < prefix_size) return bytes;
    const bool matches = std::equal(
        prefix, prefix + prefix_size, header.data(),
        [](char left, char right) {
            return std::tolower(static_cast<unsigned char>(left)) ==
                   std::tolower(static_cast<unsigned char>(right));
        });
    if (!matches) return bytes;
    *content_type = header.substr(prefix_size);
    while (!content_type->empty() &&
           (content_type->front() == ' ' || content_type->front() == '\t')) {
        content_type->erase(content_type->begin());
    }
    while (!content_type->empty() &&
           (content_type->back() == '\r' || content_type->back() == '\n')) {
        content_type->pop_back();
    }
    return bytes;
}

std::string lowercase_ascii(std::string value) {
    for (char& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}

bool parse_authority(const std::string& authority, std::string& host, std::string& port) {
    if (authority.empty() || authority.find('@') != std::string::npos) return false;
    host.clear();
    port.clear();
    if (authority.front() == '[') {
        const std::size_t close = authority.find(']');
        if (close == std::string::npos || close == 1u) return false;
        host = authority.substr(1u, close - 1u);
        if (close + 1u < authority.size()) {
            if (authority[close + 1u] != ':') return false;
            port = authority.substr(close + 2u);
        }
    } else {
        const std::size_t first = authority.find(':');
        const std::size_t last = authority.rfind(':');
        if (first != std::string::npos && first != last) return false;
        if (first == std::string::npos) {
            host = authority;
        } else {
            host = authority.substr(0u, first);
            port = authority.substr(first + 1u);
        }
    }
    if (host.empty()) return false;
    if (port.empty()) return true;
    for (char c : port) {
        if (c < '0' || c > '9') return false;
    }
    return port.size() <= 5u;
}

bool unsafe_ipv4(const sockaddr_in& addr) {
    const uint32_t v = ntohl(addr.sin_addr.s_addr);
    const uint32_t a = (v >> 24u) & 0xffu;
    const uint32_t b = (v >> 16u) & 0xffu;
    if (a == 0u || a == 10u || a == 127u || a >= 224u) return true;
    if (a == 100u && b >= 64u && b <= 127u) return true;
    if (a == 169u && b == 254u) return true;
    if (a == 172u && b >= 16u && b <= 31u) return true;
    if (a == 192u && b == 168u) return true;
    if (a == 198u && (b == 18u || b == 19u)) return true;
    if (a == 192u && b == 0u) return true;
    if (a == 198u && b == 51u) return true;
    if (a == 203u && b == 0u) return true;
    return false;
}

bool unsafe_ipv6(const sockaddr_in6& addr) {
    const unsigned char* b = addr.sin6_addr.s6_addr;
    const bool unspecified = std::all_of(b, b + 16u, [](unsigned char x) { return x == 0u; });
    if (unspecified) return true;
    const bool loopback = std::all_of(b, b + 15u, [](unsigned char x) { return x == 0u; }) && b[15] == 1u;
    if (loopback) return true;
    if ((b[0] & 0xfeu) == 0xfcu) return true;
    if (b[0] == 0xfeu && (b[1] & 0xc0u) == 0x80u) return true;
    if (b[0] == 0xffu) return true;
    if (b[0] == 0x20u && b[1] == 0x01u && b[2] == 0x0du && b[3] == 0xb8u) return true;
    return false;
}

bool resolved_destination_safe(const std::string& host, const std::string& port) {
    if (host.empty()) return false;
    std::string service = port.empty() ? "443" : port;
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_ADDRCONFIG;
    addrinfo* results = nullptr;
    const int rc = getaddrinfo(host.c_str(), service.c_str(), &hints, &results);
    if (rc != 0 || !results) return false;

    bool safe = true;
    for (addrinfo* p = results; p != nullptr; p = p->ai_next) {
        if (!p->ai_addr) {
            safe = false;
            break;
        }
        if (p->ai_family == AF_INET && p->ai_addrlen >= sizeof(sockaddr_in)) {
            if (unsafe_ipv4(*reinterpret_cast<const sockaddr_in*>(p->ai_addr))) {
                safe = false;
                break;
            }
        } else if (p->ai_family == AF_INET6 && p->ai_addrlen >= sizeof(sockaddr_in6)) {
            if (unsafe_ipv6(*reinterpret_cast<const sockaddr_in6*>(p->ai_addr))) {
                safe = false;
                break;
            }
        } else {
            safe = false;
            break;
        }
    }
    freeaddrinfo(results);
    return safe;
}

bool basic_http_url_ok(const std::string& url) {
    const bool http = url.rfind("http://", 0u) == 0;
    const bool https = url.rfind("https://", 0u) == 0;
    if (!http && !https) return false;
    const std::size_t scheme_end = url.find("://");
    if (scheme_end == std::string::npos) return false;
    const std::size_t authority_start = scheme_end + 3u;
    const std::size_t authority_end = url.find_first_of("/?#", authority_start);
    const std::string authority = url.substr(
        authority_start,
        authority_end == std::string::npos ? std::string::npos : authority_end - authority_start);
    std::string host;
    std::string port;
    if (!parse_authority(authority, host, port)) return false;
    const std::string lowered = lowercase_ascii(host);
    if (lowered == "localhost" || lowered == "localhost.localdomain") return false;
    return resolved_destination_safe(host, port.empty() ? (https ? "443" : "80") : port);
}

#define NIYAH_SETOPT(handle, option, value) \
    do { \
        const CURLcode setopt_code = curl_easy_setopt((handle), (option), (value)); \
        if (setopt_code != CURLE_OK) { \
            result.error = curl_easy_strerror(setopt_code); \
            curl_easy_cleanup(curl); \
            return result; \
        } \
    } while (0)

}  // namespace

FetchResult http_get(const std::string& url, const std::string& user_agent, const FetchLimits& limits) {
    FetchResult result;
    if (!basic_http_url_ok(url)) {
        result.error = "URL rejected by DNS/IP fetch policy";
        return result;
    }
    if (limits.max_body_bytes == 0u || limits.connect_timeout_seconds <= 0 ||
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
    const char* effective_user_agent = user_agent.empty() ? "Niyah.Engine/0.1" : user_agent.c_str();

    NIYAH_SETOPT(curl, CURLOPT_URL, url.c_str());
    NIYAH_SETOPT(curl, CURLOPT_USERAGENT, effective_user_agent);
    NIYAH_SETOPT(curl, CURLOPT_FOLLOWLOCATION, 0L);
    NIYAH_SETOPT(curl, CURLOPT_MAXREDIRS, 0L);
    NIYAH_SETOPT(curl, CURLOPT_CONNECTTIMEOUT, limits.connect_timeout_seconds);
    NIYAH_SETOPT(curl, CURLOPT_TIMEOUT, limits.total_timeout_seconds);
    NIYAH_SETOPT(curl, CURLOPT_WRITEFUNCTION, write_body);
    NIYAH_SETOPT(curl, CURLOPT_WRITEDATA, &context);
    NIYAH_SETOPT(curl, CURLOPT_HEADERFUNCTION, write_headers);
    NIYAH_SETOPT(curl, CURLOPT_HEADERDATA, &content_type);
#if LIBCURL_VERSION_NUM >= 0x075500
    NIYAH_SETOPT(curl, CURLOPT_PROTOCOLS_STR, "http,https");
    NIYAH_SETOPT(curl, CURLOPT_REDIR_PROTOCOLS_STR, "http,https");
    NIYAH_SETOPT(curl, CURLOPT_DISALLOW_USERNAME_IN_URL, 1L);
#else
    NIYAH_SETOPT(curl, CURLOPT_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS);
    NIYAH_SETOPT(curl, CURLOPT_REDIR_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS);
#endif
    NIYAH_SETOPT(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    NIYAH_SETOPT(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    NIYAH_SETOPT(curl, CURLOPT_ACCEPT_ENCODING, "");
    NIYAH_SETOPT(curl, CURLOPT_HTTPGET, 1L);

    const CURLcode perform_code = curl_easy_perform(curl);
    if (perform_code != CURLE_OK) {
        result.error = context.exceeded ? "response body limit exceeded" : curl_easy_strerror(perform_code);
        curl_easy_cleanup(curl);
        return result;
    }

    long response_code = 0;
    const CURLcode info_code = curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    if (info_code != CURLE_OK) {
        result.error = curl_easy_strerror(info_code);
        curl_easy_cleanup(curl);
        return result;
    }
    result.status = response_code;

    char* effective_url = nullptr;
    const CURLcode effective_info_code = curl_easy_getinfo(curl, CURLINFO_EFFECTIVE_URL, &effective_url);
    if (effective_info_code != CURLE_OK) {
        result.error = curl_easy_strerror(effective_info_code);
        curl_easy_cleanup(curl);
        return result;
    }
    if (effective_url) result.effective_url = effective_url;
    result.content_type = std::move(content_type);
    result.body = std::move(body);
    curl_easy_cleanup(curl);
    return result;
}

#undef NIYAH_SETOPT

}  // namespace niyah::search
