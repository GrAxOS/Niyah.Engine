#include "search_engine.hpp"

#include "niyah_url.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <utility>

namespace niyah::search {
namespace {

std::uint64_t unix_ms()
{
    const auto now = std::chrono::system_clock::now();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());
}

std::uint64_t mono_ns()
{
    const auto now = std::chrono::steady_clock::now();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
}

std::uint64_t elapsed_ms(std::uint64_t start_ns)
{
    const std::uint64_t now = mono_ns();
    return now >= start_ns ? (now - start_ns) / 1000000u : 0u;
}

}  // namespace

SearchEngine::SearchEngine(SearchConfig config)
    : config_(std::move(config)),
      frontier_storage_(std::max<std::size_t>(config_.max_documents, 1u))
{
    niyah_crawl_frontier_init(&frontier_, frontier_storage_.data(), frontier_storage_.size());
    niyah_index_init(&index_, 1.2, 0.75);

    const NiyahTelemetryConfig telemetry_config{
        config_.telemetry_enabled,
        false,
        config_.telemetry_path.empty() ? nullptr : config_.telemetry_path.c_str()};
    telemetry_ready_ = niyah_telemetry_init(&telemetry_, &telemetry_config, unix_ms(), mono_ns());
    if (telemetry_ready_)
        (void)niyah_telemetry_event(&telemetry_, "engine_init", 0, 0, 0, false);
}

SearchEngine::~SearchEngine()
{
    niyah_index_free(&index_);
    if (telemetry_ready_) niyah_telemetry_close(&telemetry_);
}

bool SearchEngine::seed(const std::string& url)
{
    const std::uint64_t start = mono_ns();
    char canonical[2048] = {0};
    const bool valid = niyah_url_canonicalize(url.c_str(), canonical, sizeof(canonical));
    const bool accepted = valid &&
                          fetched_documents_ < config_.max_documents &&
                          niyah_crawl_enqueue(&frontier_, canonical, 0, unix_ms()) == NIYAH_FETCH_ALLOW;
    if (telemetry_ready_)
        (void)niyah_telemetry_event(&telemetry_, "crawl_seed", elapsed_ms(start), 0, 0, !accepted);
    return accepted;
}

std::size_t SearchEngine::crawl_once()
{
    const std::uint64_t start = mono_ns();
    if (fetched_documents_ >= config_.max_documents) {
        if (telemetry_ready_)
            (void)niyah_telemetry_event(&telemetry_, "crawl_limit", elapsed_ms(start), 0, 0, false);
        return 0;
    }

    NiyahCrawlItem item{};
    if (!niyah_crawl_pop(&frontier_, &item)) {
        if (telemetry_ready_)
            (void)niyah_telemetry_event(&telemetry_, "crawl_empty", elapsed_ms(start), 0, 0, false);
        return 0;
    }
    if (item.depth > config_.max_depth) {
        if (telemetry_ready_)
            (void)niyah_telemetry_event(&telemetry_, "crawl_depth_blocked", elapsed_ms(start), 0, 0, false);
        return 0;
    }

    const FetchResult fetched = http_get(item.url, config_.user_agent, config_.fetch_limits);
    if (fetched.status < 200 || fetched.status >= 300 || fetched.body.empty()) {
        if (telemetry_ready_)
            (void)niyah_telemetry_event(&telemetry_, "crawl_fetch_failed", elapsed_ms(start),
                                        static_cast<std::uint64_t>(fetched.body.size()), 0, true);
        return 0;
    }

    const std::string effective = fetched.effective_url.empty() ? item.url : fetched.effective_url;
    char effective_canonical[2048] = {0};
    if (!niyah_url_canonicalize(effective.c_str(), effective_canonical,
                               sizeof(effective_canonical))) {
        if (telemetry_ready_)
            (void)niyah_telemetry_event(&telemetry_, "crawl_invalid_effective_url", elapsed_ms(start), 0, 0, true);
        return 0;
    }

    const HtmlDocument document = extract_html(fetched.body, effective_canonical);

    NiyahDocument indexed{};
    indexed.document_id = next_document_id_++;
    std::snprintf(indexed.url, sizeof(indexed.url), "%s", effective_canonical);
    std::snprintf(indexed.title, sizeof(indexed.title), "%s", document.title.c_str());
    std::snprintf(indexed.text, sizeof(indexed.text), "%s", document.text.c_str());

    if (!niyah_index_add_document(&index_, &indexed)) {
        if (telemetry_ready_)
            (void)niyah_telemetry_event(&telemetry_, "index_failed", elapsed_ms(start),
                                        static_cast<std::uint64_t>(fetched.body.size()), 0, true);
        return 0;
    }

    ++fetched_documents_;
    if (item.depth < config_.max_depth) {
        for (const std::string& link : document.links) {
            if (fetched_documents_ >= config_.max_documents) break;
            char canonical[2048] = {0};
            if (!niyah_url_resolve(effective_canonical, link.c_str(), canonical, sizeof(canonical)))
                continue;
            (void)niyah_crawl_enqueue(&frontier_, canonical, item.depth + 1u, unix_ms());
        }
    }

    if (telemetry_ready_)
        (void)niyah_telemetry_event(&telemetry_, "crawl_indexed", elapsed_ms(start),
                                    static_cast<std::uint64_t>(fetched.body.size()),
                                    static_cast<std::uint64_t>(document.text.size()), false);
    return 1;
}

std::vector<SearchResult> SearchEngine::search(const std::string& query, std::size_t limit) const
{
    const std::uint64_t start = mono_ns();
    if (limit == 0u || index_.document_count == 0u) {
        if (telemetry_ready_)
            (void)niyah_telemetry_event(&telemetry_, "search_empty", elapsed_ms(start),
                                        static_cast<std::uint64_t>(query.size()), 0, false);
        return {};
    }

    const std::size_t capacity = std::min(limit, index_.document_count);
    std::vector<NiyahSearchHit> hits(capacity);
    const std::size_t count = niyah_index_search(&index_, query.c_str(), hits.data(), hits.size());

    std::vector<SearchResult> results;
    results.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const NiyahDocument* document = niyah_index_document(&index_, hits[i].document_id);
        if (!document) continue;
        results.push_back(SearchResult{document->document_id, hits[i].score,
                                       document->url, document->title});
    }

    if (telemetry_ready_)
        (void)niyah_telemetry_event(&telemetry_, "search", elapsed_ms(start),
                                    static_cast<std::uint64_t>(query.size()),
                                    static_cast<std::uint64_t>(results.size()), false);
    return results;
}

std::size_t SearchEngine::document_count() const noexcept
{
    return index_.document_count;
}

const NiyahTelemetryStats *SearchEngine::telemetry_stats() const noexcept
{
    return telemetry_ready_ ? niyah_telemetry_stats(&telemetry_) : nullptr;
}

}  // namespace niyah::search
