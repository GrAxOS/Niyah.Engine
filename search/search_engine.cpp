#include "search_engine.hpp"

#include "niyah_url.h"

#include <algorithm>
#include <cstring>

namespace niyah::search {

SearchEngine::SearchEngine(SearchConfig config)
    : config_(std::move(config)),
      frontier_storage_(std::max<std::size_t>(config_.max_documents, 1u)) {
    niyah_crawl_frontier_init(&frontier_, frontier_storage_.data(), frontier_storage_.size());
    niyah_index_init(&index_, 1.2, 0.75);
}

SearchEngine::~SearchEngine() {
    niyah_index_free(&index_);
}

bool SearchEngine::seed(const std::string& url) {
    char canonical[2048] = {0};
    if (!niyah_url_canonicalize(url.c_str(), canonical, sizeof(canonical))) return false;
    if (fetched_documents_ >= config_.max_documents) return false;
    return niyah_crawl_enqueue(&frontier_, canonical, 0, 0) == NIYAH_FETCH_ALLOW;
}

std::size_t SearchEngine::crawl_once() {
    if (fetched_documents_ >= config_.max_documents) return 0;

    NiyahCrawlItem item{};
    if (!niyah_crawl_pop(&frontier_, &item)) return 0;
    if (item.depth > config_.max_depth) return 0;

    const FetchResult fetched = http_get(item.url, config_.user_agent, config_.fetch_limits);
    if (fetched.status < 200 || fetched.status >= 300 || fetched.body.empty()) return 0;

    const HtmlDocument document = extract_html(fetched.body, fetched.effective_url.empty() ? item.url : fetched.effective_url);
    const std::string effective = fetched.effective_url.empty() ? item.url : fetched.effective_url;

    NiyahDocument indexed{};
    indexed.document_id = next_document_id_++;
    std::snprintf(indexed.url, sizeof(indexed.url), "%s", effective.c_str());
    std::snprintf(indexed.title, sizeof(indexed.title), "%s", document.title.c_str());
    std::snprintf(indexed.text, sizeof(indexed.text), "%s", document.text.c_str());

    if (!niyah_index_add_document(&index_, &indexed)) return 0;

    ++fetched_documents_;
    if (item.depth < config_.max_depth) {
        for (const std::string& link : document.links) {
            if (fetched_documents_ >= config_.max_documents) break;
            char canonical[2048] = {0};
            if (!niyah_url_canonicalize(link.c_str(), canonical, sizeof(canonical))) continue;
            (void)niyah_crawl_enqueue(&frontier_, canonical, item.depth + 1u, 0);
        }
    }
    return 1;
}

std::vector<SearchResult> SearchEngine::search(const std::string& query, std::size_t limit) const {
    if (limit == 0 || index_.document_count == 0) return {};

    const std::size_t capacity = std::min(limit, index_.document_count);
    std::vector<NiyahSearchHit> hits(capacity);
    const std::size_t count = niyah_index_search(&index_, query.c_str(), hits.data(), hits.size());

    std::vector<SearchResult> results;
    results.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const NiyahDocument* document = niyah_index_document(&index_, hits[i].document_id);
        if (!document) continue;
        results.push_back(SearchResult{
            document->document_id,
            hits[i].score,
            document->url,
            document->title});
    }
    return results;
}

std::size_t SearchEngine::document_count() const noexcept {
    return index_.document_count;
}

}  // namespace niyah::search
