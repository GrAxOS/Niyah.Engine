#pragma once

#include "http_fetch.hpp"
#include "html_extract.hpp"
#include "niyah_crawler.h"
#include "niyah_index.h"
#include "../native/niyah_telemetry.h"

#include <cstdint>
#include <string>
#include <vector>

namespace niyah::search {

struct SearchConfig {
    std::string user_agent = "Niyah.Engine/0.1 (+local-search)";
    std::uint32_t max_depth = 2;
    std::size_t max_documents = 256;
    FetchLimits fetch_limits{};
    bool telemetry_enabled = false;
    std::string telemetry_path;
};

struct SearchResult {
    std::uint64_t document_id = 0;
    double score = 0.0;
    std::string url;
    std::string title;
};

class SearchEngine {
public:
    explicit SearchEngine(SearchConfig config = {});
    ~SearchEngine();

    SearchEngine(const SearchEngine&) = delete;
    SearchEngine& operator=(const SearchEngine&) = delete;

    bool seed(const std::string& url);
    std::size_t crawl_once();
    std::vector<SearchResult> search(const std::string& query, std::size_t limit = 10) const;
    std::size_t document_count() const noexcept;
    const NiyahTelemetryStats *telemetry_stats() const noexcept;

private:
    SearchConfig config_;
    std::vector<NiyahCrawlItem> frontier_storage_;
    NiyahCrawlFrontier frontier_{};
    NiyahInvertedIndex index_{};
    std::uint64_t next_document_id_ = 1;
    std::size_t fetched_documents_ = 0;
    mutable NiyahTelemetry telemetry_{};
    bool telemetry_ready_ = false;
};

}  // namespace niyah::search
