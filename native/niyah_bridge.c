#include "niyah.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Bridge API for UI

const char* niyah_bridge_version(void) {
    return niyah_version();
}

// Simple search stub - returns hardcoded results
// TODO: Implement actual search when niyah.h has search types
typedef struct {
    const char* id;
    float score;
} SearchResult;

static SearchResult g_results[] = {
    {"doc1", 0.95f},
    {"doc2", 0.87f},
    {"doc3", 0.76f}
};

int niyah_bridge_search(const char* query, SearchResult** results, int* count) {
    (void)query;
    *results = g_results;
    *count = 3;
    return 0;
}

// Document stub
int niyah_bridge_add_document(const char* content, const char** doc_id) {
    static const char* id = "doc_new";
    (void)content;
    *doc_id = id;
    return 0;
}

const char* niyah_get_version(void) {
    return niyah_version();
}

const char* niyah_get_truth_string(NiyahTruth truth) {
    return niyah_truth_to_string(truth);
}
