#include "niyah.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// Bridge API for UI - proper C exports for P/Invoke

EXPORT const char* niyah_bridge_version(void) {
    return niyah_version();
}

// Simple search stub
EXPORT int niyah_bridge_search(const char* query, void** results, int* count) {
    (void)query;
    // Return dummy results
    static const char* dummy[] = {"doc1", "doc2", "doc3"};
    static float scores[] = {0.95f, 0.87f, 0.76f};
    *count = 3;
    // For simplicity, return pointer to first string
    *results = (void*)dummy;
    return 0;
}

// Document stub
EXPORT int niyah_bridge_add_document(const char* content, const char** doc_id) {
    static const char* id = "doc_new";
    (void)content;
    *doc_id = id;
    return 0;
}

EXPORT const char* niyah_get_version(void) {
    return niyah_version();
}

EXPORT const char* niyah_get_truth_string(NiyahTruth truth) {
    return niyah_truth_to_string(truth);
}
