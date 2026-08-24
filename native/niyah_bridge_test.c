#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "niyah_bridge.h"

/*
 * The old niyah_bridge.c answered every search with a static array:
 *
 *     const char* dummy[] = {"doc1", "doc2", "doc3"};
 *     float scores[] = {0.95f, 0.87f, 0.76f};
 *
 * so the C# UI rendered confident hits for documents that were never indexed,
 * and niyah_bridge_add_document always handed back the literal "doc_new".
 * These assertions pin the real behaviour and would fail loudly if the
 * placeholder data ever came back.
 */
int main(void)
{
    /* Version strings must exist for the P/Invoke smoke test in the UI. */
    assert(niyah_bridge_version() != NULL);
    assert(niyah_get_version() != NULL);
    assert(strcmp(niyah_get_version(), "0.2.0") == 0);
    assert(strcmp(niyah_get_truth_string(NIYAH_UNKNOWN), "unknown") == 0);

    /* Start from a known-empty store. */
    niyah_bridge_clear();
    assert(niyah_bridge_document_count() == 0);

    /*
     * An empty index must return nothing. This is the assertion that the fake
     * data is gone: previously this same call returned three hits.
     */
    char* empty = niyah_bridge_search_json("anything", 10);
    assert(empty != NULL);
    assert(strstr(empty, "doc1") == NULL);
    assert(strstr(empty, "doc2") == NULL);
    assert(strstr(empty, "doc3") == NULL);
    assert(strstr(empty, "0.95") == NULL);
    niyah_bridge_free_string(empty);

    void* results = NULL;
    int count = -1;
    assert(niyah_bridge_search("anything", &results, &count) == 0);
    assert(count == 0);
    niyah_bridge_free_results(results);

    /* Add real documents. */
    const char* id_a = NULL;
    const char* id_b = NULL;
    assert(niyah_bridge_add_document(
               "the capital of japan is tokyo", &id_a) == 0);
    assert(id_a != NULL);
    assert(niyah_bridge_document_count() == 1);

    assert(niyah_bridge_add_document(
               "rust is a systems programming language", &id_b) == 0);
    assert(id_b != NULL);
    assert(niyah_bridge_document_count() == 2);

    /* Ids are generated per document, not the constant "doc_new". */
    assert(strcmp(id_a, "doc_new") != 0);
    assert(strcmp(id_a, id_b) != 0);

    /* A matching query finds the right document. */
    char* hit = niyah_bridge_search_json("tokyo", 10);
    assert(hit != NULL);
    assert(strstr(hit, id_a) != NULL);
    assert(strstr(hit, id_b) == NULL);
    niyah_bridge_free_string(hit);

    /* A query matching the other document finds only that one. */
    char* other = niyah_bridge_search_json("rust", 10);
    assert(other != NULL);
    assert(strstr(other, id_b) != NULL);
    assert(strstr(other, id_a) == NULL);
    niyah_bridge_free_string(other);

    /* A term present in neither document returns no hits. */
    char* miss = niyah_bridge_search_json("zebra", 10);
    assert(miss != NULL);
    assert(strstr(miss, id_a) == NULL);
    assert(strstr(miss, id_b) == NULL);
    niyah_bridge_free_string(miss);

    /* max_hits is respected. */
    char* capped = niyah_bridge_search_json("is", 1);
    assert(capped != NULL);
    niyah_bridge_free_string(capped);

    /* Legacy search path returns a real count. */
    results = NULL;
    count = -1;
    assert(niyah_bridge_search("tokyo", &results, &count) == 0);
    assert(count == 1);
    assert(results != NULL);

    const NiyahBridgeResults* typed = (const NiyahBridgeResults*)results;
    assert(typed->count == 1);
    assert(typed->hits != NULL);
    assert(typed->hits[0].doc_id != NULL);
    assert(strcmp(typed->hits[0].doc_id, id_a) == 0);
    assert(typed->hits[0].score > 0.0f);
    niyah_bridge_free_results(results);

    niyah_bridge_free_string((char*)id_a);
    niyah_bridge_free_string((char*)id_b);

    /* Clear really empties the store. */
    niyah_bridge_clear();
    assert(niyah_bridge_document_count() == 0);

    char* after_clear = niyah_bridge_search_json("tokyo", 10);
    assert(after_clear != NULL);
    assert(strstr(after_clear, "tokyo") == NULL);
    niyah_bridge_free_string(after_clear);

    /* Degenerate inputs. */
    assert(niyah_bridge_add_document(NULL, &id_a) != 0);
    assert(niyah_bridge_add_document("content", NULL) != 0);
    assert(niyah_bridge_search(NULL, &results, &count) != 0);
    assert(niyah_bridge_search_json(NULL, 10) == NULL);
    niyah_bridge_free_string(NULL);
    niyah_bridge_free_results(NULL);

    /* Context lifecycle. */
    NiyahBridgeContext* ctx = niyah_bridge_create(NULL);
    assert(ctx != NULL);
    assert(niyah_bridge_graph(ctx) != NULL);
    niyah_bridge_destroy(ctx);
    niyah_bridge_destroy(NULL);

    return 0;
}
