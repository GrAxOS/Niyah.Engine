#undef NDEBUG
#include <assert.h>

#include "niyah_identity.h"
#include "niyah_sha256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ==========================================================================
 * SHA-256 against the FIPS 180-4 / RFC 6234 vectors
 * ========================================================================== */

static void expect_hash(const char* input, const char* expected_hex)
{
    uint8_t digest[NIYAH_SHA256_BYTES];
    char hex[NIYAH_SHA256_HEX_BYTES];

    niyah_sha256_buffer(input, strlen(input), digest);
    niyah_sha256_to_hex(digest, hex);

    if (strcmp(hex, expected_hex) != 0) {
        fprintf(stderr, "sha256 mismatch\n  in  : %s\n  got : %s\n  want: %s\n",
                input, hex, expected_hex);
        assert(0);
    }
}

static void test_sha256(void)
{
    expect_hash("",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    expect_hash("abc",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    expect_hash("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    /* Streaming in awkward chunks must equal the one-shot result. */
    NiyahSha256 ctx;
    uint8_t streamed[NIYAH_SHA256_BYTES];
    uint8_t oneshot[NIYAH_SHA256_BYTES];

    const char* text =
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";

    niyah_sha256_init(&ctx);
    niyah_sha256_update(&ctx, text, 1);
    niyah_sha256_update(&ctx, text + 1, 30);
    niyah_sha256_update(&ctx, text + 31, strlen(text) - 31);
    niyah_sha256_final(&ctx, streamed);

    niyah_sha256_buffer(text, strlen(text), oneshot);
    assert(memcmp(streamed, oneshot, NIYAH_SHA256_BYTES) == 0);

    /* A one-million-'a' message crosses many block boundaries. */
    niyah_sha256_init(&ctx);
    for (int i = 0; i < 1000; ++i) {
        char chunk[1000];
        memset(chunk, 'a', sizeof(chunk));
        niyah_sha256_update(&ctx, chunk, sizeof(chunk));
    }
    uint8_t million[NIYAH_SHA256_BYTES];
    char million_hex[NIYAH_SHA256_HEX_BYTES];
    niyah_sha256_final(&ctx, million);
    niyah_sha256_to_hex(million, million_hex);
    assert(strcmp(million_hex,
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0")
        == 0);

    printf("  sha256: OK\n");
}

/* ==========================================================================
 * Identity
 * ========================================================================== */

static void write_file(const char* path, const char* text)
{
    FILE* f = fopen(path, "wb");
    assert(f != NULL);
    fputs(text, f);
    fclose(f);
}

static void test_empty_identity(void)
{
    NiyahIdentity id;
    assert(niyah_identity_capture(&id, NULL) == NIYAH_OK);

    assert(strcmp(id.engine_name, "niyah") == 0);
    assert(strcmp(id.engine_version, niyah_version()) == 0);
    assert(id.weights_loaded == false);

    /* With nothing loaded, nothing about the model is knowable. */
    assert(id.provenance_known == NIYAH_UNKNOWN);
    assert(niyah_identity_is(&id, "llama") == NIYAH_UNKNOWN);
    assert(niyah_identity_is(&id, "gpt-4") == NIYAH_UNKNOWN);

    /* The engine name is a compile-time fact and needs no manifest. */
    assert(niyah_identity_is(&id, "Niyah") == NIYAH_TRUE);
    assert(niyah_identity_is(&id, "NIYAH ENGINE") == NIYAH_TRUE);

    printf("  empty identity: OK\n");
}

static void test_manifest_and_tamper(void)
{
    const char* path = "niyah_identity_test_manifest.json";

    /* Hash of a stand-in weight buffer. */
    const char* fake_weights = "pretend these are float32 weights";
    uint8_t digest[NIYAH_SHA256_BYTES];
    char hex[NIYAH_SHA256_HEX_BYTES];
    niyah_sha256_buffer(fake_weights, strlen(fake_weights), digest);
    niyah_sha256_to_hex(digest, hex);

    /* --- honest manifest --- */
    char manifest[1024];
    snprintf(manifest, sizeof(manifest),
             "{\n"
             "  \"model_name\": \"Qwen2.5-0.5B-Instruct\",\n"
             "  \"origin\": \"https://huggingface.co/Qwen/Qwen2.5-0.5B\",\n"
             "  \"license\": \"Apache-2.0\",\n"
             "  \"weights_sha256\": \"%s\"\n"
             "}\n", hex);
    write_file(path, manifest);

    NiyahIdentity id;
    niyah_identity_init(&id);

    /* Pretend the hash was measured from memory. */
    memcpy(id.weights_sha256, hex, sizeof(hex));
    id.weights_hashed = true;

    assert(niyah_identity_load_manifest(&id, path) == NIYAH_OK);
    assert(id.provenance_known == NIYAH_TRUE);
    assert(id.weights_match_manifest == NIYAH_TRUE);
    assert(strcmp(id.model_name, "Qwen2.5-0.5B-Instruct") == 0);
    assert(strcmp(id.license, "Apache-2.0") == 0);

    /* Claims can now be answered with evidence. */
    assert(niyah_identity_is(&id, "qwen") == NIYAH_TRUE);
    assert(niyah_identity_is(&id, "Qwen2.5") == NIYAH_TRUE);
    assert(niyah_identity_is(&id, "llama") == NIYAH_FALSE);
    assert(niyah_identity_is(&id, "gpt-4") == NIYAH_FALSE);

    /* --- tampered: manifest kept, weights swapped --------------------- */
    niyah_identity_init(&id);

    const char* other_weights = "different bytes entirely";
    niyah_sha256_buffer(other_weights, strlen(other_weights), digest);
    niyah_sha256_to_hex(digest, id.weights_sha256);
    id.weights_hashed = true;

    assert(niyah_identity_load_manifest(&id, path) == NIYAH_OK);
    assert(id.weights_match_manifest == NIYAH_FALSE);

    /*
     * The manifest still says Qwen, but it is no longer evidence of anything.
     * Every model claim must collapse to UNKNOWN - not TRUE, and not FALSE
     * either, because FALSE would itself be an unsupported assertion.
     */
    assert(niyah_identity_is(&id, "qwen") == NIYAH_UNKNOWN);
    assert(niyah_identity_is(&id, "llama") == NIYAH_UNKNOWN);

    /* A manifest with no model name proves nothing. */
    write_file(path, "{ \"license\": \"MIT\" }\n");
    niyah_identity_init(&id);
    assert(niyah_identity_load_manifest(&id, path) == NIYAH_ERR_NOT_FOUND);
    assert(id.provenance_known == NIYAH_UNKNOWN);

    assert(niyah_identity_load_manifest(&id, "no_such_manifest.json")
           == NIYAH_ERR_IO);

    remove(path);
    printf("  manifest + tamper detection: OK\n");
}

static void test_self_query_detection(void)
{
    assert(niyah_identity_is_self_query("who are you?"));
    assert(niyah_identity_is_self_query("WHO ARE YOU"));
    assert(niyah_identity_is_self_query("So, what model are you exactly?"));
    assert(niyah_identity_is_self_query("are you gpt-4 under the hood"));
    assert(niyah_identity_is_self_query("Introduce yourself"));

    /* Arabic, MSA and dialect. */
    assert(niyah_identity_is_self_query(
        "\xD9\x85\xD9\x86 \xD8\xA3\xD9\x86\xD8\xAA"));
    assert(niyah_identity_is_self_query(
        "\xD9\x85\xD9\x8A\xD9\x86 \xD8\xA7\xD9\x86\xD8\xAA\xD8\x9F"));
    assert(niyah_identity_is_self_query(
        "\xD9\x88\xD8\xB4 \xD8\xA7\xD9\x86\xD8\xAA \xD8\xA8\xD8\xB6\xD8\xA8\xD8\xB7"));
    assert(niyah_identity_is_self_query(
        "\xD8\xA3\xD9\x8A \xD9\x86\xD9\x85\xD9\x88\xD8\xB0\xD8\xAC \xD8\xA3\xD9\x86\xD8\xAA"));

    /* Ordinary prompts must not be hijacked. */
    assert(!niyah_identity_is_self_query("summarise this document"));
    assert(!niyah_identity_is_self_query("what is the capital of Japan"));
    assert(!niyah_identity_is_self_query(""));
    assert(!niyah_identity_is_self_query(NULL));

    printf("  self-query detection: OK\n");
}

static void test_report(void)
{
    NiyahIdentity id;
    assert(niyah_identity_capture(&id, NULL) == NIYAH_OK);

    char buffer[4096];
    size_t written = 0;

    assert(niyah_identity_report(&id, buffer, sizeof(buffer), &written)
           == NIYAH_OK);
    assert(written > 0);
    assert(written < sizeof(buffer));
    assert(strstr(buffer, "niyah engine") != NULL);
    assert(strstr(buffer, niyah_version()) != NULL);
    assert(strstr(buffer, "no model loaded") != NULL);
    assert(strstr(buffer, "unknown") != NULL);

    /* Deterministic: same input, identical bytes. */
    char again[4096];
    size_t written_again = 0;
    assert(niyah_identity_report(&id, again, sizeof(again), &written_again)
           == NIYAH_OK);
    assert(written == written_again);
    assert(strcmp(buffer, again) == 0);

    /* Truncation is reported, never silently accepted. */
    char tiny[16];
    size_t needed = 0;
    assert(niyah_identity_report(&id, tiny, sizeof(tiny), &needed)
           == NIYAH_ERR_OVERFLOW);
    assert(needed >= sizeof(tiny));

    /* JSON variant parses as a flat object. */
    assert(niyah_identity_report_json(&id, buffer, sizeof(buffer), &written)
           == NIYAH_OK);
    assert(buffer[0] == '{');
    assert(strstr(buffer, "\"engine_name\": \"niyah\"") != NULL);
    assert(strstr(buffer, "\"provenance_known\": \"unknown\"") != NULL);

    printf("  report: OK\n");
}

int main(void)
{
    printf("niyah_identity_test\n");
    test_sha256();
    test_empty_identity();
    test_manifest_and_tamper();
    test_self_query_detection();
    test_report();
    printf("niyah_identity_test: OK\n");
    return 0;
}
