#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "niyah_proof.h"

static void test_deterministic_and_tamper_evident(void)
{
    static const char prompt[] = "explain transaction rollback";
    static const char output[] = "the transaction is restored to its prior state";
    static const char rules[] = "local-only";

    NiyahProofV1 a;
    NiyahProofV1 b;

    assert(niyah_proof_v1_generate(
               prompt, strlen(prompt),
               output, strlen(output),
               rules, strlen(rules),
               &a) == NIYAH_OK);
    assert(niyah_proof_v1_generate(
               prompt, strlen(prompt),
               output, strlen(output),
               rules, strlen(rules),
               &b) == NIYAH_OK);
    assert(memcmp(&a, &b, sizeof(a)) == 0);

    static const char changed[] = "different output";
    assert(niyah_proof_v1_generate(
               prompt, strlen(prompt),
               changed, strlen(changed),
               rules, strlen(rules),
               &b) == NIYAH_OK);
    assert(memcmp(a.digest, b.digest, NIYAH_SHA256_BYTES) != 0);
}

static void test_receipt_contains_hashes_not_payload(void)
{
    static const char prompt[] = "PRIVATE_PROMPT_SENTINEL";
    static const char output[] = "PRIVATE_OUTPUT_SENTINEL";
    static const char rules[] = "PRIVATE_RULE_SENTINEL";

    NiyahProofV1 proof;
    assert(niyah_proof_v1_generate(
               prompt, strlen(prompt),
               output, strlen(output),
               rules, strlen(rules),
               &proof) == NIYAH_OK);

    char receipt[384];
    size_t written = 0u;
    assert(niyah_proof_v1_serialize(
               &proof, receipt, sizeof(receipt), &written) == NIYAH_OK);
    assert(written > 0u);
    assert(strstr(receipt, NIYAH_PROOF_V1_HEADER) != NULL);
    assert(strstr(receipt, "prompt_hash: ") != NULL);
    assert(strstr(receipt, "output_hash: ") != NULL);
    assert(strstr(receipt, "rules_hash: ") != NULL);
    assert(strstr(receipt, prompt) == NULL);
    assert(strstr(receipt, output) == NULL);
    assert(strstr(receipt, rules) == NULL);
}

static void test_saved_receipt_verification(void)
{
    static const char path[] = "niyah_proof_v1_test.proof";
    static const char prompt[] = "hello";
    static const char output[] = "world";
    static const char rules[] = "rules-v1";

    NiyahProofV1 proof;
    assert(niyah_proof_v1_generate(
               prompt, strlen(prompt),
               output, strlen(output),
               rules, strlen(rules),
               &proof) == NIYAH_OK);
    assert(niyah_proof_v1_save(path, &proof) == NIYAH_OK);

    bool matches = false;
    assert(niyah_proof_v1_verify_file(
               path,
               prompt, strlen(prompt),
               output, strlen(output),
               rules, strlen(rules),
               &matches) == NIYAH_OK);
    assert(matches);

    static const char tampered[] = "tampered";
    matches = true;
    assert(niyah_proof_v1_verify_file(
               path,
               prompt, strlen(prompt),
               tampered, strlen(tampered),
               rules, strlen(rules),
               &matches) == NIYAH_OK);
    assert(!matches);

    assert(remove(path) == 0);
}

static void test_invalid_arguments_fail_closed(void)
{
    NiyahProofV1 proof;
    bool matches = true;

    assert(niyah_proof_v1_generate(
               NULL, 1u, NULL, 0u, NULL, 0u, &proof)
           == NIYAH_ERR_INVALID_ARG);
    assert(niyah_proof_v1_generate(
               NULL, 0u, NULL, 0u, NULL, 0u, NULL)
           == NIYAH_ERR_INVALID_ARG);
    assert(niyah_proof_v1_verify_file(
               NULL, NULL, 0u, NULL, 0u, NULL, 0u, &matches)
           == NIYAH_ERR_INVALID_ARG);
    assert(!matches);
}

int main(void)
{
    test_deterministic_and_tamper_evident();
    test_receipt_contains_hashes_not_payload();
    test_saved_receipt_verification();
    test_invalid_arguments_fail_closed();

    puts("niyah_proof_v1: ok");
    return 0;
}
