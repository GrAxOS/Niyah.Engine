#include "niyah_tokenizer.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static const NiyahVocabEntry VOCAB[8] = {
    {"l", 1u},
    {"o", 1u},
    {"w", 1u},
    {"e", 1u},
    {"r", 1u},
    {"lo", 2u},
    {"low", 3u},
    {"er", 2u}
};

static const NiyahMergeRule MERGES[3] = {
    {0u, 1u, 5u, 0u},
    {5u, 2u, 6u, 1u},
    {3u, 4u, 7u, 2u}
};

static void test_vocab_find(void) {
    uint32_t id = 0u;
    assert(niyah_tokenizer_vocab_find(VOCAB, 8u, "low", 3u, &id));
    assert(id == 6u);
    assert(!niyah_tokenizer_vocab_find(VOCAB, 8u, "zzz", 3u, &id));
}

static void test_encode_lower_matches_textbook_bpe(void) {
    const char *text = "lower";
    uint32_t scratch[5] = {0};
    size_t count = 0u;

    assert(niyah_tokenizer_encode_bytes(text, 5u, VOCAB, 8u, MERGES, 3u,
                                         scratch, 5u, &count));
    assert(count == 2u);
    assert(scratch[0] == 6u);
    assert(scratch[1] == 7u);
}

static void test_decode_round_trip(void) {
    const uint32_t tokens[2] = {6u, 7u};
    char buffer[16] = {0};
    size_t length = 0u;

    assert(niyah_tokenizer_decode(tokens, 2u, VOCAB, 8u, buffer,
                                   sizeof(buffer), &length));
    assert(length == 5u);
    assert(memcmp(buffer, "lower", 5u) == 0);
}

static void test_encode_rejects_unknown_byte(void) {
    const char *text = "l0w";
    uint32_t scratch[3] = {0};
    size_t count = 0u;

    assert(!niyah_tokenizer_encode_bytes(text, 3u, VOCAB, 8u, MERGES, 3u,
                                          scratch, 3u, &count));
}

static void test_encode_rejects_undersized_scratch(void) {
    const char *text = "lower";
    uint32_t scratch[2] = {0};
    size_t count = 0u;

    assert(!niyah_tokenizer_encode_bytes(text, 5u, VOCAB, 8u, MERGES, 3u,
                                          scratch, 2u, &count));
}

static void test_decode_rejects_undersized_buffer(void) {
    const uint32_t tokens[2] = {6u, 7u};
    char buffer[3] = {0};
    size_t length = 0u;

    assert(!niyah_tokenizer_decode(tokens, 2u, VOCAB, 8u, buffer,
                                    sizeof(buffer), &length));
}

int main(void) {
    test_vocab_find();
    test_encode_lower_matches_textbook_bpe();
    test_decode_round_trip();
    test_encode_rejects_unknown_byte();
    test_encode_rejects_undersized_scratch();
    test_decode_rejects_undersized_buffer();
    return 0;
}
