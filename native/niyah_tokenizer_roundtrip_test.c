/*
 * Regression test for the byte-fallback id collision.
 *
 * niyah_tokenizer_load() assigns vocab ids sequentially from 0. The previous
 * byte fallback emitted the raw byte value as a token id, so byte 0xD8 (216)
 * and vocab entry #216 were indistinguishable, and niyah_detokenize()
 * resolves vocab entries before byte values.
 *
 * That is not an exotic edge case. Every Arabic letter in U+0600..U+06FF is
 * encoded with a 0xD8 or 0xD9 lead byte, so essentially all Arabic input was
 * corrupted on the way back out.
 *
 * The vocab below deliberately places 256 "<0xXX>" byte tokens *after* a
 * handful of ordinary pieces, so the byte tokens occupy ids 7..262 and the
 * raw byte values 0..255 collide with real vocab ids. This is the exact
 * configuration that used to fail.
 */
#undef NDEBUG
#include <assert.h>

#include "niyah.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* kPrefixPieces[] = {
    "<unk>", "<s>", "</s>", " ", "the", "code", "engine",
};

#define PREFIX_COUNT ((int32_t)(sizeof(kPrefixPieces) / sizeof(kPrefixPieces[0])))

static void write_vocab(const char* path)
{
    FILE* f = fopen(path, "wb");
    assert(f != NULL);

    for (int32_t i = 0; i < PREFIX_COUNT; ++i) {
        fprintf(f, "%s\n", kPrefixPieces[i]);
    }
    for (int b = 0; b < 256; ++b) {
        fprintf(f, "<0x%02X>\n", (unsigned)b);
    }

    fclose(f);
}

static void roundtrip(NiyahTokenizer* tk, const char* label, const char* text)
{
    int32_t tokens[512];

    const int32_t n = niyah_tokenize(tk, text, tokens, 512);
    assert(n > 0);

    char* back = niyah_detokenize(tk, tokens, n);
    assert(back != NULL);

    if (strcmp(back, text) != 0) {
        fprintf(stderr,
                "round trip failed [%s]\n  in : %s\n  out: %s\n",
                label, text, back);
        free(back);
        assert(0);
    }

    free(back);
}

int main(void)
{
    const char* vocab_path = "niyah_roundtrip_vocab.txt";
    write_vocab(vocab_path);

    NiyahTokenizer tk;
    assert(niyah_tokenizer_load(&tk, vocab_path) == NIYAH_OK);
    assert(tk.vocab.n_vocab == PREFIX_COUNT + 256);

    /* Sanity: the byte tokens really do sit on top of colliding ids. */
    assert(niyah_tokenizer_lookup(&tk, "<0xD8>") == PREFIX_COUNT + 0xD8);

    /* ASCII, using ordinary multi-character pieces. */
    roundtrip(&tk, "ascii", "the code engine");

    /* Arabic: every letter starts with a 0xD8 or 0xD9 lead byte. */
    roundtrip(&tk, "arabic",
              "\xD8\xA7\xD9\x84\xD8\xB9\xD8\xB1\xD8\xA8\xD9\x8A\xD8\xA9");

    /* Mixed script, digits and punctuation. */
    roundtrip(&tk, "mixed", "Niyah \xD9\x86\xD9\x8A\xD8\xA9 v1.0");

    /* Four-byte UTF-8. */
    roundtrip(&tk, "emoji", "\xF0\x9F\x8C\xB9");

    /* Arabic-Indic digits and a tatweel, which sit in the same blocks. */
    roundtrip(&tk, "arabic-digits", "\xD9\xA1\xD9\xA2\xD9\xA3 \xD9\x80");

    /* Detokenising nothing yields an allocated empty string, never NULL. */
    char* empty = niyah_detokenize(&tk, NULL, 0);
    assert(empty != NULL);
    assert(empty[0] == '\0');
    free(empty);

    niyah_tokenizer_free(&tk);
    remove(vocab_path);

    printf("niyah_tokenizer_roundtrip_test: OK\n");
    return 0;
}
