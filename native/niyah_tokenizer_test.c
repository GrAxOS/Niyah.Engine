#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "niyah.h"

/* Writes a newline-delimited vocab file; the line number is the token id. */
static void write_vocab(const char* path)
{
    FILE* f = fopen(path, "wb");
    assert(f != NULL);
    /* id 0 = "he", 1 = "hello", 2 = " ", 3 = "world", 4 = "l" */
    fputs("he\nhello\n \nworld\nl\n", f);
    fclose(f);
}

int main(void)
{
    const char* path = "niyah_tokenizer_test_vocab.txt";
    write_vocab(path);

    NiyahTokenizer tok;
    memset(&tok, 0, sizeof(tok));

    assert(niyah_tokenizer_load(&tok, path) == NIYAH_OK);
    assert(tok.vocab.n_vocab == 5);
    assert(tok.owns_vocab == true);

    /* Lookup maps a piece back to its line number. */
    assert(niyah_tokenizer_lookup(&tok, "he") == 0);
    assert(niyah_tokenizer_lookup(&tok, "hello") == 1);
    assert(niyah_tokenizer_lookup(&tok, "world") == 3);
    assert(niyah_tokenizer_lookup(&tok, "nope") < 0);
    assert(niyah_tokenizer_lookup(&tok, NULL) < 0);

    /*
     * Longest match wins. "hello" must tokenize as the single piece "hello"
     * (id 1), not as "he" + "l" + "l" + "o". This is the property that
     * distinguishes greedy longest-match from naive first-match.
     */
    int32_t tokens[64];
    int32_t n = niyah_tokenize(&tok, "hello", tokens, 64);
    assert(n == 1);
    assert(tokens[0] == 1);

    /* "hello world" -> "hello", " ", "world" */
    n = niyah_tokenize(&tok, "hello world", tokens, 64);
    assert(n == 3);
    assert(tokens[0] == 1);
    assert(tokens[1] == 2);
    assert(tokens[2] == 3);

    /* Round trip through detokenize. */
    char* text = niyah_detokenize(&tok, tokens, n);
    assert(text != NULL);
    assert(strcmp(text, "hello world") == 0);
    free(text);

    /* "he" alone resolves to the shorter piece. */
    n = niyah_tokenize(&tok, "he", tokens, 64);
    assert(n == 1);
    assert(tokens[0] == 0);

    /* max_tokens is respected rather than overrunning the caller's buffer. */
    int32_t small[2];
    n = niyah_tokenize(&tok, "hello world", small, 2);
    assert(n <= 2);

    /* Unknown bytes fall back to byte tokens instead of being dropped. */
    n = niyah_tokenize(&tok, "zq", tokens, 64);
    assert(n > 0);

    /* Degenerate inputs. */
    assert(niyah_tokenize(NULL, "hi", tokens, 64) < 0);
    assert(niyah_tokenize(&tok, NULL, tokens, 64) < 0);
    assert(niyah_detokenize(&tok, NULL, 3) == NULL);

    niyah_tokenizer_free(&tok);
    assert(tok.vocab.n_vocab == 0);

    /* A missing vocab file is an IO error, not a crash. */
    NiyahTokenizer missing;
    memset(&missing, 0, sizeof(missing));
    assert(niyah_tokenizer_load(&missing, "no_such_vocab_file.txt") != NIYAH_OK);

    remove(path);
    return 0;
}
