/*
 * niyah_token_tax -- measure what tokenisation costs, per language.
 *
 * Why this exists
 * ---------------
 * Byte-level and character-hash input layers (ByT5, CANINE, MEGABYTE, BLT)
 * are an active research direction, and the reported gains concentrate in
 * exactly the place this project lives: multilingual text, morphologically
 * rich languages, noisy input and orthography. They also cost something
 * real, because byte sequences are longer and attention is quadratic.
 *
 * That is a trade, and a trade should be decided by a measurement taken on
 * this machine with this vocab, not by a claim copied out of a paper.
 *
 * This tool takes that measurement. It is an instrument, not part of the
 * inference path, and it links against the same tokeniser the engine uses
 * so the number it prints is the number the engine actually pays.
 *
 * What it reports
 * ---------------
 *   fertility            tokens per whitespace-delimited word
 *   chars per token      how much meaning one token carries
 *   bytes per token      compression against raw UTF-8
 *   byte-fallback        tokens that are a single raw byte
 *   split characters     token boundaries landing mid-codepoint
 *   dropped bytes        input the tokeniser could not represent at all
 *   tax ratio            Arabic tokens / English tokens, same meaning
 *   byte baseline        what a tokeniser-free model would consume
 *
 * And a morphology probe: for a set of words sharing one Arabic root, does
 * any token survive across every member? If the answer is no, the root is
 * invisible to the model at the input layer. kataba / kaatib / maktuub are
 * one root to a reader and three unrelated id sequences to the model.
 *
 * Usage
 * -----
 *   niyah_token_tax --vocab <path> --parallel <path.tsv>
 *                   [--families <path.txt>] [--json]
 *
 *   vocab      newline-delimited, one piece per line; line number is the id
 *              (this is what niyah_tokenizer_load expects)
 *   parallel   TAB-separated: <arabic>\t<english>, one pair per line
 *   families   whitespace-separated words per line, one root family per line
 *
 *   Starter data ships in native/testdata/.
 *
 * Honest limits, stated up front because the report is only as good as these
 * assumptions:
 *   - "word" means whitespace-delimited. Arabic clitics (wa-, bi-, al-) are
 *     not separated, so Arabic fertility here is, if anything, understated.
 *   - Punctuation is not stripped; it is charged to whichever word it touches.
 *   - The parallel corpus is small. It shows the shape of the problem, not a
 *     publishable figure. Point --parallel at real data before quoting it.
 *   - A shared token across a root family is necessary evidence of root
 *     visibility, not sufficient. The tool prints the shared piece so the
 *     claim can be checked by eye.
 */

#include "niyah.h"
#include "niyah_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAX_MAX_LINE          8192
#define TAX_MAX_FAMILY_WORDS    32

/* ========================================================================
 * UTF-8
 * ======================================================================== */

static int is_ascii_space(unsigned char c)
{
    return c == ' ' || c == '\t' || c == '\n' ||
           c == '\r' || c == '\v' || c == '\f';
}

static int is_utf8_continuation(unsigned char c)
{
    return (c & 0xC0u) == 0x80u;
}

/*
 * Decodes one codepoint and advances *pos. Returns -1 for malformed input,
 * having advanced past at least one byte so the caller always terminates.
 */
static long utf8_next(const char* s, size_t len, size_t* pos)
{
    if (*pos >= len) {
        return -1;
    }

    const unsigned char b0 = (unsigned char)s[*pos];
    size_t need;
    long   cp;

    if (b0 < 0x80u) {
        *pos += 1u;
        return (long)b0;
    } else if ((b0 & 0xE0u) == 0xC0u) {
        need = 1u;
        cp   = (long)(b0 & 0x1Fu);
    } else if ((b0 & 0xF0u) == 0xE0u) {
        need = 2u;
        cp   = (long)(b0 & 0x0Fu);
    } else if ((b0 & 0xF8u) == 0xF0u) {
        need = 3u;
        cp   = (long)(b0 & 0x07u);
    } else {
        *pos += 1u;   /* stray continuation byte or invalid lead */
        return -1;
    }

    if (*pos + need >= len) {   /* truncated sequence at end of input */
        *pos = len;
        return -1;
    }

    for (size_t i = 1u; i <= need; ++i) {
        const unsigned char bc = (unsigned char)s[*pos + i];
        if (!is_utf8_continuation(bc)) {
            *pos += 1u;
            return -1;
        }
        cp = (cp << 6) | (long)(bc & 0x3Fu);
    }

    *pos += need + 1u;
    return cp;
}

static size_t codepoint_count(const char* s)
{
    const size_t len = strlen(s);
    size_t pos = 0u;
    size_t n   = 0u;

    while (pos < len) {
        if (utf8_next(s, len, &pos) >= 0) {
            ++n;
        }
    }
    return n;
}

/*
 * Arabic block, Supplement, Extended-A, and the two presentation-form
 * blocks. Presentation forms should not appear in normalised text, but they
 * do appear in scraped text, and counting them is how you find out.
 */
static int is_arabic_codepoint(long cp)
{
    return (cp >= 0x0600L && cp <= 0x06FFL) ||
           (cp >= 0x0750L && cp <= 0x077FL) ||
           (cp >= 0x08A0L && cp <= 0x08FFL) ||
           (cp >= 0xFB50L && cp <= 0xFDFFL) ||
           (cp >= 0xFE70L && cp <= 0xFEFFL);
}

/* ========================================================================
 * Vocab id index
 * ======================================================================== */

typedef struct {
    int32_t* by_id;
    int32_t  max_id;
} IdIndex;

static int idindex_build(IdIndex* idx, const NiyahTokenizer* tk)
{
    idx->by_id  = NULL;
    idx->max_id = -1;

    const int32_t n = tk->vocab.vocab ? tk->vocab.n_vocab : 0;

    for (int32_t i = 0; i < n; ++i) {
        const int32_t vid = tk->vocab.ids ? tk->vocab.ids[i] : i;
        if (vid > idx->max_id) {
            idx->max_id = vid;
        }
    }
    if (idx->max_id < 0) {
        return 1;   /* empty vocab: valid, just useless */
    }

    idx->by_id = (int32_t*)malloc((size_t)(idx->max_id + 1) * sizeof(int32_t));
    if (!idx->by_id) {
        return 0;
    }
    for (int32_t i = 0; i <= idx->max_id; ++i) {
        idx->by_id[i] = -1;
    }
    for (int32_t i = 0; i < n; ++i) {
        const int32_t vid = tk->vocab.ids ? tk->vocab.ids[i] : i;
        if (vid >= 0 && vid <= idx->max_id && idx->by_id[vid] < 0) {
            idx->by_id[vid] = i;
        }
    }
    return 1;
}

static void idindex_free(IdIndex* idx)
{
    free(idx->by_id);
    idx->by_id  = NULL;
    idx->max_id = -1;
}

static const char* piece_for_id(const IdIndex* idx,
                                const NiyahTokenizer* tk,
                                int32_t id)
{
    if (id < 0 || id > idx->max_id || !idx->by_id) {
        return NULL;
    }
    const int32_t vi = idx->by_id[id];
    if (vi < 0) {
        return NULL;
    }
    return tk->vocab.vocab[vi];
}

/*
 * "<0x41>" -> 0x41, else -1.
 *
 * This duplicates byte_token_value() in niyah_tokenizer.c on purpose. An
 * instrument that shares code with the thing it measures cannot detect that
 * thing's bugs. If these two ever disagree, that disagreement is a finding.
 */
static int tax_byte_token_value(const char* piece)
{
    if (!piece) {
        return -1;
    }
    if (piece[0] != '<' || piece[1] != '0' ||
        (piece[2] != 'x' && piece[2] != 'X')) {
        return -1;
    }

    int value  = 0;
    int digits = 0;
    const char* p = piece + 3;

    for (; *p && *p != '>'; ++p, ++digits) {
        int d;
        if (*p >= '0' && *p <= '9') {
            d = *p - '0';
        } else if (*p >= 'a' && *p <= 'f') {
            d = *p - 'a' + 10;
        } else if (*p >= 'A' && *p <= 'F') {
            d = *p - 'A' + 10;
        } else {
            return -1;
        }
        value = value * 16 + d;
    }

    if (*p != '>' || p[1] != '\0' || digits == 0 || digits > 2) {
        return -1;
    }
    return value;
}

/* ========================================================================
 * Statistics
 * ======================================================================== */

typedef struct {
    const char* label;
    size_t lines;
    size_t words;
    size_t codepoints;
    size_t arabic_codepoints;
    size_t bytes;
    size_t tokens;
    size_t byte_fallback_tokens;
    size_t unresolved_tokens;
    size_t split_characters;
    size_t dropped_bytes;
} TaxStats;

static double ratio(size_t a, size_t b)
{
    return b ? (double)a / (double)b : 0.0;
}

/*
 * Tokenise `text` and fold the result into `st`.
 *
 * Byte offsets are recovered by replaying the emitted stream over the
 * original input rather than by trusting strlen(piece): a "<0xD8>" piece is
 * six characters long but consumes exactly one input byte. Getting this
 * wrong would silently invent both boundary violations and dropped bytes.
 */
static void measure_text(NiyahTokenizer* tk,
                         const IdIndex* idx,
                         const char* text,
                         TaxStats* st)
{
    const size_t len = strlen(text);
    if (len == 0u) {
        return;
    }

    st->lines += 1u;
    st->bytes += len;

    size_t pos = 0u;
    while (pos < len) {
        const long cp = utf8_next(text, len, &pos);
        if (cp < 0) {
            continue;
        }
        st->codepoints += 1u;
        if (is_arabic_codepoint(cp)) {
            st->arabic_codepoints += 1u;
        }
    }

    int in_word = 0;
    for (size_t i = 0u; i < len; ++i) {
        if (is_ascii_space((unsigned char)text[i])) {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            st->words += 1u;
        }
    }

    /* Worst case is one token per byte, so len + 1 always suffices. */
    if (len > (size_t)0x7FFFFFFE) {
        fprintf(stderr, "niyah_token_tax: line too long, skipped\n");
        return;
    }
    int32_t* tokens = (int32_t*)malloc((len + 1u) * sizeof(int32_t));
    if (!tokens) {
        fprintf(stderr, "niyah_token_tax: out of memory\n");
        return;
    }

    const int32_t n = niyah_tokenize(tk, text, tokens, (int32_t)(len + 1u));
    if (n > 0) {
        st->tokens += (size_t)n;
    }

    size_t offset = 0u;
    for (int32_t t = 0; t < n; ++t) {
        /*
         * A boundary that lands on a UTF-8 continuation byte has cut a
         * character in half. For Arabic every letter is two bytes, so this
         * counter is the direct measure of sub-character shredding.
         */
        if (offset > 0u && offset < len &&
            is_utf8_continuation((unsigned char)text[offset])) {
            st->split_characters += 1u;
        }

        const char* piece = piece_for_id(idx, tk, tokens[t]);
        size_t consumed;

        if (piece && tax_byte_token_value(piece) >= 0) {
            consumed = 1u;
            st->byte_fallback_tokens += 1u;
        } else if (piece) {
            consumed = strlen(piece);
        } else if (tokens[t] >= 0 && tokens[t] <= 255 &&
                   tokens[t] >= tk->vocab.n_vocab) {
            /* Mirrors the tokeniser's bare-byte fallback. */
            consumed = 1u;
            st->byte_fallback_tokens += 1u;
        } else {
            st->unresolved_tokens += 1u;
            consumed = 0u;
        }

        if (consumed > len - offset) {
            offset = len;
            break;
        }
        offset += consumed;
    }

    /* Whatever the stream did not consume, the tokeniser dropped. */
    if (offset < len) {
        st->dropped_bytes += len - offset;
    }

    free(tokens);

    /* niyah_tokenize parks the caller's buffer on the struct. Do not leave a
     * dangling pointer behind for the next call to trip over. */
    tk->tokens   = NULL;
    tk->n_tokens = 0;
}

/* ========================================================================
 * Reporting
 * ======================================================================== */

static void print_row(const char* name, size_t a, size_t b)
{
    printf("  %-28s %12zu %12zu\n", name, a, b);
}

static void print_frow(const char* name, double a, double b)
{
    printf("  %-28s %12.2f %12.2f\n", name, a, b);
}

static void report_text(const TaxStats* ar,
                        const TaxStats* en,
                        const char* vocab_path,
                        int32_t n_vocab,
                        const char* corpus_path)
{
    printf("\nNiyah tokenisation tax report\n");
    printf("=============================\n");
    printf("vocab   : %s  (%d pieces)\n", vocab_path, (int)n_vocab);
    printf("corpus  : %s  (%zu parallel pairs)\n\n", corpus_path, ar->lines);

    printf("  %-28s %12s %12s\n", "", "Arabic", "English");
    printf("  ---------------------------------------------------------\n");
    print_row("words",                ar->words,      en->words);
    print_row("characters",           ar->codepoints, en->codepoints);
    print_row("bytes (UTF-8)",        ar->bytes,      en->bytes);
    print_row("tokens",               ar->tokens,     en->tokens);
    printf("  ---------------------------------------------------------\n");
    print_frow("tokens per word",
               ratio(ar->tokens, ar->words), ratio(en->tokens, en->words));
    print_frow("characters per token",
               ratio(ar->codepoints, ar->tokens),
               ratio(en->codepoints, en->tokens));
    print_frow("bytes per token",
               ratio(ar->bytes, ar->tokens), ratio(en->bytes, en->tokens));
    printf("  ---------------------------------------------------------\n");
    print_row("byte-fallback tokens",
              ar->byte_fallback_tokens, en->byte_fallback_tokens);
    print_row("characters cut in half",
              ar->split_characters, en->split_characters);
    print_row("bytes dropped",   ar->dropped_bytes,      en->dropped_bytes);
    print_row("unresolved tokens", ar->unresolved_tokens, en->unresolved_tokens);

    const double tax = ratio(ar->tokens, en->tokens);

    printf("\n  TAX  Arabic tokens / English tokens = %.2f\n", tax);
    if (tax > 0.0) {
        printf("       The same meaning costs %.2fx more to represent in"
               " Arabic.\n", tax);
        printf("       That is compute, context window and money, per"
               " request, forever.\n");
    }

    printf("\n  Byte-level baseline (what a tokeniser-free model consumes):\n");
    printf("       Arabic  : %zu bytes vs %zu tokens  (compression %.2fx)\n",
           ar->bytes, ar->tokens, ratio(ar->bytes, ar->tokens));
    printf("       English : %zu bytes vs %zu tokens  (compression %.2fx)\n",
           en->bytes, en->tokens, ratio(en->bytes, en->tokens));
    printf("       A byte model pays the byte column and owes no vocabulary"
           " table.\n");

    if (ar->split_characters > 0u) {
        printf("\n  NOTE %zu Arabic characters were split across token"
               " boundaries.\n", ar->split_characters);
        printf("       Every Arabic letter is two bytes in UTF-8. A boundary"
               " inside one\n");
        printf("       means the model never sees that letter as a letter.\n");
    }
    if (ar->dropped_bytes > 0u) {
        printf("\n  WARNING %zu Arabic bytes were dropped entirely. This"
               " vocab cannot\n", ar->dropped_bytes);
        printf("          round-trip Arabic. Fix the vocab before trusting"
               " any other number.\n");
    }
}

static void report_json(const TaxStats* ar, const TaxStats* en)
{
    const TaxStats* s[2];
    s[0] = ar;
    s[1] = en;

    printf("{\n  \"languages\": {\n");
    for (int i = 0; i < 2; ++i) {
        printf("    \"%s\": {\n", s[i]->label);
        printf("      \"lines\": %zu,\n",       s[i]->lines);
        printf("      \"words\": %zu,\n",       s[i]->words);
        printf("      \"characters\": %zu,\n",  s[i]->codepoints);
        printf("      \"arabic_characters\": %zu,\n",
               s[i]->arabic_codepoints);
        printf("      \"bytes\": %zu,\n",       s[i]->bytes);
        printf("      \"tokens\": %zu,\n",      s[i]->tokens);
        printf("      \"tokens_per_word\": %.4f,\n",
               ratio(s[i]->tokens, s[i]->words));
        printf("      \"characters_per_token\": %.4f,\n",
               ratio(s[i]->codepoints, s[i]->tokens));
        printf("      \"bytes_per_token\": %.4f,\n",
               ratio(s[i]->bytes, s[i]->tokens));
        printf("      \"byte_fallback_tokens\": %zu,\n",
               s[i]->byte_fallback_tokens);
        printf("      \"split_characters\": %zu,\n",
               s[i]->split_characters);
        printf("      \"dropped_bytes\": %zu,\n", s[i]->dropped_bytes);
        printf("      \"unresolved_tokens\": %zu\n",
               s[i]->unresolved_tokens);
        printf("    }%s\n", i == 0 ? "," : "");
    }
    printf("  },\n");
    printf("  \"tax_ratio\": %.4f\n", ratio(ar->tokens, en->tokens));
    printf("}\n");
}

/* ========================================================================
 * Morphology probe
 * ======================================================================== */

/*
 * For each family of words sharing one root, look for a token id present in
 * every member. A shared multi-character piece means the tokeniser exposes
 * something the model can latch onto. No shared piece means the root is
 * invisible at the input layer, and every derived form must be learned as an
 * unrelated string.
 */
static int probe_families(NiyahTokenizer* tk,
                          const IdIndex* idx,
                          const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "niyah_token_tax: cannot open %s\n", path);
        return 0;
    }

    printf("\n  Morphology probe (shared token across one root family)\n");
    printf("  ---------------------------------------------------------\n");

    char line[TAX_MAX_LINE];
    size_t families = 0u;
    size_t preserved = 0u;

    while (fgets(line, sizeof(line), f)) {
        size_t l = strlen(line);
        while (l > 0u && (line[l - 1u] == '\n' || line[l - 1u] == '\r')) {
            line[--l] = '\0';
        }
        if (l == 0u || line[0] == '#') {
            continue;
        }

        /* Split on whitespace, in place. */
        char*  words[TAX_MAX_FAMILY_WORDS];
        size_t n_words = 0u;
        size_t i = 0u;

        while (i < l && n_words < TAX_MAX_FAMILY_WORDS) {
            while (i < l && is_ascii_space((unsigned char)line[i])) {
                ++i;
            }
            if (i >= l) {
                break;
            }
            words[n_words++] = line + i;
            while (i < l && !is_ascii_space((unsigned char)line[i])) {
                ++i;
            }
            if (i < l) {
                line[i++] = '\0';
            }
        }

        if (n_words < 2u) {
            continue;
        }
        ++families;

        /* Tokenise every member. */
        int32_t* ids[TAX_MAX_FAMILY_WORDS];
        int32_t  counts[TAX_MAX_FAMILY_WORDS];
        int      ok = 1;

        for (size_t w = 0u; w < n_words; ++w) {
            const size_t wl = strlen(words[w]);
            ids[w] = (int32_t*)malloc((wl + 1u) * sizeof(int32_t));
            if (!ids[w]) {
                ok = 0;
                counts[w] = 0;
                break;
            }
            counts[w] = niyah_tokenize(tk, words[w], ids[w],
                                       (int32_t)(wl + 1u));
            tk->tokens   = NULL;
            tk->n_tokens = 0;
        }

        const char* shared = NULL;

        if (ok) {
            for (int32_t a = 0; a < counts[0] && !shared; ++a) {
                const int32_t cand = ids[0][a];
                const char* piece = piece_for_id(idx, tk, cand);

                /* A single character shared by every Arabic word proves
                 * nothing. Require at least two characters. */
                if (!piece || codepoint_count(piece) < 2u) {
                    continue;
                }
                if (tax_byte_token_value(piece) >= 0) {
                    continue;
                }

                int in_all = 1;
                for (size_t w = 1u; w < n_words && in_all; ++w) {
                    int found = 0;
                    for (int32_t b = 0; b < counts[w]; ++b) {
                        if (ids[w][b] == cand) {
                            found = 1;
                            break;
                        }
                    }
                    in_all = found;
                }
                if (in_all) {
                    shared = piece;
                }
            }
        }

        if (shared) {
            ++preserved;
            printf("  root visible    %-34s shared piece: \"%s\"\n",
                   words[0], shared);
        } else {
            printf("  root SHATTERED  %-34s no piece common to all %zu"
                   " forms\n", words[0], n_words);
        }

        for (size_t w = 0u; w < n_words; ++w) {
            free(ids[w]);
            ids[w] = NULL;
        }
    }

    fclose(f);

    printf("  ---------------------------------------------------------\n");
    printf("  %zu of %zu root families expose a shared token (%.0f%%).\n",
           preserved, families,
           families ? 100.0 * (double)preserved / (double)families : 0.0);

    if (families && preserved * 2u < families) {
        printf("\n  Most roots are invisible to the model at the input"
               " layer.\n");
        printf("  Arabic is root-and-pattern. A tokeniser that hides the"
               " root forces\n");
        printf("  the model to memorise every derived form separately,"
               " which is\n");
        printf("  exactly the behaviour that reads as repetition rather"
               " than\n");
        printf("  understanding.\n");
    }

    return 1;
}

/* ========================================================================
 * main
 * ======================================================================== */

static void usage(const char* argv0)
{
    fprintf(stderr,
        "usage: %s --vocab <path> --parallel <file.tsv>\n"
        "          [--families <file.txt>] [--json]\n"
        "\n"
        "  --vocab     newline-delimited vocab, one piece per line\n"
        "  --parallel  TAB-separated <arabic>\\t<english> pairs\n"
        "  --families  whitespace-separated root families, one per line\n"
        "  --json      machine-readable output for CI\n"
        "\n"
        "  Starter data: native/testdata/parallel_ar_en.tsv\n"
        "                native/testdata/root_families_ar.txt\n",
        argv0);
}

int main(int argc, char** argv)
{
    const char* vocab_path    = NULL;
    const char* parallel_path = NULL;
    const char* families_path = NULL;
    int json = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (strcmp(argv[i], "--parallel") == 0 && i + 1 < argc) {
            parallel_path = argv[++i];
        } else if (strcmp(argv[i], "--families") == 0 && i + 1 < argc) {
            families_path = argv[++i];
        } else if (strcmp(argv[i], "--json") == 0) {
            json = 1;
        } else {
            fprintf(stderr, "unknown or incomplete argument: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    if (!vocab_path || !parallel_path) {
        usage(argv[0]);
        return 2;
    }

    NiyahTokenizer tk;
    memset(&tk, 0, sizeof(tk));

    const NiyahStatus st = niyah_tokenizer_load(&tk, vocab_path);
    if (st != NIYAH_OK) {
        fprintf(stderr, "niyah_token_tax: cannot load vocab %s: %s\n",
                vocab_path, niyah_status_to_string(st));
        return 1;
    }
    if (tk.vocab.n_vocab <= 0) {
        fprintf(stderr, "niyah_token_tax: vocab %s is empty\n", vocab_path);
        niyah_tokenizer_free(&tk);
        return 1;
    }

    IdIndex idx;
    if (!idindex_build(&idx, &tk)) {
        fprintf(stderr, "niyah_token_tax: out of memory building id index\n");
        niyah_tokenizer_free(&tk);
        return 1;
    }

    FILE* f = fopen(parallel_path, "rb");
    if (!f) {
        fprintf(stderr, "niyah_token_tax: cannot open %s\n", parallel_path);
        idindex_free(&idx);
        niyah_tokenizer_free(&tk);
        return 1;
    }

    TaxStats ar;
    TaxStats en;
    memset(&ar, 0, sizeof(ar));
    memset(&en, 0, sizeof(en));
    ar.label = "arabic";
    en.label = "english";

    char   line[TAX_MAX_LINE];
    size_t skipped = 0u;

    while (fgets(line, sizeof(line), f)) {
        size_t l = strlen(line);
        const int had_newline = (l > 0u && line[l - 1u] == '\n');

        while (l > 0u && (line[l - 1u] == '\n' || line[l - 1u] == '\r')) {
            line[--l] = '\0';
        }

        if (!had_newline && l + 1u == sizeof(line)) {
            /* Truncated by the buffer. Counting a fragment would understate
             * the tax, so refuse it and say so. */
            ++skipped;
            int c;
            while ((c = fgetc(f)) != EOF && c != '\n') {
                /* discard remainder */
            }
            continue;
        }

        if (l == 0u || line[0] == '#') {
            continue;
        }

        char* tab = strchr(line, '\t');
        if (!tab) {
            ++skipped;
            continue;
        }
        *tab = '\0';

        measure_text(&tk, &idx, line,    &ar);
        measure_text(&tk, &idx, tab + 1, &en);
    }

    fclose(f);

    if (ar.lines == 0u) {
        fprintf(stderr, "niyah_token_tax: no usable pairs in %s\n",
                parallel_path);
        idindex_free(&idx);
        niyah_tokenizer_free(&tk);
        return 1;
    }

    if (json) {
        report_json(&ar, &en);
    } else {
        report_text(&ar, &en, vocab_path, tk.vocab.n_vocab, parallel_path);
        if (skipped > 0u) {
            printf("\n  %zu malformed or over-long lines were skipped.\n",
                   skipped);
        }
        if (families_path) {
            probe_families(&tk, &idx, families_path);
        }
        printf("\n  Measured with vocab \"%s\". Nothing here is a claim about"
               " any\n", vocab_path);
        printf("  other tokeniser. Point --vocab at a different vocab to"
               " compare.\n\n");
    }

    idindex_free(&idx);
    niyah_tokenizer_free(&tk);
    return 0;
}
