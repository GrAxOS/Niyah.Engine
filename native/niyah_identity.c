#include "niyah_identity.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ==========================================================================
 * Small helpers
 * ========================================================================== */

static char lower_ascii(char c)
{
    return (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
}

/* Case-insensitive substring search, ASCII only. Arabic needles are matched
 * byte-exactly, which is correct: they contain no ASCII case to fold. */
static bool contains_ci(const char* haystack, const char* needle)
{
    if (!haystack || !needle || !needle[0]) {
        return false;
    }

    for (const char* h = haystack; *h; ++h) {
        const char* a = h;
        const char* b = needle;
        while (*a && *b && lower_ascii(*a) == lower_ascii(*b)) {
            ++a;
            ++b;
        }
        if (!*b) {
            return true;
        }
    }
    return false;
}

static void copy_bounded(char* dst, size_t dst_size, const char* src)
{
    if (!dst || dst_size == 0) {
        return;
    }
    if (!src) {
        dst[0] = '\0';
        return;
    }
    const size_t len = strlen(src);
    const size_t take = (len < dst_size - 1u) ? len : dst_size - 1u;
    memcpy(dst, src, take);
    dst[take] = '\0';
}

/* Appends to buffer, tracking how much has been written. Never overflows;
 * truncation is reported by the caller comparing *used against buffer_size. */
static void append(char* buffer, size_t buffer_size, size_t* used,
                   const char* fmt, ...)
{
    if (!buffer || !used || *used >= buffer_size) {
        if (used) {
            /* Keep counting so the caller can size a second attempt. */
            va_list ap;
            va_start(ap, fmt);
            const int n = vsnprintf(NULL, 0, fmt, ap);
            va_end(ap);
            if (n > 0) {
                *used += (size_t)n;
            }
        }
        return;
    }

    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(buffer + *used, buffer_size - *used, fmt, ap);
    va_end(ap);

    if (n > 0) {
        *used += (size_t)n;
    }
}

/* ==========================================================================
 * Flat JSON string extraction, same deliberate limits as niyah_model.c:
 * no nesting, no arrays, no escape handling. Manifests are ours.
 * ========================================================================== */

static bool json_find_string(const char* buf, const char* key,
                             char* out, size_t out_size)
{
    char pattern[128];
    const int written = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (written <= 0 || (size_t)written >= sizeof(pattern)) {
        return false;
    }

    const char* p = strstr(buf, pattern);
    if (!p) {
        return false;
    }
    p += written;

    while (*p && *p != ':' && *p != ',' && *p != '}') {
        ++p;
    }
    if (*p != ':') {
        return false;
    }
    ++p;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
        ++p;
    }
    if (*p != '"') {
        return false;
    }
    ++p;

    const char* end = strchr(p, '"');
    if (!end) {
        return false;
    }

    size_t len = (size_t)(end - p);
    if (len > out_size - 1u) {
        len = out_size - 1u;
    }
    memcpy(out, p, len);
    out[len] = '\0';
    return true;
}

/* ==========================================================================
 * Capture
 * ========================================================================== */

void niyah_identity_init(NiyahIdentity* identity)
{
    if (!identity) {
        return;
    }

    memset(identity, 0, sizeof(*identity));

    copy_bounded(identity->engine_name, sizeof(identity->engine_name),
                 "niyah");
    copy_bounded(identity->engine_version, sizeof(identity->engine_version),
                 niyah_version());
    copy_bounded(identity->engine_build_date,
                 sizeof(identity->engine_build_date), __DATE__);

    identity->provenance_known = NIYAH_UNKNOWN;
    identity->weights_match_manifest = NIYAH_UNKNOWN;
}

NiyahStatus niyah_identity_capture(NiyahIdentity* identity,
                                   const NiyahLLM* llm)
{
    if (!identity) {
        return NIYAH_ERR_INVALID_ARG;
    }

    niyah_identity_init(identity);

    if (!llm) {
        /* Honest empty state rather than an error. */
        return NIYAH_OK;
    }

    identity->config = llm->model.config;
    niyah_model_config_normalize(&identity->config);

    identity->weights_loaded = (llm->model.weights != NULL)
                            && (llm->model.weights_size > 0);
    identity->weights_bytes = llm->model.weights_size;

    if (identity->weights_loaded) {
        /*
         * Measured, not declared: the blob is float32, so the parameter count
         * is exactly the number of floats actually resident.
         */
        identity->parameter_count =
            (uint64_t)(llm->model.weights_size / sizeof(float));
    } else {
        /* Fall back to what the shape implies, clearly not a measurement. */
        const size_t expected =
            niyah_model_expected_floats(&identity->config);
        identity->parameter_count = (uint64_t)expected;
    }

    return NIYAH_OK;
}

NiyahStatus niyah_identity_hash_weights(NiyahIdentity* identity,
                                        const NiyahLLM* llm)
{
    if (!identity || !llm) {
        return NIYAH_ERR_INVALID_ARG;
    }
    if (!llm->model.weights || llm->model.weights_size == 0) {
        return NIYAH_ERR_NO_WEIGHTS;
    }

    uint8_t digest[NIYAH_SHA256_BYTES];
    niyah_sha256_buffer(llm->model.weights, llm->model.weights_size, digest);
    niyah_sha256_to_hex(digest, identity->weights_sha256);
    identity->weights_hashed = true;

    /* A manifest may already have been loaded; settle the comparison now. */
    if (identity->declared_weights_sha256[0] != '\0') {
        identity->weights_match_manifest =
            (strcmp(identity->weights_sha256,
                    identity->declared_weights_sha256) == 0)
                ? NIYAH_TRUE : NIYAH_FALSE;
    }

    return NIYAH_OK;
}

NiyahStatus niyah_identity_load_manifest(NiyahIdentity* identity,
                                         const char* manifest_path)
{
    if (!identity || !manifest_path) {
        return NIYAH_ERR_INVALID_ARG;
    }

    FILE* f = fopen(manifest_path, "rb");
    if (!f) {
        return NIYAH_ERR_IO;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NIYAH_ERR_IO;
    }
    const long size = ftell(f);
    if (size <= 0 || size > (1L << 20)) {
        fclose(f);
        return NIYAH_ERR_IO;
    }
    rewind(f);

    char* buf = (char*)malloc((size_t)size + 1u);
    if (!buf) {
        fclose(f);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    const size_t read = fread(buf, 1, (size_t)size, f);
    fclose(f);
    buf[read] = '\0';

    const bool have_name = json_find_string(buf, "model_name",
                                            identity->model_name,
                                            sizeof(identity->model_name));
    json_find_string(buf, "origin", identity->origin,
                     sizeof(identity->origin));
    json_find_string(buf, "license", identity->license,
                     sizeof(identity->license));
    json_find_string(buf, "weights_sha256",
                     identity->declared_weights_sha256,
                     sizeof(identity->declared_weights_sha256));
    json_find_string(buf, "corpus_manifest_sha256",
                     identity->corpus_manifest_sha256,
                     sizeof(identity->corpus_manifest_sha256));

    free(buf);

    if (!have_name || identity->model_name[0] == '\0') {
        /* A manifest with no model name proves nothing. */
        identity->provenance_known = NIYAH_UNKNOWN;
        return NIYAH_ERR_NOT_FOUND;
    }

    identity->provenance_known = NIYAH_TRUE;

    if (identity->weights_hashed
        && identity->declared_weights_sha256[0] != '\0') {
        identity->weights_match_manifest =
            (strcmp(identity->weights_sha256,
                    identity->declared_weights_sha256) == 0)
                ? NIYAH_TRUE : NIYAH_FALSE;
    }

    return NIYAH_OK;
}

/* ==========================================================================
 * Claims
 * ========================================================================== */

NiyahTruth niyah_identity_is(const NiyahIdentity* identity, const char* claim)
{
    if (!identity || !claim || !claim[0]) {
        return NIYAH_UNKNOWN;
    }

    /*
     * The engine name is a compile-time fact, so it can be answered without
     * any manifest. Note this says nothing about the weights.
     */
    if (contains_ci(claim, identity->engine_name)) {
        return NIYAH_TRUE;
    }

    /* Everything below is a claim about the weights. */

    if (identity->provenance_known != NIYAH_TRUE) {
        return NIYAH_UNKNOWN;   /* no manifest: refuse to guess */
    }

    /*
     * A manifest whose hash does not match the loaded weights is not evidence
     * of anything. Do not return FALSE either - that would itself be a claim
     * the engine cannot support.
     */
    if (identity->weights_match_manifest == NIYAH_FALSE) {
        return NIYAH_UNKNOWN;
    }

    if (contains_ci(identity->model_name, claim)) {
        return NIYAH_TRUE;
    }
    return NIYAH_FALSE;
}

bool niyah_identity_is_self_query(const char* prompt)
{
    if (!prompt) {
        return false;
    }

    static const char* kEnglish[] = {
        "who are you",
        "what are you",
        "what model are you",
        "which model are you",
        "what llm",
        "are you gpt",
        "are you llama",
        "are you qwen",
        "are you claude",
        "are you chatgpt",
        "introduce yourself",
        "tell me about yourself",
    };

    /*
     * Arabic needles as hex escapes so the file stays pure ASCII and no
     * toolchain has to be told about the source encoding.
     *
     *   \xD9\x86\xD9\x85\xD9\x88\xD8\xB0\xD8\xAC          = "model"
     *   \xD9\x85\xD9\x86 \xD8\xA3\xD9\x86\xD8\xAA          = "who are you"
     *   \xD9\x85\xD9\x8A\xD9\x86 \xD8\xA7\xD9\x86\xD8\xAA  = "who are you" (dialect)
     *   \xD9\x88\xD8\xB4 \xD8\xA7\xD9\x86\xD8\xAA          = "what are you" (dialect)
     */
    static const char* kArabic[] = {
        "\xD9\x86\xD9\x85\xD9\x88\xD8\xB0\xD8\xAC",
        "\xD9\x85\xD9\x86 \xD8\xA3\xD9\x86\xD8\xAA",
        "\xD9\x85\xD9\x8A\xD9\x86 \xD8\xA7\xD9\x86\xD8\xAA",
        "\xD9\x88\xD8\xB4 \xD8\xA7\xD9\x86\xD8\xAA",
    };

    for (size_t i = 0; i < sizeof(kEnglish) / sizeof(kEnglish[0]); ++i) {
        if (contains_ci(prompt, kEnglish[i])) {
            return true;
        }
    }
    for (size_t i = 0; i < sizeof(kArabic) / sizeof(kArabic[0]); ++i) {
        if (strstr(prompt, kArabic[i]) != NULL) {
            return true;
        }
    }
    return false;
}

/* ==========================================================================
 * Reports
 * ========================================================================== */

static const char* truth_word(NiyahTruth t)
{
    return niyah_truth_to_string(t);
}

NiyahStatus niyah_identity_report(const NiyahIdentity* identity,
                                  char* buffer,
                                  size_t buffer_size,
                                  size_t* out_size)
{
    if (!identity || !buffer || buffer_size == 0) {
        return NIYAH_ERR_INVALID_ARG;
    }

    size_t used = 0;
    buffer[0] = '\0';

    append(buffer, buffer_size, &used,
           "%s engine %s (built %s)\n\n",
           identity->engine_name, identity->engine_version,
           identity->engine_build_date);

    append(buffer, buffer_size, &used,
           "This report is measured from the running process. It is not\n"
           "generated by the model and does not pass through the sampler.\n\n");

    /* --- weights --- */
    append(buffer, buffer_size, &used, "WEIGHTS\n");
    if (!identity->weights_loaded) {
        append(buffer, buffer_size, &used,
               "  loaded            no\n"
               "  This engine has no model loaded and cannot answer as one.\n\n");
    } else {
        append(buffer, buffer_size, &used,
               "  loaded            yes\n"
               "  size              %.1f MB\n"
               "  parameters        %llu\n",
               (double)identity->weights_bytes / 1.0e6,
               (unsigned long long)identity->parameter_count);

        if (identity->weights_hashed) {
            append(buffer, buffer_size, &used,
                   "  sha256            %s\n", identity->weights_sha256);
        } else {
            append(buffer, buffer_size, &used,
                   "  sha256            not computed\n");
        }
        append(buffer, buffer_size, &used,
               "  matches manifest  %s\n\n",
               truth_word(identity->weights_match_manifest));
    }

    /* --- shape --- */
    append(buffer, buffer_size, &used,
           "SHAPE\n"
           "  vocab             %d\n"
           "  embedding dim     %d\n"
           "  heads             %d\n"
           "  kv heads          %d\n"
           "  layers            %d\n"
           "  ffn hidden        %d\n"
           "  context           %d\n"
           "  rope theta        %.1f\n"
           "  norm eps          %g\n"
           "  tied embeddings   %s\n\n",
           identity->config.n_vocab,
           identity->config.n_embd,
           identity->config.n_head,
           identity->config.n_kv_head,
           identity->config.n_layer,
           identity->config.n_ff,
           identity->config.n_ctx,
           (double)identity->config.rope_theta,
           (double)identity->config.norm_eps,
           identity->config.tie_word_embeddings ? "yes" : "no");

    /* --- provenance --- */
    append(buffer, buffer_size, &used, "PROVENANCE\n");
    if (identity->provenance_known != NIYAH_TRUE) {
        append(buffer, buffer_size, &used,
               "  unknown. No manifest was loaded, so the origin of these\n"
               "  weights cannot be established. This engine will not guess.\n\n");
    } else {
        append(buffer, buffer_size, &used, "  model             %s\n",
               identity->model_name);
        if (identity->origin[0]) {
            append(buffer, buffer_size, &used, "  origin            %s\n",
                   identity->origin);
        }
        if (identity->license[0]) {
            append(buffer, buffer_size, &used, "  license           %s\n",
                   identity->license);
        }
        if (identity->corpus_manifest_sha256[0]) {
            append(buffer, buffer_size, &used, "  corpus manifest   %s\n",
                   identity->corpus_manifest_sha256);
        }
        if (identity->weights_match_manifest == NIYAH_FALSE) {
            append(buffer, buffer_size, &used,
                   "\n  WARNING: the loaded weights do not match the hash this\n"
                   "  manifest declares. The weights were replaced after the\n"
                   "  manifest was written. Treat every field above as void.\n");
        }
        append(buffer, buffer_size, &used, "\n");
    }

    append(buffer, buffer_size, &used,
           "The engine is %s. The weights are a separate artefact and are not.\n",
           identity->engine_name);

    if (out_size) {
        *out_size = used;
    }
    return (used < buffer_size) ? NIYAH_OK : NIYAH_ERR_OVERFLOW;
}

NiyahStatus niyah_identity_report_json(const NiyahIdentity* identity,
                                       char* buffer,
                                       size_t buffer_size,
                                       size_t* out_size)
{
    if (!identity || !buffer || buffer_size == 0) {
        return NIYAH_ERR_INVALID_ARG;
    }

    size_t used = 0;
    buffer[0] = '\0';

    append(buffer, buffer_size, &used,
           "{\n"
           "  \"engine_name\": \"%s\",\n"
           "  \"engine_version\": \"%s\",\n"
           "  \"engine_build_date\": \"%s\",\n"
           "  \"weights_loaded\": %s,\n"
           "  \"weights_bytes\": %llu,\n"
           "  \"parameter_count\": %llu,\n"
           "  \"weights_sha256\": \"%s\",\n"
           "  \"n_vocab\": %d,\n"
           "  \"n_embd\": %d,\n"
           "  \"n_head\": %d,\n"
           "  \"n_kv_head\": %d,\n"
           "  \"n_layer\": %d,\n"
           "  \"n_ff\": %d,\n"
           "  \"n_ctx\": %d,\n"
           "  \"rope_theta\": %.1f,\n"
           "  \"norm_eps\": %g,\n"
           "  \"tie_word_embeddings\": %s,\n"
           "  \"provenance_known\": \"%s\",\n"
           "  \"model_name\": \"%s\",\n"
           "  \"origin\": \"%s\",\n"
           "  \"license\": \"%s\",\n"
           "  \"corpus_manifest_sha256\": \"%s\",\n"
           "  \"weights_match_manifest\": \"%s\"\n"
           "}\n",
           identity->engine_name,
           identity->engine_version,
           identity->engine_build_date,
           identity->weights_loaded ? "true" : "false",
           (unsigned long long)identity->weights_bytes,
           (unsigned long long)identity->parameter_count,
           identity->weights_hashed ? identity->weights_sha256 : "",
           identity->config.n_vocab,
           identity->config.n_embd,
           identity->config.n_head,
           identity->config.n_kv_head,
           identity->config.n_layer,
           identity->config.n_ff,
           identity->config.n_ctx,
           (double)identity->config.rope_theta,
           (double)identity->config.norm_eps,
           identity->config.tie_word_embeddings ? "true" : "false",
           truth_word(identity->provenance_known),
           identity->model_name,
           identity->origin,
           identity->license,
           identity->corpus_manifest_sha256,
           truth_word(identity->weights_match_manifest));

    if (out_size) {
        *out_size = used;
    }
    return (used < buffer_size) ? NIYAH_OK : NIYAH_ERR_OVERFLOW;
}
