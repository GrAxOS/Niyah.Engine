#include "niyah_telemetry.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static bool write_json_string(FILE *fp, const char *value) {
    if (!fp || !value) return false;
    if (fputc('"', fp) == EOF) return false;
    for (const unsigned char *p = (const unsigned char *)value; *p; ++p) {
        switch (*p) {
        case '"': if (fputs("\\\"", fp) == EOF) return false; break;
        case '\\': if (fputs("\\\\", fp) == EOF) return false; break;
        case '\b': if (fputs("\\b", fp) == EOF) return false; break;
        case '\f': if (fputs("\\f", fp) == EOF) return false; break;
        case '\n': if (fputs("\\n", fp) == EOF) return false; break;
        case '\r': if (fputs("\\r", fp) == EOF) return false; break;
        case '\t': if (fputs("\\t", fp) == EOF) return false; break;
        default:
            if (*p < 0x20u) {
                if (fprintf(fp, "\\u%04x", (unsigned int)*p) < 0) return false;
            } else if (fputc((int)*p, fp) == EOF) {
                return false;
            }
            break;
        }
    }
    return fputc('"', fp) != EOF;
}

bool niyah_telemetry_init(NiyahTelemetry *telemetry,
                          const NiyahTelemetryConfig *config,
                          uint64_t now_unix_ms,
                          uint64_t now_monotonic_ns) {
    if (!telemetry || !config) return false;
    memset(telemetry, 0, sizeof(*telemetry));
    telemetry->config = *config;
    telemetry->started_monotonic_ns = now_monotonic_ns;
    telemetry->stats.started_unix_ms = now_unix_ms;

    if (!config->enabled) return true;
    if (!config->path || config->path[0] == '\0') return false;

    FILE *fp = fopen(config->path, "a");
    if (!fp) return false;
    const bool ok = fputs("{\"type\":\"telemetry_started\",\"version\":1}\n", fp) >= 0;
    if (ok) (void)fflush(fp);
    fclose(fp);
    return ok;
}

void niyah_telemetry_close(NiyahTelemetry *telemetry) {
    if (!telemetry) return;
    if (telemetry->config.enabled && telemetry->config.path) {
        FILE *fp = fopen(telemetry->config.path, "a");
        if (fp) {
            (void)fputs("{\"type\":\"telemetry_stopped\"}\n", fp);
            (void)fflush(fp);
            fclose(fp);
        }
    }
    memset(telemetry, 0, sizeof(*telemetry));
}

bool niyah_telemetry_event(NiyahTelemetry *telemetry,
                           const char *name,
                           uint64_t duration_ms,
                           uint64_t bytes_in,
                           uint64_t bytes_out,
                           bool error) {
    if (!telemetry || !name || name[0] == '\0') return false;

    ++telemetry->stats.events;
    if (error) ++telemetry->stats.errors;
    telemetry->stats.bytes_in += bytes_in;
    telemetry->stats.bytes_out += bytes_out;

    if (!telemetry->config.enabled || !telemetry->config.path) return true;

    FILE *fp = fopen(telemetry->config.path, "a");
    if (!fp) return false;

    bool ok = fputs("{\"type\":\"event\",\"name\":", fp) >= 0;
    ok = ok && write_json_string(fp, name);
    ok = ok && fprintf(fp,
        ",\"duration_ms\":%" PRIu64
        ",\"bytes_in\":%" PRIu64
        ",\"bytes_out\":%" PRIu64
        ",\"error\":%s}\n",
        duration_ms, bytes_in, bytes_out,
        error ? "true" : "false") >= 0;
    if (ok) (void)fflush(fp);
    fclose(fp);
    return ok;
}

const NiyahTelemetryStats *niyah_telemetry_stats(const NiyahTelemetry *telemetry) {
    return telemetry ? &telemetry->stats : NULL;
}
