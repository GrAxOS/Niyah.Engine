#include "niyah_telemetry.h"

#include <stdio.h>
#include <string.h>
#include <inttypes.h>

static const char *json_escape_name(const char *name) {
    return name ? name : "unknown";
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
    fputs("{\"type\":\"telemetry_started\",\"version\":1}", fp);
    fputc('\n', fp);
    fclose(fp);
    return true;
}

void niyah_telemetry_close(NiyahTelemetry *telemetry) {
    if (!telemetry) return;
    if (telemetry->config.enabled && telemetry->config.path) {
        FILE *fp = fopen(telemetry->config.path, "a");
        if (fp) {
            fputs("{\"type\":\"telemetry_stopped\"}", fp);
            fputc('\n', fp);
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

    /* Deliberately record metadata only. No URL, prompt, page body, source text,
       headers, cookies, tokens, or credentials are emitted by this API. */
    int written = fprintf(fp,
        "{\"type\":\"event\",\"name\":\"%s\",\"duration_ms\":%" PRIu64
        ",\"bytes_in\":%" PRIu64 ",\"bytes_out\":%" PRIu64 ",\"error\":%s}\n",
        json_escape_name(name), duration_ms, bytes_in, bytes_out,
        error ? "true" : "false");
    fclose(fp);
    return written > 0;
}

const NiyahTelemetryStats *niyah_telemetry_stats(const NiyahTelemetry *telemetry) {
    return telemetry ? &telemetry->stats : NULL;
}
