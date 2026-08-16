#ifndef NIYAH_TELEMETRY_H
#define NIYAH_TELEMETRY_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bool enabled;
    bool include_content;
    const char *path;
} NiyahTelemetryConfig;

typedef struct {
    uint64_t started_unix_ms;
    uint64_t events;
    uint64_t errors;
    uint64_t bytes_in;
    uint64_t bytes_out;
} NiyahTelemetryStats;

typedef struct {
    NiyahTelemetryConfig config;
    NiyahTelemetryStats stats;
    uint64_t started_monotonic_ns;
} NiyahTelemetry;

bool niyah_telemetry_init(NiyahTelemetry *telemetry,
                          const NiyahTelemetryConfig *config,
                          uint64_t now_unix_ms,
                          uint64_t now_monotonic_ns);
void niyah_telemetry_close(NiyahTelemetry *telemetry);

bool niyah_telemetry_event(NiyahTelemetry *telemetry,
                           const char *name,
                           uint64_t duration_ms,
                           uint64_t bytes_in,
                           uint64_t bytes_out,
                           bool error);

const NiyahTelemetryStats *niyah_telemetry_stats(const NiyahTelemetry *telemetry);

#ifdef __cplusplus
}
#endif

#endif
