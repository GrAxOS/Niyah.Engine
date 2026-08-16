#include "niyah_telemetry.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    const char *path = "niyah-telemetry-test.jsonl";
    remove(path);

    NiyahTelemetry telemetry;
    NiyahTelemetryConfig config = {
        true,
        false,
        path
    };

    assert(niyah_telemetry_init(&telemetry, &config, 1u, 2u));
    assert(niyah_telemetry_event(&telemetry, "search", 7u, 10u, 3u, false));
    assert(niyah_telemetry_event(&telemetry, "fetch_error", 12u, 0u, 0u, true));

    const NiyahTelemetryStats *stats = niyah_telemetry_stats(&telemetry);
    assert(stats != NULL);
    assert(stats->events == 2u);
    assert(stats->errors == 1u);
    assert(stats->bytes_in == 10u);
    assert(stats->bytes_out == 3u);

    niyah_telemetry_close(&telemetry);

    FILE *fp = fopen(path, "r");
    assert(fp != NULL);
    char line[512];
    size_t lines = 0u;
    while (fgets(line, sizeof(line), fp) != NULL) {
        ++lines;
        assert(strstr(line, "search") != NULL ||
               strstr(line, "fetch_error") != NULL ||
               strstr(line, "telemetry_") != NULL);
        assert(strstr(line, "prompt") == NULL);
        assert(strstr(line, "cookie") == NULL);
        assert(strstr(line, "token") == NULL);
    }
    fclose(fp);
    remove(path);

    assert(lines == 4u);
    return 0;
}
