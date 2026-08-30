#include "niyah.h"

#include <string.h>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <time.h>
#endif

/* Was a stub. Monotonic nanosecond clock; wall-clock time would let NTP
 * adjustments produce negative durations. */

int64_t niyah_telemetry_now_ns(void)
{
#if defined(_WIN32)
    static LARGE_INTEGER frequency;
    static int initialized = 0;
    LARGE_INTEGER counter;

    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = 1;
    }
    QueryPerformanceCounter(&counter);

    if (frequency.QuadPart == 0) {
        return 0;
    }
    return (int64_t)((counter.QuadPart * 1000000000LL) / frequency.QuadPart);
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
#endif
}

void niyah_telemetry_start(NiyahTelemetry* telemetry)
{
    if (!telemetry) {
        return;
    }
    memset(telemetry, 0, sizeof(*telemetry));
    telemetry->start_time = niyah_telemetry_now_ns();
}

void niyah_telemetry_end(NiyahTelemetry* telemetry)
{
    if (!telemetry) {
        return;
    }
    telemetry->end_time = niyah_telemetry_now_ns();
}

double niyah_telemetry_elapsed_ms(const NiyahTelemetry* telemetry)
{
    if (!telemetry || telemetry->end_time <= telemetry->start_time) {
        return 0.0;
    }
    return (double)(telemetry->end_time - telemetry->start_time) / 1.0e6;
}

double niyah_telemetry_tokens_per_second(const NiyahTelemetry* telemetry)
{
    if (!telemetry || telemetry->tokens_processed <= 0) {
        return 0.0;
    }
    const double ms = niyah_telemetry_elapsed_ms(telemetry);
    if (ms <= 0.0) {
        return 0.0;
    }
    return (double)telemetry->tokens_processed * 1000.0 / ms;
}
