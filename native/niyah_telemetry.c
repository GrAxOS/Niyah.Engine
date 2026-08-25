#include "niyah.h"
<<<<<<< HEAD
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Telemetry ─────────────────────────────────────────────────────────── */

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>

static int64_t get_ms(void) {
    return (int64_t)GetTickCount64();
}
static int64_t get_memory_bytes(void) {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (int64_t)pmc.WorkingSetSize;
    return 0;
}
#  pragma comment(lib, "psapi.lib")
#  include <psapi.h>

#else
#  include <sys/time.h>
#  include <sys/resource.h>

static int64_t get_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000 + (int64_t)tv.tv_usec / 1000;
}
static int64_t get_memory_bytes(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#  ifdef __APPLE__
    return ru.ru_maxrss;
#  else
    return ru.ru_maxrss * 1024;
#  endif
}
#endif

void niyah_telemetry_start(NiyahTelemetry* t) {
    if (!t) return;
    t->start_time      = get_ms();
    t->end_time        = 0;
    t->memory_used     = get_memory_bytes();
    t->tokens_processed = 0;
}

void niyah_telemetry_end(NiyahTelemetry* t) {
    if (!t) return;
    t->end_time    = get_ms();
    t->memory_used = get_memory_bytes() - t->memory_used;
=======

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
>>>>>>> origin/main
}
