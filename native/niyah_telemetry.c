#include "niyah.h"
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
}
