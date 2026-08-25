#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <string.h>

#include "niyah.h"

int main(void)
{
    /* The clock must be monotonic and actually advance. */
    const int64_t t0 = niyah_telemetry_now_ns();
    assert(t0 > 0);

    int64_t t1 = t0;
    volatile double spin = 0.0;
    for (int i = 0; i < 2000000 && t1 <= t0; ++i) {
        spin += (double)i * 0.5;
        t1 = niyah_telemetry_now_ns();
    }
    (void)spin;
    assert(t1 >= t0);

    NiyahTelemetry tel;
    memset(&tel, 0, sizeof(tel));

    niyah_telemetry_start(&tel);
    assert(tel.start_time > 0);

    volatile double work = 0.0;
    for (int i = 0; i < 500000; ++i) {
        work += (double)i * 0.25;
    }
    (void)work;

    niyah_telemetry_end(&tel);
    assert(tel.end_time >= tel.start_time);

    const double ms = niyah_telemetry_elapsed_ms(&tel);
    assert(ms >= 0.0);
    assert(isfinite(ms));

    /* No tokens recorded yet, so throughput must be 0 rather than a divide
     * by zero or a fabricated number. */
    assert(niyah_telemetry_tokens_per_second(&tel) == 0.0);

    tel.tokens_processed = 100;
    const double tps = niyah_telemetry_tokens_per_second(&tel);
    assert(tps >= 0.0);
    assert(isfinite(tps));

    /* A zero-duration window must not produce inf or NaN. */
    NiyahTelemetry instant;
    memset(&instant, 0, sizeof(instant));
    instant.start_time = 1000;
    instant.end_time = 1000;
    instant.tokens_processed = 50;
    const double zero_tps = niyah_telemetry_tokens_per_second(&instant);
    assert(isfinite(zero_tps));
    assert(niyah_telemetry_elapsed_ms(&instant) == 0.0);

    /* Degenerate inputs. */
    niyah_telemetry_start(NULL);
    niyah_telemetry_end(NULL);
    assert(niyah_telemetry_elapsed_ms(NULL) == 0.0);
    assert(niyah_telemetry_tokens_per_second(NULL) == 0.0);

    return 0;
}
