#ifndef NIYAH_TRUTH_H
#define NIYAH_TRUTH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NIYAH_TRUTH_FACT = 0,
    NIYAH_TRUTH_INFERENCE = 1,
    NIYAH_TRUTH_UNCERTAIN = 2,
    NIYAH_TRUTH_UNKNOWN = 3,
    NIYAH_TRUTH_CONFLICTED = 4
} NiyahTruthClass;

typedef struct {
    NiyahTruthClass classification;
    uint32_t source_count;
    uint32_t conflict_count;
} NiyahEvidenceStatus;

const char *niyah_truth_class_name(NiyahTruthClass classification);

#ifdef __cplusplus
}
#endif

#endif
