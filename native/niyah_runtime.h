#ifndef NIYAH_RUNTIME_H
#define NIYAH_RUNTIME_H

#include "niyah.h"

NiyahRuntime* niyah_runtime_create(const NiyahRuntimeConfig* config);
void niyah_runtime_destroy(NiyahRuntime* runtime);

#endif // NIYAH_RUNTIME_H
