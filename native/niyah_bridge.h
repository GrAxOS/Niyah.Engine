#ifndef NIYAH_BRIDGE_H
#define NIYAH_BRIDGE_H

#include "niyah.h"

NiyahBridgeContext* niyah_bridge_create(NiyahLLM* llm);
void niyah_bridge_destroy(NiyahBridgeContext* ctx);

#endif // NIYAH_BRIDGE_H
