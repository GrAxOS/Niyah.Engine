#include "niyah_bridge.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

int main(void) {
    NiyahBridge *bridge = NULL;
    assert(niyah_bridge_create(&bridge) == NIYAH_BRIDGE_OK);
    assert(bridge != NULL);

    assert(niyah_bridge_model_validate(
               bridge,
               32000u,
               4096u,
               512u,
               8u,
               8u,
               4u,
               1536u) == NIYAH_BRIDGE_OK);

    assert(niyah_bridge_add_document(
               bridge,
               1u,
               "https://example.invalid/one",
               "First",
               "Niyah evidence engine search") == NIYAH_BRIDGE_OK);
    assert(niyah_bridge_add_document(
               bridge,
               2u,
               "https://example.invalid/two",
               "Second",
               "Local search document") == NIYAH_BRIDGE_OK);

    char output[256];
    memset(output, 0, sizeof(output));
    assert(niyah_bridge_search(
               bridge,
               "evidence",
               8u,
               output,
               sizeof(output)) == NIYAH_BRIDGE_OK);
    assert(output[0] != '\0');

    niyah_bridge_destroy(bridge);
    return 0;
}
