#include "niyah_mini_config.h"
#include "../niyah.h"

#include <stdio.h>
#include <string.h>

const char* status_to_string(NiyahStatus status) {
    switch (status) {
        case NIYAH_OK: return "ok";
        case NIYAH_ERR_INVALID_ARG: return "invalid argument";
        case NIYAH_ERR_OUT_OF_MEMORY: return "out of memory";
        case NIYAH_ERR_IO: return "i/o error";
        case NIYAH_ERR_UNSUPPORTED: return "unsupported";
        case NIYAH_ERR_NO_WEIGHTS: return "no weights loaded";
        case NIYAH_ERR_SHAPE: return "shape mismatch";
        case NIYAH_ERR_NOT_FOUND: return "not found";
        case NIYAH_ERR_OVERFLOW: return "arithmetic overflow";
        default: return "unknown status";
    }
}

int main(void) {
    fprintf(stderr, "Testing NiyahMini configuration...\n");

    /* Test preset configurations */
    for (int variant = 0; variant <= NIYAH_MINI_MEDIUM; variant++) {
        NiyahMiniConfig config;
        niyah_mini_config_init(&config, variant);

        fprintf(stderr, "\nVariant %d:\n", variant);
        fprintf(stderr, "  Layers: %d\n", config.n_layers);
        fprintf(stderr, "  Dim: %d\n", config.n_dim);
        fprintf(stderr, "  Heads: %d\n", config.n_heads);
        fprintf(stderr, "  KV Heads: %d\n", config.n_kv_heads);
        fprintf(stderr, "  FF Dim: %d\n", config.n_ff);
        fprintf(stderr, "  Vocab: %d\n", config.n_vocab);
        fprintf(stderr, "  Context: %d\n", config.n_ctx);
        fprintf(stderr, "  Params: %lu\n", niyah_mini_config_n_params(&config));

        /* Validate */
        NiyahStatus status = niyah_mini_config_validate(&config);
        if (status != NIYAH_OK) {
            fprintf(stderr, "  ERROR: Validation failed: %s\n", status_to_string(status));
            return 1;
        }
    }

    /* Test default initialization */
    {
        NiyahMiniConfig config;
        niyah_mini_config_init(&config, -1);  /* Should default to BASE */
        if (config.variant != NIYAH_MINI_BASE) {
            fprintf(stderr, "ERROR: Default variant should be BASE\n");
            return 1;
        }
    }

    /* Test invalid configurations */
    {
        NiyahMiniConfig config;
        memset(&config, 0, sizeof(config));
        config.n_layers = -1;  /* Invalid */
        NiyahStatus status = niyah_mini_config_validate(&config);
        if (status == NIYAH_OK) {
            fprintf(stderr, "ERROR: Should reject negative layers\n");
            return 1;
        }
    }

    {
        NiyahMiniConfig config;
        memset(&config, 0, sizeof(config));
        config.n_layers = 12;
        config.n_dim = 512;
        config.n_heads = 7;  /* Not a divisor of dim */
        NiyahStatus status = niyah_mini_config_validate(&config);
        if (status == NIYAH_OK) {
            fprintf(stderr, "ERROR: Should reject n_heads not dividing n_dim\n");
            return 1;
        }
    }

    fprintf(stderr, "\nAll configuration tests passed!\n");
    return 0;
}
