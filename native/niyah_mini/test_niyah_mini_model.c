#include "niyah_mini_model.h"
#include "../niyah.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    fprintf(stderr, "Testing NiyahMini model...\n");

    /* Test model initialization */
    {
        NiyahMiniModel model;
        memset(&model, 0, sizeof(model));

        NiyahMiniConfig config;
        niyah_mini_config_init(&config, NIYAH_MINI_TINY);

        NiyahStatus status = niyah_mini_model_init(&model, &config);
        if (status != NIYAH_OK) {
            fprintf(stderr, "ERROR: Failed to initialize model\n");
            return 1;
        }

        /* Check weights were allocated */
        if (!model.weights.embedding) {
            fprintf(stderr, "ERROR: Embedding weights not allocated\n");
            niyah_mini_model_free(&model);
            return 1;
        }

        /* Check config was copied */
        if (model.config.n_layers != config.n_layers) {
            fprintf(stderr, "ERROR: Config not copied correctly\n");
            niyah_mini_model_free(&model);
            return 1;
        }

        niyah_mini_model_free(&model);
    }

    /* Test forward state */
    {
        NiyahMiniConfig config;
        niyah_mini_config_init(&config, NIYAH_MINI_TINY);

        NiyahMiniForwardState state;
        memset(&state, 0, sizeof(state));

        NiyahStatus status = niyah_mini_forward_state_init(&state, &config, 128);
        if (status != NIYAH_OK) {
            fprintf(stderr, "ERROR: Failed to initialize forward state\n");
            return 1;
        }

        if (!state.hidden) {
            fprintf(stderr, "ERROR: Hidden state not allocated\n");
            niyah_mini_forward_state_free(&state);
            return 1;
        }

        niyah_mini_forward_state_free(&state);
    }

    /* Test weight memory calculation */
    {
        NiyahMiniConfig config;
        niyah_mini_config_init(&config, NIYAH_MINI_TINY);

        size_t size = niyah_mini_weights_memory_size(&config);
        if (size == 0) {
            fprintf(stderr, "ERROR: Weight memory size should not be zero\n");
            return 1;
        }

        fprintf(stderr, "  Tiny model weights: %zu bytes\n", size);
    }

    /* Test forward state memory calculation */
    {
        NiyahMiniConfig config;
        niyah_mini_config_init(&config, NIYAH_MINI_TINY);

        size_t size = niyah_mini_forward_state_memory_size(&config, 128);
        if (size == 0) {
            fprintf(stderr, "ERROR: Forward state memory size should not be zero\n");
            return 1;
        }

        fprintf(stderr, "  Forward state (seq=128): %zu bytes\n", size);
    }

    /* Test parameter count */
    {
        NiyahMiniConfig config;
        niyah_mini_config_init(&config, NIYAH_MINI_TINY);

        uint64_t n_params = niyah_mini_config_n_params(&config);
        if (n_params == 0) {
            fprintf(stderr, "ERROR: Parameter count should not be zero\n");
            return 1;
        }

        fprintf(stderr, "  Tiny model parameters: %lu\n", n_params);
    }

    fprintf(stderr, "\nAll model tests passed!\n");
    return 0;
}
