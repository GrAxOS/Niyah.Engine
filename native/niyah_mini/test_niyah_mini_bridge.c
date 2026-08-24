#include "niyah_mini_bridge.h"
#include "../niyah.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    fprintf(stderr, "Testing NiyahMini bridge...\n");

    /* Test wrapped model initialization */
    {
        NiyahMiniWrappedModel wrapped;
        memset(&wrapped, 0, sizeof(wrapped));

        NiyahMiniConfig config;
        niyah_mini_config_init(&config, NIYAH_MINI_TINY);

        NiyahStatus status = niyah_mini_wrapped_init(&wrapped, &config);
        if (status != NIYAH_OK) {
            fprintf(stderr, "ERROR: Failed to initialize wrapped model\n");
            return 1;
        }

        if (wrapped.weights_loaded) {
            fprintf(stderr, "ERROR: weights_loaded should be false initially\n");
            niyah_mini_wrapped_free(&wrapped);
            return 1;
        }

        niyah_mini_wrapped_free(&wrapped);
    }

    /* Test evidence envelope */
    {
        NiyahMiniEvidenceEnvelope envelope;
        memset(&envelope, 0, sizeof(envelope));

        NiyahStatus status = niyah_mini_evidence_envelope_init(&envelope);
        if (status != NIYAH_OK) {
            fprintf(stderr, "ERROR: Failed to initialize evidence envelope\n");
            return 1;
        }

        if (envelope.label != NIYAH_MINI_EVIDENCE_UNKNOWN) {
            fprintf(stderr, "ERROR: Default label should be UNKNOWN\n");
            niyah_mini_evidence_envelope_free(&envelope);
            return 1;
        }

        /* Test formatting */
        const char* text = "This is a FACT statement.";
        status = niyah_mini_format_evidence_output(&envelope, text, NULL, 0);
        if (status != NIYAH_OK) {
            fprintf(stderr, "ERROR: Failed to format evidence output\n");
            niyah_mini_evidence_envelope_free(&envelope);
            return 1;
        }

        if (envelope.label != NIYAH_MINI_EVIDENCE_FACT) {
            fprintf(stderr, "ERROR: Should detect FACT label\n");
            niyah_mini_evidence_envelope_free(&envelope);
            return 1;
        }

        if (!envelope.answer || strcmp(envelope.answer, text) != 0) {
            fprintf(stderr, "ERROR: Answer not set correctly\n");
            niyah_mini_evidence_envelope_free(&envelope);
            return 1;
        }

        niyah_mini_evidence_envelope_free(&envelope);
    }

    /* Test generation with no weights */
    {
        NiyahMiniWrappedModel wrapped;
        memset(&wrapped, 0, sizeof(wrapped));

        NiyahMiniConfig config;
        niyah_mini_config_init(&config, NIYAH_MINI_TINY);

        NiyahStatus status = niyah_mini_wrapped_init(&wrapped, &config);
        if (status != NIYAH_OK) {
            fprintf(stderr, "ERROR: Failed to initialize wrapped model\n");
            return 1;
        }

        /* Don't load weights - should return NIYAH_ERR_NO_WEIGHTS */
        NiyahLLMOutput output = niyah_mini_wrapped_generate(
            &wrapped, NULL, "Test prompt", 10);

        if (output.status != NIYAH_ERR_NO_WEIGHTS) {
            fprintf(stderr, "ERROR: Should return NIYAH_ERR_NO_WEIGHTS\n");
            niyah_mini_wrapped_free(&wrapped);
            return 1;
        }

        if (output.text != NULL) {
            fprintf(stderr, "ERROR: text should be NULL when no weights\n");
            niyah_mini_wrapped_free(&wrapped);
            return 1;
        }

        niyah_mini_wrapped_free(&wrapped);
    }

    /* Test conversion to NiyahModel */
    {
        NiyahMiniWrappedModel wrapped;
        memset(&wrapped, 0, sizeof(wrapped));

        NiyahMiniConfig config;
        niyah_mini_config_init(&config, NIYAH_MINI_TINY);

        NiyahStatus status = niyah_mini_wrapped_init(&wrapped, &config);
        if (status != NIYAH_OK) {
            fprintf(stderr, "ERROR: Failed to initialize wrapped model\n");
            return 1;
        }

        NiyahModel niyah_model;
        memset(&niyah_model, 0, sizeof(niyah_model));

        status = niyah_mini_to_niyah_model(&niyah_model, &wrapped);
        if (status != NIYAH_OK) {
            fprintf(stderr, "ERROR: Failed to convert to NiyahModel\n");
            niyah_mini_wrapped_free(&wrapped);
            return 1;
        }

        if (niyah_model.config.n_vocab != config.n_vocab) {
            fprintf(stderr, "ERROR: Config not copied correctly\n");
            niyah_mini_wrapped_free(&wrapped);
            return 1;
        }

        niyah_mini_wrapped_free(&wrapped);
    }

    fprintf(stderr, "\nAll bridge tests passed!\n");
    return 0;
}
