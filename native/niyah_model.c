#include "niyah.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Minimal model file loader (.niyah binary format) ───────────────────── */
/*
 * File header layout (little-endian):
 *   [4] magic = "NIYH"
 *   [4] version (uint32)
 *   [4] n_vocab (int32)
 *   [4] n_embd  (int32)
 *   [4] n_head  (int32)
 *   [4] n_layer (int32)
 *   [4] n_ctx   (int32)
 *   [4] type    (int32)
 *   [rest] weight bytes
 */
#define NIYAH_MODEL_MAGIC "NIYH"

int niyah_model_load(NiyahModel* model, const char* path) {
    if (!model || !path) return -1;

    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    char magic[5] = {0};
    fread(magic, 1, 4, f);
    if (strncmp(magic, NIYAH_MODEL_MAGIC, 4) != 0) { fclose(f); return -2; }

    uint32_t version = 0;
    fread(&version, 4, 1, f);
    (void)version;

    int32_t fields[7];
    if (fread(fields, sizeof(int32_t), 7, f) != 7) { fclose(f); return -3; }

    model->config.n_vocab = fields[0];
    model->config.n_embd  = fields[1];
    model->config.n_head  = fields[2];
    model->config.n_layer = fields[3];
    model->config.n_ctx   = fields[4];
    model->config.type    = fields[5];

    /* Read remaining bytes as weights */
    long start = ftell(f);
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    fseek(f, start, SEEK_SET);

    model->weights_size = (size_t)(end - start);
    if (model->weights_size > 0) {
        model->weights = malloc(model->weights_size);
        if (!model->weights) { fclose(f); return -4; }
        fread(model->weights, 1, model->weights_size, f);
    } else {
        model->weights = NULL;
    }

    fclose(f);
    return 0;
}

void niyah_model_free(NiyahModel* model) {
    if (!model) return;
    free(model->weights);
    model->weights      = NULL;
    model->weights_size = 0;
}
