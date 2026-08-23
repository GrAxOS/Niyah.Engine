#ifndef NIYAH_ATTENTION_H
#define NIYAH_ATTENTION_H

#include "niyah.h"

void niyah_attention_forward(NiyahAttentionState* state, const float* x, float* out);
void niyah_multihead_attention_forward(NiyahMultiHeadAttentionState* state);

#endif // NIYAH_ATTENTION_H
