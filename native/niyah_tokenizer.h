#ifndef NIYAH_TOKENIZER_H
#define NIYAH_TOKENIZER_H

#include "niyah.h"

int32_t niyah_tokenize(NiyahTokenizer* tokenizer, const char* text, int32_t* tokens, int32_t max_tokens);
char* niyah_detokenize(NiyahTokenizer* tokenizer, const int32_t* tokens, int32_t n_tokens);

#endif // NIYAH_TOKENIZER_H
