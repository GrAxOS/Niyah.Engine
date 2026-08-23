#ifndef NIYAH_LLM_H
#define NIYAH_LLM_H

#include "niyah.h"

NiyahLLMOutput niyah_llm_generate(NiyahLLM* llm, const char* prompt, int32_t max_tokens);

#endif // NIYAH_LLM_H
