#include "niyah_source.h"
#include <string.h>

bool niyah_source_is_usable(const NiyahSource *source)
{
    if (!source || !source->uri || !source->uri[0]) {
        return false;
    }
    if (source->kind == NIYAH_SOURCE_UNKNOWN) {
        return false;
    }
    if (!source->verified) {
        return false;
    }
    // ضد الحشو: يرفض الهاش المزيف
    if (source->content_hash) {
        int all_zero = 1;
        for (int i = 0; i < 32; i++) {
            if (source->content_hash[i] != 0) { all_zero = 0; break; }
        }
        if (all_zero) return false;
    }
    return true;
}

int niyah_source_donkey_type(const NiyahSource *source) {
    if (!source || !source->uri || source->kind == NIYAH_SOURCE_UNKNOWN) return 1; // GPT
    if (!source->verified) return 2; // Grok
    if (!source->content_hash) return 3; // Gemini
    return 0; // human
}
