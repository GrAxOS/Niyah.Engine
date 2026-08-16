#include "niyah_source.h"

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

    return true;
}
