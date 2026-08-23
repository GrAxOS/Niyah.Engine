#include "niyah.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    printf("Niyah version: %s\n", niyah_version());
    printf("Truth test: %s\n", niyah_truth_to_string(NIYAH_TRUE));
    printf("All tests passed.\n");
    return 0;
}
