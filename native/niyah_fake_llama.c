#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int readable(const char* path)
{
    FILE* f = path ? fopen(path, "rb") : NULL;
    if (!f) return 0;
    fclose(f);
    return 1;
}

int main(int argc, char** argv)
{
    const char* model = NULL;
    const char* adapter = NULL;
    int prompt = 0, tokens = 0, single_turn = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model = argv[++i];
        else if (strcmp(argv[i], "--lora") == 0 && i + 1 < argc) adapter = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) { ++i; prompt = 1; }
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) { ++i; tokens = 1; }
        else if (strcmp(argv[i], "--single-turn") == 0) single_turn = 1;
    }

    if (!readable(model) || !readable(adapter) || !prompt || !tokens || !single_turn) return 90;

    const char* output = getenv("NIYAH_FAKE_OUTPUT");
    if (output && fwrite(output, 1u, strlen(output), stdout) != strlen(output)) return 91;
    if (fflush(stdout) != 0) return 91;

    const char* rc = getenv("NIYAH_FAKE_EXIT");
    return rc && rc[0] ? atoi(rc) : 0;
}
