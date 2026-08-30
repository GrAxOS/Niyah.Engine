#include "niyah.h"
#include "niyah_sha256.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <direct.h>
#  include <process.h>
#  define PATH_SEP '\\'
#else
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <unistd.h>
#  define PATH_SEP '/'
#endif

#define NIYAH_PATH_MAX 2048
#define NIYAH_URL_MAX  2048
#define NIYAH_MAX_ARTIFACTS 16

typedef struct {
    char role[32];
    char sha256[NIYAH_SHA256_HEX_BYTES];
    char url[NIYAH_URL_MAX];
} PackageArtifact;

typedef struct {
    char name[64];
    char version[64];
    PackageArtifact artifacts[NIYAH_MAX_ARTIFACTS];
    size_t artifact_count;
} PackageManifest;

static int is_sep(char c)
{
    return c == '/' || c == '\\';
}

static int mkdir_one(const char* path)
{
#ifdef _WIN32
    if (_mkdir(path) == 0 || errno == EEXIST) {
        return 0;
    }
#else
    if (mkdir(path, 0755) == 0 || errno == EEXIST) {
        return 0;
    }
#endif
    return -1;
}

static int mkdir_p(const char* path)
{
    char tmp[NIYAH_PATH_MAX];
    size_t len;

    if (!path) {
        return -1;
    }

    len = strlen(path);
    if (len == 0 || len >= sizeof(tmp)) {
        return -1;
    }

    memcpy(tmp, path, len + 1u);

    while (len > 1u && is_sep(tmp[len - 1u])) {
        tmp[--len] = '\0';
    }

    for (char* p = tmp + 1; *p; ++p) {
        if (!is_sep(*p)) {
            continue;
        }

        const char saved = *p;
        *p = '\0';

#ifdef _WIN32
        if (!(strlen(tmp) == 2u && tmp[1] == ':'))
#endif
        {
            if (tmp[0] && mkdir_one(tmp) != 0) {
                *p = saved;
                return -1;
            }
        }

        *p = saved;
    }

    return mkdir_one(tmp);
}

static int path_join(char* out, size_t cap, const char* a, const char* b)
{
    const size_t alen = strlen(a);
    const int needs_sep = alen > 0 && !is_sep(a[alen - 1u]);

    int n;
    if (needs_sep) {
        n = snprintf(out, cap, "%s%c%s", a, PATH_SEP, b);
    } else {
        n = snprintf(out, cap, "%s%s", a, b);
    }

    return (n >= 0 && (size_t)n < cap) ? 0 : -1;
}

static int get_niyah_home(char* out, size_t cap)
{
    const char* explicit_home = getenv("NIYAH_HOME");
    if (explicit_home && explicit_home[0]) {
        return snprintf(out, cap, "%s", explicit_home) < (int)cap ? 0 : -1;
    }

#ifdef _WIN32
    const char* root = getenv("LOCALAPPDATA");
    if (!root || !root[0]) {
        root = getenv("USERPROFILE");
    }
    if (!root || !root[0]) {
        return -1;
    }
    return snprintf(out, cap, "%s\\Niyah", root) < (int)cap ? 0 : -1;
#else
    const char* root = getenv("HOME");
    if (!root || !root[0]) {
        return -1;
    }
    return snprintf(out, cap, "%s/.niyah", root) < (int)cap ? 0 : -1;
#endif
}

static int valid_package_name(const char* name)
{
    if (!name || !name[0]) {
        return 0;
    }

    for (const unsigned char* p = (const unsigned char*)name; *p; ++p) {
        if (!(isalnum(*p) || *p == '-' || *p == '_' || *p == '.')) {
            return 0;
        }
    }

    return 1;
}

static int file_exists(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        return 0;
    }
    fclose(f);
    return 1;
}

static int hash_text_valid(const char* hash)
{
    if (!hash || strlen(hash) != 64u) {
        return 0;
    }

    for (size_t i = 0; i < 64u; ++i) {
        if (!isxdigit((unsigned char)hash[i])) {
            return 0;
        }
    }

    return 1;
}

static int hash_equal_ci(const char* a, const char* b)
{
    for (size_t i = 0; i < 64u; ++i) {
        if (tolower((unsigned char)a[i]) !=
            tolower((unsigned char)b[i])) {
            return 0;
        }
    }
    return a[64] == '\0' && b[64] == '\0';
}

static int verify_file(const char* path, const char* expected)
{
    uint8_t digest[NIYAH_SHA256_BYTES];
    char actual[NIYAH_SHA256_HEX_BYTES];

    if (!hash_text_valid(expected)) {
        return -1;
    }

    if (niyah_sha256_file(path, digest) != NIYAH_OK) {
        return -1;
    }

    niyah_sha256_to_hex(digest, actual);
    return hash_equal_ci(actual, expected) ? 1 : 0;
}

static int run_curl(const char* url, const char* output, int resume)
{
#ifdef _WIN32
    const char* normal[] = {
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--output", output,
        url,
        NULL
    };

    const char* resumed[] = {
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--continue-at", "-",
        "--output", output,
        url,
        NULL
    };

    const intptr_t rc = _spawnvp(
        _P_WAIT,
        "curl",
        resume ? resumed : normal);

    return rc == 0 ? 0 : -1;
#else
    char* const normal[] = {
        (char*)"curl",
        (char*)"--fail",
        (char*)"--location",
        (char*)"--silent",
        (char*)"--show-error",
        (char*)"--output", (char*)output,
        (char*)url,
        NULL
    };

    char* const resumed[] = {
        (char*)"curl",
        (char*)"--fail",
        (char*)"--location",
        (char*)"--silent",
        (char*)"--show-error",
        (char*)"--continue-at", (char*)"-",
        (char*)"--output", (char*)output,
        (char*)url,
        NULL
    };

    pid_t pid = fork();
    if (pid < 0) {
        return -1;
    }

    if (pid == 0) {
        execvp("curl", resume ? resumed : normal);
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return -1;
    }

    return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : -1;
#endif
}

static char* trim(char* text)
{
    while (*text && isspace((unsigned char)*text)) {
        ++text;
    }

    char* end = text + strlen(text);
    while (end > text && isspace((unsigned char)end[-1])) {
        *--end = '\0';
    }

    return text;
}

static int copy_field(char* dst, size_t cap, const char* src)
{
    const size_t len = strlen(src);
    if (len >= cap) {
        return -1;
    }

    memcpy(dst, src, len + 1u);
    return 0;
}

static int parse_manifest(const char* path, PackageManifest* manifest)
{
    FILE* f;
    char line[NIYAH_URL_MAX + 256];
    int saw_header = 0;

    if (!path || !manifest) {
        return -1;
    }

    memset(manifest, 0, sizeof(*manifest));

    f = fopen(path, "rb");
    if (!f) {
        return -1;
    }

    while (fgets(line, sizeof(line), f)) {
        char* p = trim(line);

        if (!p[0] || p[0] == '#') {
            continue;
        }

        if (!saw_header) {
            if (strcmp(p, "NIYAH-PACKAGE 1") != 0) {
                fclose(f);
                return -1;
            }

            saw_header = 1;
            continue;
        }

        if (strncmp(p, "name ", 5) == 0) {
            if (copy_field(
                    manifest->name,
                    sizeof(manifest->name),
                    trim(p + 5)) != 0) {
                fclose(f);
                return -1;
            }
            continue;
        }

        if (strncmp(p, "version ", 8) == 0) {
            if (copy_field(
                    manifest->version,
                    sizeof(manifest->version),
                    trim(p + 8)) != 0) {
                fclose(f);
                return -1;
            }
            continue;
        }

        if (strncmp(p, "artifact ", 9) == 0) {
            if (manifest->artifact_count >= NIYAH_MAX_ARTIFACTS) {
                fclose(f);
                return -1;
            }

            PackageArtifact* a =
                &manifest->artifacts[manifest->artifact_count];

            char extra[2];
            const int fields = sscanf(
                p + 9,
                "%31s %64s %2047s %1s",
                a->role,
                a->sha256,
                a->url,
                extra);

            if (fields != 3 || !hash_text_valid(a->sha256)) {
                fclose(f);
                return -1;
            }

            ++manifest->artifact_count;
            continue;
        }

        fclose(f);
        return -1;
    }

    fclose(f);

    return saw_header &&
           valid_package_name(manifest->name) &&
           manifest->version[0] &&
           manifest->artifact_count > 0
        ? 0
        : -1;
}

static int build_manifest_url(
    char* out,
    size_t cap,
    const char* registry,
    const char* package)
{
    const size_t len = strlen(registry);
    const char* slash = len > 0 && registry[len - 1u] == '/' ? "" : "/";

    const int n = snprintf(
        out,
        cap,
        "%s%s%s.manifest",
        registry,
        slash,
        package);

    return n >= 0 && (size_t)n < cap ? 0 : -1;
}

static int copy_file(const char* src, const char* dst)
{
    FILE* in = fopen(src, "rb");
    if (!in) {
        return -1;
    }

    FILE* out = fopen(dst, "wb");
    if (!out) {
        fclose(in);
        return -1;
    }

    unsigned char buffer[64 * 1024];
    int ok = 1;

    for (;;) {
        const size_t n = fread(buffer, 1, sizeof(buffer), in);

        if (n > 0 && fwrite(buffer, 1, n, out) != n) {
            ok = 0;
            break;
        }

        if (n < sizeof(buffer)) {
            if (ferror(in)) {
                ok = 0;
            }
            break;
        }
    }

    if (fclose(out) != 0) {
        ok = 0;
    }
    fclose(in);

    return ok ? 0 : -1;
}

static int atomic_replace(const char* src, const char* dst)
{
#ifdef _WIN32
    return MoveFileExA(
        src,
        dst,
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)
        ? 0
        : -1;
#else
    return rename(src, dst);
#endif
}

static int write_text_atomic(const char* path, const char* text)
{
    char tmp[NIYAH_PATH_MAX];

    if (snprintf(tmp, sizeof(tmp), "%s.part", path) >= (int)sizeof(tmp)) {
        return -1;
    }

    FILE* f = fopen(tmp, "wb");
    if (!f) {
        return -1;
    }

    const size_t len = strlen(text);
    const int ok =
        fwrite(text, 1, len, f) == len &&
        fclose(f) == 0;

    if (!ok) {
        return -1;
    }

    return atomic_replace(tmp, path);
}

static int pull_package(const char* package)
{
    char home[NIYAH_PATH_MAX];
    char cache_dir[NIYAH_PATH_MAX];
    char blobs_dir[NIYAH_PATH_MAX];
    char packages_dir[NIYAH_PATH_MAX];
    char package_root[NIYAH_PATH_MAX];
    char version_dir[NIYAH_PATH_MAX];
    char cache_manifest[NIYAH_PATH_MAX];
    char manifest_url[NIYAH_URL_MAX];

    const char* registry = getenv("NIYAH_REGISTRY");
    if (!registry || !registry[0]) {
        registry =
            "https://raw.githubusercontent.com/"
            "GrAxOS/Niyah.Engine/main/manifests/packages";
    }

    if (!valid_package_name(package)) {
        fprintf(stderr, "niyah: invalid package name\n");
        return 2;
    }

    if (get_niyah_home(home, sizeof(home)) != 0 ||
        path_join(cache_dir, sizeof(cache_dir), home, "cache") != 0 ||
        path_join(blobs_dir, sizeof(blobs_dir), home, "blobs") != 0 ||
        path_join(packages_dir, sizeof(packages_dir), home, "packages") != 0) {
        fprintf(stderr, "niyah: failed to resolve home directory\n");
        return 2;
    }

    if (mkdir_p(cache_dir) != 0 ||
        mkdir_p(blobs_dir) != 0 ||
        mkdir_p(packages_dir) != 0) {
        fprintf(stderr, "niyah: failed to create local store\n");
        return 2;
    }

    char manifest_name[96];
    if (snprintf(
            manifest_name,
            sizeof(manifest_name),
            "%s.manifest",
            package) >= (int)sizeof(manifest_name) ||
        path_join(
            cache_manifest,
            sizeof(cache_manifest),
            cache_dir,
            manifest_name) != 0 ||
        build_manifest_url(
            manifest_url,
            sizeof(manifest_url),
            registry,
            package) != 0) {
        fprintf(stderr, "niyah: path too long\n");
        return 2;
    }

    printf("resolving %s\n", package);

    if (run_curl(manifest_url, cache_manifest, 0) != 0) {
        fprintf(
            stderr,
            "niyah: could not fetch manifest: %s\n",
            manifest_url);
        return 3;
    }

    PackageManifest manifest;
    if (parse_manifest(cache_manifest, &manifest) != 0) {
        fprintf(stderr, "niyah: invalid package manifest\n");
        return 4;
    }

    if (strcmp(manifest.name, package) != 0) {
        fprintf(
            stderr,
            "niyah: manifest package mismatch: expected %s, got %s\n",
            package,
            manifest.name);
        return 4;
    }

    for (size_t i = 0; i < manifest.artifact_count; ++i) {
        const PackageArtifact* a = &manifest.artifacts[i];

        char blob_path[NIYAH_PATH_MAX];
        char part_path[NIYAH_PATH_MAX];

        if (path_join(
                blob_path,
                sizeof(blob_path),
                blobs_dir,
                a->sha256) != 0 ||
            snprintf(
                part_path,
                sizeof(part_path),
                "%s.part",
                blob_path) >= (int)sizeof(part_path)) {
            fprintf(stderr, "niyah: artifact path too long\n");
            return 2;
        }

        if (file_exists(blob_path)) {
            const int verified = verify_file(blob_path, a->sha256);

            if (verified == 1) {
                printf("verified %s (cached)\n", a->role);
                continue;
            }

            fprintf(
                stderr,
                "niyah: existing blob failed verification; refusing overwrite\n");
            return 5;
        }

        printf("pulling %s\n", a->role);

        if (run_curl(a->url, part_path, 1) != 0) {
            fprintf(stderr, "niyah: download failed for %s\n", a->role);
            return 3;
        }

        const int verified = verify_file(part_path, a->sha256);
        if (verified != 1) {
            fprintf(stderr, "niyah: SHA-256 mismatch for %s\n", a->role);
            remove(part_path);
            return 5;
        }

        /*
         * Do not replace an existing blob. Another process may have completed
         * the same hash while this download was running.
         */
        if (file_exists(blob_path)) {
            if (verify_file(blob_path, a->sha256) != 1) {
                fprintf(
                    stderr,
                    "niyah: concurrent blob has wrong hash; refusing overwrite\n");
                return 5;
            }

            remove(part_path);
        } else if (rename(part_path, blob_path) != 0) {
            fprintf(stderr, "niyah: failed to install verified blob\n");
            return 6;
        }

        printf("verified %s\n", a->role);
    }

    if (path_join(
            package_root,
            sizeof(package_root),
            packages_dir,
            manifest.name) != 0 ||
        path_join(
            version_dir,
            sizeof(version_dir),
            package_root,
            manifest.version) != 0 ||
        mkdir_p(version_dir) != 0) {
        fprintf(stderr, "niyah: failed to create package directory\n");
        return 6;
    }

    char installed_manifest[NIYAH_PATH_MAX];
    char installed_manifest_part[NIYAH_PATH_MAX];
    char current_path[NIYAH_PATH_MAX];

    if (path_join(
            installed_manifest,
            sizeof(installed_manifest),
            version_dir,
            "manifest.niyah") != 0 ||
        snprintf(
            installed_manifest_part,
            sizeof(installed_manifest_part),
            "%s.part",
            installed_manifest) >= (int)sizeof(installed_manifest_part) ||
        path_join(
            current_path,
            sizeof(current_path),
            package_root,
            "current") != 0) {
        fprintf(stderr, "niyah: package metadata path too long\n");
        return 6;
    }

    if (copy_file(cache_manifest, installed_manifest_part) != 0 ||
        atomic_replace(installed_manifest_part, installed_manifest) != 0 ||
        write_text_atomic(current_path, manifest.version) != 0) {
        fprintf(stderr, "niyah: failed to register package\n");
        return 6;
    }

    printf(
        "installed %s@%s (%zu artifacts)\n",
        manifest.name,
        manifest.version,
        manifest.artifact_count);

    return 0;
}

static void print_help(void)
{
    puts("Niyah package/runtime CLI");
    puts("");
    puts("Usage:");
    puts("  niyah pull <package>");
    puts("  niyah version");
    puts("");
    puts("Environment:");
    puts("  NIYAH_HOME      local package store");
    puts("  NIYAH_REGISTRY  package manifest registry");
}

int main(int argc, char** argv)
{
    if (argc == 1) {
        print_help();
        return 0;
    }

    if (strcmp(argv[1], "--version") == 0 ||
        strcmp(argv[1], "version") == 0) {
        printf("niyah %s\n", niyah_version());
        return 0;
    }

    if (strcmp(argv[1], "pull") == 0) {
        if (argc != 3) {
            fprintf(stderr, "usage: niyah pull <package>\n");
            return 2;
        }

        return pull_package(argv[2]);
    }

    if (strcmp(argv[1], "help") == 0 ||
        strcmp(argv[1], "--help") == 0 ||
        strcmp(argv[1], "-h") == 0) {
        print_help();
        return 0;
    }

    fprintf(stderr, "niyah: unknown command: %s\n", argv[1]);
    return 2;
}
