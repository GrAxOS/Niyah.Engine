#include "niyah_proof.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <fcntl.h>
#  include <io.h>
#  include <process.h>
#  define PATH_SEP '\\'
#else
#  include <libgen.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <unistd.h>
#  define PATH_SEP '/'
#endif

#define NIYAH_PATH_MAX 2048
#define NIYAH_URL_MAX 2048
#define NIYAH_MAX_ARTIFACTS 16
#define NIYAH_MAX_TOKENS_LIMIT 32768ul
#define NIYAH_RUN_PROOF_CONTRACT_V1 "NIYAH-RUN-PROOF-V1"

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

static int path_join(char* out, size_t cap, const char* a, const char* b)
{
    const size_t alen = strlen(a);
    const int needs_sep = alen > 0 && !is_sep(a[alen - 1u]);
    const int n = needs_sep
        ? snprintf(out, cap, "%s%c%s", a, PATH_SEP, b)
        : snprintf(out, cap, "%s%s", a, b);
    return n >= 0 && (size_t)n < cap ? 0 : -1;
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

static int valid_component(const char* value)
{
    if (!value || !value[0] || strcmp(value, ".") == 0 || strcmp(value, "..") == 0) {
        return 0;
    }
    for (const unsigned char* p = (const unsigned char*)value; *p; ++p) {
        if (!(isalnum(*p) || *p == '-' || *p == '_' || *p == '.')) {
            return 0;
        }
    }
    return 1;
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
        if (tolower((unsigned char)a[i]) != tolower((unsigned char)b[i])) {
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

static int parse_manifest(const char* path, PackageManifest* manifest)
{
    FILE* f;
    char line[NIYAH_URL_MAX + 256];
    int saw_header = 0;

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
            if (copy_field(manifest->name, sizeof(manifest->name), trim(p + 5)) != 0) {
                fclose(f);
                return -1;
            }
            continue;
        }
        if (strncmp(p, "version ", 8) == 0) {
            if (copy_field(manifest->version, sizeof(manifest->version), trim(p + 8)) != 0) {
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
            PackageArtifact* a = &manifest->artifacts[manifest->artifact_count];
            char extra[2];
            const int fields = sscanf(p + 9, "%31s %64s %2047s %1s", a->role, a->sha256, a->url, extra);
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
    return saw_header && valid_component(manifest->name) && valid_component(manifest->version) && manifest->artifact_count > 0
        ? 0 : -1;
}

static int read_current_version(const char* path, char* out, size_t cap)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        return -1;
    }
    if (!fgets(out, (int)cap, f)) {
        fclose(f);
        return -1;
    }
    fclose(f);
    char* p = trim(out);
    if (p != out) {
        memmove(out, p, strlen(p) + 1u);
    }
    return valid_component(out) ? 0 : -1;
}

static const PackageArtifact* find_role(const PackageManifest* manifest, const char* role)
{
    for (size_t i = 0; i < manifest->artifact_count; ++i) {
        if (strcmp(manifest->artifacts[i].role, role) == 0) {
            return &manifest->artifacts[i];
        }
    }
    return NULL;
}

static int run_process(char* const argv[])
{
#ifdef _WIN32
    const intptr_t rc = _spawnvp(_P_WAIT, argv[0], (const char* const*)argv);
    if (rc < 0) {
        return 127;
    }
    return (int)rc;
#else
    pid_t pid = fork();
    if (pid < 0) {
        return 127;
    }
    if (pid == 0) {
        execvp(argv[0], argv);
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return 127;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return 128;
#endif
}

static int write_stdout_all(const uint8_t* data, size_t size)
{
    size_t offset = 0u;
    while (offset < size) {
#ifdef _WIN32
        const unsigned int left = (unsigned int)(size - offset);
        const int wrote = _write(_fileno(stdout), data + offset, left);
        if (wrote <= 0) {
            return -1;
        }
#else
        const ssize_t wrote = write(STDOUT_FILENO, data + offset, size - offset);
        if (wrote < 0 && errno == EINTR) {
            continue;
        }
        if (wrote <= 0) {
            return -1;
        }
#endif
        offset += (size_t)wrote;
    }
    return 0;
}

static int run_process_hash_stdout(char* const argv[],
                                   uint8_t output_hash[NIYAH_SHA256_BYTES],
                                   int* capture_ok)
{
    if (!argv || !argv[0] || !output_hash || !capture_ok) {
        return 127;
    }
    *capture_ok = 0;

#ifdef _WIN32
    int pipefd[2];
    if (_pipe(pipefd, 8192u, _O_BINARY) != 0) {
        return 127;
    }

    fflush(stdout);
    const int stdout_fd = _fileno(stdout);
    const int saved_stdout = _dup(stdout_fd);
    if (saved_stdout < 0 || _dup2(pipefd[1], stdout_fd) != 0) {
        if (saved_stdout >= 0) _close(saved_stdout);
        _close(pipefd[0]);
        _close(pipefd[1]);
        return 127;
    }
    _close(pipefd[1]);

    const intptr_t process = _spawnvp(_P_NOWAIT, argv[0], (const char* const*)argv);
    const int spawn_error = errno;
    const int restore_ok = _dup2(saved_stdout, stdout_fd) == 0;
    _close(saved_stdout);

    if (process < 0) {
        _close(pipefd[0]);
        errno = spawn_error;
        return 127;
    }

    NiyahSha256 hash;
    niyah_sha256_init(&hash);
    int stream_ok = restore_ok;
    uint8_t buffer[8192];
    for (;;) {
        const int got = _read(pipefd[0], buffer, (unsigned int)sizeof(buffer));
        if (got == 0) {
            break;
        }
        if (got < 0) {
            stream_ok = 0;
            break;
        }
        niyah_sha256_update(&hash, buffer, (size_t)got);
        if (write_stdout_all(buffer, (size_t)got) != 0) {
            stream_ok = 0;
        }
    }
    _close(pipefd[0]);

    int status = 0;
    if (_cwait(&status, process, _WAIT_CHILD) < 0) {
        return 127;
    }
    niyah_sha256_final(&hash, output_hash);
    *capture_ok = stream_ok;
    return status;
#else
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        return 127;
    }

    fflush(stdout);
    const pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return 127;
    }
    if (pid == 0) {
        close(pipefd[0]);
        if (dup2(pipefd[1], STDOUT_FILENO) < 0) {
            _exit(127);
        }
        close(pipefd[1]);
        execvp(argv[0], argv);
        _exit(127);
    }

    close(pipefd[1]);
    NiyahSha256 hash;
    niyah_sha256_init(&hash);
    int stream_ok = 1;
    uint8_t buffer[8192];
    for (;;) {
        const ssize_t got = read(pipefd[0], buffer, sizeof(buffer));
        if (got == 0) {
            break;
        }
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            stream_ok = 0;
            break;
        }
        niyah_sha256_update(&hash, buffer, (size_t)got);
        if (write_stdout_all(buffer, (size_t)got) != 0) {
            stream_ok = 0;
        }
    }
    close(pipefd[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            return 127;
        }
    }
    niyah_sha256_final(&hash, output_hash);
    *capture_ok = stream_ok;
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return 128;
#endif
}

static int parse_max_tokens(const char* text, unsigned long* value)
{
    if (!text || !text[0] || !value) {
        return -1;
    }
    for (const unsigned char* p = (const unsigned char*)text; *p; ++p) {
        if (!isdigit(*p)) {
            return -1;
        }
    }
    errno = 0;
    char* end = NULL;
    const unsigned long parsed = strtoul(text, &end, 10);
    if (errno == ERANGE || !end || *end != '\0' || parsed == 0ul || parsed > NIYAH_MAX_TOKENS_LIMIT) {
        return -1;
    }
    *value = parsed;
    return 0;
}

static int proof_path_state(const char* path)
{
    errno = 0;
    FILE* f = fopen(path, "rb");
    if (f) {
        fclose(f);
        return 1;
    }
    return errno == ENOENT ? 0 : -1;
}

static void lowercase_hash(const char* hash, char out[NIYAH_SHA256_HEX_BYTES])
{
    for (size_t i = 0u; i < 64u; ++i) {
        out[i] = (char)tolower((unsigned char)hash[i]);
    }
    out[64] = '\0';
}

static int build_execution_contract(char* out,
                                    size_t cap,
                                    const PackageManifest* manifest,
                                    const PackageArtifact* base,
                                    const PackageArtifact* adapter,
                                    unsigned long max_tokens)
{
    char base_hash[NIYAH_SHA256_HEX_BYTES];
    char adapter_hash[NIYAH_SHA256_HEX_BYTES];
    lowercase_hash(base->sha256, base_hash);
    lowercase_hash(adapter->sha256, adapter_hash);

    const int n = snprintf(
        out,
        cap,
        "runtime_contract_version=%s\n"
        "proof_version=%s\n"
        "package_name=%s\n"
        "package_version=%s\n"
        "base_sha256=%s\n"
        "adapter_sha256=%s\n"
        "max_tokens=%lu\n"
        "runtime=llama.cpp\n"
        "local_only=true\n"
        "single_turn=true\n",
        NIYAH_RUN_PROOF_CONTRACT_V1,
        NIYAH_PROOF_V1_HEADER,
        manifest->name,
        manifest->version,
        base_hash,
        adapter_hash,
        max_tokens);
    return n >= 0 && (size_t)n < cap ? n : -1;
}

static int save_proof_atomic_no_replace(const char* path, const NiyahProofV1* proof)
{
    char receipt[384];
    size_t receipt_size = 0u;
    if (niyah_proof_v1_serialize(proof, receipt, sizeof(receipt), &receipt_size) != NIYAH_OK) {
        return -1;
    }

    char temp_path[NIYAH_PATH_MAX];
#ifdef _WIN32
    const int pid = _getpid();
#else
    const long pid = (long)getpid();
#endif
    const int n = snprintf(temp_path, sizeof(temp_path), "%s.tmp.%ld", path, (long)pid);
    if (n < 0 || (size_t)n >= sizeof(temp_path)) {
        return -1;
    }

    FILE* f = fopen(temp_path, "wbx");
    if (!f) {
        return -1;
    }

    int ok = fwrite(receipt, 1u, receipt_size, f) == receipt_size;
    if (ok) {
        ok = fflush(f) == 0;
    }
    if (fclose(f) != 0) {
        ok = 0;
    }
    if (!ok) {
        remove(temp_path);
        return -1;
    }

#ifdef _WIN32
    if (!MoveFileA(temp_path, path)) {
        remove(temp_path);
        return -1;
    }
#else
    if (link(temp_path, path) != 0) {
        unlink(temp_path);
        return -1;
    }
    (void)unlink(temp_path);
#endif
    return 0;
}

static int run_package(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: niyah run <package> --prompt TEXT [--max-tokens N] [--proof PATH]\n");
        return 2;
    }

    const char* package = argv[2];
    if (!valid_component(package)) {
        fprintf(stderr, "niyah: invalid package name\n");
        return 2;
    }

    const char* prompt = NULL;
    const char* max_tokens = "256";
    const char* proof_path = NULL;
    int saw_prompt = 0;
    int saw_max_tokens = 0;
    int saw_proof = 0;

    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--prompt") == 0) {
            if (saw_prompt || i + 1 >= argc) {
                fprintf(stderr, "niyah: --prompt requires exactly one value\n");
                return 2;
            }
            saw_prompt = 1;
            prompt = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--max-tokens") == 0) {
            if (saw_max_tokens || i + 1 >= argc) {
                fprintf(stderr, "niyah: --max-tokens requires exactly one value\n");
                return 2;
            }
            saw_max_tokens = 1;
            max_tokens = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--proof") == 0) {
            if (saw_proof || i + 1 >= argc) {
                fprintf(stderr, "niyah: --proof requires exactly one value\n");
                return 2;
            }
            saw_proof = 1;
            proof_path = argv[++i];
            continue;
        }
        fprintf(stderr, "niyah: unknown run option: %s\n", argv[i]);
        return 2;
    }

    if (!prompt || !prompt[0]) {
        fprintf(stderr, "niyah: --prompt is required\n");
        return 2;
    }
    unsigned long max_tokens_value = 0ul;
    if (parse_max_tokens(max_tokens, &max_tokens_value) != 0) {
        fprintf(stderr, "niyah: invalid --max-tokens value (expected 1..%lu)\n", NIYAH_MAX_TOKENS_LIMIT);
        return 2;
    }
    if (proof_path && !proof_path[0]) {
        fprintf(stderr, "niyah: --proof path must not be empty\n");
        return 2;
    }
    if (proof_path) {
        const int state = proof_path_state(proof_path);
        if (state != 0) {
            fprintf(stderr, state > 0
                ? "niyah: refusing to overwrite existing proof: %s\n"
                : "niyah: cannot validate proof target: %s\n",
                proof_path);
            return 6;
        }
    }

    char home[NIYAH_PATH_MAX];
    char packages_dir[NIYAH_PATH_MAX];
    char package_root[NIYAH_PATH_MAX];
    char current_path[NIYAH_PATH_MAX];
    char version[64];
    char version_dir[NIYAH_PATH_MAX];
    char manifest_path[NIYAH_PATH_MAX];
    char blobs_dir[NIYAH_PATH_MAX];

    if (get_niyah_home(home, sizeof(home)) != 0 ||
        path_join(packages_dir, sizeof(packages_dir), home, "packages") != 0 ||
        path_join(package_root, sizeof(package_root), packages_dir, package) != 0 ||
        path_join(current_path, sizeof(current_path), package_root, "current") != 0 ||
        read_current_version(current_path, version, sizeof(version)) != 0 ||
        path_join(version_dir, sizeof(version_dir), package_root, version) != 0 ||
        path_join(manifest_path, sizeof(manifest_path), version_dir, "manifest.niyah") != 0 ||
        path_join(blobs_dir, sizeof(blobs_dir), home, "blobs") != 0) {
        fprintf(stderr, "niyah: package not installed or metadata path invalid: %s\n", package);
        return 7;
    }

    PackageManifest manifest;
    if (parse_manifest(manifest_path, &manifest) != 0 || strcmp(manifest.name, package) != 0 || strcmp(manifest.version, version) != 0) {
        fprintf(stderr, "niyah: installed package metadata is invalid\n");
        return 4;
    }

    const PackageArtifact* base = find_role(&manifest, "base");
    const PackageArtifact* adapter = find_role(&manifest, "adapter");
    if (!base || !adapter) {
        fprintf(stderr, "niyah: package must contain base and adapter artifacts\n");
        return 4;
    }

    char base_path[NIYAH_PATH_MAX];
    char adapter_path[NIYAH_PATH_MAX];
    if (path_join(base_path, sizeof(base_path), blobs_dir, base->sha256) != 0 ||
        path_join(adapter_path, sizeof(adapter_path), blobs_dir, adapter->sha256) != 0) {
        fprintf(stderr, "niyah: artifact path too long\n");
        return 2;
    }

    if (verify_file(base_path, base->sha256) != 1 || verify_file(adapter_path, adapter->sha256) != 1) {
        fprintf(stderr, "niyah: refusing to run: package artifact verification failed\n");
        return 5;
    }

    const char* llama_cli = getenv("NIYAH_LLAMA_CLI");
    if (!llama_cli || !llama_cli[0]) {
        llama_cli = "llama-cli";
    }

    fprintf(stderr, "niyah: running verified package %s@%s\n", manifest.name, manifest.version);

    char* child_argv[] = {
        (char*)llama_cli,
        (char*)"-m", base_path,
        (char*)"--lora", adapter_path,
        (char*)"-p", (char*)prompt,
        (char*)"-n", (char*)max_tokens,
        (char*)"--single-turn",
        NULL
    };

    if (!proof_path) {
        const int rc = run_process(child_argv);
        if (rc == 127) {
            fprintf(stderr, "niyah: llama-cli not found; set NIYAH_LLAMA_CLI to the executable path\n");
        }
        return rc;
    }

    char contract[768];
    const int contract_size = build_execution_contract(
        contract, sizeof(contract), &manifest, base, adapter, max_tokens_value);
    if (contract_size < 0) {
        fprintf(stderr, "niyah: execution contract is too large\n");
        return 6;
    }

    uint8_t prompt_hash[NIYAH_SHA256_BYTES];
    uint8_t output_hash[NIYAH_SHA256_BYTES];
    uint8_t contract_hash[NIYAH_SHA256_BYTES];
    niyah_sha256_buffer(prompt, strlen(prompt), prompt_hash);
    niyah_sha256_buffer(contract, (size_t)contract_size, contract_hash);

    int capture_ok = 0;
    const int rc = run_process_hash_stdout(child_argv, output_hash, &capture_ok);
    if (rc == 127) {
        fprintf(stderr, "niyah: llama-cli not found; set NIYAH_LLAMA_CLI to the executable path\n");
    }
    if (rc != 0) {
        return rc;
    }
    if (!capture_ok) {
        fprintf(stderr, "niyah: output capture/hash stream failed; proof not written\n");
        return 6;
    }

    NiyahProofV1 proof;
    if (niyah_proof_v1_generate_hashes(prompt_hash, output_hash, contract_hash, &proof) != NIYAH_OK) {
        fprintf(stderr, "niyah: proof generation failed\n");
        return 6;
    }
    if (save_proof_atomic_no_replace(proof_path, &proof) != 0) {
        fprintf(stderr, "niyah: proof serialization/write failed; no success recorded\n");
        return 6;
    }

    return 0;
}

static int sibling_core_cli(char* out, size_t cap, const char* argv0)
{
#ifdef _WIN32
    const char* last_slash = strrchr(argv0, '\\');
    const char* last_fslash = strrchr(argv0, '/');
    const char* last = last_slash;
    if (!last || (last_fslash && last_fslash > last)) {
        last = last_fslash;
    }
    if (!last) {
        return snprintf(out, cap, "niyah-core-cli.exe") < (int)cap ? 0 : -1;
    }
    const size_t dir_len = (size_t)(last - argv0);
    return snprintf(out, cap, "%.*s\\niyah-core-cli.exe", (int)dir_len, argv0) < (int)cap ? 0 : -1;
#else
    const char* last = strrchr(argv0, '/');
    if (!last) {
        return snprintf(out, cap, "niyah-core-cli") < (int)cap ? 0 : -1;
    }
    const size_t dir_len = (size_t)(last - argv0);
    return snprintf(out, cap, "%.*s/niyah-core-cli", (int)dir_len, argv0) < (int)cap ? 0 : -1;
#endif
}

static int forward_core_cli(int argc, char** argv)
{
    char core_path[NIYAH_PATH_MAX];
    if (sibling_core_cli(core_path, sizeof(core_path), argv[0]) != 0) {
        fprintf(stderr, "niyah: failed to resolve core CLI path\n");
        return 127;
    }

    char** child_argv = (char**)calloc((size_t)argc + 1u, sizeof(char*));
    if (!child_argv) {
        return 127;
    }
    child_argv[0] = core_path;
    for (int i = 1; i < argc; ++i) {
        child_argv[i] = argv[i];
    }
    child_argv[argc] = NULL;
    const int rc = run_process(child_argv);
    free(child_argv);
    return rc;
}

int main(int argc, char** argv)
{
    if (argc >= 2 && strcmp(argv[1], "run") == 0) {
        return run_package(argc, argv);
    }
    return forward_core_cli(argc, argv);
}
