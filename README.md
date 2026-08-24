# Niyah.Engine

A C11 inference and retrieval engine built around an explicit epistemic model:
three-valued (Kleene) truth, evidence envelopes with provenance, and a
constraint solver, wired to a BM25 retrieval layer and a local model runtime.

The guiding rule of this codebase is that the engine must never fabricate.
When weights are missing, `niyah_llm_generate` returns
`NIYAH_ERR_NO_WEIGHTS` and a `NULL` text pointer rather than a plausible
string, and that behaviour is pinned by a test.

## Repository layout

| Path | Language | Contents |
| --- | --- | --- |
| `native/` | C11 | The engine: kernels, transformer, tokenizer, sampler, model loader, evidence subsystem, CSP solver, and the C ABI bridge |
| `search/` | C + C++ | BM25 inverted index (`niyah_index.c`), URL handling, HTTP fetch, HTML extraction, search engine front end |
| `storage/` | C + SQL | `local_store.c` plus Postgres schema and migrations |
| `neutral/` | Python | Corpus cleaning, manifest validation, training and inference scripts |
| `tools/` | Python + shell | `gguf_to_niyah.py` weight converter, build scripts, and the local check runner |
| `rag/` | JSON | `official_sources.json`, the allowed-source list for retrieval |
| `knowledge/` | JSON | `domains.json`, domain taxonomy |
| `docs/` | Markdown | `CHARACTER.md` and design notes |
| `ui/Niyah.App/` | C# | Desktop front end, calls `native/` through P/Invoke |

There is no top-level `tests/` directory. The native test suites live next to
the code they cover in `native/`, and the retrieval tests live in `search/`.

## Building

### CMake (recommended)

```sh
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native
ctest --test-dir build/native --output-on-failure
```

This produces a static `niyah` library and a shared `niyah` library. The
shared one is what `ui/Niyah.App` loads.

### Plain make

```sh
cd native
make lib
make test
```

### Retrieval layer

```sh
cmake -S search -B build/search -DCMAKE_BUILD_TYPE=Release
cmake --build build/search
ctest --test-dir build/search --output-on-failure
```

When compiling any `native/` translation unit by hand, define
`NIYAH_BRIDGE_EXPORTS`. Without it the evidence and CSP headers resolve their
export macro to `__declspec(dllimport)` on Windows and those files will not
compile.

## Checks

There is no hosted CI. All checks run locally through one script:

```sh
sh tools/ci.sh            # native, plain make, search, python tooling
sh tools/ci.sh search     # single stage: native | make | search | python
```

Windows:

```bat
tools\ci.cmd
tools\ci.cmd native
```

Stages, in order: CMake build plus `ctest` for `native/`, `make lib` and
`make test` in `native/` (POSIX only), CMake build plus `ctest` for `search/`,
and `compileall` over `tools/` and `neutral/`. `BUILD_TYPE` overrides the
default `Release`. The test suites `#undef NDEBUG`, so assertions run in
Release too.

## Status

Implemented and covered by assertions:

| Area | Notes |
| --- | --- |
| Kleene truth logic | Full NOT / AND / OR / IMPLIES tables |
| GEMM, matvec | Blocked, cache-tiled; `_bt` variant for on-disk layout |
| Softmax | Max-subtracted for numerical stability, plus temperature and log variants |
| RMSNorm, LayerNorm | In-place and out-of-place |
| SwiGLU, SiLU, GELU | Overflow-safe sigmoid |
| RoPE | NeoX / GGUF half-split convention |
| Attention | Causal MHA, GQA decode against a KV cache |
| Transformer block | Pre-norm residual, weighted and weightless paths |
| Sampler | Greedy, temperature, top-k, top-p on xoshiro256\*\*; repetition penalty |
| Tokenizer | Greedy longest-match with byte fallback |
| Runtime | 64-byte aligned arena allocator |
| Model loader | Reads the flat float32 blob from `tools/gguf_to_niyah.py`, detects tied embeddings |
| Inference | Prefill plus incremental decode with EOS handling |
| Evidence | Envelopes, DAG, aggregation, reasoner verdicts |
| CSP solver | Backtracking search over six relational operators |
| Bridge | Real document store behind the C ABI |
| BM25 retrieval | Inverted index with checkpoint/rollback on insert failure; real per-document term frequencies |

Not implemented:

| Area | Notes |
| --- | --- |
| GPU execution | `NiyahRuntimeConfig.use_gpu` exists but is forced to `false`; there is no device backend |
| Quantised weights | The loader reads float32 only. GGUF Q4/Q8 tensors must be dequantised by the converter |
| Batched inference | `batch` fields are present in the state structs but only batch size 1 is exercised |
| Multi-head attention state | `niyah_multihead_attention_forward` operates on a pre-split q/k/v state and has no projection weights of its own |

## Preparing weights

`tools/gguf_to_niyah.py` converts a GGUF checkpoint into a config JSON plus a
flat float32 blob. The blob order is fixed and the loader asserts against it:

1. `embedding` — `vocab_size * dim`
2. per layer — `attn_norm`, `wq`, `wk`, `wv`, `wo`, `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_down`
3. `final_norm` — `dim`
4. `lm_head` — `vocab_size * dim`, omitted when embeddings are tied

All projection matrices are row-major `[out_features][in_features]`.

## Usage

```c
#include "niyah.h"

int main(void) {
    NiyahModelConfig config;
    if (niyah_model_load_config_json(&config, "model/config.json") != NIYAH_OK) {
        return 1;
    }

    NiyahLLM llm;
    memset(&llm, 0, sizeof(llm));

    if (niyah_model_load(&llm.model, &config, "model/weights.bin") != NIYAH_OK) {
        return 1;
    }
    if (niyah_tokenizer_load(&llm.tokenizer, "model/vocab.txt") != NIYAH_OK) {
        return 1;
    }

    NiyahLLMOutput out = niyah_llm_generate(&llm, "Explain BM25 briefly.", 128);

    /* Always check status. With no weights loaded this is
     * NIYAH_ERR_NO_WEIGHTS and out.text is NULL -- the engine does not
     * invent a completion. */
    if (out.status == NIYAH_OK && out.text != NULL) {
        printf("%s\n", out.text);
        printf("%.1f tok/s\n", niyah_telemetry_tokens_per_second(&out.telemetry));
    } else {
        fprintf(stderr, "generate failed: %s\n",
                niyah_status_to_string(out.status));
    }

    niyah_llm_output_free(&out);
    niyah_tokenizer_free(&llm.tokenizer);
    niyah_model_free(&llm.model);
    return 0;
}
```

## Retrieval semantics

`search/niyah_index.c` keeps one posting per (term, document) pair and stores
the number of occurrences of the term in that document, counted from the token
buffer, so the BM25 saturation term `tf * (k1 + 1) / (tf + norm)` is live.
Document length is the number of tokens actually indexed, capped at
`NIYAH_DOCUMENT_TOKEN_LIMIT` (1024); terms are truncated to `NIYAH_TERM_MAX`
(64 bytes including the NUL). Scores from before the term-frequency fix are not
comparable with current ones.

## Known issues

- Term lookup, posting scans, and document lookup are linear scans
  (`find_term`, `document_position`), so indexing is quadratic in vocabulary
  size and search is quadratic in corpus size. A hash map over terms and
  document ids is the next retrieval change.
- Document text is borrowed, not copied: `NiyahDocument.text` must outlive the
  index. Nothing enforces this.
- `search/niyah_index.h` and `native/niyah_document.h` both define
  `NiyahDocument` for different concepts and cannot be included in the same
  translation unit; the header raises an `#error` if both are pulled in.

## License

See repository metadata.
