#!/usr/bin/env bash
set -euo pipefail

NIYAH=$1
FAKE=$2
HOME_DIR=$(mktemp -d '/tmp/niyah run proof.XXXXXX')
trap 'rm -rf "$HOME_DIR"' EXIT
export NIYAH_HOME="$HOME_DIR"
export NIYAH_LLAMA_CLI="$FAKE"

hash_file() { sha256sum "$1" | awk '{print $1}'; }
field() { awk -v key="$2" '$1 == key { print $2 }' "$1"; }

install_package() {
    local name=$1 version=$2 base=$3 adapter=$4
    local base_tmp="$HOME_DIR/base.tmp" adapter_tmp="$HOME_DIR/adapter.tmp"
    printf '%s' "$base" > "$base_tmp"
    printf '%s' "$adapter" > "$adapter_tmp"
    local base_hash adapter_hash
    base_hash=$(hash_file "$base_tmp")
    adapter_hash=$(hash_file "$adapter_tmp")
    mkdir -p "$HOME_DIR/blobs" "$HOME_DIR/packages/$name/$version"
    cp "$base_tmp" "$HOME_DIR/blobs/$base_hash"
    cp "$adapter_tmp" "$HOME_DIR/blobs/$adapter_hash"
    printf '%s\n' "$version" > "$HOME_DIR/packages/$name/current"
    cat > "$HOME_DIR/packages/$name/$version/manifest.niyah" <<MANIFEST
NIYAH-PACKAGE 1
name $name
version $version
artifact base $base_hash https://example.invalid/base
artifact adapter $adapter_hash https://example.invalid/adapter
MANIFEST
}

run_capture() {
    local output=$1 rc=$2 capture=$3
    shift 3
    export NIYAH_FAKE_OUTPUT="$output"
    if [[ -n "$rc" ]]; then export NIYAH_FAKE_EXIT="$rc"; else unset NIYAH_FAKE_EXIT || true; fi
    set +e
    "$NIYAH" "$@" > "$capture"
    RUN_RC=$?
    set -e
}

assert_rc() { [[ $RUN_RC -eq $1 ]] || { echo "expected rc $1, got $RUN_RC" >&2; exit 1; }; }
assert_ne_zero() { [[ $RUN_RC -ne 0 ]] || { echo "expected nonzero rc" >&2; exit 1; }; }

PACKAGE=oracle-engine
PROMPT=PROMPT_SENTINEL_7341
OUTPUT_A=OUTPUT_SENTINEL_A_7341
OUTPUT_B=OUTPUT_SENTINEL_B_7341
CAPTURE="$HOME_DIR/captured output.txt"
install_package "$PACKAGE" 1.0.0 base-a adapter-a

PROOF_A="$HOME_DIR/run proof a.proof"
run_capture "$OUTPUT_A" '' "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --max-tokens 80 --proof "$PROOF_A"
assert_rc 0
printf '%s' "$OUTPUT_A" > "$HOME_DIR/expected.txt"
cmp "$HOME_DIR/expected.txt" "$CAPTURE"
[[ -f "$PROOF_A" ]]
! grep -Fq "$PROMPT" "$PROOF_A"
! grep -Fq "$OUTPUT_A" "$PROOF_A"
[[ $(field "$PROOF_A" 'output_hash:') == $(hash_file "$CAPTURE") ]]
RULES_A=$(field "$PROOF_A" 'rules_hash:')
OUT_A=$(field "$PROOF_A" 'output_hash:')

PROOF_SAME="$HOME_DIR/run proof same.proof"
run_capture "$OUTPUT_A" '' "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --max-tokens 80 --proof "$PROOF_SAME"
assert_rc 0
cmp "$PROOF_A" "$PROOF_SAME"

PROOF_B="$HOME_DIR/run proof b.proof"
run_capture "$OUTPUT_B" '' "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --max-tokens 80 --proof "$PROOF_B"
assert_rc 0
[[ $(field "$PROOF_B" 'output_hash:') != "$OUT_A" ]]
[[ $(field "$PROOF_B" 'rules_hash:') == "$RULES_A" ]]

PROOF_MAX="$HOME_DIR/run proof max.proof"
run_capture "$OUTPUT_A" '' "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --max-tokens 81 --proof "$PROOF_MAX"
assert_rc 0
[[ $(field "$PROOF_MAX" 'rules_hash:') != "$RULES_A" ]]

install_package "$PACKAGE" 1.0.0 base-b adapter-a
PROOF_BASE="$HOME_DIR/run proof base.proof"
run_capture "$OUTPUT_A" '' "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --max-tokens 80 --proof "$PROOF_BASE"
assert_rc 0
RULES_BASE=$(field "$PROOF_BASE" 'rules_hash:')
[[ "$RULES_BASE" != "$RULES_A" ]]

install_package "$PACKAGE" 1.0.0 base-b adapter-b
PROOF_ADAPTER="$HOME_DIR/run proof adapter.proof"
run_capture "$OUTPUT_A" '' "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --max-tokens 80 --proof "$PROOF_ADAPTER"
assert_rc 0
RULES_ADAPTER=$(field "$PROOF_ADAPTER" 'rules_hash:')
[[ "$RULES_ADAPTER" != "$RULES_BASE" ]]

install_package oracle-engine-alt 1.0.0 base-b adapter-b
PROOF_PACKAGE="$HOME_DIR/run proof package.proof"
run_capture "$OUTPUT_A" '' "$CAPTURE" run oracle-engine-alt --prompt "$PROMPT" --max-tokens 80 --proof "$PROOF_PACKAGE"
assert_rc 0
[[ $(field "$PROOF_PACKAGE" 'rules_hash:') != "$RULES_ADAPTER" ]]

install_package "$PACKAGE" 1.0.1 base-b adapter-b
PROOF_VERSION="$HOME_DIR/run proof version.proof"
run_capture "$OUTPUT_A" '' "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --max-tokens 80 --proof "$PROOF_VERSION"
assert_rc 0
[[ $(field "$PROOF_VERSION" 'rules_hash:') != "$RULES_ADAPTER" ]]

FAILED="$HOME_DIR/run proof failed.proof"
run_capture PARTIAL_OUTPUT 9 "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --max-tokens 80 --proof "$FAILED"
assert_rc 9
[[ $(cat "$CAPTURE") == PARTIAL_OUTPUT ]]
[[ ! -e "$FAILED" ]]

cp "$PROOF_A" "$HOME_DIR/before.proof"
run_capture SHOULD_NOT_RUN '' "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --max-tokens 80 --proof "$PROOF_A"
assert_ne_zero
[[ ! -s "$CAPTURE" ]]
cmp "$HOME_DIR/before.proof" "$PROOF_A"

EMPTY="$HOME_DIR/run proof empty.proof"
run_capture '' '' "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --max-tokens 80 --proof "$EMPTY"
assert_rc 0
[[ ! -s "$CAPTURE" ]]
printf '' > "$HOME_DIR/empty.txt"
[[ $(field "$EMPTY" 'output_hash:') == $(hash_file "$HOME_DIR/empty.txt") ]]

UNWRITABLE="$HOME_DIR/missing dir/proof.proof"
run_capture "$OUTPUT_A" '' "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --max-tokens 80 --proof "$UNWRITABLE"
assert_ne_zero
cmp "$HOME_DIR/expected.txt" "$CAPTURE"
[[ ! -e "$UNWRITABLE" ]]

for bad in 0 32769 abc; do
    run_capture SHOULD_NOT_RUN '' "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --max-tokens "$bad"
    assert_rc 2
    [[ ! -s "$CAPTURE" ]]
done

run_capture SHOULD_NOT_RUN '' "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --max-tokens
assert_rc 2
run_capture SHOULD_NOT_RUN '' "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --proof
assert_rc 2
run_capture SHOULD_NOT_RUN '' "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --max-tokens 80 --max-tokens 81
assert_rc 2
run_capture SHOULD_NOT_RUN '' "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --proof a.proof --proof b.proof
assert_rc 2
run_capture SHOULD_NOT_RUN '' "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --prompt other
assert_rc 2
run_capture SHOULD_NOT_RUN '' "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --proof ''
assert_rc 2

run_capture LEGACY_RUN_OUTPUT '' "$CAPTURE" run "$PACKAGE" --prompt "$PROMPT" --max-tokens 80
assert_rc 0
[[ $(cat "$CAPTURE") == LEGACY_RUN_OUTPUT ]]

echo 'niyah_run_proof: ok'
