#!/usr/bin/env bash
# Download Qwen2.5-0.5B-Instruct GGUF and convert to Niyah native format.
# Usage: ./tools/download_and_convert.sh
#
# Override the checkpoint with GGUF_GLOB, e.g.:
#     GGUF_GLOB='*q4_0*' ./tools/download_and_convert.sh
#
# Quantisation support in tools/convert_gguf_to_niyah.py: F32, F16, Q4_0,
# Q4_1, Q4_K and Q6_K. That covers llama-quantize's q4_k_m preset, which is
# what this script pulls by default. Still undecodable: Q2_K, Q3_K, Q5_K,
# Q8_K and the Q5_0/Q5_1/Q8_0/Q8_1 family. Requantise those first:
#
#     llama-quantize model.gguf model-q4_0.gguf Q4_0
#
# Speed: the Q4_K and Q6_K decoders are per-element Python loops. On a
# fixture that is invisible; on a full 0.5B checkpoint it is slow. --progress
# is passed below so you can see per-tensor advance rather than a silent
# wait. Conversion is a one-time cost per checkpoint.

set -euo pipefail

MODEL_DIR="./models"
OUTPUT_DIR="./weights"
GGUF_GLOB="${GGUF_GLOB:-*q4_k_m*.gguf}"
F32_FILE="${OUTPUT_DIR}/qwen2.5-0.5b-f32.bin"
CONFIG_FILE="${OUTPUT_DIR}/config.json"
REPO="Qwen/Qwen2.5-0.5B-Instruct-GGUF"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

die() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
    exit 1
}

find_gguf() {
    find "${MODEL_DIR}" -type f -name '*.gguf' 2>/dev/null | head -n 1
}

log "Creating directories..."
mkdir -p "${MODEL_DIR}" "${OUTPUT_DIR}"

# The converter itself is stdlib-only; huggingface-cli is what needs deps.
if ! command -v huggingface-cli >/dev/null 2>&1; then
    log "Installing Python dependencies..."
    pip3 install -r tools/requirements.txt || warn "Failed to install dependencies"
fi

GGUF_FILE="$(find_gguf)"
if [ -z "${GGUF_FILE}" ]; then
    log "Downloading ${GGUF_GLOB} from ${REPO}..."
    # --include rather than an exact filename: some GGUF repos publish
    # shard-suffixed names like ...-00001-of-00001.gguf.
    huggingface-cli download "${REPO}" --include "${GGUF_GLOB}" \
        --local-dir "${MODEL_DIR}" \
        || die "Download failed. Browse available files at https://huggingface.co/${REPO}/tree/main"
    GGUF_FILE="$(find_gguf)"
    [ -n "${GGUF_FILE}" ] || die "Download reported success but no .gguf landed in ${MODEL_DIR}"
else
    log "GGUF file already present: ${GGUF_FILE}"
fi

# Reject only what the converter still cannot decode. Q4_K and Q6_K are
# supported now, so they must NOT appear here.
lower_name="$(printf '%s' "${GGUF_FILE}" | tr '[:upper:]' '[:lower:]')"
for unsupported in q2_k q3_k q5_k q8_k q5_0 q5_1 q8_0 q8_1; do
    case "${lower_name}" in
        *"${unsupported}"*)
            die "${GGUF_FILE} looks like ${unsupported}, which this converter cannot decode yet. Requantise: llama-quantize ${GGUF_FILE} ${MODEL_DIR}/model-q4_0.gguf Q4_0"
            ;;
    esac
done

log "Converting ${GGUF_FILE} to Niyah native float32 format..."
warn "K-quant decoding is pure Python; a full checkpoint takes a while."
# Positional input/output; --config is optional. This differs from the old
# gguf_to_niyah.py CLI, which used --input/--output.
python3 tools/convert_gguf_to_niyah.py \
    "${GGUF_FILE}" \
    "${F32_FILE}" \
    --config "${CONFIG_FILE}" \
    --progress

if [ -f "${F32_FILE}" ] && [ -f "${CONFIG_FILE}" ]; then
    log "Conversion successful!"
    log "  Weights: ${F32_FILE} ($(du -h "${F32_FILE}" | cut -f1))"
    log "  Config:  ${CONFIG_FILE}"
    echo ""
    echo "Qwen2.5-0.5B ties its embeddings, so the blob carries no separate"
    echo "lm_head and the config says tie_word_embeddings: true. Expected."
    echo ""
    echo "The config carries both key schemas, so either loader works:"
    echo "  niyah_model_load_config_json(&config, \"${CONFIG_FILE}\");"
    echo "  niyah_mini_model_load(&model, \"${CONFIG_FILE}\", \"${F32_FILE}\");"
else
    die "Conversion failed. Check error messages above."
fi
