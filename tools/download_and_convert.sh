#!/usr/bin/env bash
# Download Qwen2.5-0.5B-Instruct GGUF and convert to Niyah native format.
# Usage: ./tools/download_and_convert.sh
#
# Note on quantisation: tools/convert_gguf_to_niyah.py decodes F32, F16,
# Q4_0 and Q4_1. The K-quants (Q4_K / Q5_K / Q6_K) use 256-element
# superblocks with 6-bit packed scales and are not implemented yet, so this
# script pulls the q4_0 build rather than q4_k_m. If you already have a
# K-quant file, requantise it first:
#
#     llama-quantize model.gguf model-q4_0.gguf Q4_0

set -euo pipefail

MODEL_DIR="./models"
OUTPUT_DIR="./weights"
GGUF_NAME="qwen2.5-0.5b-instruct-q4_0.gguf"
GGUF_FILE="${MODEL_DIR}/${GGUF_NAME}"
F32_FILE="${OUTPUT_DIR}/qwen2.5-0.5b-f32.bin"
CONFIG_FILE="${OUTPUT_DIR}/config.json"

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

log "Creating directories..."
mkdir -p "${MODEL_DIR}" "${OUTPUT_DIR}"

# The converter itself is stdlib-only; huggingface-cli is what needs deps.
if ! command -v huggingface-cli >/dev/null 2>&1; then
    log "Installing Python dependencies..."
    pip3 install -r tools/requirements.txt || warn "Failed to install dependencies"
fi

if [ ! -f "${GGUF_FILE}" ]; then
    log "Downloading ${GGUF_NAME}..."
    huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF "${GGUF_NAME}" \
        --local-dir "${MODEL_DIR}" \
        || die "Download failed. List available files with: huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF --repo-type model --quiet 2>/dev/null || true"
else
    log "GGUF file already exists: ${GGUF_FILE}"
fi

case "${GGUF_FILE}" in
    *_k_m.gguf|*_k_s.gguf|*_K_M.gguf|*_K_S.gguf|*q[0-9]_k*.gguf)
        die "${GGUF_FILE} is a K-quant. Requantise first: llama-quantize ${GGUF_FILE} ${MODEL_DIR}/model-q4_0.gguf Q4_0"
        ;;
esac

log "Converting GGUF to Niyah native float32 format..."
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
    echo "The config carries both key schemas, so either loader works:"
    echo "  niyah_model_load_config_json(&config, \"${CONFIG_FILE}\");"
    echo "  niyah_mini_model_load(&model, \"${CONFIG_FILE}\", \"${F32_FILE}\");"
else
    die "Conversion failed. Check error messages above."
fi
