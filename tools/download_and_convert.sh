#!/usr/bin/env bash
# Download Qwen2.5-0.5B-Instruct GGUF and convert to Niyah native format
# Usage: ./tools/download_and_convert.sh

set -euo pipefail

MODEL_DIR="./models"
OUTPUT_DIR="./weights"
GGUF_FILE="${MODEL_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf"
F32_FILE="${OUTPUT_DIR}/qwen2.5-0.5b-f32.bin"
CONFIG_FILE="${OUTPUT_DIR}/config.json"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Step 1: Create directories
log "Creating directories..."
mkdir -p "${MODEL_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Step 2: Install Python dependencies
log "Installing Python dependencies..."
pip3 install -r tools/requirements.txt || warn "Failed to install dependencies"

# Step 3: Download GGUF (if not exists)
if [ ! -f "${GGUF_FILE}" ]; then
    log "Downloading Qwen2.5-0.5B-Instruct GGUF (~500MB)..."
    huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir "${MODEL_DIR}"
else
    log "GGUF file already exists: ${GGUF_FILE}"
fi

# Step 4: Convert to native format
log "Converting GGUF to Niyah native float32 format..."
python3 tools/gguf_to_niyah.py \
    --input "${GGUF_FILE}" \
    --output "${F32_FILE}" \
    --config "${CONFIG_FILE}"

# Step 5: Verify output
if [ -f "${F32_FILE}" ] && [ -f "${CONFIG_FILE}" ]; then
    log "Conversion successful!"
    log "  Weights: ${F32_FILE} ($(du -h "${F32_FILE}" | cut -f1))"
    log "  Config: ${CONFIG_FILE}"
    
    echo ""
    echo "To load weights in C:"
    echo "  uint8_t *buffer = load_file(\"${F32_FILE}\");"
    echo "  niyah_llm_weights_load_from_buffer(&weights, &config, buffer, size);"
else
    warn "Conversion failed. Check error messages above."
    exit 1
fi
