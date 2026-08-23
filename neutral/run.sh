#!/usr/bin/env bash
# Niyah.Neutral: Complete pipeline from data to inference
# Usage: ./neutral/run.sh [download|clean|validate|train|infer]

set -euo pipefail

# Configuration
MODEL="Qwen/Qwen2.5-7B-Instruct"
DATA_DIR="./raw_data"
CORPUS_DIR="./corpus"
OUTPUT_DIR="./qwen_neutral"
MANIFEST_FILE="${CORPUS_DIR}/manifest.jsonl"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
    exit 1
}

# Step 1: Download sample data (PubMed, RFC, etc.)
download_data() {
    log "Downloading sample data..."
    
    mkdir -p "${DATA_DIR}/pubmed"
    mkdir -p "${DATA_DIR}/rfc"
    
    # Example: Download 10 PubMed articles (replace with actual download logic)
    # For now, create placeholder files
    for i in {1..10}; do
        echo "Sample PubMed article ${i} - This is placeholder content." > "${DATA_DIR}/pubmed/article_${i}.txt"
    done
    
    # Example: Download 5 RFCs (replace with actual download from ietf.org)
    for i in 791 821 2119 2324 8174; do
        curl -s "https://www.rfc-editor.org/rfc/rfc${i}.txt" -o "${DATA_DIR}/rfc/rfc${i}.txt" || warn "Failed to download RFC${i}"
    done
    
    log "Data downloaded to ${DATA_DIR}"
}

# Step 2: Clean corpus (provenance-preserving)
clean_corpus() {
    log "Cleaning corpus..."
    
    mkdir -p "${CORPUS_DIR}"
    
    # Clean PubMed articles
    python3 neutral/clean_corpus.py \
        --input "${DATA_DIR}/pubmed" \
        --output "${CORPUS_DIR}/pubmed.jsonl" \
        --source-name "PubMedCentral" \
        --source-url-prefix "https://www.ncbi.nlm.nih.gov/pmc" \
        --domain "medicine" \
        --language "en" \
        --license "CC-BY"
    
    # Clean RFCs
    python3 neutral/clean_corpus.py \
        --input "${DATA_DIR}/rfc" \
        --output "${CORPUS_DIR}/rfc.jsonl" \
        --source-name "IETF" \
        --source-url-prefix "https://www.rfc-editor.org/rfc" \
        --domain "networking" \
        --language "en" \
        --license "Public Domain"
    
    # Merge all JSONL files
    cat "${CORPUS_DIR}"/*.jsonl > "${MANIFEST_FILE}"
    
    log "Corpus cleaned: ${MANIFEST_FILE}"
}

# Step 3: Validate manifest
validate_manifest() {
    log "Validating manifest..."
    
    python3 neutral/validate_manifest.py "${MANIFEST_FILE}"
    
    if [ $? -eq 0 ]; then
        log "Manifest validation passed."
    else
        error "Manifest validation failed. Check rejected.jsonl for details."
    fi
}

# Step 4: Train (QLoRA fine-tuning)
train() {
    log "Training QLoRA adapter..."
    
    python3 neutral/train.py \
        --model "${MODEL}" \
        --data "${MANIFEST_FILE}" \
        --output "${OUTPUT_DIR}" \
        --epochs 1.0 \
        --max-seq-length 2048 \
        --truthrl-coeff 0.1
    
    log "Training complete: ${OUTPUT_DIR}"
}

# Step 5: Inference (placeholder - not yet implemented)
infer() {
    log "Inference not yet implemented."
    warn "This is a placeholder. Implement inference.py with LVU + Peer Prediction + MMR audit."
}

# Main
case "${1:-all}" in
    download)
        download_data
        ;;
    clean)
        clean_corpus
        ;;
    validate)
        validate_manifest
        ;;
    train)
        train
        ;;
    infer)
        infer
        ;;
    all)
        download_data
        clean_corpus
        validate_manifest
        train
        infer
        ;;
    *)
        echo "Usage: $0 [download|clean|validate|train|infer|all]"
        exit 1
        ;;
esac
