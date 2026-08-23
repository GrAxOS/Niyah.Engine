# Niyah.Neutral (inside Niyah.Engine)

**Neutral, sovereign LLM pipeline** — fine-tuned on verified scientific/technical/medical data only.

## Quick Start

```bash
# 1. Install
pip install -r neutral/requirements.txt

# 2. Download base model (Qwen2.5-7B-Instruct from HuggingFace)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf --local-dir ./models

# 3. Prepare data (example: PubMed XML files in ./raw_data/pubmed)
python neutral/clean_corpus.py --input ./raw_data/pubmed --output ./corpus_pubmed.jsonl --source-name PubMedCentral --source-url-prefix https://www.ncbi.nlm.nih.gov/pmc --domain medicine --language en --license CC-BY

# 4. Validate manifest
python neutral/validate_manifest.py ./corpus_pubmed.jsonl

# 5. Fine-tune (QLoRA, ~2 weeks on 1x A100 40GB)
python neutral/train.py --model Qwen/Qwen2.5-7B-Instruct --data ./corpus_pubmed.jsonl --output ./qwen_neutral_pubmed --epochs 1.0

# 6. Inference (placeholder - not yet implemented)
# python neutral/inference.py --model ./qwen_neutral_pubmed --prompt "ماهي أعراض التهاب الزائدة الدودية؟"
```

## Files

| File | Purpose |
|---|---|
| `requirements.txt` | Python dependencies |
| `data_sources.md` | Data source policy + admitted registries |
| `clean_corpus.py` | Build provenance-preserving JSONL corpus |
| `validate_manifest.py` | Validate JSONL manifest (required fields, license, checksum) |
| `train.py` | QLoRA fine-tuning scaffold (SYSTEM_POLICY: no flattery, no simulated emotion) |
| `epistemic_schema.md` | FACT/INFERENCE/UNKNOWN/CONFLICTED output schema |
| `inference_contract.md` | Inference contract (provenance, validation, audit, evaluation) |

## Cost: ~$4050 (GPU rental + storage)

## License: Apache 2.0 (same as Qwen base)

## Acknowledgments

- Qwen Team (base model)
- PubMed Central, WHO, FDA, IETF, POSIX, arXiv (data sources)
- Sulaiman Alshammari (philosophy + MMR audit design)
