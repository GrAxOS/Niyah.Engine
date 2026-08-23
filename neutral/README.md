# Niyah.Neutral (inside Niyah.Engine)

**Neutral, sovereign LLM pipeline** — fine-tuned on verified scientific/technical/medical/production-code data only.

## Features

- **TruthRL reward**: Penalizes hallucinations, rewards honest abstention (UNKNOWN)
- **LVU consistency**: Measures agreement across 5 samples to detect overconfidence
- **Peer prediction**: Verifies consistency across 3 paraphrased prompts
- **MMR audit log**: Append-only cryptographic audit of every inference
- **Production code training**: Full repositories (Linux, PostgreSQL, Nginx) with tests + docs

## Quick Start

```bash
# 1. Install dependencies
pip install -r neutral/requirements.txt

# 2. Run complete pipeline (download → train → infer)
bash neutral/run.sh all

# Or run step-by-step:
bash neutral/run.sh download    # Download PubMed, RFCs, etc.
bash neutral/run.sh clean       # Clean corpus with provenance
bash neutral/run.sh validate    # Validate manifest
bash neutral/run.sh train       # QLoRA fine-tuning
bash neutral/run.sh infer       # Inference (placeholder)
```

## Files

| File | Purpose |
|---|---|
| `requirements.txt` | Python dependencies (torch, transformers, peft, trl, bitsandbytes) |
| `data_sources.md` | Data source policy + admitted registries (including production code) |
| `clean_corpus.py` | Build provenance-preserving JSONL corpus |
| `validate_manifest.py` | Validate JSONL manifest (required fields, license, checksum) |
| `train.py` | QLoRA fine-tuning with TruthRL reward, LVU, peer prediction |
| `epistemic_schema.md` | FACT/INFERENCE/UNKNOWN/CONFLICTED output schema with LVU fields |
| `inference_contract.md` | Inference contract (provenance, validation, audit, evaluation) |
| `run.sh` | Complete pipeline script (download → clean → validate → train → infer) |

## Data Sources

| Domain | Source | License |
|---|---|---|
| Medicine | PubMed Central Open Access Subset | CC-BY |
| Public health | WHO publications | CC-BY |
| Drug labels | DailyMed | Public Domain |
| Networking | IETF RFCs | Public Domain |
| Computing | POSIX / Open Group | Proprietary (free access) |
| Research | arXiv (peer-reviewed only) | Various (CC-BY, arXiv license) |
| **Production code** | Linux kernel, PostgreSQL, Nginx, SQLite | GPL, BSD, MIT |

## Cost: ~$4050 (GPU rental + storage)

- GPU: 1x A100 40GB for 2 months (~$4000)
- Storage: 1TB S3 (~$50)
- Total: ~$4050

## License: Apache 2.0 (same as Qwen base)

## Known Biases (Inherited from Qwen2.5-7B base)

- Training corpus: Internet data (includes social media, marketing, Western-centric sources)
- Language bias: Stronger in English than Arabic
- Cultural bias: Reflects Qwen team's alignment choices (Chinese + Western)
- Safety filters: May over-refuse on sensitive topics (health, legal)

**This fine-tuning improves epistemic honesty (FACT/INFERENCE/UNKNOWN) and reduces repetition, but does NOT eliminate base model biases.**

## Acknowledgments

- Qwen Team (base model)
- PubMed Central, WHO, FDA, IETF, POSIX, arXiv (data sources)
- Linux kernel, PostgreSQL, Nginx, SQLite, Kubernetes, Rust stdlib, Go stdlib (production code)
- Sulaiman Alshammari (philosophy + MMR audit design)
