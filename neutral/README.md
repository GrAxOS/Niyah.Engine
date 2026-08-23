# Niyah.Neutral (inside Niyah.Engine)

**Neutral, sovereign LLM pipeline** — fine-tuned on verified scientific/technical/medical data only.

## Quick Start

```bash
# 1. Install
pip install transformers peft trl datasets accelerate bitsandbytes torch

# 2. Download base model
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf --local-dir ./models

# 3. Clean data
python neutral/clean_corpus.py --input ./raw_data --output ./clean_corpus.jsonl

# 4. Fine-tune (QLoRA, ~2 weeks on 1x A100)
python neutral/train.py --data ./clean_corpus.jsonl --output ./qwen_neutral

# 5. Train classifier
python neutral/train_classifier.py --data ./epistemic_labels.jsonl --output ./classifier

# 6. Inference
python neutral/inference.py --model ./qwen_neutral --classifier ./classifier --prompt "ماهي أعراض التهاب الزائدة الدودية؟"
```

## Cost: ~$4050 (GPU rental + storage)

## License: Apache 2.0
