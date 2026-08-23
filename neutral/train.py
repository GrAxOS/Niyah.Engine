#!/usr/bin/env python3
"""Supervised fine-tuning scaffold.

This script deliberately does not claim that QLoRA removes properties learned
by a base model. It trains an adapter, logs its configuration, and requires a
provenance-preserving corpus manifest.
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

SYSTEM_POLICY = '''You are Niyah. Answer without flattery, simulated emotion, advertising, or claims of certainty without evidence. Separate FACT, INFERENCE, UNKNOWN, and CONFLICTED. Do not fabricate sources. For health, legal, or safety-critical questions, state limits and describe verification steps.'''

def format_example(row):
    text = row['text']
    return f'<|system|>\n{SYSTEM_POLICY}<|end|>\n<|user|>\nSummarize this source faithfully and preserve uncertainty.\n<|end|>\n<|assistant|>\nSOURCE: {row["source_url"]}\n{text}<|end|>\n'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='HF causal-LM checkpoint, not a GGUF file')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--epochs', type=float, default=1.0)
    p.add_argument('--max-seq-length', type=int, default=2048)
    args = p.parse_args()

    dataset = load_dataset('json', data_files=args.data, split='train')
    required = {'text', 'source_url', 'license', 'content_sha256'}
    if not required.issubset(set(dataset.column_names)):
        raise ValueError(f'Corpus lacks required provenance fields: {sorted(required)}')

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype='float16')
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, device_map='auto', trust_remote_code=False)
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    model = get_peft_model(model, lora)

    train_args = TrainingArguments(output_dir=str(args.output), num_train_epochs=args.epochs, per_device_train_batch_size=1, gradient_accumulation_steps=16, learning_rate=2e-4, logging_steps=10, save_steps=500, save_total_limit=2, fp16=True, report_to='none')
    trainer = SFTTrainer(model=model, args=train_args, train_dataset=dataset, formatting_func=format_example, max_seq_length=args.max_seq_length, tokenizer=tokenizer)
    trainer.train()
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    (args.output / 'training_manifest.json').write_text(json.dumps({'base_model': args.model, 'data': args.data, 'epochs': args.epochs, 'max_seq_length': args.max_seq_length, 'system_policy': SYSTEM_POLICY}, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
