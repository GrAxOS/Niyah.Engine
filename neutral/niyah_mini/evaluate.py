#!/usr/bin/env python3
"""
Evaluation Framework for NiyahMini

This script evaluates the model on:
1. Perplexity on held-out data
2. Accuracy on specific tasks
3. Evidence-awareness metrics
4. Multilingual performance
5. Provenance tracking

NO external evaluation datasets - all evaluations use our own data.
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np


class NiyahMiniEvaluator:
    """Evaluator for NiyahMini model."""
    
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}
    
    def evaluate_perplexity(self, data: List[Dict], batch_size: int = 8) -> float:
        """Evaluate perplexity on data."""
        total_loss = 0.0
        num_tokens = 0
        
        num_batches = len(data) // batch_size
        if len(data) % batch_size != 0:
            num_batches += 1
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(data))
            batch_items = data[start:end]
            
            max_len = max(len(item['input_ids']) for item in batch_items)
            
            input_ids = np.zeros((len(batch_items), max_len), dtype=np.int32)
            targets = np.zeros((len(batch_items), max_len), dtype=np.int32)
            
            for i, item in enumerate(batch_items):
                seq_len = len(item['input_ids'])
                input_ids[i, :seq_len] = item['input_ids']
                targets[i, :seq_len] = item['targets']
                num_tokens += seq_len
            
            # Forward pass
            logits = self.model.forward(input_ids)
            
            # Compute loss
            loss = self.compute_loss(logits, targets)
            total_loss += loss * num_tokens
        
        # Compute perplexity
        avg_loss = total_loss / max(num_tokens, 1)
        perplexity = math.exp(avg_loss)
        
        self.results['perplexity'] = perplexity
        self.results['avg_loss'] = avg_loss
        
        return perplexity
    
    def compute_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        
        logits_max = np.max(logits_flat, axis=1, keepdims=True)
        logits_exp = np.exp(logits_flat - logits_max)
        logits_sum = np.sum(logits_exp, axis=1, keepdims=True)
        log_probs = np.log(logits_exp / logits_sum)
        
        log_probs_target = log_probs[np.arange(len(targets_flat)), targets_flat]
        
        loss = -np.mean(log_probs_target)
        
        return float(loss)
    
    def evaluate_accuracy(self, tasks: List[Dict]) -> Dict[str, float]:
        """Evaluate accuracy on specific tasks."""
        results = {}
        
        for task in tasks:
            task_name = task.get('name', 'unknown')
            examples = task.get('examples', [])
            
            if not examples:
                continue
            
            correct = 0
            total = 0
            
            for example in examples:
                input_text = example.get('input', '')
                expected = example.get('expected', '')
                
                # Tokenize input
                # In practice, use the real tokenizer
                input_ids = [ord(c) for c in input_text[:128]]
                
                # Get model output
                # In practice, use the real model
                output_id = self.model.forward(np.array([input_ids]))[0, -1, :].argmax()
                output_char = chr(output_id)
                
                if output_char == expected:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            results[task_name] = accuracy
        
        self.results['accuracy'] = results
        return results
    
    def evaluate_evidence_awareness(self, examples: List[Dict]) -> Dict[str, float]:
        """Evaluate model's evidence-awareness."""
        results = {
            'fact_correct': 0.0,
            'inference_correct': 0.0,
            'unknown_correct': 0.0,
            'conflicted_correct': 0.0,
            'total': 0
        }
        
        for example in examples:
            prompt = example.get('prompt', '')
            expected_label = example.get('expected_label', 'UNKNOWN')
            
            # Get model output
            # In practice, use the real model
            output_text = self.model.generate(prompt, max_tokens=50)
            
            # Check for evidence labels
            if 'FACT' in output_text:
                predicted = 'FACT'
            elif 'INFERENCE' in output_text:
                predicted = 'INFERENCE'
            elif 'UNKNOWN' in output_text:
                predicted = 'UNKNOWN'
            elif 'CONFLICTED' in output_text:
                predicted = 'CONFLICTED'
            else:
                predicted = 'UNKNOWN'
            
            if predicted == expected_label:
                results[f'{expected_label.lower()}_correct'] += 1.0
            results['total'] += 1
        
        # Compute percentages
        for key in ['fact_correct', 'inference_correct', 'unknown_correct', 'conflicted_correct']:
            results[key] = (results[key] / max(results['total'], 1)) * 100.0
        
        self.results['evidence_awareness'] = results
        return results
    
    def evaluate_multilingual(self, examples: List[Dict]) -> Dict[str, float]:
        """Evaluate multilingual performance."""
        results = {}
        
        by_language = {}
        for example in examples:
            lang = example.get('language', 'unknown')
            if lang not in by_language:
                by_language[lang] = []
            by_language[lang].append(example)
        
        for lang, lang_examples in by_language.items():
            if not lang_examples:
                continue
            
            # Evaluate perplexity for this language
            # In practice, use the real evaluation
            results[lang] = self.evaluate_perplexity(lang_examples)
        
        self.results['multilingual'] = results
        return results
    
    def generate_report(self) -> str:
        """Generate evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("NIYAHMINI EVALUATION REPORT")
        report.append("=" * 60)
        
        if 'perplexity' in self.results:
            report.append(f"\nPerplexity: {self.results['perplexity']:.2f}")
            report.append(f"Average Loss: {self.results['avg_loss']:.4f}")
        
        if 'accuracy' in self.results:
            report.append("\nAccuracy by Task:")
            for task, acc in self.results['accuracy'].items():
                report.append(f"  {task}: {acc:.2%}")
        
        if 'evidence_awareness' in self.results:
            report.append("\nEvidence Awareness:")
            for key, value in self.results['evidence_awareness'].items():
                if key != 'total':
                    report.append(f"  {key}: {value:.2f}%")
        
        if 'multilingual' in self.results:
            report.append("\nMultilingual Performance:")
            for lang, ppl in self.results['multilingual'].items():
                report.append(f"  {lang}: {ppl:.2f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def create_evaluation_tasks() -> List[Dict]:
    """Create evaluation tasks."""
    tasks = []
    
    # Simple character prediction
    tasks.append({
        'name': 'char_prediction',
        'examples': [
            {'input': 'The quick brown fox jumps over the lazy ', 'expected': 'd'},
            {'input': 'Hello, world! This is a ', 'expected': 't'},
            {'input': 'مرحبا بالعالم! هذا ', 'expected': 'ن'},
        ]
    })
    
    # Simple arithmetic
    tasks.append({
        'name': 'arithmetic',
        'examples': [
            {'input': 'What is 2 + 2? ', 'expected': '4'},
            {'input': 'What is 3 * 5? ', 'expected': '1'},
        ]
    })
    
    return tasks


def create_evidence_examples() -> List[Dict]:
    """Create evidence-awareness evaluation examples."""
    examples = []
    
    # FACT examples
    examples.append({
        'prompt': 'What is the capital of France?',
        'expected_label': 'FACT'
    })
    
    examples.append({
        'prompt': 'What is 2 + 2?',
        'expected_label': 'FACT'
    })
    
    # INFERENCE examples
    examples.append({
        'prompt': 'What will the weather be like tomorrow?',
        'expected_label': 'INFERENCE'
    })
    
    examples.append({
        'prompt': 'Will it rain next week?',
        'expected_label': 'INFERENCE'
    })
    
    # UNKNOWN examples
    examples.append({
        'prompt': 'What is the meaning of life?',
        'expected_label': 'UNKNOWN'
    })
    
    examples.append({
        'prompt': 'What will happen in 100 years?',
        'expected_label': 'UNKNOWN'
    })
    
    # CONFLICTED examples
    examples.append({
        'prompt': 'Is coffee good or bad for health?',
        'expected_label': 'CONFLICTED'
    })
    
    return examples


def main():
    parser = argparse.ArgumentParser(description='Evaluate NiyahMini model')
    parser.add_argument('--model-dir', type=Path, required=True, help='Model directory')
    parser.add_argument('--data-dir', type=Path, required=True, help='Data directory')
    parser.add_argument('--output', type=Path, default=Path('evaluation_report.txt'), help='Output report')
    args = parser.parse_args()
    
    # Load model (placeholder)
    # In practice, load the real model
    class DummyModel:
        def forward(self, input_ids):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            n_vocab = 32768
            return np.random.randn(batch_size, seq_len, n_vocab).astype(np.float32) * 0.01
        
        def generate(self, prompt, max_tokens=50):
            return "This is a test output with FACT and UNKNOWN labels."
    
    model = DummyModel()
    
    # Create evaluator
    evaluator = NiyahMiniEvaluator(model)
    
    # Load evaluation data
    train_path = args.data_dir / 'train.jsonl'
    val_path = args.data_dir / 'val.jsonl'
    test_path = args.data_dir / 'test.jsonl'
    
    def load_data(path):
        if not path.exists():
            return []
        with path.open('r') as f:
            return [json.loads(line) for line in f if line.strip()]
    
    test_data = load_data(test_path)
    
    if not test_data:
        print("No test data found, creating dummy data")
        test_data = [
            {'input_ids': [ord(c) for c in 'This is a test'],
             'targets': [ord(c) for c in ' sentence.']},
            {'input_ids': [ord(c) for c in 'مرحبا بالعالم'],
             'targets': [ord(c) for c in '!']},
        ]
    
    # Evaluate
    print("Evaluating perplexity...")
    ppl = evaluator.evaluate_perplexity(test_data)
    print(f"Perplexity: {ppl:.2f}")
    
    print("\nEvaluating accuracy...")
    tasks = create_evaluation_tasks()
    accuracy = evaluator.evaluate_accuracy(tasks)
    print(f"Accuracy: {accuracy}")
    
    print("\nEvaluating evidence awareness...")
    evidence_examples = create_evidence_examples()
    evidence = evaluator.evaluate_evidence_awareness(evidence_examples)
    print(f"Evidence awareness: {evidence}")
    
    print("\nEvaluating multilingual...")
    # For multilingual, we'd need data in multiple languages
    multilingual = evaluator.evaluate_multilingual(test_data)
    print(f"Multilingual: {multilingual}")
    
    # Generate report
    report = evaluator.generate_report()
    print(report)
    
    # Save report
    with args.output.open('w') as f:
        f.write(report)
    
    print(f"\nReport saved to {args.output}")


if __name__ == '__main__':
    main()
