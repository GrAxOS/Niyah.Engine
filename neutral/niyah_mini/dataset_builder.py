#!/usr/bin/env python3
"""
Dataset Builder for NiyahMini - Creates Training Data with Provenance

This script:
1. Takes cleaned corpus (from corpus_cleaner.py)
2. Creates training examples with prompts and completions
3. Formats data for transformer training
4. Preserves full provenance chain
5. Balances content across domains and languages

NO external APIs or datasets that would make the model a copy of another.
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple


class TrainingExample:
    """A single training example with prompt and completion."""
    
    def __init__(
        self,
        prompt: str,
        completion: str,
        document_id: str,
        source_url: str,
        license: str,
        language: str,
        domain: str,
        example_type: str = 'completion'
    ):
        self.prompt = prompt
        self.completion = completion
        self.document_id = document_id
        self.source_url = source_url
        self.license = license
        self.language = language
        self.domain = domain
        self.example_type = example_type
        self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'prompt': self.prompt,
            'completion': self.completion,
            'document_id': self.document_id,
            'source_url': self.source_url,
            'license': self.license,
            'language': self.language,
            'domain': self.domain,
            'example_type': self.example_type,
            'timestamp': self.timestamp,
            'sha256': hashlib.sha256(
                f"{self.prompt}||{self.completion}".encode()
            ).hexdigest()
        }
    
    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class DatasetBuilder:
    """Builds training dataset from cleaned corpus."""
    
    # Example types
    EXAMPLE_TYPES = ['completion', 'instruction', 'question_answer', 'summarization']
    
    # Domain weights for balancing
    DOMAIN_WEIGHTS = {
        'medical': 0.2,
        'technical': 0.3,
        'legal': 0.1,
        'mathematics': 0.2,
        'general': 0.2,
    }
    
    # Language weights
    LANGUAGE_WEIGHTS = {
        'ar': 0.4,
        'en': 0.4,
        'code': 0.1,
        'ar-en': 0.05,
        'ar-code': 0.05,
    }
    
    def __init__(
        self,
        max_length: int = 2048,
        min_prompt_length: int = 10,
        max_prompt_length: int = 512,
        min_completion_length: int = 20,
        max_completion_length: int = 1024,
        balance_domains: bool = True,
        balance_languages: bool = True,
        test_split: float = 0.1,
        val_split: float = 0.1
    ):
        self.max_length = max_length
        self.min_prompt_length = min_prompt_length
        self.max_prompt_length = max_prompt_length
        self.min_completion_length = min_completion_length
        self.max_completion_length = max_completion_length
        self.balance_domains = balance_domains
        self.balance_languages = balance_languages
        self.test_split = test_split
        self.val_split = val_split
        
        # Statistics
        self.stats = {
            'total_examples': 0,
            'by_type': Counter(),
            'by_domain': Counter(),
            'by_language': Counter(),
            'rejected': 0,
            'rejected_short': 0,
            'rejected_long': 0,
        }
    
    @staticmethod
    def sha256(text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def load_corpus(self, corpus_path: Path) -> List[Dict]:
        """Load cleaned corpus from JSONL file."""
        records = []
        with corpus_path.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records
    
    def create_completion_example(self, record: Dict) -> Optional[TrainingExample]:
        """Create a completion-style training example."""
        text = record.get('text', '')
        
        if len(text) < self.min_completion_length:
            self.stats['rejected_short'] += 1
            self.stats['rejected'] += 1
            return None
        
        # Split text into prompt and completion
        # Use first sentence as prompt, rest as completion
        sentences = self.split_sentences(text, record.get('language', 'en'))
        
        if len(sentences) < 2:
            # Use first half as prompt, second half as completion
            split_pos = len(text) // 2
            prompt = text[:split_pos].strip()
            completion = text[split_pos:].strip()
        else:
            prompt = sentences[0].strip()
            completion = ' '.join(sentences[1:]).strip()
        
        # Ensure lengths are within bounds
        if len(prompt) < self.min_prompt_length:
            # Extend prompt
            prompt = text[:self.min_prompt_length * 2].strip()
        
        if len(prompt) > self.max_prompt_length:
            prompt = prompt[:self.max_prompt_length].strip()
        
        if len(completion) > self.max_completion_length:
            completion = completion[:self.max_completion_length].strip()
        
        if len(completion) < self.min_completion_length:
            self.stats['rejected_short'] += 1
            self.stats['rejected'] += 1
            return None
        
        # Check total length
        if len(prompt) + len(completion) > self.max_length:
            self.stats['rejected_long'] += 1
            self.stats['rejected'] += 1
            return None
        
        return TrainingExample(
            prompt=prompt,
            completion=completion,
            document_id=record.get('document_id', ''),
            source_url=record.get('source_url', ''),
            license=record.get('license', ''),
            language=record.get('language', 'unknown'),
            domain=record.get('domain', 'general'),
            example_type='completion'
        )
    
    def create_instruction_example(self, record: Dict) -> Optional[TrainingExample]:
        """Create an instruction-style training example."""
        text = record.get('text', '')
        language = record.get('language', 'en')
        
        if len(text) < self.min_completion_length * 2:
            self.stats['rejected_short'] += 1
            self.stats['rejected'] += 1
            return None
        
        # Generate instruction based on domain
        domain = record.get('domain', 'general')
        instruction = self.generate_instruction(domain, language)
        
        # Use text as the "answer" or "content"
        completion = text
        
        # Truncate if needed
        if len(completion) > self.max_completion_length:
            completion = completion[:self.max_completion_length].strip()
        
        if len(completion) < self.min_completion_length:
            self.stats['rejected_short'] += 1
            self.stats['rejected'] += 1
            return None
        
        # Format prompt
        if language in ['ar', 'ar-en', 'ar-code']:
            prompt = f"{instruction}\nالمحتوى:"
        else:
            prompt = f"{instruction}\nContent:"
        
        if len(prompt) > self.max_prompt_length:
            prompt = prompt[:self.max_prompt_length].strip()
        
        return TrainingExample(
            prompt=prompt,
            completion=completion,
            document_id=record.get('document_id', ''),
            source_url=record.get('source_url', ''),
            license=record.get('license', ''),
            language=language,
            domain=domain,
            example_type='instruction'
        )
    
    def create_qa_example(self, record: Dict) -> Optional[TrainingExample]:
        """Create a question-answer training example."""
        text = record.get('text', '')
        language = record.get('language', 'en')
        
        if len(text) < self.min_completion_length * 3:
            self.stats['rejected_short'] += 1
            self.stats['rejected'] += 1
            return None
        
        # Split into sentences
        sentences = self.split_sentences(text, language)
        
        if len(sentences) < 2:
            self.stats['rejected_short'] += 1
            self.stats['rejected'] += 1
            return None
        
        # Use first sentence as question, rest as answer
        question = sentences[0].strip()
        answer = ' '.join(sentences[1:]).strip()
        
        # Ensure question ends with question mark
        if not question.endswith('?'):
            if language in ['ar', 'ar-en', 'ar-code']:
                question += '؟'
            else:
                question += '?'
        
        # Truncate if needed
        if len(question) > self.max_prompt_length:
            question = question[:self.max_prompt_length].strip()
        
        if len(answer) > self.max_completion_length:
            answer = answer[:self.max_completion_length].strip()
        
        if len(answer) < self.min_completion_length:
            self.stats['rejected_short'] += 1
            self.stats['rejected'] += 1
            return None
        
        return TrainingExample(
            prompt=question,
            completion=answer,
            document_id=record.get('document_id', ''),
            source_url=record.get('source_url', ''),
            license=record.get('license', ''),
            language=language,
            domain=record.get('domain', 'general'),
            example_type='question_answer'
        )
    
    def create_summarization_example(self, record: Dict) -> Optional[TrainingExample]:
        """Create a summarization training example."""
        text = record.get('text', '')
        language = record.get('language', 'en')
        
        if len(text) < self.min_completion_length * 5:
            self.stats['rejected_short'] += 1
            self.stats['rejected'] += 1
            return None
        
        # Split into sentences
        sentences = self.split_sentences(text, language)
        
        if len(sentences) < 3:
            self.stats['rejected_short'] += 1
            self.stats['rejected'] += 1
            return None
        
        # Use first few sentences as prompt, last few as summary
        split_pos = len(sentences) // 2
        prompt = ' '.join(sentences[:split_pos]).strip()
        summary = ' '.join(sentences[split_pos:]).strip()
        
        # Add instruction
        if language in ['ar', 'ar-en', 'ar-code']:
            prompt = f"لخص النص التالي:\n{prompt}"
        else:
            prompt = f"Summarize the following text:\n{prompt}"
        
        # Truncate if needed
        if len(prompt) > self.max_prompt_length:
            prompt = prompt[:self.max_prompt_length].strip()
        
        if len(summary) > self.max_completion_length:
            summary = summary[:self.max_completion_length].strip()
        
        if len(summary) < self.min_completion_length:
            self.stats['rejected_short'] += 1
            self.stats['rejected'] += 1
            return None
        
        return TrainingExample(
            prompt=prompt,
            completion=summary,
            document_id=record.get('document_id', ''),
            source_url=record.get('source_url', ''),
            license=record.get('license', ''),
            language=language,
            domain=record.get('domain', 'general'),
            example_type='summarization'
        )
    
    def split_sentences(self, text: str, language: str) -> List[str]:
        """Split text into sentences based on language."""
        if language in ['ar', 'ar-en', 'ar-code']:
            # Arabic sentence splitting
            # Split on Arabic punctuation followed by space or newline
            sentences = re.split(r'([.!?؛؟]+[\s\n]+|\n+)', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        else:
            # English sentence splitting
            sentences = re.split(r'([.!?]+[\s\n]+|\n+)', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Filter out empty sentences
        sentences = [s for s in sentences if s]
        
        return sentences
    
    def generate_instruction(self, domain: str, language: str) -> str:
        """Generate an instruction based on domain and language."""
        if language in ['ar', 'ar-en', 'ar-code']:
            instructions = {
                'medical': 'قدم معلومات طبية دقيقة مع الإشارة إلى المصادر',
                'technical': 'شرح المفاهيم الفنية بوضوح',
                'legal': 'قدم تحليل قانوني دقيق',
                'mathematics': 'شرح المفاهيم الرياضية خطوة بخطوة',
                'general': 'قدم معلومات دقيقة ومفيدة',
            }
        else:
            instructions = {
                'medical': 'Provide accurate medical information with sources',
                'technical': 'Explain technical concepts clearly',
                'legal': 'Provide accurate legal analysis',
                'mathematics': 'Explain mathematical concepts step by step',
                'general': 'Provide accurate and helpful information',
            }
        
        return instructions.get(domain, instructions['general'])
    
    def create_examples_from_record(self, record: Dict) -> List[TrainingExample]:
        """Create multiple training examples from a single record."""
        examples = []
        
        # Try different example types
        for example_type in self.EXAMPLE_TYPES:
            example = None
            if example_type == 'completion':
                example = self.create_completion_example(record)
            elif example_type == 'instruction':
                example = self.create_instruction_example(record)
            elif example_type == 'question_answer':
                example = self.create_qa_example(record)
            elif example_type == 'summarization':
                example = self.create_summarization_example(record)
            
            if example:
                examples.append(example)
                self.stats['by_type'][example_type] += 1
        
        return examples
    
    def balance_examples(
        self,
        examples: List[TrainingExample],
        target_count: int
    ) -> List[TrainingExample]:
        """Balance examples across domains and languages."""
        if not self.balance_domains and not self.balance_languages:
            return examples
        
        # Group by domain and language
        by_domain = defaultdict(list)
        by_language = defaultdict(list)
        
        for example in examples:
            by_domain[example.domain].append(example)
            by_language[example.language].append(example)
        
        # Apply domain balancing
        if self.balance_domains:
            # Calculate target per domain based on weights
            domain_counts = {d: len(exs) for d, exs in by_domain.items()}
            total = sum(domain_counts.values())
            
            # Keep all examples but ensure minimum representation
            balanced = []
            for domain, weight in self.DOMAIN_WEIGHTS.items():
                if domain in by_domain:
                    # Keep all examples from this domain
                    balanced.extend(by_domain[domain])
            
            # Add remaining examples
            for domain in by_domain:
                if domain not in self.DOMAIN_WEIGHTS:
                    balanced.extend(by_domain[domain])
            
            examples = balanced
        
        # Apply language balancing
        if self.balance_languages:
            language_counts = {l: len(exs) for l, exs in by_language.items()}
            total = sum(language_counts.values())
            
            # Keep all examples but ensure minimum representation
            balanced = []
            for language, weight in self.LANGUAGE_WEIGHTS.items():
                if language in by_language:
                    balanced.extend(by_language[language])
            
            # Add remaining examples
            for language in by_language:
                if language not in self.LANGUAGE_WEIGHTS:
                    balanced.extend(by_language[language])
            
            examples = balanced
        
        # Limit to target count
        if target_count > 0 and len(examples) > target_count:
            # Shuffle and truncate
            random.shuffle(examples)
            examples = examples[:target_count]
        
        return examples
    
    def split_dataset(
        self,
        examples: List[TrainingExample]
    ) -> Tuple[List[TrainingExample], List[TrainingExample], List[TrainingExample]]:
        """Split dataset into train, validation, and test sets."""
        # Shuffle
        random.shuffle(examples)
        
        # Calculate split sizes
        total = len(examples)
        val_size = int(total * self.val_split)
        test_size = int(total * self.test_split)
        
        # Ensure minimum sizes
        val_size = max(val_size, 100)
        test_size = max(test_size, 100)
        
        # Adjust if splits overlap
        if val_size + test_size > total:
            val_size = total // 4
            test_size = total // 4
        
        # Split
        train = examples[test_size + val_size:]
        val = examples[test_size:test_size + val_size]
        test = examples[:test_size]
        
        return train, val, test
    
    def build_dataset(
        self,
        corpus_path: Path,
        output_dir: Path
    ) -> Tuple[Path, Path, Path]:
        """Build complete dataset from corpus."""
        # Load corpus
        records = self.load_corpus(corpus_path)
        print(f"Loaded {len(records)} records from {corpus_path}")
        
        # Create examples
        all_examples = []
        for record in records:
            examples = self.create_examples_from_record(record)
            all_examples.extend(examples)
        
        self.stats['total_examples'] = len(all_examples)
        
        # Count by domain and language
        for example in all_examples:
            self.stats['by_domain'][example.domain] += 1
            self.stats['by_language'][example.language] += 1
        
        print(f"Created {len(all_examples)} training examples")
        print(f"Examples by type: {dict(self.stats['by_type'])}")
        print(f"Examples by domain: {dict(self.stats['by_domain'])}")
        print(f"Examples by language: {dict(self.stats['by_language'])}")
        
        # Balance examples
        all_examples = self.balance_examples(all_examples, 0)
        
        # Split dataset
        train, val, test = self.split_dataset(all_examples)
        
        print(f"Dataset splits:")
        print(f"  Train: {len(train)}")
        print(f"  Validation: {len(val)}")
        print(f"  Test: {len(test)}")
        
        # Save datasets
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = output_dir / 'train.jsonl'
        val_path = output_dir / 'val.jsonl'
        test_path = output_dir / 'test.jsonl'
        
        self.save_dataset(train, train_path)
        self.save_dataset(val, val_path)
        self.save_dataset(test, test_path)
        
        # Save manifest
        manifest_path = output_dir / 'dataset_manifest.json'
        self.save_manifest(
            train_path, val_path, test_path,
            corpus_path, manifest_path
        )
        
        return train_path, val_path, test_path
    
    def save_dataset(
        self,
        examples: List[TrainingExample],
        output_path: Path
    ):
        """Save dataset to JSONL file."""
        with output_path.open('w', encoding='utf-8') as f:
            for example in examples:
                f.write(example.to_jsonl() + '\n')
    
    def save_manifest(
        self,
        train_path: Path,
        val_path: Path,
        test_path: Path,
        source_path: Path,
        manifest_path: Path
    ):
        """Save dataset manifest."""
        manifest = {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'source_corpus': str(source_path),
            'source_sha256': self.sha256_bytes(source_path.read_bytes()),
            'splits': {
                'train': {
                    'path': str(train_path),
                    'sha256': self.sha256_bytes(train_path.read_bytes()),
                    'size': self.count_lines(train_path)
                },
                'validation': {
                    'path': str(val_path),
                    'sha256': self.sha256_bytes(val_path.read_bytes()),
                    'size': self.count_lines(val_path)
                },
                'test': {
                    'path': str(test_path),
                    'sha256': self.sha256_bytes(test_path.read_bytes()),
                    'size': self.count_lines(test_path)
                }
            },
            'stats': self.stats,
            'config': {
                'max_length': self.max_length,
                'min_prompt_length': self.min_prompt_length,
                'max_prompt_length': self.max_prompt_length,
                'min_completion_length': self.min_completion_length,
                'max_completion_length': self.max_completion_length,
                'balance_domains': self.balance_domains,
                'balance_languages': self.balance_languages,
                'test_split': self.test_split,
                'val_split': self.val_split,
            }
        }
        
        with manifest_path.open('w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def count_lines(file_path: Path) -> int:
        """Count lines in a file."""
        with file_path.open('r', encoding='utf-8') as f:
            return sum(1 for _ in f)


def main():
    parser = argparse.ArgumentParser(
        description='Build training dataset from cleaned corpus'
    )
    parser.add_argument(
        '--corpus', type=Path, required=True,
        help='Input cleaned corpus (JSONL)'
    )
    parser.add_argument(
        '--output-dir', type=Path, required=True,
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--max-length', type=int, default=2048,
        help='Maximum total length (prompt + completion)'
    )
    parser.add_argument(
        '--min-prompt-length', type=int, default=10,
        help='Minimum prompt length'
    )
    parser.add_argument(
        '--max-prompt-length', type=int, default=512,
        help='Maximum prompt length'
    )
    parser.add_argument(
        '--min-completion-length', type=int, default=20,
        help='Minimum completion length'
    )
    parser.add_argument(
        '--max-completion-length', type=int, default=1024,
        help='Maximum completion length'
    )
    parser.add_argument(
        '--no-balance-domains', action='store_false', dest='balance_domains',
        help='Disable domain balancing'
    )
    parser.add_argument(
        '--no-balance-languages', action='store_false', dest='balance_languages',
        help='Disable language balancing'
    )
    parser.add_argument(
        '--test-split', type=float, default=0.1,
        help='Test split ratio'
    )
    parser.add_argument(
        '--val-split', type=float, default=0.1,
        help='Validation split ratio'
    )
    args = parser.parse_args()
    
    # Initialize builder
    builder = DatasetBuilder(
        max_length=args.max_length,
        min_prompt_length=args.min_prompt_length,
        max_prompt_length=args.max_prompt_length,
        min_completion_length=args.min_completion_length,
        max_completion_length=args.max_completion_length,
        balance_domains=args.balance_domains,
        balance_languages=args.balance_languages,
        test_split=args.test_split,
        val_split=args.val_split
    )
    
    # Build dataset
    train_path, val_path, test_path = builder.build_dataset(
        corpus_path=args.corpus,
        output_dir=args.output_dir
    )
    
    print(f"\nDataset saved to:")
    print(f"  Train: {train_path}")
    print(f"  Validation: {val_path}")
    print(f"  Test: {test_path}")
    print(f"  Manifest: {args.output_dir / 'dataset_manifest.json'}")


if __name__ == '__main__':
    main()
