#!/usr/bin/env python3
"""
Dataset Validator for NiyahMini - Validates Training Data

This script validates:
1. Dataset manifest structure
2. Individual record structure
3. Provenance chain
4. Content quality
5. SHA-256 hashes
6. License compliance
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set


class DatasetValidator:
    """Validates NiyahMini dataset files."""
    
    # Required fields for training examples
    REQUIRED_FIELDS = {
        'prompt', 'completion', 'document_id', 'source_url',
        'license', 'language', 'domain', 'example_type', 'timestamp', 'sha256'
    }
    
    # Valid example types
    VALID_EXAMPLE_TYPES = {'completion', 'instruction', 'question_answer', 'summarization'}
    
    # Valid languages
    VALID_LANGUAGES = {'ar', 'en', 'code', 'ar-en', 'ar-code', 'en-code', 'unknown'}
    
    # Valid domains
    VALID_DOMAINS = {'medical', 'technical', 'legal', 'mathematics', 'general'}
    
    # Valid licenses
    VALID_LICENSES = {
        'CC-BY', 'CC-BY-SA', 'CC0', 'Public Domain',
        'MIT', 'Apache-2.0', 'BSD', 'GPL', 'LGPL',
        'Unlicense', 'WTFPL', 'ISC', 'BSL-1.0'
    }
    
    def __init__(self):
        self.errors: List[Dict] = []
        self.warnings: List[Dict] = []
        self.stats: Dict[str, Any] = {
            'total_records': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'by_type': {},
            'by_language': {},
            'by_domain': {},
            'by_license': {},
        }
    
    @staticmethod
    def sha256(text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def sha256_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()
    
    def validate_manifest(self, manifest_path: Path) -> bool:
        """Validate dataset manifest."""
        try:
            with manifest_path.open('r', encoding='utf-8') as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.errors.append({
                'type': 'manifest_error',
                'path': str(manifest_path),
                'message': f'Failed to load manifest: {e}'
            })
            return False
        
        # Check required fields
        required_manifest_fields = {
            'created_at', 'source_corpus', 'source_sha256', 'splits', 'stats', 'config'
        }
        
        for field in required_manifest_fields:
            if field not in manifest:
                self.errors.append({
                    'type': 'manifest_missing_field',
                    'path': str(manifest_path),
                    'field': field
                })
        
        # Validate splits
        if 'splits' in manifest:
            for split_name, split_info in manifest['splits'].items():
                if split_name not in {'train', 'validation', 'test'}:
                    self.warnings.append({
                        'type': 'manifest_unknown_split',
                        'path': str(manifest_path),
                        'split': split_name
                    })
                
                for field in {'path', 'sha256', 'size'}:
                    if field not in split_info:
                        self.errors.append({
                            'type': 'manifest_split_missing_field',
                            'path': str(manifest_path),
                            'split': split_name,
                            'field': field
                        })
        
        return len(self.errors) == 0
    
    def validate_record(self, record: Dict, line_no: int, file_path: str) -> bool:
        """Validate a single training example record."""
        is_valid = True
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in record:
                self.errors.append({
                    'type': 'missing_field',
                    'path': file_path,
                    'line': line_no,
                    'field': field
                })
                is_valid = False
        
        # Validate example_type
        example_type = record.get('example_type')
        if example_type not in self.VALID_EXAMPLE_TYPES:
            self.errors.append({
                'type': 'invalid_example_type',
                'path': file_path,
                'line': line_no,
                'value': example_type,
                'valid_values': list(self.VALID_EXAMPLE_TYPES)
            })
            is_valid = False
        
        # Validate language
        language = record.get('language')
        if language not in self.VALID_LANGUAGES:
            self.warnings.append({
                'type': 'unknown_language',
                'path': file_path,
                'line': line_no,
                'value': language,
                'valid_values': list(self.VALID_LANGUAGES)
            })
        
        # Validate domain
        domain = record.get('domain')
        if domain not in self.VALID_DOMAINS:
            self.warnings.append({
                'type': 'unknown_domain',
                'path': file_path,
                'line': line_no,
                'value': domain,
                'valid_values': list(self.VALID_DOMAINS)
            })
        
        # Validate license
        license = record.get('license')
        if license not in self.VALID_LICENSES:
            self.errors.append({
                'type': 'invalid_license',
                'path': file_path,
                'line': line_no,
                'value': license,
                'valid_values': list(self.VALID_LICENSES)
            })
            is_valid = False
        
        # Validate SHA-256
        prompt = record.get('prompt', '')
        completion = record.get('completion', '')
        stored_sha = record.get('sha256', '')
        
        if stored_sha:
            computed_sha = self.sha256(f"{prompt}||{completion}")
            if stored_sha != computed_sha:
                self.errors.append({
                    'type': 'sha256_mismatch',
                    'path': file_path,
                    'line': line_no,
                    'stored': stored_sha,
                    'computed': computed_sha
                })
                is_valid = False
        
        # Validate lengths
        if prompt and len(prompt) == 0:
            self.errors.append({
                'type': 'empty_prompt',
                'path': file_path,
                'line': line_no
            })
            is_valid = False
        
        if completion and len(completion) == 0:
            self.errors.append({
                'type': 'empty_completion',
                'path': file_path,
                'line': line_no
            })
            is_valid = False
        
        # Validate source_url
        source_url = record.get('source_url', '')
        if not source_url:
            self.warnings.append({
                'type': 'missing_source_url',
                'path': file_path,
                'line': line_no
            })
        
        # Validate document_id
        document_id = record.get('document_id', '')
        if not document_id:
            self.warnings.append({
                'type': 'missing_document_id',
                'path': file_path,
                'line': line_no
            })
        
        # Update stats
        if is_valid:
            self.stats['valid_records'] += 1
        else:
            self.stats['invalid_records'] += 1
        
        self.stats['total_records'] += 1
        
        if example_type:
            self.stats['by_type'][example_type] = self.stats['by_type'].get(example_type, 0) + 1
        if language:
            self.stats['by_language'][language] = self.stats['by_language'].get(language, 0) + 1
        if domain:
            self.stats['by_domain'][domain] = self.stats['by_domain'].get(domain, 0) + 1
        if license:
            self.stats['by_license'][license] = self.stats['by_license'].get(license, 0) + 1
        
        return is_valid
    
    def validate_file(self, file_path: Path) -> bool:
        """Validate a single JSONL file."""
        print(f"Validating {file_path}...")
        
        try:
            with file_path.open('r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        self.errors.append({
                            'type': 'json_error',
                            'path': str(file_path),
                            'line': line_no,
                            'message': str(e)
                        })
                        continue
                    
                    self.validate_record(record, line_no, str(file_path))
        except IOError as e:
            self.errors.append({
                'type': 'file_error',
                'path': str(file_path),
                'message': str(e)
            })
            return False
        
        return True
    
    def validate_dataset(self, dataset_dir: Path) -> bool:
        """Validate entire dataset."""
        # Validate manifest
        manifest_path = dataset_dir / 'dataset_manifest.json'
        if manifest_path.exists():
            self.validate_manifest(manifest_path)
        else:
            self.warnings.append({
                'type': 'missing_manifest',
                'path': str(dataset_dir)
            })
        
        # Validate dataset files
        dataset_files = [
            dataset_dir / 'train.jsonl',
            dataset_dir / 'val.jsonl',
            dataset_dir / 'test.jsonl'
        ]
        
        for file_path in dataset_files:
            if file_path.exists():
                self.validate_file(file_path)
            else:
                self.warnings.append({
                    'type': 'missing_file',
                    'path': str(file_path)
                })
        
        # Validate source corpus if available
        manifest_path = dataset_dir / 'dataset_manifest.json'
        if manifest_path.exists():
            try:
                with manifest_path.open('r', encoding='utf-8') as f:
                    manifest = json.load(f)
                    source_corpus = Path(manifest.get('source_corpus', ''))
                    if source_corpus.exists():
                        self.validate_corpus(source_corpus)
            except Exception:
                pass
        
        return len(self.errors) == 0
    
    def validate_corpus(self, corpus_path: Path) -> bool:
        """Validate source corpus file."""
        print(f"Validating source corpus {corpus_path}...")
        
        # Required fields for corpus records
        corpus_required = {
            'document_id', 'source_name', 'source_url', 'license',
            'content_sha256', 'language', 'domain', 'text',
            'retrieved_at_utc', 'transformations'
        }
        
        try:
            with corpus_path.open('r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        self.errors.append({
                            'type': 'corpus_json_error',
                            'path': str(corpus_path),
                            'line': line_no,
                            'message': str(e)
                        })
                        continue
                    
                    # Check required fields
                    for field in corpus_required:
                        if field not in record:
                            self.errors.append({
                                'type': 'corpus_missing_field',
                                'path': str(corpus_path),
                                'line': line_no,
                                'field': field
                            })
                    
                    # Validate SHA-256
                    text = record.get('text', '')
                    stored_sha = record.get('content_sha256', '')
                    
                    if stored_sha:
                        computed_sha = self.sha256(text)
                        if stored_sha != computed_sha:
                            self.errors.append({
                                'type': 'corpus_sha256_mismatch',
                                'path': str(corpus_path),
                                'line': line_no,
                                'stored': stored_sha,
                                'computed': computed_sha
                            })
        except IOError as e:
            self.errors.append({
                'type': 'corpus_file_error',
                'path': str(corpus_path),
                'message': str(e)
            })
            return False
        
        return True
    
    def print_report(self):
        """Print validation report."""
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)
        
        print(f"\nTotal records: {self.stats['total_records']}")
        print(f"Valid records: {self.stats['valid_records']}")
        print(f"Invalid records: {self.stats['invalid_records']}")
        
        print(f"\nErrors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        
        if self.stats['by_type']:
            print(f"\nBy type:")
            for example_type, count in sorted(self.stats['by_type'].items()):
                print(f"  {example_type}: {count}")
        
        if self.stats['by_language']:
            print(f"\nBy language:")
            for language, count in sorted(self.stats['by_language'].items()):
                print(f"  {language}: {count}")
        
        if self.stats['by_domain']:
            print(f"\nBy domain:")
            for domain, count in sorted(self.stats['by_domain'].items()):
                print(f"  {domain}: {count}")
        
        if self.stats['by_license']:
            print(f"\nBy license:")
            for license, count in sorted(self.stats['by_license'].items()):
                print(f"  {license}: {count}")
        
        if self.errors:
            print(f"\n{'ERRORS':-60}")
            for error in self.errors:
                error_type = error.get('type', 'unknown')
                path = error.get('path', '')
                line = error.get('line', '')
                message = error.get('message', '')
                
                if line:
                    print(f"  [{error_type}] {path}:{line} - {message}")
                else:
                    print(f"  [{error_type}] {path} - {message}")
        
        if self.warnings:
            print(f"\n{'WARNINGS':-60}")
            for warning in self.warnings:
                warning_type = warning.get('type', 'unknown')
                path = warning.get('path', '')
                line = warning.get('line', '')
                message = warning.get('message', '')
                
                if line:
                    print(f"  [{warning_type}] {path}:{line} - {message}")
                else:
                    print(f"  [{warning_type}] {path} - {message}")
        
        print("\n" + "=" * 60)
        if len(self.errors) == 0:
            print("VALIDATION PASSED")
        else:
            print("VALIDATION FAILED")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Validate NiyahMini dataset'
    )
    parser.add_argument(
        'path', type=Path, nargs='?',
        help='Dataset directory or file to validate'
    )
    parser.add_argument(
        '--manifest', type=Path,
        help='Validate manifest only'
    )
    parser.add_argument(
        '--corpus', type=Path,
        help='Validate corpus only'
    )
    args = parser.parse_args()
    
    validator = DatasetValidator()
    
    if args.manifest:
        validator.validate_manifest(args.manifest)
    elif args.corpus:
        validator.validate_corpus(args.corpus)
    elif args.path:
        if args.path.is_dir():
            validator.validate_dataset(args.path)
        else:
            validator.validate_file(args.path)
    else:
        print("Please specify a path to validate")
        sys.exit(1)
    
    validator.print_report()
    
    if len(validator.errors) > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
