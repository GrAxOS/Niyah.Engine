#!/usr/bin/env python3
"""
Corpus Cleaner for NiyahMini - Provenance-Aware Data Processing

This script:
1. Cleans raw text from various sources
2. Preserves provenance (source_url, license, SHA-256)
3. Removes PII, duplicates, and low-quality content
4. Handles Arabic, English, code, and mixed content
5. Outputs a clean, structured corpus with manifest

NO external APIs or datasets that would make the model a copy of another.
All data is processed with full provenance tracking.
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import unicodedata


class ProvenanceRecord:
    """Represents a single document with full provenance."""
    
    def __init__(
        self,
        document_id: str,
        source_name: str,
        source_url: str,
        license: str,
        content_sha256: str,
        language: str,
        domain: str,
        text: str,
        publication_date: Optional[str] = None,
        version: Optional[str] = None,
        transformations: Optional[List[str]] = None,
        quality_flags: Optional[List[str]] = None,
        retrieved_at_utc: Optional[str] = None
    ):
        self.document_id = document_id
        self.source_name = source_name
        self.source_url = source_url
        self.license = license
        self.content_sha256 = content_sha256
        self.language = language
        self.domain = domain
        self.text = text
        self.publication_date = publication_date
        self.version = version
        self.transformations = transformations or []
        self.quality_flags = quality_flags or []
        self.retrieved_at_utc = retrieved_at_utc or datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'source_name': self.source_name,
            'source_url': self.source_url,
            'license': self.license,
            'content_sha256': self.content_sha256,
            'language': self.language,
            'domain': self.domain,
            'publication_date': self.publication_date,
            'version': self.version,
            'transformations': self.transformations,
            'quality_flags': self.quality_flags,
            'text': self.text,
            'retrieved_at_utc': self.retrieved_at_utc,
        }
    
    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class CorpusCleaner:
    """Cleans and processes raw text into a structured corpus."""
    
    # Required fields for manifest
    REQUIRED_FIELDS = {
        'document_id', 'source_name', 'source_url', 'license',
        'content_sha256', 'language', 'domain', 'text',
        'retrieved_at_utc', 'transformations'
    }
    
    # Quality thresholds
    MIN_TEXT_LENGTH = 100  # Minimum characters
    MAX_TEXT_LENGTH = 100000  # Maximum characters
    MIN_WORD_COUNT = 20  # Minimum words
    MAX_WORD_COUNT = 20000  # Maximum words
    
    # PII patterns to remove
    PII_PATTERNS = [
        # Email addresses
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        # Phone numbers (various formats)
        r'(\+?\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}',
        # Social security numbers (US)
        r'\d{3}-\d{2}-\d{4}',
        # IP addresses
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
        # Credit card numbers
        r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',
        # URLs (we keep source_url but remove embedded URLs)
        r'https?://[^\s]+',
    ]
    
    # Arabic-specific patterns
    ARABIC_PATTERNS = {
        'tatweel': re.compile(r'[ـ]+'),  # Tatweel (elongation)
        'diacritics': re.compile(r'[\u064B-\u065F\u0670]+'),  # Arabic diacritics
        'punctuation': re.compile(r'[،؛؟!]'),  # Arabic punctuation
    }
    
    # English-specific patterns
    ENGLISH_PATTERNS = {
        'whitespace': re.compile(r'[\s]+'),
        'punctuation': re.compile(r'[\.,;:!?]'),
    }
    
    # Code-specific patterns
    CODE_PATTERNS = {
        'comments': re.compile(r'//.*|#.*|/\*.*\*/', re.DOTALL),
        'strings': re.compile(r'"[^"]*"|'\'[^\']*\''),
        'whitespace': re.compile(r'[\s]+'),
    }
    
    def __init__(
        self,
        output_dir: Path,
        min_length: int = MIN_TEXT_LENGTH,
        max_length: int = MAX_TEXT_LENGTH,
        remove_pii: bool = True,
        deduplicate: bool = True,
        preserve_provenance: bool = True,
        allowed_licenses: Optional[Set[str]] = None,
        allowed_domains: Optional[Set[str]] = None
    ):
        self.output_dir = output_dir
        self.min_length = min_length
        self.max_length = max_length
        self.remove_pii = remove_pii
        self.deduplicate = deduplicate
        self.preserve_provenance = preserve_provenance
        self.allowed_licenses = allowed_licenses or {
            'CC-BY', 'CC-BY-SA', 'CC0', 'Public Domain',
            'MIT', 'Apache-2.0', 'BSD', 'GPL', 'LGPL',
            'Unlicense', 'WTFPL', 'ISC', 'BSL-1.0'
        }
        self.allowed_domains = allowed_domains or {
            'ncbi.nlm.nih.gov', 'who.int', 'wikipedia.org',
            'ietf.org', 'github.com', 'arxiv.org',
            'kernel.org', 'postgresql.org', 'nginx.org'
        }
        
        # Track seen hashes for deduplication
        self.seen_hashes: Set[str] = set()
        
        # Statistics
        self.stats = {
            'total_input': 0,
            'accepted': 0,
            'rejected': 0,
            'rejected_pii': 0,
            'rejected_duplicate': 0,
            'rejected_license': 0,
            'rejected_short': 0,
            'rejected_long': 0,
            'rejected_quality': 0,
        }
    
    @staticmethod
    def sha256(text: str) -> str:
        """Compute SHA-256 hash of text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def sha256_bytes(data: bytes) -> str:
        """Compute SHA-256 hash of bytes."""
        return hashlib.sha256(data).hexdigest()
    
    def detect_language(self, text: str) -> str:
        """Detect primary language of text."""
        # Count Arabic characters
        arabic_chars = sum(
            1 for c in text 
            if (0x0600 <= ord(c) <= 0x06FF) or 
               (0x0750 <= ord(c) <= 0x077F) or
               (0x08A0 <= ord(c) <= 0x08FF)
        )
        
        # Count English characters
        english_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        
        # Count code-like characters
        code_chars = sum(1 for c in text if c in '{}[]()<>@#$%^&*+-=/\\|~`')
        
        # Count digits
        digit_chars = sum(1 for c in text if c.isdigit())
        
        # Total non-whitespace characters
        total_chars = sum(1 for c in text if not c.isspace())
        
        if total_chars == 0:
            return 'unknown'
        
        # Determine primary language
        if arabic_chars > english_chars and arabic_chars > code_chars:
            if arabic_chars / total_chars > 0.5:
                return 'ar'
            elif english_chars / total_chars > 0.3:
                return 'ar-en'  # Mixed Arabic-English
            else:
                return 'ar'
        elif english_chars > arabic_chars and english_chars > code_chars:
            if code_chars / total_chars > 0.2:
                return 'en-code'  # Mixed English-code
            else:
                return 'en'
        elif code_chars > arabic_chars and code_chars > english_chars:
            return 'code'
        else:
            # Mixed content
            if arabic_chars > 0 and english_chars > 0:
                return 'ar-en'
            elif arabic_chars > 0 and code_chars > 0:
                return 'ar-code'
            elif english_chars > 0 and code_chars > 0:
                return 'en-code'
            else:
                return 'unknown'
    
    def detect_domain(self, text: str, source_url: str = '') -> str:
        """Detect domain of text."""
        # Check for medical terms
        medical_terms = ['patient', 'diagnosis', 'treatment', 'disease', 'medical',
                        'مريض', 'تشخيص', 'علاج', 'مرض', 'طبي']
        if any(term in text.lower() for term in medical_terms):
            return 'medical'
        
        # Check for technical terms
        tech_terms = ['algorithm', 'function', 'variable', 'code', 'program',
                      'خوارزمية', 'دالة', 'متغير', 'كود', 'برمجة']
        if any(term in text.lower() for term in tech_terms):
            return 'technical'
        
        # Check for legal terms
        legal_terms = ['law', 'legal', 'contract', 'rights', 'license',
                       'قانون', 'عقد', 'حقوق', 'ترخيص']
        if any(term in text.lower() for term in legal_terms):
            return 'legal'
        
        # Check for mathematical terms
        math_terms = ['theorem', 'proof', 'equation', 'formula', 'math',
                      'مبرهنة', 'برهان', 'معادلة', 'صيغة', 'رياضيات']
        if any(term in text.lower() for term in math_terms):
            return 'mathematics'
        
        # Check for general knowledge
        if 'wikipedia' in source_url.lower():
            return 'general'
        
        # Default
        return 'general'
    
    def remove_pii(self, text: str) -> str:
        """Remove personally identifiable information from text."""
        if not self.remove_pii:
            return text
        
        result = text
        for pattern in self.PII_PATTERNS:
            result = re.sub(pattern, '[REDACTED]', result)
        
        return result
    
    def normalize_text(self, text: str, language: str) -> str:
        """Normalize text according to language."""
        # Remove control characters
        text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
        
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Language-specific normalization
        if language in ['ar', 'ar-en', 'ar-code']:
            # Remove excessive tatweel (elongation)
            text = self.ARABIC_PATTERNS['tatweel'].sub('ـ', text)
            # Remove tatweel at end of words
            text = re.sub(r'([^ـ])ـ+$', r'\1', text, flags=re.MULTILINE)
            # Normalize spaces around Arabic punctuation
            text = re.sub(r'\s+([،؛؟!])', r'\1', text)
            text = re.sub(r'([،؛؟!])\s+', r'\1 ', text)
        
        # Normalize whitespace
        text = re.sub(r'[\s]+', ' ', text)
        text = re.sub(r'\s+(\n)', r'\1', text)  # Preserve newlines
        text = re.sub(r'(\n)\s+', r'\1', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def clean_text(self, text: str, language: str) -> str:
        """Clean text: remove PII, normalize, etc."""
        # Remove PII
        text = self.remove_pii(text)
        
        # Normalize
        text = self.normalize_text(text, language)
        
        return text
    
    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a corpus record."""
        errors = []
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in record:
                errors.append(f'missing_field:{field}')
        
        # Check text length
        text = record.get('text', '')
        if len(text) < self.min_length:
            errors.append(f'too_short:{len(text)}')
        if len(text) > self.max_length:
            errors.append(f'too_long:{len(text)}')
        
        # Check word count
        word_count = len(text.split())
        if word_count < self.MIN_WORD_COUNT:
            errors.append(f'too_few_words:{word_count}')
        if word_count > self.MAX_WORD_COUNT:
            errors.append(f'too_many_words:{word_count}')
        
        # Check license
        license = record.get('license', '')
        if license and license not in self.allowed_licenses:
            errors.append(f'unknown_license:{license}')
        
        # Check source URL
        source_url = record.get('source_url', '')
        if source_url:
            # Extract domain
            domain = self.extract_domain(source_url)
            if domain and domain not in self.allowed_domains:
                errors.append(f'unknown_domain:{domain}')
        
        # Check SHA-256
        content_sha256 = record.get('content_sha256', '')
        if content_sha256:
            computed_sha = self.sha256(text)
            if content_sha256 != computed_sha:
                errors.append('sha256_mismatch')
        
        return len(errors) == 0, errors
    
    @staticmethod
    def extract_domain(url: str) -> str:
        """Extract domain from URL."""
        if not url:
            return ''
        
        # Remove protocol
        if '://' in url:
            url = url.split('://', 1)[1]
        
        # Get domain
        domain = url.split('/', 1)[0]
        
        # Remove port
        domain = domain.split(':', 1)[0]
        
        # Remove www.
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return domain.lower()
    
    def process_record(
        self,
        text: str,
        source_name: str,
        source_url: str,
        license: str,
        publication_date: Optional[str] = None,
        version: Optional[str] = None,
        transformations: Optional[List[str]] = None,
        quality_flags: Optional[List[str]] = None
    ) -> Optional[ProvenanceRecord]:
        """Process a single text record into a cleaned corpus entry."""
        self.stats['total_input'] += 1
        
        # Compute content hash (before cleaning)
        content_sha256 = self.sha256(text)
        
        # Check for duplicate
        if self.deduplicate and content_sha256 in self.seen_hashes:
            self.stats['rejected_duplicate'] += 1
            self.stats['rejected'] += 1
            return None
        
        # Detect language
        language = self.detect_language(text)
        
        # Detect domain
        domain = self.detect_domain(source_url) if source_url else 'unknown'
        
        # Clean text
        cleaned_text = self.clean_text(text, language)
        
        # Recompute hash after cleaning
        cleaned_sha256 = self.sha256(cleaned_text)
        
        # Check license
        if license and license not in self.allowed_licenses:
            self.stats['rejected_license'] += 1
            self.stats['rejected'] += 1
            return None
        
        # Check length after cleaning
        if len(cleaned_text) < self.min_length:
            self.stats['rejected_short'] += 1
            self.stats['rejected'] += 1
            return None
        
        if len(cleaned_text) > self.max_length:
            self.stats['rejected_long'] += 1
            self.stats['rejected'] += 1
            return None
        
        # Track seen hashes
        if self.deduplicate:
            self.seen_hashes.add(cleaned_sha256)
        
        # Create record
        record = ProvenanceRecord(
            document_id=cleaned_sha256,
            source_name=source_name,
            source_url=source_url,
            license=license,
            content_sha256=cleaned_sha256,
            language=language,
            domain=domain,
            text=cleaned_text,
            publication_date=publication_date,
            version=version,
            transformations=transformations or ['cleaned', 'pii_removed', 'normalized'],
            quality_flags=quality_flags,
        )
        
        self.stats['accepted'] += 1
        return record
    
    def process_file(
        self,
        file_path: Path,
        source_name: str,
        source_url: str,
        license: str,
        publication_date: Optional[str] = None,
        version: Optional[str] = None
    ) -> List[ProvenanceRecord]:
        """Process a single file into corpus records."""
        records = []
        
        # Read file
        try:
            content = file_path.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)
            return records
        
        # Split into documents if it's a JSONL file
        if file_path.suffix == '.jsonl':
            lines = content.strip().split('\n')
            for line in lines:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    text = data.get('text', data.get('content', ''))
                    record = self.process_record(
                        text=text,
                        source_name=source_name,
                        source_url=data.get('source_url', source_url),
                        license=data.get('license', license),
                        publication_date=data.get('publication_date', publication_date),
                        version=data.get('version', version)
                    )
                    if record:
                        records.append(record)
                except json.JSONDecodeError:
                    # Try to process as plain text
                    record = self.process_record(
                        text=line,
                        source_name=source_name,
                        source_url=source_url,
                        license=license,
                        publication_date=publication_date,
                        version=version
                    )
                    if record:
                        records.append(record)
        else:
            # Process as single document
            record = self.process_record(
                text=content,
                source_name=source_name,
                source_url=source_url,
                license=license,
                publication_date=publication_date,
                version=version
            )
            if record:
                records.append(record)
        
        return records
    
    def process_directory(
        self,
        dir_path: Path,
        source_name: str,
        source_url_prefix: str,
        license: str,
        domain: str = 'general'
    ) -> List[ProvenanceRecord]:
        """Process all files in a directory."""
        records = []
        
        for file_path in dir_path.rglob('*'):
            if not file_path.is_file():
                continue
            
            # Skip hidden files
            if file_path.name.startswith('.'):
                continue
            
            # Build source URL
            source_url = f"{source_url_prefix.rstrip('/')}/{file_path.relative_to(dir_path)}"
            
            # Process file
            file_records = self.process_file(
                file_path=file_path,
                source_name=source_name,
                source_url=source_url,
                license=license,
                domain=domain
            )
            records.extend(file_records)
        
        return records
    
    def save_corpus(
        self,
        records: List[ProvenanceRecord],
        output_path: Path,
        manifest_path: Optional[Path] = None
    ):
        """Save cleaned corpus to JSONL file."""
        # Write corpus
        with output_path.open('w', encoding='utf-8') as f:
            for record in records:
                f.write(record.to_jsonl() + '\n')
        
        # Write manifest
        if manifest_path:
            manifest = {
                'created_at': datetime.now(timezone.utc).isoformat(),
                'total_records': len(records),
                'stats': self.stats,
                'sha256': self.sha256_bytes(output_path.read_bytes()),
                'config': {
                    'min_length': self.min_length,
                    'max_length': self.max_length,
                    'remove_pii': self.remove_pii,
                    'deduplicate': self.deduplicate,
                    'allowed_licenses': list(self.allowed_licenses),
                    'allowed_domains': list(self.allowed_domains),
                }
            }
            with manifest_path.open('w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def save_rejected(self, rejected: List[Dict], output_path: Path):
        """Save rejected records with reasons."""
        with output_path.open('w', encoding='utf-8') as f:
            for record in rejected:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Clean corpus with provenance tracking for NiyahMini'
    )
    parser.add_argument(
        '--input', type=Path, required=True,
        help='Input file or directory'
    )
    parser.add_argument(
        '--output', type=Path, required=True,
        help='Output JSONL file'
    )
    parser.add_argument(
        '--manifest', type=Path,
        help='Output manifest file (optional)'
    )
    parser.add_argument(
        '--rejected', type=Path,
        help='Output rejected records file (optional)'
    )
    parser.add_argument(
        '--source-name', required=True,
        help='Name of the source'
    )
    parser.add_argument(
        '--source-url',
        help='Base URL for the source'
    )
    parser.add_argument(
        '--license', required=True,
        help='License for the content'
    )
    parser.add_argument(
        '--domain',
        help='Domain of the content'
    )
    parser.add_argument(
        '--min-length', type=int, default=100,
        help='Minimum text length'
    )
    parser.add_argument(
        '--max-length', type=int, default=100000,
        help='Maximum text length'
    )
    parser.add_argument(
        '--no-pii', action='store_false', dest='remove_pii',
        help='Disable PII removal'
    )
    parser.add_argument(
        '--no-deduplicate', action='store_false', dest='deduplicate',
        help='Disable deduplication'
    )
    args = parser.parse_args()
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize cleaner
    cleaner = CorpusCleaner(
        output_dir=args.output.parent,
        min_length=args.min_length,
        max_length=args.max_length,
        remove_pii=args.remove_pii,
        deduplicate=args.deduplicate
    )
    
    # Process input
    if args.input.is_file():
        records = cleaner.process_file(
            file_path=args.input,
            source_name=args.source_name,
            source_url=args.source_url or '',
            license=args.license,
            domain=args.domain
        )
    else:
        records = cleaner.process_directory(
            dir_path=args.input,
            source_name=args.source_name,
            source_url_prefix=args.source_url or '',
            license=args.license,
            domain=args.domain or 'general'
        )
    
    # Save corpus
    output_path = cleaner.save_corpus(
        records=records,
        output_path=args.output,
        manifest_path=args.manifest
    )
    
    print(f"Processed {cleaner.stats['total_input']} inputs")
    print(f"Accepted: {cleaner.stats['accepted']}")
    print(f"Rejected: {cleaner.stats['rejected']}")
    print(f"  - Duplicates: {cleaner.stats['rejected_duplicate']}")
    print(f"  - License: {cleaner.stats['rejected_license']}")
    print(f"  - Too short: {cleaner.stats['rejected_short']}")
    print(f"  - Too long: {cleaner.stats['rejected_long']}")
    print(f"Saved corpus to {output_path}")
    
    # Save manifest if requested
    if args.manifest:
        print(f"Saved manifest to {args.manifest}")


if __name__ == '__main__':
    main()
