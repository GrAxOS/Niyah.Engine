#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

MIN_CHARS = 400

def normalize(text: str) -> str:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def record(path: Path, source_name: str, source_url_prefix: str, domain: str, language: str, license_name: str):
    raw = path.read_text(encoding='utf-8', errors='replace')
    text = normalize(raw)
    if len(text) < MIN_CHARS:
        return None, 'too_short'
    digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()
    return {
        'document_id': digest,
        'source_name': source_name,
        'source_url': f'{source_url_prefix.rstrip("/")}/{path.name}',
        'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
        'license': license_name,
        'content_sha256': digest,
        'language': language,
        'domain': domain,
        'publication_date': None,
        'version': None,
        'transformations': ['newline_normalization', 'whitespace_normalization'],
        'quality_flags': [],
        'text': text,
    }, None

def main():
    parser = argparse.ArgumentParser(description='Build a provenance-preserving JSONL corpus from approved local files.')
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rejected', type=Path, default=Path('rejected.jsonl'))
    parser.add_argument('--source-name', required=True)
    parser.add_argument('--source-url-prefix', required=True)
    parser.add_argument('--domain', required=True)
    parser.add_argument('--language', default='und')
    parser.add_argument('--license', dest='license_name', required=True)
    args = parser.parse_args()

    accepted = rejected = 0
    with args.output.open('w', encoding='utf-8') as out, args.rejected.open('w', encoding='utf-8') as bad:
        for path in sorted(p for p in args.input.rglob('*') if p.is_file()):
            item, reason = record(path, args.source_name, args.source_url_prefix, args.domain, args.language, args.license_name)
            if item is None:
                bad.write(json.dumps({'path': str(path), 'reason': reason}, ensure_ascii=False) + '\n')
                rejected += 1
                continue
            out.write(json.dumps(item, ensure_ascii=False) + '\n')
            accepted += 1
    print(json.dumps({'accepted': accepted, 'rejected': rejected, 'output': str(args.output)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
