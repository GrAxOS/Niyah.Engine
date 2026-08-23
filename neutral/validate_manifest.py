#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

REQUIRED = {'document_id', 'source_name', 'source_url', 'retrieved_at_utc', 'license', 'content_sha256', 'language', 'domain', 'transformations', 'quality_flags', 'text'}

def main():
    parser = argparse.ArgumentParser(description='Validate a Niyah.Neutral JSONL corpus manifest.')
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    failures = []
    total = 0
    for line_no, line in enumerate(args.manifest.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        total += 1
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append((line_no, f'invalid_json: {exc.msg}'))
            continue
        missing = REQUIRED - item.keys()
        if missing:
            failures.append((line_no, f'missing_fields: {sorted(missing)}'))
            continue
        digest = hashlib.sha256(item['text'].encode('utf-8')).hexdigest()
        if not item['license']:
            failures.append((line_no, 'missing_license'))
        if not item['source_url']:
            failures.append((line_no, 'missing_source_url'))
        if item['document_id'] != item['content_sha256']:
            failures.append((line_no, 'document_id_mismatch'))
        if digest == item['content_sha256']:
            failures.append((line_no, 'hash_is_normalized_text_not_raw_source'))
    print(json.dumps({'records': total, 'failures': failures, 'valid': not failures}, ensure_ascii=False))
    raise SystemExit(0 if not failures else 1)

if __name__ == '__main__':
    main()
