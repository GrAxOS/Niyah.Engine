#!/usr/bin/env python3
"""Unit tests for Niyah.Neutral pipeline."""
import json
import pytest
import hashlib
from pathlib import Path
from neutral.clean_corpus import normalize, record
from neutral.validate_manifest import REQUIRED

def test_normalize():
    """Test text normalization."""
    assert normalize('Hello\r\nWorld') == 'Hello\nWorld'
    assert normalize('Hello   World') == 'Hello World'
    assert normalize('Hello\n\n\nWorld') == 'Hello\n\nWorld'

def test_record_min_length():
    """Test that records below MIN_CHARS are rejected."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write('Too short')
        f.flush()
        item, reason = record(Path(f.name), 'test', 'https://example.com', 'test', 'en', 'MIT')
        assert item is None
        assert reason == 'too_short'

def test_record_success():
    """Test successful record creation."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write('A' * 500)  # Above MIN_CHARS
        f.flush()
        item, reason = record(Path(f.name), 'test', 'https://example.com', 'test', 'en', 'MIT')
        assert item is not None
        assert 'document_id' in item
        assert 'content_sha256' in item
        assert item['source_name'] == 'test'

def test_validate_manifest_required_fields():
    """Test that all REQUIRED fields are present."""
    assert 'document_id' in REQUIRED
    assert 'source_url' in REQUIRED
    assert 'license' in REQUIRED
    assert 'content_sha256' in REQUIRED

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
