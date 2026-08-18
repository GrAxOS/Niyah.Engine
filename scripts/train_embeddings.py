#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def assert_offline() -> None:
    original = socket.socket
    class BlockedSocket(original):
        def connect(self, *args, **kwargs):
            raise RuntimeError("network egress disabled: embedding training must be local-only")
    socket.socket = BlockedSocket


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dimensions", required=True, type=int)
    parser.add_argument("--version", default="1")
    parser.add_argument("--manifest", default=str(ROOT / "data" / "embedding_space.json"))
    args = parser.parse_args()

    if args.dimensions <= 0:
        raise SystemExit("dimensions must be > 0")

    assert_offline()
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name": args.model_name,
        "dimensions": args.dimensions,
        "version": args.version,
        "distance": "cosine",
        "single_model_per_space": True,
        "network": "disabled",
    }

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("model_name") != args.model_name or existing.get("dimensions") != args.dimensions:
            raise SystemExit("single_model_per_space violation: manifest already binds a different model or dimension")

    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", **payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
