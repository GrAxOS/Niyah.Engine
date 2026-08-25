#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_graph(chunks_dir: Path, output_path: Path, manifest_path: Path | None) -> dict:
    """Merge all chunk files into a single canonical graph."""
    full = {"nodes": [], "edges": [], "evidence": [], "schemas": [], "constraints": []}
    manifest = {"chunks": [], "errors": []}
    
    chunk_files = sorted(chunks_dir.glob("khawrizm_graph_chunk_*.json"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {chunks_dir}")
    
    for f in chunk_files:
        try:
            text = f.read_text(encoding="utf-8-sig")
            if not text.strip() or len(text) < 20:
                manifest["errors"].append({"file": f.name, "error": "empty_or_too_small"})
                continue
            
            d = json.loads(text)
            g = d.get("graph", {})
            
            full["nodes"].extend(g.get("nodes", []))
            full["edges"].extend(g.get("edges", []))
            full["evidence"].extend(g.get("evidence", []))
            full["schemas"].extend(g.get("schemas", []))
            full["constraints"].extend(g.get("constraints", []))
            
            sha = hashlib.sha256(text.encode()).hexdigest()[:12]
            manifest["chunks"].append({
                "id": f.name,
                "nodes": len(g.get("nodes", [])),
                "edges": len(g.get("edges", [])),
                "sha": sha,
            })
            print(f"[OK] {f.name}: {len(g.get('nodes', []))} nodes, {len(g.get('edges', []))} edges")
        except json.JSONDecodeError as e:
            manifest["errors"].append({"file": f.name, "error": f"json_decode: {e}"})
            print(f"[SKIP] {f.name}: JSON decode error: {e}", file=sys.stderr)
        except Exception as e:
            manifest["errors"].append({"file": f.name, "error": str(e)})
            print(f"[SKIP] {f.name}: {e}", file=sys.stderr)
    
    print(f"\nTOTAL: {len(full['nodes'])} nodes, {len(full['edges'])} edges, {len(full['evidence'])} evidence")
    print(f"Processed {len(manifest['chunks'])} chunks, {len(manifest['errors'])} errors")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote canonical graph to {output_path}")
    
    if manifest_path:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote manifest to {manifest_path}")
    
    return full


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge graph chunks into canonical graph")
    parser.add_argument("--chunks-dir", default=str(ROOT / "chunks"), help="Directory containing chunk files")
    parser.add_argument("--output", default=str(ROOT / "data" / "canonical_graph_v2.json"), help="Output canonical graph file")
    parser.add_argument("--manifest", default=str(ROOT / "data" / "build_manifest.json"), help="Output manifest file")
    args = parser.parse_args()
    
    try:
        build_graph(
            chunks_dir=Path(args.chunks_dir),
            output_path=Path(args.output),
            manifest_path=Path(args.manifest) if args.manifest else None,
        )
        return 0
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())