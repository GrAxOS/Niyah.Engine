#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export_graphml(graph: dict, output_path: Path) -> None:
    """Export graph to GraphML format."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"',
        '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns',
        '         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">',
        '  <key id="label" for="node" attr.name="label" attr.type="string"/>',
        '  <key id="type" for="node" attr.name="type" attr.type="string"/>',
        '  <key id="edge_type" for="edge" attr.name="type" attr.type="string"/>',
        '  <graph id="G" edgedefault="directed">',
    ]
    
    for node in nodes:
        node_id = node.get("id", "")
        label = node.get("label", "").replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        node_type = node.get("type", "")
        lines.append(f'    <node id="{node_id}">')
        lines.append(f'      <data key="label">{label}</data>')
        lines.append(f'      <data key="type">{node_type}</data>')
        lines.append('    </node>')
    
    for i, edge in enumerate(edges):
        source = edge.get("source", "")
        target = edge.get("target", "")
        edge_type = edge.get("type", "")
        edge_id = edge.get("id", f"e{i}")
        lines.append(f'    <edge id="{edge_id}" source="{source}" target="{target}">')
        lines.append(f'      <data key="edge_type">{edge_type}</data>')
        lines.append('    </edge>')
    
    lines.append('  </graph>')
    lines.append('</graphml>')
    
    output_path.write_text('\n'.join(lines) + '\n', encoding="utf-8")


def export_cytoscape(graph: dict, output_path: Path) -> None:
    """Export graph to Cytoscape.js JSON format."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    cyto = {
        "elements": {
            "nodes": [
                {
                    "data": {
                        "id": n.get("id"),
                        "label": n.get("label"),
                        "type": n.get("type"),
                    }
                }
                for n in nodes
            ],
            "edges": [
                {
                    "data": {
                        "id": e.get("id"),
                        "source": e.get("source"),
                        "target": e.get("target"),
                        "type": e.get("type"),
                    }
                }
                for e in edges
            ],
        }
    }
    
    output_path.write_text(json.dumps(cyto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export graph to various formats")
    parser.add_argument("--graph", default=str(ROOT / "data" / "canonical_graph_v2.json"))
    parser.add_argument("--format", choices=["graphml", "cytoscape"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    try:
        graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.format == "graphml":
            export_graphml(graph, output_path)
        elif args.format == "cytoscape":
            export_cytoscape(graph, output_path)
        
        print(f"Exported {args.format} to {output_path}")
        return 0
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
