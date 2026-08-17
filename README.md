# Khawrizm Sovereign Graph

Sovereign, bias-null knowledge graph — Fitrah-aligned, Zero-Cloud validation.

## Structure
```
khawrizm-sovereign-graph/
├── README.md
├── schema/
│   └── sovereign_knowledge_graph_v1.0.0.json
├── chunks/
│   ├── khawrizm_graph_chunk_0001.json
│   ├── khawrizm_graph_chunk_0002.json
│   └── ...
├── manifests/
│   └── graph_manifest.json
└── audits/
    └── audit_log_2026-08-17.json
```

## Build
```python
import json, glob
full_graph = {"nodes":[], "edges":[], "evidence":[], "schemas":[], "constraints":[]}
for f in sorted(glob.glob("chunks/*.json")):
    data = json.load(open(f, encoding='utf-8'))
    g = data.get("graph", {})
    full_graph["nodes"].extend(g.get("nodes", []))
    full_graph["edges"].extend(g.get("edges", []))
    full_graph["evidence"].extend(g.get("evidence", []))

print(f"Total nodes: {len(full_graph['nodes'])}")
```

## Principles
- BIASNULL: true
- FILTERREJECT: COMPANY | STATE | PERSONAL | TELEMETRY
- FITRAH ANCHOR
- Zero-Cloud
