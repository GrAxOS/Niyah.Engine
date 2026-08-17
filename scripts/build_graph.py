import json, glob, pathlib, hashlib
base = pathlib.Path("chunks")
full={"nodes":[],"edges":[],"evidence":[],"schemas":[],"constraints":[]}
manifest={"chunks":[]}
for f in sorted(base.glob("khawrizm_graph_chunk_*.json")):
  try:
    text=f.read_text(encoding="utf-8-sig")
    if not text.strip() or len(text) < 20: continue
    d=json.loads(text)
    g=d.get("graph",{})
    full["nodes"].extend(g.get("nodes",[]))
    full["edges"].extend(g.get("edges",[]))
    full["evidence"].extend(g.get("evidence",[]))
    full["schemas"].extend(g.get("schemas",[]))
    full["constraints"].extend(g.get("constraints",[]))
    sha=hashlib.sha256(text.encode()).hexdigest()[:12]
    manifest["chunks"].append({"id":f.name,"nodes":len(g.get("nodes",[])),"sha":sha})
    print(f"[OK] {f.name}")
  except Exception as e:
    print(f"[SKIP] {f.name}: {e}")
print(f"\nTOTAL nodes:{len(full['nodes'])} edges:{len(full['edges'])}")