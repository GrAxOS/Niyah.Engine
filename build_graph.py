
import json, glob, pathlib
base = pathlib.Path("chunks")
full_graph = {"nodes":[], "edges":[], "evidence":[], "schemas":[], "constraints":[]}
for f in sorted(base.glob("*.json")):
    try:
        data = json.loads(f.read_text(encoding='utf-8'))
        g = data.get("graph", {})
        full_graph["nodes"].extend(g.get("nodes", []))
        full_graph["edges"].extend(g.get("edges", []))
        full_graph["evidence"].extend(g.get("evidence", []))
        full_graph["schemas"].extend(g.get("schemas", []))
        full_graph["constraints"].extend(g.get("constraints", []))
        print(f"[OK] {f.name} -> nodes:{len(g.get('nodes',[]))} edges:{len(g.get('edges',[]))}")
    except Exception as e:
        print(f"[ERR] {f.name}: {e}")
print(f"\nTOTAL -> nodes:{len(full_graph['nodes'])} edges:{len(full_graph['edges'])}")
pathlib.Path("full_graph_built.json").write_text(json.dumps(full_graph, ensure_ascii=False, indent=2), encoding='utf-8')
