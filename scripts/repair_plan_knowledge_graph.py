import json, hashlib
from pathlib import Path
from collections import defaultdict, Counter

ROOT=Path.cwd()
AUDITS=ROOT/"audits"
AUDITS.mkdir(exist_ok=True)

def load(p):
    try: return json.loads(p.read_text(encoding="utf-8-sig")), None
    except Exception as e: return None, str(e)

nodes=defaultdict(list)
edges=[]
parse_errors=[]

for p in sorted((ROOT/"chunks").glob("*.json")):
    obj,err=load(p)
    rel=str(p.relative_to(ROOT))
    if err:
        parse_errors.append({"file":rel,"error":err})
        continue
    g=obj.get("graph",obj) if isinstance(obj,dict) else {}
    for n in g.get("nodes",[]) or []:
        if isinstance(n,dict) and n.get("id"):
            props=n.get("properties") or {}
            nodes[n["id"]].append({"file":rel,"type":n.get("type"),"label":n.get("label") or props.get("name") or props.get("label"),"sha256":hashlib.sha256(json.dumps(n,sort_keys=True,ensure_ascii=False).encode()).hexdigest()})
    for e in g.get("edges",[]) or []:
        if isinstance(e,dict):
            edges.append({"file":rel,"id":e.get("id"),"source":e.get("source"),"target":e.get("target"),"relation":e.get("relation")})

node_ids=set(nodes.keys())
duplicate_nodes={k:v for k,v in nodes.items() if len(v)>1}
dangling=[]
for e in edges:
    ms=e.get("source") not in node_ids
    mt=e.get("target") not in node_ids
    if ms or mt:
        dangling.append({**e,"missing_source":ms,"missing_target":mt})

plan={
  "summary":{
    "parse_errors":len(parse_errors),
    "unique_nodes":len(node_ids),
    "edges":len(edges),
    "duplicate_node_groups":len(duplicate_nodes),
    "duplicate_node_occurrences":sum(len(v) for v in duplicate_nodes.values()),
    "dangling_edges":len(dangling)
  },
  "parse_errors":parse_errors,
  "duplicate_nodes":[{"id":k,"canonical":v[0],"duplicates":v[1:]} for k,v in sorted(duplicate_nodes.items())],
  "dangling_edges":dangling,
  "safe_next_action":"Do not mutate chunks automatically. Review duplicate canonical choices and add missing standard nodes or retarget dangling edges explicitly."
}
(AUDITS/"knowledge_graph_repair_plan.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8")

md=["# KHZ Graph Repair Plan","","## Summary"]
for k,v in plan["summary"].items(): md.append(f"- {k}: {v}")
md+=["","## Dangling edges"]
for x in dangling: md.append(f"- file={x["file"]} id={x.get("id")} source={x.get("source")} target={x.get("target")} missing_source={x["missing_source"]} missing_target={x["missing_target"]}")
md+=["","## Duplicate node groups"]
for x in plan["duplicate_nodes"]: md.append(f"- {x["id"]}: occurrences={1+len(x["duplicates"])} canonical_file={x["canonical"]["file"]}")
(AUDITS/"knowledge_graph_repair_plan.md").write_text("\n".join(md),encoding="utf-8")
print(json.dumps(plan["summary"],ensure_ascii=False,indent=2))
print("audit=audits/knowledge_graph_repair_plan.json")
print("report=audits/knowledge_graph_repair_plan.md")
