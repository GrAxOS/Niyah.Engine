from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
LOG_DIR = PUBLIC / "logs"

AUDITS = [
    ("knowledge_graph_repo", "scripts/audit_knowledge_graph_repo.py"),
    ("json_folder_structure_diff", "scripts/audit_json_folder_structure_diff.py"),
    ("backups_structure", "scripts/audit_backups_structure.py"),
    ("knowledge_payload_quality", "scripts/audit_knowledge_payload_quality.py"),
    ("verified_lessons", "scripts/list_verified_lessons.py"),
]

def git(args):
    try:
        p = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True,
                           encoding="utf-8", errors="replace")
        return p.stdout.strip() if p.returncode == 0 else ""
    except Exception:
        return ""

def run_audits():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name, script in AUDITS:
        start = time.perf_counter()
        path = ROOT / script
        if not path.exists():
            result = {"name": name, "ok": False, "returncode": 127,
                      "duration_seconds": 0, "combined": f"Missing: {script}"}
        else:
            p = subprocess.run([sys.executable, script], cwd=ROOT, text=True,
                               capture_output=True, encoding="utf-8", errors="replace")
            combined = p.stdout or ""
            if p.stderr:
                combined += "\n" + p.stderr
            result = {"name": name, "ok": p.returncode == 0,
                      "returncode": p.returncode,
                      "duration_seconds": round(time.perf_counter() - start, 3),
                      "combined": combined}
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        (LOG_DIR / f"{safe}.log").write_text(result["combined"], encoding="utf-8")
        results.append(result)
    return results

def analyze():
    roots = ["chunks","data","knowledge","schema","manifests","sources","audits","normalized"]
    files = []
    for d in roots:
        base = ROOT / d
        if base.exists():
            files.extend(base.rglob("*.json"))
    files = sorted(set(files))

    parse_errors = []
    node_ids, edge_ids, evidence_ids = Counter(), Counter(), Counter()
    nodes, edges, evidence = [], [], []
    schemas = Counter()
    graph_files = empty_graph_files = 0

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        try:
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            parse_errors.append({"file": rel, "error": str(exc)})
            continue

        if not isinstance(obj, dict):
            continue

        schema = obj.get("schema")
        version = schema.get("version") if isinstance(schema, dict) else obj.get("schema_version")
        if version is not None:
            schemas[str(version)] += 1

        graph = obj.get("graph")
        if not isinstance(graph, dict):
            if any(isinstance(obj.get(k), list) for k in ("nodes","edges","evidence")):
                graph = obj
            else:
                continue

        graph_files += 1
        ns = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
        es = graph.get("edges") if isinstance(graph.get("edges"), list) else []
        evs = graph.get("evidence") if isinstance(graph.get("evidence"), list) else []

        if not ns and not es and not evs:
            empty_graph_files += 1

        for n in ns:
            if not isinstance(n, dict) or n.get("id") is None:
                continue
            nid = str(n["id"])
            node_ids[nid] += 1
            nodes.append({
                "id": nid,
                "label": str(n.get("label") or n.get("name") or n.get("title") or nid),
                "kind": str(n.get("type") or n.get("kind") or "node"),
                "file": rel
            })

        for e in es:
            if not isinstance(e, dict):
                continue
            eid = e.get("id")
            if eid is not None:
                edge_ids[str(eid)] += 1
            edges.append({
                "id": str(eid) if eid is not None else "",
                "source": str(e.get("source") or e.get("from") or ""),
                "target": str(e.get("target") or e.get("to") or ""),
                "label": str(e.get("label") or e.get("type") or e.get("relation") or "edge"),
                "file": rel
            })

        for ev in evs:
            if not isinstance(ev, dict):
                continue
            evid = ev.get("id")
            if evid is not None:
                evidence_ids[str(evid)] += 1
            evidence.append({
                "id": str(evid) if evid is not None else "",
                "file": rel,
                "text": str(ev.get("text") or ev.get("description") or ev.get("claim") or "")[:500]
            })

    node_set = set(node_ids)
    dangling = [e for e in edges if e["source"] and e["target"] and
                (e["source"] not in node_set or e["target"] not in node_set)]

    result = {
        "json_files": len(files),
        "parse_errors": parse_errors,
        "graph_files": graph_files,
        "empty_graph_files": empty_graph_files,
        "node_occurrences": len(nodes),
        "unique_nodes": len(node_ids),
        "edge_occurrences": len(edges),
        "evidence_occurrences": len(evidence),
        "duplicate_node_ids": {k:v for k,v in node_ids.items() if v > 1},
        "duplicate_edge_ids": {k:v for k,v in edge_ids.items() if v > 1},
        "duplicate_evidence_ids": {k:v for k,v in evidence_ids.items() if v > 1},
        "dangling_edges": dangling,
        "schema_versions": dict(schemas),
        "node_kind_counts": dict(Counter(n["kind"] for n in nodes)),
        "nodes": nodes[:1500],
        "edges": edges[:3000],
        "evidence": evidence[:1500],
    }
    return result

def build():
    PUBLIC.mkdir(parents=True, exist_ok=True)
    audits = run_audits()
    g = analyze()

    state = "PASS"
    if g["parse_errors"] or any(not a["ok"] for a in audits):
        state = "FAIL"
    elif g["duplicate_node_ids"] or g["duplicate_edge_ids"] or g["dangling_edges"]:
        state = "FAIL"

    commit = git(["rev-parse","HEAD"]) or os.getenv("CI_COMMIT_SHA","")
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "snapshot_id": "",
        "target": ".",
        "environment": {
            "ci": bool(os.getenv("CI")),
            "project": os.getenv("CI_PROJECT_PATH",""),
            "branch": git(["branch","--show-current"]) or os.getenv("CI_COMMIT_REF_NAME",""),
            "commit": commit,
            "pipeline_id": os.getenv("CI_PIPELINE_ID",""),
            "job_id": os.getenv("CI_JOB_ID",""),
            "pipeline_url": os.getenv("CI_PIPELINE_URL",""),
            "job_url": os.getenv("CI_JOB_URL",""),
        },
        "working_tree": {
            "clean": not bool(git(["status","--short"]).strip()),
            "status": git(["status","--short"])
        },
        "files": {
            "json_files": g["json_files"],
            "parse_errors": len(g["parse_errors"])
        },
        "graph": {k:g[k] for k in (
            "graph_files","empty_graph_files","node_occurrences","unique_nodes",
            "edge_occurrences","evidence_occurrences","duplicate_node_ids",
            "duplicate_edge_ids","duplicate_evidence_ids","dangling_edges",
            "schema_versions","node_kind_counts","nodes","edges","evidence"
        )},
        "audit_runs": audits,
        "parse_errors": g["parse_errors"]
    }
    payload["snapshot_id"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",",":")).encode()
    ).hexdigest()[:24]

    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    (PUBLIC / "khz_realtime_bundle.json").write_text(text, encoding="utf-8")
    (PUBLIC / "khz_realtime_snapshot.json").write_text(text, encoding="utf-8")

    return payload

HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KHZ Realtime A-Z</title>
<style>
:root{--bg:#02050a;--p:#07111f;--l:#1c3557;--t:#e6f0ff;--m:#8296b5;--g:#22c55e;--r:#ef4444;--i:#38bdf8}
*{box-sizing:border-box}html,body{margin:0;background:#02050a;color:var(--t);font-family:Consolas,monospace}body{padding:16px}
.wrap{max-width:1800px;margin:auto}.head,.card{border:1px solid var(--l);border-radius:14px;background:rgba(6,12,22,.93)}.head{padding:15px;display:flex;justify-content:space-between;position:sticky;top:8px;z-index:5}.title{color:#7dd3fc;font-size:18px}.sub,.muted{color:var(--m);font-size:9px;margin-top:5px}.live{color:#86efac;font-size:9px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:12px}.card{padding:13px}.metric{font-size:28px;font-weight:700}.pass{color:var(--g)}.fail{color:var(--r)}
.meta{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.meta div{padding:8px;border:1px solid #263d5f;border-radius:8px;background:var(--p)}.meta span{display:block;color:var(--m);font-size:9px}.meta b{font-size:10px;overflow-wrap:anywhere}
#orbit{width:100%;height:600px;background:#01030a;border-radius:10px}.table{max-height:430px;overflow:auto;border:1px solid #203b5d;border-radius:9px}table{border-collapse:collapse;width:100%;font-size:9px}th,td{padding:7px;border-bottom:1px solid #182a43;text-align:left;vertical-align:top}th{position:sticky;top:0;background:var(--p);color:#93c5fd}
.toolbar{display:flex;gap:8px}.toolbar input,.toolbar select,.toolbar button{background:#06101d;border:1px solid var(--l);color:var(--t);padding:8px;border-radius:7px}.toolbar input{flex:1}.log{white-space:pre-wrap;max-height:430px;overflow:auto;background:#01040a;padding:10px;border-radius:8px;font-size:9px;line-height:1.5}
#details{position:absolute;left:12px;bottom:12px;max-width:520px;padding:10px;border:1px solid #2b6790;border-radius:9px;background:rgba(2,6,23,.9);display:none}.open{display:block!important}.kind{color:#7dd3fc;font-size:10px}.src{color:#64748b;font-size:9px}.txt{font-size:10px;margin-top:7px}
@media(max-width:950px){.grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:600px){.grid{grid-template-columns:1fr}body{padding:8px}}
</style></head>
<body><div class="wrap">
<div class="head"><div><div class="title">KHZ REALTIME A-Z KNOWLEDGE ANALYSIS</div><div class="sub">GitLab CI snapshot · browser refresh every 10s</div></div><div class="live">● <span id="live">LIVE</span></div></div>
<div class="grid"><div class="card"><div class="muted">STATE</div><div id="state" class="metric">LOADING</div><div id="state2" class="muted"></div></div><div class="card"><div class="muted">UNIQUE NODES</div><div id="nodes" class="metric">0</div></div><div class="card"><div class="muted">EDGES</div><div id="edges" class="metric">0</div></div><div class="card"><div class="muted">EVIDENCE</div><div id="evidence" class="metric">0</div></div></div>
<div class="card" style="margin-top:12px"><div class="muted">PIPELINE / REPOSITORY</div><div id="meta" class="meta"></div></div>
<div class="card" style="margin-top:12px"><div class="muted">KNOWLEDGE ORBIT</div><div style="position:relative"><canvas id="orbit"></canvas><div id="details"><div id="dtitle"></div><div id="dkind" class="kind"></div><div id="dsrc" class="src"></div><div id="dtext" class="txt"></div></div></div></div>
<div class="card" style="margin-top:12px"><div class="muted">A-Z</div><div class="table"><table><thead><tr><th>Area</th><th>Value</th><th>Meaning</th></tr></thead><tbody id="az"></tbody></table></div></div>
<div class="card" style="margin-top:12px"><div class="muted">AUDITS</div><div class="table"><table><thead><tr><th>Name</th><th>RC</th><th>State</th><th>Sec</th></tr></thead><tbody id="audits"></tbody></table></div></div>
<div class="card" style="margin-top:12px"><div class="muted">ISSUES</div><div class="table"><table><thead><tr><th>Category</th><th>Count</th><th>Details</th></tr></thead><tbody id="issues"></tbody></table></div></div>
<div class="card" style="margin-top:12px"><div class="muted">LOG STREAM</div><div class="toolbar"><input id="filter" placeholder="filter logs"><select id="sel"></select><button id="refresh">refresh</button></div><div id="log" class="log" style="margin-top:8px"></div></div>
</div>
<script>
const U="./khz_realtime_bundle.json";let B=null,N=[],P=[],S=null,T=0,LF="",LN="";
const $=x=>document.getElementById(x),esc=x=>String(x??"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;"),col=k=>({verified_lesson:"#22c55e",atomic_fact:"#86efac",taxonomy:"#38bdf8",staging:"#f59e0b",reserved:"#64748b",source:"#a78bfa",chunk:"#ef4444",schema:"#eab308",data:"#f472b6"}[k]||"#cbd5e1");
async function load(){let r=await fetch(U+"?t="+Date.now(),{cache:"no-store"});if(!r.ok)throw Error("HTTP "+r.status);return r.json()}
function render(){let g=B.graph,e=B.environment||{};$("state").textContent=B.state;$("state").className="metric "+(B.state==="PASS"?"pass":"fail");$("state2").textContent=`${B.files.json_files} JSON · ${B.files.parse_errors} parse errors`;$("nodes").textContent=g.unique_nodes;$("edges").textContent=g.edge_occurrences;$("evidence").textContent=g.evidence_occurrences;$("live").textContent="LIVE "+B.snapshot_id;
$("meta").innerHTML=[["target",B.target],["project",e.project],["branch",e.branch],["commit",e.commit],["pipeline",e.pipeline_id],["job",e.job_id],["generated",B.generated_utc],["tree",B.working_tree.clean?"CLEAN":"DIRTY"]].map(x=>`<div><span>${esc(x[0])}</span><b>${esc(x[1])}</b></div>`).join("");
let rows=[["A Audits",B.audit_runs.length,"audit programs"],["B Branch",e.branch,"branch"],["C Commit",e.commit,"revision"],["D Documents",B.files.json_files,"JSON files"],["E Evidence",g.evidence_occurrences,"evidence"],["F Graph files",g.graph_files,"graph files"],["G Nodes",g.node_occurrences,"node occurrences"],["H Schemas",Object.keys(g.schema_versions).length,"versions"],["I Duplicate IDs",Object.keys(g.duplicate_node_ids).length,"duplicate node IDs"],["J JSON errors",B.files.parse_errors,"parse errors"],["K Dangling",g.dangling_edges.length,"missing endpoints"],["L Empty graphs",g.empty_graph_files,"empty"],["M Snapshot",B.snapshot_id,"fingerprint"],["N Unique",g.unique_nodes,"unique IDs"],["O Edges",g.edge_occurrences,"occurrences"],["P Pipeline",e.pipeline_id,"CI"],["Q Quality",B.state,"state"],["R Repo",B.target,"target"],["S Schema",JSON.stringify(g.schema_versions),"versions"],["T Time",B.generated_utc,"UTC"],["U Tree",B.working_tree.clean?"CLEAN":"DIRTY","git status"],["V Verified lessons",(B.audit_runs.find(x=>x.name==="verified_lessons")||{}).ok?"PASS":"FAIL","audit"],["W Web","GitLab Pages","delivery"],["X Execution","CI + polling","model"],["Y Yield","graph + logs","data"],["Z Source","repository","truth source"]];$("az").innerHTML=rows.map(r=>`<tr><td>${esc(r[0])}</td><td>${esc(r[1])}</td><td>${esc(r[2])}</td></tr>`).join("");
$("audits").innerHTML=B.audit_runs.map(a=>`<tr><td>${esc(a.name)}</td><td>${a.returncode}</td><td class="${a.ok?"pass":"fail"}">${a.ok?"PASS":"FAIL"}</td><td>${a.duration_seconds}</td></tr>`).join("");
let is=[["parse errors",B.files.parse_errors,B.parse_errors],["duplicate nodes",Object.keys(g.duplicate_node_ids).length,g.duplicate_node_ids],["duplicate edges",Object.keys(g.duplicate_edge_ids).length,g.duplicate_edge_ids],["duplicate evidence",Object.keys(g.duplicate_evidence_ids).length,g.duplicate_evidence_ids],["dangling edges",g.dangling_edges.length,g.dangling_edges]];$("issues").innerHTML=is.map(r=>`<tr><td>${esc(r[0])}</td><td>${r[1]}</td><td>${esc(JSON.stringify(r[2]))}</td></tr>`).join("");
$("sel").innerHTML=B.audit_runs.map((a,i)=>`<option value="${i}">${esc(a.name)}</option>`).join("");LN=B.audit_runs[0]?.name||"";showlog();N=g.nodes.map((n,i)=>{let h=hash(n.id+":"+i);return {...n,a:(h%62831)/10000,r:100+(h%700),s:.0007+(h%23)/100000,p:(h%1000)/1000}})}
function hash(s){let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}return h>>>0}
function size(){let d=Math.min(devicePixelRatio||1,2),c=$("orbit");c.width=c.clientWidth*d;c.height=c.clientHeight*d}addEventListener("resize",size);size();
function draw(){let c=$("orbit"),x=c.getContext("2d"),W=c.width,H=c.height;x.fillStyle="#02050a";x.fillRect(0,0,W,H);x.save();x.globalAlpha=.12;x.strokeStyle="#1e40af";for(let r=100;r<Math.min(W,H)*.48;r+=90){x.beginPath();x.arc(W/2,H/2,r,0,Math.PI*2);x.stroke()}x.restore();P=N.map(n=>{let a=n.a+T*n.s,z=Math.sin(a)*n.r,sc=.76+((z+700)/1400)*.42;return{n,p:{x:W/2+Math.cos(a)*n.r,y:H/2+Math.sin(a*.84+n.p)*n.r*.58,z,sc}}}).sort((a,b)=>a.p.z-b.p.z);for(let q of P){let n=q.n,p=q.p,on=S&&S.id===n.id,cc=col(n.kind),r=(on?7:4)*p.sc;x.save();x.globalAlpha=Math.max(.44,Math.min(1,.65+p.z/1400));x.shadowColor=cc;x.shadowBlur=on?28:14;x.fillStyle=cc;x.beginPath();x.arc(p.x,p.y,r,0,Math.PI*2);x.fill();x.shadowBlur=0;if(on){x.strokeStyle="#fff";x.lineWidth=2;x.stroke()}x.font=`${Math.max(8,10*p.sc)}px Consolas`;x.textAlign="center";x.textBaseline="top";x.fillStyle=on?"#fff":"rgba(226,232,240,.88)";x.fillText(String(n.label||n.id).slice(0,42),p.x,p.y+r+5);x.font=`${Math.max(7,8*p.sc)}px Consolas`;x.fillStyle=cc;x.fillText(String(n.kind).slice(0,26),p.x,p.y+r+20);x.restore()}T++;requestAnimationFrame(draw)}
function hit(e){let c=$("orbit"),r=c.getBoundingClientRect(),sx=c.width/r.width,sy=c.height/r.height,xx=(e.clientX-r.left)*sx,yy=(e.clientY-r.top)*sy,b=null,d=1e9;for(let q of P){let dd=Math.hypot(xx-q.p.x,yy-q.p.y);if(dd<24*q.p.sc&&dd<d){b=q.n;d=dd}}return b}
$("orbit").onclick=e=>{let n=hit(e);S=n&&S&&S.id===n.id?null:n||S;if(S){$("details").classList.add("open");$("dtitle").textContent=S.label||S.id;$("dkind").textContent=S.kind||"";$("dsrc").textContent=S.file||"";$("dtext").textContent=S.id||""}else $("details").classList.remove("open")};
$("sel").onchange=e=>{LN=B.audit_runs[+e.target.value]?.name||"";showlog()};$("filter").oninput=e=>{LF=e.target.value;showlog()};$("refresh").onclick=async()=>{B=await load();render()};function showlog(){let a=B?.audit_runs?.find(x=>x.name===LN)||B?.audit_runs?.[0],s=a?.combined||"";if(LF)s=s.split("\\n").filter(x=>x.toLowerCase().includes(LF.toLowerCase())).join("\\n");$("log").textContent=s||"NO LOG"}
(async()=>{B=await load();render();draw();setInterval(async()=>{try{let n=await load();if(n.snapshot_id!==B.snapshot_id){B=n;render()}}catch{}},10000)})()
</script></body></html>"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    payload = build()
    if not args.verify_only:
        (PUBLIC / "index.html").write_text(HTML, encoding="utf-8")
    print(json.dumps({
        "state": payload["state"],
        "snapshot_id": payload["snapshot_id"],
        "json_files": payload["files"]["json_files"],
        "nodes": payload["graph"]["unique_nodes"],
        "edges": payload["graph"]["edge_occurrences"],
        "evidence": payload["graph"]["evidence_occurrences"],
        "parse_errors": payload["files"]["parse_errors"],
        "duplicate_nodes": len(payload["graph"]["duplicate_node_ids"]),
        "dangling_edges": len(payload["graph"]["dangling_edges"])
    }, ensure_ascii=False, indent=2))
    return 0 if not payload["files"]["parse_errors"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
