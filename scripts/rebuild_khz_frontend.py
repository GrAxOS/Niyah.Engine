from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
BUNDLE = PUBLIC / "khz_realtime_bundle.json"

if not BUNDLE.exists():
    raise SystemExit("Missing public/khz_realtime_bundle.json")

bundle = json.loads(BUNDLE.read_text(encoding="utf-8-sig"))
bundle_json = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))

html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#050b14">
<title>KHZ Sovereign Graph Control Room</title>
<style>
:root{
--bg:#050a11;
--panel:#09111d;
--panel2:#0c1726;
--line:#1c334d;
--line2:#284766;
--text:#edf5ff;
--muted:#7f95ae;
--cyan:#57d6ff;
--green:#3ddc97;
--red:#ff657b;
--amber:#ffbd55;
--purple:#ad96ff;
}
*{box-sizing:border-box}
html,body{
margin:0;
min-height:100%;
background:
radial-gradient(circle at 20% 0,#102b45 0,transparent 32%),
radial-gradient(circle at 100% 0,#21163d 0,transparent 28%),
linear-gradient(180deg,#07101a 0%,#04080d 55%,#02050a 100%);
color:var(--text);
font-family:Inter,Segoe UI,Arial,sans-serif
}
body{padding:14px}
.shell{max-width:1800px;margin:auto}
.top{
position:sticky;
top:0;
z-index:50;
display:flex;
justify-content:space-between;
align-items:center;
gap:12px;
padding:13px 15px;
border:1px solid var(--line);
border-radius:14px;
background:rgba(4,10,17,.9);
backdrop-filter:blur(18px)
}
.brand{display:flex;gap:11px;align-items:center}
.logo{
width:38px;
height:38px;
display:grid;
place-items:center;
border:1px solid #2b5679;
border-radius:11px;
background:#081522;
color:var(--cyan);
font-weight:800;
font-size:11px
}
.title{
font-size:14px;
font-weight:750;
letter-spacing:.05em
}
.sub{
margin-top:3px;
font-size:9px;
color:var(--muted)
}
.live{
display:flex;
align-items:center;
gap:7px;
padding:6px 10px;
border:1px solid #21445b;
border-radius:999px;
background:#071621;
font-size:9px;
color:#9de8c7
}
.dot{
width:7px;
height:7px;
border-radius:50%;
background:var(--green);
box-shadow:0 0 12px var(--green)
}
.hero{
display:grid;
grid-template-columns:minmax(0,1.5fr) minmax(300px,.65fr);
gap:12px;
margin-top:12px
}
.panel{
border:1px solid var(--line);
border-radius:14px;
background:linear-gradient(180deg,rgba(10,19,31,.96),rgba(7,13,22,.96));
box-shadow:0 18px 55px rgba(0,0,0,.22)
}
.hero-main{padding:18px}
.eyebrow{
font-size:8px;
letter-spacing:.18em;
color:var(--cyan);
text-transform:uppercase
}
h1{
margin:6px 0 0;
font-size:26px;
line-height:1.08
}
.hero-copy{
max-width:760px;
margin:9px 0 0;
color:#abc0d7;
font-size:11px;
line-height:1.65
}
.chips{
display:flex;
flex-wrap:wrap;
gap:6px;
margin-top:14px
}
.chip{
padding:5px 8px;
border:1px solid #213a55;
border-radius:999px;
background:#07131f;
color:#9cb6d1;
font-size:8px
}
.hero-side{padding:13px}
.snapshot-head{
display:flex;
justify-content:space-between;
color:#93aac4;
font-size:9px
}
.snapshot{
margin-top:9px;
padding:9px;
border:1px solid #223b56;
border-radius:9px;
background:#050d16;
font:8px/1.5 Consolas,monospace;
word-break:break-all;
color:#cae1f9
}
.meta-grid{
display:grid;
grid-template-columns:1fr 1fr;
gap:7px;
margin-top:8px
}
.meta{
padding:8px;
border:1px solid #203750;
border-radius:8px;
background:#07121e
}
.meta span{
display:block;
font-size:7px;
color:var(--muted)
}
.meta b{
display:block;
margin-top:3px;
font-size:9px;
overflow-wrap:anywhere
}
.kpis{
display:grid;
grid-template-columns:repeat(6,minmax(0,1fr));
gap:9px;
margin-top:12px
}
.kpi{padding:12px 13px}
.kpi-label{
font-size:7px;
color:var(--muted);
letter-spacing:.14em;
text-transform:uppercase
}
.kpi-value{
margin-top:6px;
font-size:22px;
font-weight:800
}
.kpi-sub{
margin-top:4px;
font-size:7px;
color:#607892
}
.good{color:var(--green)}
.bad{color:var(--red)}
.info{color:var(--cyan)}
.layout{
display:grid;
grid-template-columns:minmax(0,1.55fr) minmax(330px,.65fr);
gap:12px;
margin-top:12px
}
.section{padding:13px}
.section-head{
display:flex;
justify-content:space-between;
align-items:center;
gap:10px;
margin-bottom:9px
}
.section-title{
font-size:9px;
letter-spacing:.13em;
text-transform:uppercase;
color:#a8c0db
}
.section-meta{
font-size:7px;
color:var(--muted)
}
.toolbar{
display:flex;
flex-wrap:wrap;
gap:7px;
margin-bottom:9px
}
.control{
min-height:32px;
padding:7px 9px;
border:1px solid #24405c;
border-radius:8px;
background:#06101a;
color:var(--text);
font-size:9px;
outline:none
}
.control:focus{
border-color:#3c729c;
box-shadow:0 0 0 3px rgba(87,214,255,.07)
}
.search{
flex:1;
min-width:220px
}
.orbit{
position:relative;
height:620px;
overflow:hidden;
border:1px solid #1c344d;
border-radius:10px;
background:#02070c
}
#scene{
width:100%;
height:100%;
display:block
}
.overlay{
position:absolute;
top:10px;
left:10px;
display:flex;
gap:6px;
pointer-events:none
}
.badge{
padding:4px 7px;
border:1px solid #213c59;
border-radius:999px;
background:rgba(3,9,15,.82);
color:#91abc6;
font-size:7px
}
.inspector{
position:absolute;
left:10px;
right:10px;
bottom:10px;
display:none;
padding:11px;
border:1px solid #2d5779;
border-radius:10px;
background:rgba(4,11,18,.94);
backdrop-filter:blur(12px)
}
.inspector.open{display:block}
.inspector-top{
display:flex;
justify-content:space-between;
gap:8px
}
.node-title{
font-size:12px;
font-weight:750;
overflow:hidden;
text-overflow:ellipsis;
white-space:nowrap
}
.node-kind{
margin-top:3px;
font-size:8px;
color:var(--cyan)
}
.node-source{
margin-top:6px;
font:7px/1.5 Consolas,monospace;
color:#66819e;
word-break:break-all
}
.node-detail{
margin-top:7px;
padding:8px;
border:1px solid #1d344c;
border-radius:8px;
background:#050d16;
font:8px/1.5 Consolas,monospace;
color:#b9cde2
}
.close{
width:28px;
height:28px;
border:1px solid #27445f;
border-radius:7px;
background:#07121d;
color:#adc3da
}
.stack{
display:grid;
gap:12px
}
.matrix{
display:grid;
grid-template-columns:1fr 1fr;
gap:7px
}
.matrix-item{
padding:8px;
border:1px solid #213851;
border-radius:8px;
background:#06101a
}
.matrix-item b{
display:block;
font-size:8px
}
.matrix-item span{
display:block;
margin-top:4px;
font-size:7px;
color:var(--muted)
}
.list{
display:grid;
gap:6px;
max-height:250px;
overflow:auto
}
.list-item{
padding:8px;
border:1px solid #1d354e;
border-radius:8px;
background:#06101a;
cursor:pointer
}
.list-item:hover{
border-color:#355e7c;
background:#081522
}
.list-top{
display:flex;
justify-content:space-between;
gap:8px
}
.list-name{
font-size:8px;
overflow:hidden;
text-overflow:ellipsis;
white-space:nowrap
}
.list-count{
padding:3px 6px;
border:1px solid #284460;
border-radius:999px;
font-size:7px;
color:#9fc4df
}
.list-sub{
margin-top:4px;
font-size:7px;
color:var(--muted)
}
.table-wrap{
overflow:auto;
border:1px solid #1b334c;
border-radius:9px
}
table{
width:100%;
border-collapse:collapse;
font-size:7px
}
th,td{
padding:7px 8px;
text-align:left;
vertical-align:top;
border-bottom:1px solid #13283d
}
th{
position:sticky;
top:0;
z-index:1;
background:#071320;
color:#9ec0df
}
td{
color:#b5c9dc
}
code{
font:7px/1.5 Consolas,monospace;
color:#c5d7ff
}
.log{
height:330px;
overflow:auto;
padding:10px;
border:1px solid #1b334c;
border-radius:9px;
background:#02070c;
color:#9eb4cb;
font:7px/1.6 Consolas,monospace;
white-space:pre-wrap;
word-break:break-word
}
.footer{
display:flex;
justify-content:space-between;
gap:10px;
margin:12px 2px 0;
font-size:7px;
color:#536b84
}
.empty{
padding:14px;
text-align:center;
font-size:8px;
color:var(--muted)
}
@media(max-width:1250px){
.hero,.layout{grid-template-columns:1fr}
.kpis{grid-template-columns:repeat(3,minmax(0,1fr))}
}
@media(max-width:760px){
body{padding:8px}
.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}
.orbit{height:500px}
.hero-main h1{font-size:22px}
}
</style>
</head>
<body>
<div class="shell">

<div class="top">
<div class="brand">
<div class="logo">KHZ</div>
<div>
<div class="title">SOVEREIGN GRAPH CONTROL ROOM</div>
<div class="sub">Realtime repository analysis · GitLab Pages · read-only surface</div>
</div>
</div>
<div class="live"><span class="dot"></span><span id="live-status">LIVE</span></div>
</div>

<div class="hero">

<div class="panel hero-main">
<div class="eyebrow">KHZ · GRAPH OPERATIONS</div>
<h1>Realtime Knowledge Analysis</h1>
<div class="hero-copy">
Repository-derived inspection of graph structure, evidence, schemas, audit runs and CI state.
The frontend itself is static; the published JSON snapshot is refreshed by GitLab CI and polled by the browser.
</div>
<div class="chips">
<span class="chip">GitLab Pages</span>
<span class="chip">CI snapshot</span>
<span class="chip">10s polling</span>
<span class="chip">A–Z control map</span>
<span class="chip">Node inspector</span>
<span class="chip">Audit logs</span>
</div>
</div>

<div class="panel hero-side">
<div class="snapshot-head">
<span>SNAPSHOT</span>
<span id="snapshot-time">—</span>
</div>
<div class="snapshot" id="snapshot-id">—</div>
<div class="meta-grid">
<div class="meta"><span>PROJECT</span><b id="project">—</b></div>
<div class="meta"><span>BRANCH</span><b id="branch">—</b></div>
<div class="meta"><span>COMMIT</span><b id="commit">—</b></div>
<div class="meta"><span>PIPELINE</span><b id="pipeline">—</b></div>
<div class="meta"><span>JOB</span><b id="job">—</b></div>
<div class="meta"><span>TREE</span><b id="tree">—</b></div>
</div>
</div>

</div>

<div class="kpis">

<div class="panel kpi">
<div class="kpi-label">Quality state</div>
<div class="kpi-value" id="state">—</div>
<div class="kpi-sub" id="state-sub">—</div>
</div>

<div class="panel kpi">
<div class="kpi-label">Unique nodes</div>
<div class="kpi-value info" id="unique-nodes">—</div>
<div class="kpi-sub">deduplicated identifiers</div>
</div>

<div class="panel kpi">
<div class="kpi-label">Node occurrences</div>
<div class="kpi-value" id="node-occurrences">—</div>
<div class="kpi-sub">raw node records</div>
</div>

<div class="panel kpi">
<div class="kpi-label">Edges</div>
<div class="kpi-value" id="edge-count">—</div>
<div class="kpi-sub">edge occurrences</div>
</div>

<div class="panel kpi">
<div class="kpi-label">Evidence</div>
<div class="kpi-value" id="evidence-count">—</div>
<div class="kpi-sub">evidence occurrences</div>
</div>

<div class="panel kpi">
<div class="kpi-label">JSON files</div>
<div class="kpi-value" id="json-files">—</div>
<div class="kpi-sub">repository inventory</div>
</div>

</div>

<div class="layout">

<div class="panel section">

<div class="section-head">
<div class="section-title">Knowledge orbit</div>
<div class="section-meta" id="orbit-meta">—</div>
</div>

<div class="toolbar">
<input class="control search" id="node-search" placeholder="Search node / kind / source">
<select class="control" id="kind-filter">
<option value="">All kinds</option>
</select>
<button class="control" id="reset-node">Reset</button>
</div>

<div class="orbit">
<canvas id="scene"></canvas>

<div class="overlay">
<span class="badge">hover = inspect</span>
<span class="badge">click = lock</span>
<span class="badge">Esc = clear</span>
</div>

<div class="inspector" id="inspector">

<div class="inspector-top">
<div>
<div class="node-title" id="inspector-title">—</div>
<div class="node-kind" id="inspector-kind">—</div>
</div>
<button class="close" id="inspector-close">×</button>
</div>

<div class="node-source" id="inspector-source">—</div>
<div class="node-detail" id="inspector-detail">—</div>

</div>
</div>
</div>

<div class="stack">

<div class="panel section">

<div class="section-head">
<div class="section-title">Audit matrix</div>
<div class="section-meta">execution state</div>
</div>

<div class="matrix" id="audit-matrix"></div>

</div>

<div class="panel section">

<div class="section-head">
<div class="section-title">Integrity faults</div>
<div class="section-meta">click to filter</div>
</div>

<div class="list" id="fault-list"></div>

</div>

<div class="panel section">

<div class="section-head">
<div class="section-title">Schema distribution</div>
<div class="section-meta">observed</div>
</div>

<div class="list" id="schema-list"></div>

</div>

</div>
</div>

<div class="panel section" style="margin-top:12px">

<div class="section-head">
<div class="section-title">A–Z control map</div>
<div class="section-meta">compact operational view</div>
</div>

<div class="table-wrap">
<table>
<thead>
<tr>
<th>Area</th>
<th>Value</th>
<th>Meaning</th>
</tr>
</thead>
<tbody id="az-table"></tbody>
</table>
</div>

</div>

<div class="panel section" style="margin-top:12px">

<div class="section-head">
<div class="section-title">Integrity detail</div>
<div class="section-meta">filtered findings</div>
</div>

<div class="toolbar">
<input class="control search" id="issue-search" placeholder="Search integrity findings">
<select class="control" id="issue-filter">
<option value="">All issues</option>
<option value="duplicate-node">Duplicate nodes</option>
<option value="duplicate-edge">Duplicate edges</option>
<option value="duplicate-evidence">Duplicate evidence</option>
<option value="dangling">Dangling edges</option>
<option value="parse">Parse errors</option>
</select>
</div>

<div class="table-wrap">
<table>
<thead>
<tr>
<th>Class</th>
<th>Identifier</th>
<th>Count</th>
<th>Detail</th>
</tr>
</thead>
<tbody id="issues-table"></tbody>
</table>
</div>

</div>

<div class="panel section" style="margin-top:12px">

<div class="section-head">
<div class="section-title">CI logs</div>
<div class="section-meta" id="log-meta">—</div>
</div>

<div class="toolbar">
<select class="control" id="log-select"></select>
<input class="control search" id="log-filter" placeholder="Filter current log">
<button class="control" id="log-refresh">Refresh</button>
</div>

<div class="log" id="log-view">—</div>

</div>

<div class="footer">
<span>Repository-derived snapshot · read-only frontend</span>
<span id="footer-snapshot">—</span>
</div>

</div>

<script>
"use strict";

const BUNDLE_URL = "./khz_realtime_bundle.json";
const POLL_MS = 10000;

let bundle = null;
let allNodes = [];
let visibleNodes = [];
let projected = [];
let lockedNode = null;
let hoverNode = null;
let tick = 0;
let activeLog = "";
let issueQuery = "";
let activeIssueClass = "";
let nodeQuery = "";
let activeKind = "";

const $ = (id) => document.getElementById(id);

const colors = {
verified_lesson:"#3ddc97",
atomic_fact:"#8be9a8",
taxonomy:"#57d6ff",
staging:"#ffbd55",
reserved:"#71839b",
source:"#ad96ff",
chunk:"#ff657b",
schema:"#f8d36e",
data:"#f78bd4",
manifest:"#5eead4"
};

function colorOf(kind){
return colors[kind] || "#a8bfd8";
}

function escapeHtml(value){
return String(value ?? "")
.replaceAll("&","&amp;")
.replaceAll("<","&lt;")
.replaceAll(">","&gt;")
.replaceAll('"',"&quot;")
.replaceAll("'","&#039;");
}

function hashCode(text){
let h = 2166136261;
for(let i=0;i<text.length;i++){
h ^= text.charCodeAt(i);
h = Math.imul(h,16777619);
}
return h >>> 0;
}

function short(value,n=14){
const text = String(value ?? "");
return text.length > n
? text.slice(0,n-1)+"…"
: text;
}

async function fetchBundle(){
const response = await fetch(
BUNDLE_URL + "?t=" + Date.now(),
{cache:"no-store"}
);

if(!response.ok){
throw new Error("HTTP " + response.status);
}

return response.json();
}

function normalizeNodes(){

const source = Array.isArray(bundle.graph?.nodes)
? bundle.graph.nodes
: [];

allNodes = source.map((node,index)=>{

const h = hashCode(
String(node.id || "") +
"|" +
String(node.file || "") +
"|" +
index
);

return {
...node,
angle:(h % 62831) / 10000,
radius:95 + (h % 760),
speed:0.00055 + ((h % 17) / 120000),
phase:((h >>> 8) % 1000) / 1000
};

});

applyNodeFilters();
}

function applyNodeFilters(){

const q = nodeQuery.trim().toLowerCase();

visibleNodes = allNodes.filter(node => {

const text = [
node.id,
node.label,
node.kind,
node.file
]
.join(" ")
.toLowerCase();

return (
(!q || text.includes(q)) &&
(!activeKind || node.kind === activeKind)
);

});

$("orbit-meta").textContent =
visibleNodes.length.toLocaleString() +
" visible / " +
allNodes.length.toLocaleString();

}

function renderOverview(){

const graph = bundle.graph || {};
const env = bundle.environment || {};

$("state").textContent =
bundle.state || "UNKNOWN";

$("state").className =
"kpi-value " +
(bundle.state === "PASS" ? "good" : "bad");

$("state-sub").textContent =
String(bundle.files?.parse_errors || 0) +
" parse errors · " +
Object.keys(graph.duplicate_node_ids || {}).length +
" duplicate node IDs";

$("unique-nodes").textContent =
Number(graph.unique_nodes || 0).toLocaleString();

$("node-occurrences").textContent =
Number(graph.node_occurrences || 0).toLocaleString();

$("edge-count").textContent =
Number(graph.edge_occurrences || 0).toLocaleString();

$("evidence-count").textContent =
Number(graph.evidence_occurrences || 0).toLocaleString();

$("json-files").textContent =
Number(bundle.files?.json_files || 0).toLocaleString();

$("snapshot-id").textContent =
bundle.snapshot_id || "—";

$("snapshot-time").textContent =
bundle.generated_utc
? new Date(bundle.generated_utc).toLocaleString()
: "—";

$("project").textContent =
env.project || "—";

$("branch").textContent =
env.branch || "—";

$("commit").textContent =
short(env.commit || "—");

$("pipeline").textContent =
env.pipeline_id || "—";

$("job").textContent =
env.job_id || "—";

$("tree").textContent =
bundle.working_tree?.clean
? "CLEAN"
: "DIRTY";

$("live-status").textContent =
"LIVE · " +
short(bundle.snapshot_id || "",12);

$("footer-snapshot").textContent =
"snapshot " +
(bundle.snapshot_id || "—");

const kinds =
Object.keys(graph.node_kind_counts || {})
.sort();

$("kind-filter").innerHTML =
'<option value="">All kinds</option>' +
kinds.map(kind =>
`<option value="${escapeHtml(kind)}">
${escapeHtml(kind)} · ${graph.node_kind_counts[kind]}
</option>`
).join("");

$("kind-filter").value =
activeKind;

}

function renderAudits(){

const rows = bundle.audit_runs || [];

$("audit-matrix").innerHTML =
rows.map(row => `
<div class="matrix-item">
<b style="color:${row.ok ? "var(--green)" : "var(--red)"}">
${escapeHtml(row.name.replaceAll("_"," "))}
</b>
<span>
${row.ok ? "PASS" : "FAIL"} ·
${row.duration_seconds}s
</span>
</div>
`).join("");

}

function renderFaults(){

const graph = bundle.graph || {};

const rows = [
[
"Duplicate node IDs",
Object.keys(graph.duplicate_node_ids || {}).length,
"duplicate-node"
],
[
"Duplicate edge IDs",
Object.keys(graph.duplicate_edge_ids || {}).length,
"duplicate-edge"
],
[
"Duplicate evidence IDs",
Object.keys(graph.duplicate_evidence_ids || {}).length,
"duplicate-evidence"
],
[
"Dangling edges",
(graph.dangling_edges || []).length,
"dangling"
],
[
"Parse errors",
(bundle.parse_errors || []).length,
"parse"
]
];

$("fault-list").innerHTML =
rows.map(row => `
<div class="list-item" data-issue="${row[2]}">
<div class="list-top">
<div class="list-name">
${escapeHtml(row[0])}
</div>
<div class="list-count">${row[1]}</div>
</div>
<div class="list-sub">
${row[1] ? "inspection available" : "clear"}
</div>
</div>
`).join("");

document
.querySelectorAll("[data-issue]")
.forEach(item => {

item.onclick = () => {

activeIssueClass =
item.dataset.issue || "";

$("issue-filter").value =
activeIssueClass;

renderIssues();

window.scrollTo({
top:$("issues-table")
.closest(".panel")
.offsetTop - 70,
behavior:"smooth"
});

};

});

}

function renderSchemas(){

const entries =
Object.entries(bundle.graph.schema_versions || {})
.sort((a,b)=>b[1]-a[1]);

$("schema-list").innerHTML =
entries.length
? entries.map(entry => `
<div class="list-item">
<div class="list-top">
<div class="list-name">
${escapeHtml(entry[0])}
</div>
<div class="list-count">
${entry[1]}
</div>
</div>
<div class="list-sub">
observed JSON documents
</div>
</div>
`).join("")
: '<div class="empty">No schema data</div>';

}

function renderAZ(){

const g = bundle.graph || {};
const env = bundle.environment || {};

const rows = [
["A · Audits",bundle.audit_runs.length,"audit programs"],
["B · Branch",env.branch || "—","build branch"],
["C · Commit",env.commit || "—","source revision"],
["D · Documents",bundle.files.json_files,"JSON inventory"],
["E · Evidence",g.evidence_occurrences,"evidence records"],
["F · Graph files",g.graph_files,"graph-bearing files"],
["G · Nodes",g.node_occurrences,"node occurrences"],
["H · Schemas",Object.keys(g.schema_versions || {}).length,"schema versions"],
["I · Duplicate IDs",Object.keys(g.duplicate_node_ids || {}).length,"duplicate node identifiers"],
["J · JSON errors",bundle.files.parse_errors,"parse errors"],
["K · Dangling",g.dangling_edges?.length || 0,"missing endpoints"],
["L · Empty graphs",g.empty_graph_files || 0,"empty graph files"],
["M · Snapshot",bundle.snapshot_id || "—","fingerprint"],
["N · Unique nodes",g.unique_nodes,"unique IDs"],
["O · Edges",g.edge_occurrences,"edge occurrences"],
["P · Pipeline",env.pipeline_id || "—","GitLab pipeline"],
["Q · Quality",bundle.state || "—","computed state"],
["R · Repository",env.project || "—","repository"],
["S · Schema mix",JSON.stringify(g.schema_versions || {}),"distribution"],
["T · Timestamp",bundle.generated_utc || "—","UTC"],
["U · Working tree",bundle.working_tree?.clean ? "CLEAN" : "DIRTY","repository state"],
["V · Verified lessons",((bundle.audit_runs || []).find(x=>x.name==="verified_lessons") || {}).ok ? "PASS" : "FAIL","audit"],
["W · Web","GitLab Pages","delivery"],
["X · Execution","CI + browser polling","runtime"],
["Y · Yield","graph + audits + logs","inspection"],
["Z · Source","repository snapshot","truth boundary"]
];

$("az-table").innerHTML =
rows.map(row => `
<tr>
<td>${escapeHtml(row[0])}</td>
<td>${escapeHtml(row[1])}</td>
<td>${escapeHtml(row[2])}</td>
</tr>
`).join("");

}

function makeIssueRows(){

const g = bundle.graph || {};
const rows = [];

Object.entries(
g.duplicate_node_ids || {}
)
.forEach(([id,count]) => {

rows.push({
className:"duplicate-node",
kind:"Duplicate node",
id,
count,
detail:"node ID occurs repeatedly"
});

});

Object.entries(
g.duplicate_edge_ids || {}
)
.forEach(([id,count]) => {

rows.push({
className:"duplicate-edge",
kind:"Duplicate edge",
id,
count,
detail:"edge ID occurs repeatedly"
});

});

Object.entries(
g.duplicate_evidence_ids || {}
)
.forEach(([id,count]) => {

rows.push({
className:"duplicate-evidence",
kind:"Duplicate evidence",
id,
count,
detail:"evidence ID occurs repeatedly"
});

});

(g.dangling_edges || [])
.forEach(edge => {

rows.push({
className:"dangling",
kind:"Dangling edge",
id:
String(edge.source) +
" → " +
String(edge.target),
count:1,
detail:edge.file
});

});

(bundle.parse_errors || [])
.forEach(error => {

rows.push({
className:"parse",
kind:"Parse error",
id:error.file,
count:1,
detail:error.error
});

});

return rows;

}

function renderIssues(){

const query =
issueQuery.trim().toLowerCase();

const rows =
makeIssueRows()
.filter(row => {

const classMatch =
!activeIssueClass ||
row.className === activeIssueClass;

const text = [
row.kind,
row.id,
row.detail
]
.join(" ")
.toLowerCase();

return (
classMatch &&
(!query || text.includes(query))
);

});

$("issues-table").innerHTML =
rows.length
? rows.slice(0,1200).map(row => `
<tr>
<td>${escapeHtml(row.kind)}</td>
<td><code>${escapeHtml(row.id)}</code></td>
<td style="color:var(--red);font-weight:700">
${row.count}
</td>
<td>${escapeHtml(row.detail)}</td>
</tr>
`).join("")
: '<tr><td colspan="4"><div class="empty">No matching findings</div></td></tr>';

}

function renderLogs(){

const rows =
bundle.audit_runs || [];

$("log-select").innerHTML =
rows.map((row,index) =>
`<option value="${index}">
${escapeHtml(row.name.replaceAll("_"," "))}
</option>`
).join("");

if(!activeLog){
activeLog =
rows[0]?.name || "";
}

const index =
Math.max(
0,
rows.findIndex(row => row.name === activeLog)
);

$("log-select").value =
String(index);

updateLog();

}

function updateLog(){

const rows =
bundle.audit_runs || [];

const row =
rows.find(item => item.name === activeLog) ||
rows[0];

let text =
row?.combined ||
"NO LOG OUTPUT";

const filter =
$("log-filter")
.value
.trim()
.toLowerCase();

if(filter){

text =
text
.split("\n")
.filter(line =>
line.toLowerCase().includes(filter)
)
.join("\n");

}

$("log-view").textContent =
text || "NO MATCHING LOG LINES";

$("log-meta").textContent =
row
? row.name +
" · RC " +
row.returncode +
" · " +
row.duration_seconds +
"s"
: "—";

}

function renderAll(){

renderOverview();
renderAudits();
renderFaults();
renderSchemas();
renderAZ();
renderIssues();
renderLogs();
normalizeNodes();

}

function resizeCanvas(){

const canvas =
$("scene");

const dpr =
Math.min(
window.devicePixelRatio || 1,
2
);

canvas.width =
Math.max(
1,
Math.floor(canvas.clientWidth * dpr)
);

canvas.height =
Math.max(
1,
Math.floor(canvas.clientHeight * dpr)
);

}

window.addEventListener(
"resize",
resizeCanvas
);

function project(node){

const width =
$("scene").width;

const height =
$("scene").height;

const angle =
node.angle +
tick * node.speed;

const wave =
Math.sin(
angle * .73 +
node.phase * Math.PI * 2
);

const orbit =
node.radius *
(.78 + wave * .10);

const x =
Math.cos(angle) *
orbit;

const y =
Math.sin(
angle * .84 +
node.phase
) *
orbit *
.54;

const z =
Math.sin(angle) *
orbit;

const scale =
.74 +
((z + 900) / 1800) *
.44;

return {
x:width / 2 + x,
y:height / 2 + y,
z,
scale
};

}

function draw(){

const canvas =
$("scene");

const ctx =
canvas.getContext("2d");

const width =
canvas.width;

const height =
canvas.height;

const gradient =
ctx.createRadialGradient(
width / 2,
height / 2,
0,
width / 2,
height / 2,
Math.max(width,height) * .72
);

gradient.addColorStop(
0,
"#0a1d2d"
);

gradient.addColorStop(
.5,
"#040a12"
);

gradient.addColorStop(
1,
"#010307"
);

ctx.fillStyle =
gradient;

ctx.fillRect(
0,
0,
width,
height
);

ctx.save();

ctx.globalAlpha = .14;
ctx.strokeStyle = "#28587f";

for(
let radius = 90;
radius < Math.min(width,height) * .48;
radius += 90
){

ctx.beginPath();

ctx.arc(
width / 2,
height / 2,
radius,
0,
Math.PI * 2
);

ctx.stroke();

}

ctx.restore();

projected =
visibleNodes
.map(node => ({
node,
point:project(node)
}))
.sort(
(a,b) =>
a.point.z -
b.point.z
);

for(
const item of projected
){

const node =
item.node;

const point =
item.point;

const selected =
lockedNode &&
lockedNode.id === node.id;

const hovered =
!selected &&
hoverNode &&
hoverNode.id === node.id;

const nodeColor =
colorOf(node.kind);

const radius =
(selected ? 7 : hovered ? 6 : 3.5) *
point.scale;

ctx.save();

ctx.globalAlpha =
Math.max(
.35,
Math.min(
1,
.58 + point.z / 1500
)
);

ctx.shadowColor =
nodeColor;

ctx.shadowBlur =
selected
? 30
: hovered
? 22
: 11;

ctx.fillStyle =
nodeColor;

ctx.beginPath();

ctx.arc(
point.x,
point.y,
radius,
0,
Math.PI * 2
);

ctx.fill();

if(
selected ||
hovered
){

ctx.shadowBlur = 0;
ctx.strokeStyle = "#f6fbff";
ctx.lineWidth = 1.5;
ctx.stroke();

ctx.font =
"600 " +
Math.max(9,12 * point.scale) +
"px Inter,Segoe UI,Arial";

ctx.textAlign = "left";
ctx.textBaseline = "middle";
ctx.fillStyle = "#ffffff";

const label =
String(node.label || node.id || "");

ctx.fillText(
label.length > 55
? label.slice(0,54) + "…"
: label,
point.x + radius + 8,
point.y - 5
);

ctx.font =
Math.max(8,9 * point.scale) +
"px Inter,Segoe UI,Arial";

ctx.fillStyle =
nodeColor;

ctx.fillText(
String(node.kind || "node"),
point.x + radius + 8,
point.y + 10
);

}

ctx.restore();

}

tick++;

requestAnimationFrame(
draw
);

}

function hitTest(event){

const canvas =
$("scene");

const rect =
canvas.getBoundingClientRect();

const scaleX =
canvas.width /
rect.width;

const scaleY =
canvas.height /
rect.height;

const x =
(event.clientX - rect.left) *
scaleX;

const y =
(event.clientY - rect.top) *
scaleY;

let best = null;
let bestDistance = Infinity;

for(
const item of projected
){

const distance =
Math.hypot(
x - item.point.x,
y - item.point.y
);

const threshold =
22 *
item.point.scale;

if(
distance < threshold &&
distance < bestDistance
){

best =
item.node;

bestDistance =
distance;

}

}

return best;

}

function showInspector(node){

if(!node){

$("inspector")
.classList
.remove("open");

return;

}

$("inspector")
.classList
.add("open");

$("inspector-title")
.textContent =
node.label ||
node.id ||
"Node";

$("inspector-kind")
.textContent =
node.kind ||
"node";

$("inspector-source")
.textContent =
node.file ||
"";

$("inspector-detail").textContent =
[
["ID",node.id],
["Kind",node.kind],
["Source",node.file]
]
.filter(item => item[1])
.map(
item =>
item[0] +
": " +
item[1]
)
.join(" · ");

}

$("scene")
.addEventListener(
"mousemove",
event => {

if(lockedNode){
return;
}

hoverNode =
hitTest(event);

if(hoverNode){
showInspector(
hoverNode
);
}

}
);

$("scene")
.addEventListener(
"mouseleave",
() => {

if(!lockedNode){

hoverNode =
null;

showInspector(
null
);

}

}
);

$("scene")
.addEventListener(
"click",
event => {

const node =
hitTest(event);

if(!node){

lockedNode = null;
hoverNode = null;
showInspector(null);
return;

}

lockedNode =
lockedNode &&
lockedNode.id === node.id
? null
: node;

showInspector(
lockedNode || node
);

}
);

window.addEventListener(
"keydown",
event => {

if(event.key === "Escape"){

lockedNode = null;
hoverNode = null;

showInspector(
null
);

}

}
);

$("inspector-close").onclick =
() => {

lockedNode = null;
hoverNode = null;

showInspector(
null
);

};

$("node-search").oninput =
event => {

nodeQuery =
event.target.value;

applyNodeFilters();

};

$("kind-filter").onchange =
event => {

activeKind =
event.target.value;

applyNodeFilters();

};

$("reset-node").onclick =
() => {

nodeQuery = "";
activeKind = "";

$("node-search").value = "";
$("kind-filter").value = "";

applyNodeFilters();

};

$("issue-search").oninput =
event => {

issueQuery =
event.target.value;

renderIssues();

};

$("issue-filter").onchange =
event => {

activeIssueClass =
event.target.value;

renderIssues();

};

$("log-select").onchange =
event => {

const row =
bundle.audit_runs[
Number(event.target.value)
];

activeLog =
row?.name || "";

updateLog();

};

$("log-filter").oninput =
updateLog;

$("log-refresh").onclick =
async () => {

try{

bundle =
await fetchBundle();

renderAll();

}catch(error){

$("log-view")
.textContent =
"Refresh failed: " +
error.message;

}

};

async function boot(){

bundle =
await fetchBundle();

renderAll();

resizeCanvas();

draw();

}

boot()
.catch(error => {

$("state")
.textContent =
"OFFLINE";

$("state")
.className =
"kpi-value bad";

$("state-sub")
.textContent =
error.message;

});

setInterval(
async () => {

try{

const next =
await fetchBundle();

if(
!bundle ||
next.snapshot_id !==
bundle.snapshot_id
){

bundle =
next;

renderAll();

}

}catch(_error){}

},
POLL_MS
);

</script>
</body>
</html>
"""

(PUBLIC / "index.html").write_text(html, encoding="utf-8")
print("KHZ_FRONTEND_REBUILT")
