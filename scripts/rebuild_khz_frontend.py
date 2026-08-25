from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
BUNDLE = PUBLIC / "khz_realtime_bundle.json"

if not BUNDLE.exists():
    raise SystemExit("ERROR: public/khz_realtime_bundle.json not found")

bundle = json.loads(BUNDLE.read_text(encoding="utf-8-sig"))
data = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))

html = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#02040a">
<title>KHAWRIZM — SOVEREIGN COSMOS II</title>
<style>
:root{--bg:#02040a;--text:#d8e4ff;--muted:#546481;--line:rgba(104,130,186,.15);--cyan:#39dcff;--green:#54ff9a;--red:#ff523f;--yellow:#ffd43d;--purple:#b96cff;--blue:#657bff;--orange:#ff9f36}
*{box-sizing:border-box}
html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#02040a;color:var(--text);font-family:Consolas,"Cascadia Mono",monospace}
canvas{position:fixed;inset:0;width:100vw;height:100vh;display:block}
#hud{position:fixed;inset:0;pointer-events:none}
#header{position:absolute;top:18px;left:20px;right:20px;display:flex;justify-content:space-between;align-items:flex-start}
.brand-title{font-size:14px;letter-spacing:.16em;color:#a7c2ff;text-shadow:0 0 18px rgba(100,130,255,.45)}
.brand-sub{margin-top:4px;font-size:8px;color:#4c5c78;letter-spacing:.08em}
.live{padding:7px 10px;border:1px solid rgba(90,119,174,.18);border-radius:999px;background:rgba(2,5,12,.54);font-size:8px;color:#8effbf;backdrop-filter:blur(9px)}
.dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:#54ff9a;box-shadow:0 0 12px #54ff9a;margin-right:6px}
#legend{position:absolute;top:62px;left:20px;display:flex;flex-wrap:wrap;gap:5px;max-width:74vw}
.legend-item{padding:4px 7px;border:1px solid rgba(88,113,165,.12);border-radius:999px;background:rgba(2,5,11,.46);font-size:7px}
#left{position:absolute;left:20px;bottom:18px;width:290px;padding:11px;border:1px solid rgba(88,113,165,.16);border-radius:11px;background:rgba(3,7,14,.55);backdrop-filter:blur(9px);pointer-events:auto}
#right{position:absolute;right:20px;bottom:18px;width:300px;max-height:50vh;overflow:auto;padding:11px;border:1px solid rgba(88,113,165,.16);border-radius:11px;background:rgba(3,7,14,.55);backdrop-filter:blur(9px);pointer-events:auto}
.section{font-size:8px;color:#667896;letter-spacing:.15em;text-transform:uppercase;margin-bottom:8px}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.metric{padding:7px;border:1px solid rgba(88,113,165,.11);border-radius:7px;background:rgba(7,12,22,.46)}
.metric span{display:block;font-size:7px;color:#45546e}
.metric b{display:block;margin-top:3px;font-size:11px;color:#b8c8e6}
#search{width:100%;margin-top:8px;padding:7px 8px;border:1px solid rgba(88,113,165,.14);border-radius:7px;background:rgba(5,9,17,.72);color:#c5d7f8;outline:none;font:8px Consolas,monospace}
#search::placeholder{color:#3f4d66}
.list{display:grid;gap:5px;margin-top:8px}
.item{padding:7px;border:1px solid rgba(88,113,165,.10);border-radius:7px;background:rgba(6,11,20,.40);cursor:pointer}
.item:hover{border-color:rgba(95,137,202,.28);background:rgba(11,18,31,.64)}
.item-name{font-size:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.item-meta{margin-top:3px;font-size:7px;color:#4f6280}
#inspect{position:absolute;left:50%;bottom:18px;transform:translateX(-50%);width:min(560px,calc(100vw - 40px));display:none;padding:10px 12px;border:1px solid rgba(112,140,198,.20);border-radius:10px;background:rgba(3,7,14,.80);backdrop-filter:blur(11px);pointer-events:auto}
#inspect.open{display:block}
.inspect-title{font-size:11px;color:#e4edff}
.inspect-kind{margin-top:3px;font-size:8px}
.inspect-source{margin-top:5px;font-size:7px;color:#526786;word-break:break-all}
.inspect-id{margin-top:6px;padding:6px;border-radius:7px;background:rgba(7,12,21,.58);font-size:7px;color:#8ea4c9;word-break:break-all}
#snapshot{position:absolute;right:20px;top:62px;font-size:7px;color:#2f3a51}
@media(max-width:800px){
#left,#right{width:calc(100vw - 40px)}
#right{top:100px;bottom:auto;max-height:23vh}
#left{bottom:18px;max-height:25vh;overflow:auto}
#inspect{bottom:28vh}
#legend{top:84px;max-width:calc(100vw - 40px)}
}
</style>
</head>
<body>
<canvas id="cosmos"></canvas>

<div id="hud">
  <div id="header">
    <div>
      <div class="brand-title">KHAWRIZM — SOVEREIGN COSMOS II</div>
      <div class="brand-sub">KNOWLEDGE AS CELESTIAL SYSTEM · LIVE REPOSITORY STATE</div>
    </div>
    <div class="live"><span class="dot"></span><span id="live-text">LIVE</span></div>
  </div>

  <div id="legend"></div>
  <div id="snapshot">snapshot: —</div>

  <div id="left">
    <div class="section">COSMOS TELEMETRY</div>
    <div class="metrics">
      <div class="metric"><span>STATE</span><b id="m-state">—</b></div>
      <div class="metric"><span>UNIQUE NODES</span><b id="m-nodes">—</b></div>
      <div class="metric"><span>EDGES</span><b id="m-edges">—</b></div>
      <div class="metric"><span>EVIDENCE</span><b id="m-evidence">—</b></div>
      <div class="metric"><span>JSON</span><b id="m-json">—</b></div>
      <div class="metric"><span>DUPLICATE IDs</span><b id="m-dup">—</b></div>
    </div>
    <input id="search" placeholder="search knowledge..." autocomplete="off">
    <div class="list" id="node-list"></div>
  </div>

  <div id="right">
    <div class="section">SYSTEMS</div>
    <div class="list" id="system-list"></div>
  </div>

  <div id="inspect">
    <div class="inspect-title" id="i-title">—</div>
    <div class="inspect-kind" id="i-kind">—</div>
    <div class="inspect-source" id="i-source">—</div>
    <div class="inspect-id" id="i-id">—</div>
  </div>
</div>

<script>
"use strict";

const DATA = __DATA__;
const POLL_MS = 10000;

const canvas = document.getElementById("cosmos");
const ctx = canvas.getContext("2d");

let W = 0;
let H = 0;
let DPR = 1;
let tick = 0;
let nodes = [];
let projected = [];
let activeNode = null;
let hoverNode = null;
let search = "";

const COLORS = {
  verified_lesson:"#54ff9a",
  atomic_fact:"#8cffbd",
  taxonomy:"#39dcff",
  staging:"#ffb642",
  reserved:"#596d88",
  source:"#b96cff",
  chunk:"#ff523f",
  schema:"#ffd43d",
  data:"#ff62dc",
  manifest:"#39e6cd",
  node:"#91a7cb"
};

function colorFor(kind){
  return COLORS[kind] || COLORS.node;
}

function hash32(text){
  let h = 2166136261;
  for(let i=0;i<text.length;i++){
    h ^= text.charCodeAt(i);
    h = Math.imul(h,16777619);
  }
  return h >>> 0;
}

function esc(value){
  return String(value ?? "")
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

function resize(){
  DPR = Math.min(window.devicePixelRatio || 1,2);
  W = canvas.width = Math.max(1,Math.floor(innerWidth * DPR));
  H = canvas.height = Math.max(1,Math.floor(innerHeight * DPR));
}

window.addEventListener("resize",resize);

function initNodes(){
  nodes = (DATA.graph?.nodes || []).map((node,index)=>{
    const h = hash32(
      String(node.id || "") + "|" +
      String(node.file || "") + "|" +
      index
    );
    const ring = index % 18;
    return {
      ...node,
      angle:(h % 62831) / 10000,
      radius:150 + ring * 77 + (h % 50),
      speed:0.00018 + ((h % 17) / 170000),
      phase:((h >>> 8) % 1000) / 1000
    };
  });
}

function filteredNodes(){
  const q = search.trim().toLowerCase();
  if(!q) return nodes;
  return nodes.filter(n=>[
    n.id,n.label,n.kind,n.file
  ].join(" ").toLowerCase().includes(q));
}

function updateHud(){
  const g = DATA.graph || {};
  document.getElementById("m-state").textContent = DATA.state || "—";
  document.getElementById("m-nodes").textContent = Number(g.unique_nodes || 0).toLocaleString();
  document.getElementById("m-edges").textContent = Number(g.edge_occurrences || 0).toLocaleString();
  document.getElementById("m-evidence").textContent = Number(g.evidence_occurrences || 0).toLocaleString();
  document.getElementById("m-json").textContent = Number(DATA.files?.json_files || 0).toLocaleString();
  document.getElementById("m-dup").textContent = Object.keys(g.duplicate_node_ids || {}).length.toLocaleString();
  document.getElementById("live-text").textContent = "LIVE · " + String(DATA.snapshot_id || "").slice(0,12);
  document.getElementById("snapshot").textContent = "snapshot: " + String(DATA.snapshot_id || "").slice(0,18);

  const counts = g.node_kind_counts || {};
  document.getElementById("legend").innerHTML = Object.keys(counts).sort().map(k=>`
    <div class="legend-item" style="color:${colorFor(k)}">
      ${esc(k)} · ${counts[k]}
    </div>
  `).join("");
}

function renderLists(){
  const visible = filteredNodes();

  document.getElementById("node-list").innerHTML =
    visible.slice(0,26).map(n=>`
      <div class="item" data-node="${encodeURIComponent(n.id)}">
        <div class="item-name" style="color:${colorFor(n.kind)}">${esc(n.label || n.id)}</div>
        <div class="item-meta">${esc(n.kind)} · ${esc(n.file)}</div>
      </div>
    `).join("") ||
    '<div class="item-meta">no matching knowledge</div>';

  document.querySelectorAll("[data-node]").forEach(el=>{
    el.onclick=()=>{
      const id = decodeURIComponent(el.dataset.node);
      const node = nodes.find(n=>n.id===id);
      if(node){
        activeNode = node;
        showInspector(node);
      }
    };
  });

  const systems = nodes.filter(n=>/system|engine|os|core|lab|layer|world|cosmos/i.test(
    String(n.label || "") + " " + String(n.kind || "")
  )).slice(0,30);

  document.getElementById("system-list").innerHTML =
    systems.map(n=>`
      <div class="item" data-system="${encodeURIComponent(n.id)}">
        <div class="item-name" style="color:${colorFor(n.kind)}">${esc(n.label || n.id)}</div>
        <div class="item-meta">${esc(n.kind)}</div>
      </div>
    `).join("") ||
    '<div class="item-meta">no system nodes found</div>';

  document.querySelectorAll("[data-system]").forEach(el=>{
    el.onclick=()=>{
      const id = decodeURIComponent(el.dataset.system);
      const node = nodes.find(n=>n.id===id);
      if(node){
        activeNode=node;
        showInspector(node);
      }
    };
  });
}

function showInspector(node){
  const box = document.getElementById("inspect");

  if(!node){
    box.classList.remove("open");
    return;
  }

  box.classList.add("open");
  document.getElementById("i-title").textContent = node.label || node.id || "Knowledge Node";
  document.getElementById("i-kind").textContent = node.kind || "node";
  document.getElementById("i-kind").style.color = colorFor(node.kind);
  document.getElementById("i-source").textContent = node.file || "";
  document.getElementById("i-id").textContent = "ID: " + String(node.id || "");
}

function center(){
  return {x:W/2,y:H/2};
}

function project(node){
  const c = center();
  const angle = node.angle + tick * node.speed;
  const wave = Math.sin(angle * .83 + node.phase * Math.PI * 2);
  const orbit = node.radius * (.82 + wave * .08);
  const z = Math.sin(angle) * orbit;
  const scale = .72 + ((z + 1700) / 3400) * .38;

  return {
    x:c.x + Math.cos(angle) * orbit,
    y:c.y + wave * orbit * .55,
    z,
    scale
  };
}

function stars(){
  const count = Math.max(120,Math.floor((W*H)/(23000*DPR)));
  for(let i=0;i<count;i++){
    const x = hash32("x"+i) % W;
    const y = hash32("y"+i) % H;
    const a = .04 + ((hash32("a"+i)%60)/1000);
    const r = (1 + (hash32("r"+i)%3)/2) * DPR;
    ctx.fillStyle = `rgba(145,170,218,${a})`;
    ctx.beginPath();
    ctx.arc(x,y,r,0,Math.PI*2);
    ctx.fill();
  }
}

function orbits(){
  const c=center();
  ctx.save();
  ctx.globalAlpha=.08;
  ctx.strokeStyle="#53698e";

  for(let i=1;i<=12;i++){
    const radius=105+i*92;
    ctx.beginPath();
    ctx.ellipse(c.x,c.y,radius,radius*.55,0,0,Math.PI*2);
    ctx.stroke();
  }

  ctx.restore();
}

function sovereign(){
  const c=center();

  const glow = ctx.createRadialGradient(c.x,c.y,0,c.x,c.y,180);
  glow.addColorStop(0,"rgba(255,251,181,.98)");
  glow.addColorStop(.18,"rgba(255,219,73,.90)");
  glow.addColorStop(.42,"rgba(255,175,23,.22)");
  glow.addColorStop(1,"rgba(255,145,0,0)");

  ctx.fillStyle=glow;
  ctx.beginPath();
  ctx.arc(c.x,c.y,180,0,Math.PI*2);
  ctx.fill();

  ctx.save();
  ctx.shadowColor="#ffe067";
  ctx.shadowBlur=50;
  ctx.fillStyle="#ffe268";
  ctx.beginPath();
  ctx.arc(c.x,c.y,39,0,Math.PI*2);
  ctx.fill();
  ctx.restore();

  ctx.fillStyle="#ac9130";
  ctx.font="8px Consolas";
  ctx.textAlign="center";
  ctx.fillText("THE SOVEREIGN",c.x,c.y+59);
}

function drawNode(node,p){
  const color=colorFor(node.kind);
  const selected=activeNode && activeNode.id===node.id;
  const hovered=!selected && hoverNode && hoverNode.id===node.id;
  const size=(selected?10:hovered?8:4.2)*p.scale*DPR;

  ctx.save();
  ctx.globalAlpha=Math.max(.32,Math.min(1,.70+p.z/2500));

  const glow=ctx.createRadialGradient(
    p.x,p.y,0,
    p.x,p.y,size*6
  );

  glow.addColorStop(0,color);
  glow.addColorStop(1,"rgba(0,0,0,0)");

  ctx.fillStyle=glow;
  ctx.beginPath();
  ctx.arc(p.x,p.y,size*6,0,Math.PI*2);
  ctx.fill();

  ctx.shadowColor=color;
  ctx.shadowBlur=selected?30:hovered?22:14;
  ctx.fillStyle=color;
  ctx.beginPath();
  ctx.arc(p.x,p.y,size,0,Math.PI*2);
  ctx.fill();

  if(selected || hovered){
    ctx.shadowBlur=0;
    ctx.strokeStyle="#fff";
    ctx.lineWidth=1.2*DPR;
    ctx.stroke();

    ctx.textAlign="center";
    ctx.textBaseline="top";
    ctx.font=`${Math.max(8,9*p.scale)}px Consolas`;
    ctx.fillStyle="#dce7ff";

    const text=String(node.label || node.id || "");
    ctx.fillText(
      text.length>40 ? text.slice(0,39)+"…" : text,
      p.x,
      p.y+size+6*DPR
    );

    ctx.font=`${Math.max(7,7*p.scale)}px Consolas`;
    ctx.fillStyle=color;
    ctx.fillText(
      String(node.kind || "node"),
      p.x,
      p.y+size+18*DPR
    );
  }

  ctx.restore();
}

function draw(){
  ctx.clearRect(0,0,W,H);

  const bg=ctx.createRadialGradient(
    W/2,H/2,0,
    W/2,H/2,Math.max(W,H)*.75
  );

  bg.addColorStop(0,"#070b15");
  bg.addColorStop(.55,"#02050c");
  bg.addColorStop(1,"#000106");

  ctx.fillStyle=bg;
  ctx.fillRect(0,0,W,H);

  stars();
  orbits();
  sovereign();

  projected=filteredNodes().map(node=>({
    node,
    point:project(node)
  })).sort((a,b)=>a.point.z-b.point.z);

  for(const item of projected){
    drawNode(item.node,item.point);
  }

  tick++;
  requestAnimationFrame(draw);
}

function hit(event){
  const rect=canvas.getBoundingClientRect();
  const sx=W/rect.width;
  const sy=H/rect.height;
  const x=(event.clientX-rect.left)*sx;
  const y=(event.clientY-rect.top)*sy;

  let best=null;
  let bestDistance=Infinity;

  for(const item of projected){
    const d=Math.hypot(
      x-item.point.x,
      y-item.point.y
    );

    const threshold=22*item.point.scale*DPR;

    if(d<threshold && d<bestDistance){
      best=item.node;
      bestDistance=d;
    }
  }

  return best;
}

canvas.addEventListener("mousemove",event=>{
  if(activeNode) return;
  hoverNode=hit(event);
  if(hoverNode) showInspector(hoverNode);
});

canvas.addEventListener("mouseleave",()=>{
  if(!activeNode){
    hoverNode=null;
    showInspector(null);
  }
});

canvas.addEventListener("click",event=>{
  const node=hit(event);

  if(!node){
    activeNode=null;
    hoverNode=null;
    showInspector(null);
    return;
  }

  activeNode =
    activeNode && activeNode.id===node.id
    ? null
    : node;

  showInspector(activeNode || node);
});

document.getElementById("search").addEventListener("input",event=>{
  search=event.target.value;
  renderLists();
});

window.addEventListener("keydown",event=>{
  if(event.key==="Escape"){
    activeNode=null;
    hoverNode=null;
    showInspector(null);
  }
});

resize();
initNodes();
updateHud();
renderLists();
draw();

setInterval(async()=>{
  try{
    const response=await fetch(
      "./khz_realtime_bundle.json?t="+Date.now(),
      {cache:"no-store"}
    );

    if(!response.ok) return;

    const next=await response.json();

    if(next.snapshot_id !== DATA.snapshot_id){
      location.reload();
    }
  }catch(_error){}
},POLL_MS);
</script>
</body>
</html>"""

html = html.replace("__DATA__", data)

(PUBLIC / "index.html").write_text(
    html,
    encoding="utf-8",
    newline="\n"
)

print("KHZ_SOVEREIGN_COSMOS_BUILT")
