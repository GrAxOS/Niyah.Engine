#!/usr/bin/env python3
import json
import re
import shutil
from pathlib import Path
from collections import Counter

ROOT = Path.cwd()
SRC = ROOT / "sources" / "90_legacy" / "legacy_graph_sources"
STAGE = ROOT / "knowledge" / "40_staging" / "auto_classified_legacy"
RESERVED = ROOT / "knowledge" / "70_reserved" / "legacy_empty_slots"
AUDITS = ROOT / "audits"

STAGE.mkdir(parents=True, exist_ok=True)
RESERVED.mkdir(parents=True, exist_ok=True)
AUDITS.mkdir(parents=True, exist_ok=True)

RULES = {
    "cryptography": [
        "fips", "nist", "ml-kem", "mlkem", "kyber", "ml-dsa", "mldsa", "slh-dsa", "slhdsa",
        "falcon", "xmss", "sha", "aes", "rsa", "signature", "kem", "hash", "pqc", "cryptographic"
    ],
    "mathematics": [
        "ntt", "number theoretic", "polynomial", "ring", "modular", "barrett", "montgomery",
        "matrix", "vector", "multiplication", "cooley", "tukey", "algebra", "complexity_bounds"
    ],
    "hardware_isa": [
        "x86", "x86-64", "arm", "armv8", "aarch64", "risc-v", "riscv", "avx", "avx2", "avx512",
        "neon", "simd", "sse", "ymm", "xmm", "zmm", "register", "opcode", "instruction"
    ],
    "operating_systems": [
        "linux", "windows", "kernel", "systemd", "syscall", "driver", "process", "memory manager",
        "filesystem", "ubuntu", "kali"
    ],
    "networking": [
        "bgp", "ospf", "isis", "tcp", "udp", "ip", "ipv6", "dns", "dhcp", "vpn", "vpc",
        "subnet", "routing", "switching", "ethernet", "mpls", "ccna", "ccnp", "ccie", "ripe"
    ],
    "programming_languages": [
        "c++", "cpp", " c ", "csharp", "c#", "java", "javascript", "typescript", "html", "css",
        "assembly", "asm", "python", "rust", "go", "sql", "compiler", "abi", "posix"
    ],
    "cloud": [
        "azure", "aws", "gcp", "google cloud", "huawei cloud", "compute", "storage", "iam",
        "cloud", "kubernetes", "aks", "eks", "gke", "s3", "blob", "ec2"
    ],
    "databases": [
        "postgres", "postgresql", "mysql", "sql", "database", "graph database", "vector database",
        "index", "transaction", "wal", "acid"
    ],
    "human_languages": [
        "arabic", "english", "morphology", "tokenization", "ner", "root", "pattern", "grammar",
        "translation", "lemma", "stemming"
    ],
    "security": [
        "zero trust", "access control", "privacy", "compliance", "iso", "iec", "security",
        "enclave", "sev", "tdx", "confidential"
    ]
}

def rel(p):
    return str(p.relative_to(ROOT)).replace("\\", "/")

def slug(s):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_").lower()

def load_json(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))

def flatten(x):
    if isinstance(x, dict):
        return " ".join(str(k) + " " + flatten(v) for k, v in x.items())
    if isinstance(x, list):
        return " ".join(flatten(v) for v in x)
    return str(x)

def graph_of(x):
    if isinstance(x, dict) and isinstance(x.get("graph"), dict):
        return x["graph"]
    if isinstance(x, dict) and any(k in x for k in ["nodes", "edges", "evidence"]):
        return x
    return {}

def classify(text):
    t = " " + text.lower() + " "
    scores = Counter()
    for domain, words in RULES.items():
        for w in words:
            if w.lower() in t:
                scores[domain] += 1
    if not scores:
        return "unclassified", {}
    top = scores.most_common()
    return top[0][0], dict(scores)

rows = []
written = []

for p in sorted(SRC.rglob("*.json")):
    try:
        obj = load_json(p)
        parse_ok = True
        err = None
    except Exception as e:
        obj = {}
        parse_ok = False
        err = str(e)

    g = graph_of(obj)
    nodes = g.get("nodes", []) or []
    edges = g.get("edges", []) or []
    evidence = g.get("evidence", []) or []
    is_empty = parse_ok and not nodes and not edges and not evidence

    source_rel = rel(p)
    name = slug(source_rel.replace("/", "__"))

    if is_empty:
        marker = {
            "schema": {"name": "khz_reserved_empty_legacy_slot", "version": "0.1.0"},
            "status": "EMPTY_RESERVED",
            "source_file": source_rel,
            "reason": "Legacy empty graph file preserved for future expansion.",
            "policy": "Do not treat as knowledge. Fill through verified lesson pipeline only."
        }
        out = RESERVED / (name + ".reserved.json")
        out.write_text(json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(rel(out))
        domain = "reserved_empty"
        scores = {}
    else:
        text = flatten(obj)
        domain, scores = classify(text)
        out_dir = STAGE / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / (name + ".json")
        shutil.copy2(p, out)

        meta = {
            "schema": {"name": "khz_auto_classified_legacy_source", "version": "0.1.0"},
            "status": "AUTO_CLASSIFIED_CANDIDATE",
            "source_file": source_rel,
            "staged_file": rel(out),
            "domain": domain,
            "scores": scores,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "evidence_count": len(evidence),
            "policy": "Candidate only. Not verified lesson. Not canonical."
        }
        meta_out = out_dir / (name + ".meta.json")
        meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.extend([rel(out), rel(meta_out)])

    rows.append({
        "source_file": source_rel,
        "parse_ok": parse_ok,
        "error": err,
        "empty": is_empty,
        "domain": domain,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "evidence_count": len(evidence),
        "scores": scores
    })

summary = {
    "source_root": rel(SRC),
    "staging_root": rel(STAGE),
    "reserved_root": rel(RESERVED),
    "files_scanned": len(rows),
    "empty_reserved": sum(1 for r in rows if r["empty"]),
    "non_empty_candidates": sum(1 for r in rows if not r["empty"] and r["parse_ok"]),
    "parse_errors": sum(1 for r in rows if not r["parse_ok"]),
    "by_domain": dict(Counter(r["domain"] for r in rows)),
    "written_count": len(written)
}

audit = {"summary": summary, "rows": rows, "written": written}

(AUDITS / "legacy_auto_classification_audit.json").write_text(
    json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8"
)

(AUDITS / "legacy_auto_classification_audit.md").write_text(
    "# Legacy Auto Classification Audit\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in summary.items())
    + "\n\n## Domain counts\n"
    + "\n".join(f"- {k}: {v}" for k, v in sorted(summary["by_domain"].items()))
    + "\n",
    encoding="utf-8"
)

print(json.dumps(summary, ensure_ascii=False, indent=2))


