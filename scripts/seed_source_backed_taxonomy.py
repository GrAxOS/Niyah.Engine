#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path.cwd()
BASE = ROOT / "knowledge" / "10_domains"
AUDITS = ROOT / "audits"
SCRIPT = ROOT / "scripts"

AUDITS.mkdir(exist_ok=True)
SCRIPT.mkdir(exist_ok=True)

SOURCE_PACKS = [
    {
        "prefix": "cloud/azure",
        "provider": "Microsoft Azure",
        "source_type": "official_product_category_page",
        "sources": [
            {
                "title": "Azure Products - Browse by Category",
                "url": "https://azure.microsoft.com/en-us/products/category/",
                "supports": [
                    "Azure product categories",
                    "AI + Machine Learning",
                    "Analytics",
                    "Compute",
                    "Containers",
                    "Databases",
                    "Developer Tools",
                    "DevOps",
                    "Hybrid + multicloud",
                    "Identity",
                    "Integration",
                    "Internet of Things",
                    "Management and Governance",
                    "Migration",
                    "Networking",
                    "Security",
                    "Storage",
                    "Virtual desktop infrastructure",
                    "Web"
                ]
            }
        ]
    },
    {
        "prefix": "cloud/aws",
        "provider": "Amazon Web Services",
        "source_type": "official_services_by_category_page",
        "sources": [
            {
                "title": "AWS services by category - Overview of Amazon Web Services",
                "url": "https://docs.aws.amazon.com/whitepapers/latest/aws-overview/amazon-web-services-cloud-platform.html",
                "supports": [
                    "AWS services by category",
                    "Analytics",
                    "Application integration",
                    "Compute",
                    "Containers",
                    "Databases",
                    "Developer tools",
                    "Internet of Things",
                    "Machine Learning and Artificial Intelligence",
                    "Management and governance",
                    "Migration and transfer",
                    "Networking and content delivery",
                    "Security identity and compliance",
                    "Storage"
                ]
            }
        ]
    },
    {
        "prefix": "cloud/gcp",
        "provider": "Google Cloud",
        "source_type": "official_products_page",
        "sources": [
            {
                "title": "Google Cloud Products and Services",
                "url": "https://cloud.google.com/products",
                "supports": [
                    "Google Cloud products by category",
                    "AI/ML",
                    "Infrastructure",
                    "Databases and analytics",
                    "Developer tools",
                    "App development",
                    "Integration services",
                    "Management tools",
                    "Security and identity",
                    "Web and app hosting",
                    "Productivity and collaboration"
                ]
            }
        ]
    },
    {
        "prefix": "cloud/huawei_cloud",
        "provider": "Huawei Cloud",
        "source_type": "official_products_page",
        "sources": [
            {
                "title": "Huawei Cloud Products",
                "url": "https://www.huaweicloud.com/intl/en-us/product/",
                "supports": [
                    "Huawei Cloud products by category",
                    "AI",
                    "Analytics",
                    "Business Applications",
                    "Compute",
                    "Containers & Middleware",
                    "Content Delivery & Edge Computing",
                    "Databases",
                    "Developer Services",
                    "Huawei Cloud Stack",
                    "Media Services",
                    "Migration & O&M",
                    "Management",
                    "Networking",
                    "Security",
                    "Storage"
                ]
            },
            {
                "title": "Huawei Cloud Saudi Arabia Region",
                "url": "https://activity.huaweicloud.com/intl/en-us/saudi_arabia_region.html",
                "supports": [
                    "Huawei Cloud services in Saudi Arabia span infrastructure, databases, containers, big data, and AI",
                    "Huawei Cloud has a data center in Riyadh according to the cited page"
                ]
            }
        ]
    },
    {
        "prefix": "operating_systems/linux/ubuntu",
        "provider": "Ubuntu",
        "source_type": "official_documentation",
        "sources": [
            {
                "title": "Ubuntu Server documentation",
                "url": "https://ubuntu.com/server/docs/",
                "supports": [
                    "Ubuntu Server documentation",
                    "installation",
                    "system basics",
                    "networking",
                    "security",
                    "managing system",
                    "data and storage",
                    "web and mail services"
                ]
            },
            {
                "title": "About Netplan - Ubuntu Server documentation",
                "url": "https://ubuntu.com/server/docs/explanation/networking/about-netplan/",
                "supports": [
                    "Netplan handles network configuration on Ubuntu",
                    "Netplan uses YAML configuration files",
                    "Netplan integrates with NetworkManager and systemd-networkd"
                ]
            }
        ]
    },
    {
        "prefix": "operating_systems/linux/kali",
        "provider": "Kali Linux",
        "source_type": "official_documentation",
        "sources": [
            {
                "title": "Kali Docs",
                "url": "https://www.kali.org/docs/",
                "supports": [
                    "Kali official documentation",
                    "installation",
                    "virtualization",
                    "USB",
                    "ARM",
                    "containers",
                    "WSL",
                    "cloud",
                    "tools",
                    "troubleshooting",
                    "development"
                ]
            },
            {
                "title": "Kali Tools documentation",
                "url": "https://www.kali.org/docs/tools/",
                "supports": [
                    "Tools inside of Kali",
                    "Kali Tools",
                    "Metasploit Framework",
                    "removed tools",
                    "submitting tools to Kali"
                ]
            }
        ]
    },
    {
        "prefix": "human_languages/arabic",
        "provider": "Arabic language",
        "source_type": "language_reference",
        "sources": [
            {
                "title": "Arabic Morphology Fundamentals",
                "url": "https://arabicagenticai.com/encyclopedia/arabic-morphology/",
                "supports": [
                    "Arabic morphology is organized around a root-pattern system",
                    "Most Arabic words derive from a consonantal root",
                    "The root k-t-b relates to writing"
                ]
            },
            {
                "title": "Arabic Root System",
                "url": "https://blog.alifbee.com/arabic-root-system/",
                "supports": [
                    "Arabic root system forms words from consonantal bases",
                    "Most roots consist of three consonants",
                    "Patterns transform roots into specific words"
                ]
            }
        ]
    },
    {
        "prefix": "human_languages/english",
        "provider": "English language",
        "source_type": "language_reference",
        "sources": [
            {
                "title": "Britannica - Part of speech",
                "url": "https://www.britannica.com/topic/part-of-speech",
                "supports": [
                    "Traditional English grammar has eight parts of speech",
                    "Noun",
                    "Pronoun",
                    "Verb",
                    "Adjective",
                    "Adverb",
                    "Conjunction",
                    "Preposition",
                    "Interjection"
                ]
            },
            {
                "title": "Cambridge Dictionary - Part of speech",
                "url": "https://dictionary.cambridge.org/grammar/british-grammar/part-of-speech",
                "supports": [
                    "Part of speech is part of English grammar reference"
                ]
            }
        ]
    },
    {
        "prefix": "science/physics",
        "provider": "Physics",
        "source_type": "education_reference",
        "sources": [
            {
                "title": "David Tong Teaching",
                "url": "https://davidtong.org/teaching/",
                "supports": [
                    "Classical Mechanics",
                    "Dynamics and Relativity",
                    "Electromagnetism",
                    "Quantum Mechanics",
                    "Solid State Physics",
                    "Fluid Mechanics",
                    "Statistical Physics"
                ]
            },
            {
                "title": "What Is Taught in Physics",
                "url": "https://scienceinsights.org/what-is-taught-in-physics-from-mechanics-to-relativity/",
                "supports": [
                    "Physics courses cover motion, energy, forces, matter",
                    "Core areas include classical mechanics, electromagnetism, quantum mechanics, thermodynamics, and modern physics"
                ]
            }
        ]
    }
]

def rel(p):
    return str(p.relative_to(ROOT)).replace("\\", "/")

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def find_leaf_dirs(prefix):
    start = BASE / prefix
    if not start.exists():
        return []
    leaves = []
    for p in sorted(start.rglob("*")):
        if not p.is_dir():
            continue
        if p.name in {"L0", "L1", "L2", "L3", "L4", "L5"}:
            continue
        children = [c for c in p.iterdir() if c.is_dir() and c.name not in {"L0", "L1", "L2", "L3", "L4", "L5"}]
        has_levels = all((p / lvl).exists() for lvl in ["L0", "L1", "L2", "L3", "L4", "L5"])
        if has_levels and not children:
            leaves.append(p)
    return leaves

created = []
for pack in SOURCE_PACKS:
    leaves = find_leaf_dirs(pack["prefix"])
    for leaf in leaves:
        parts = rel(leaf).split("/")
        index = {
            "schema": {
                "name": "khz_source_backed_learning_leaf",
                "version": "0.1.0"
            },
            "id": "leaf_" + "_".join(parts[2:]),
            "path": rel(leaf),
            "status": "SOURCE_BACKED_TAXONOMY",
            "provider_or_subject": pack["provider"],
            "source_type": pack["source_type"],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "claims": [
                {
                    "claim": "This leaf is part of a staged KHZ learning taxonomy. It is not yet a full lesson.",
                    "confidence": 1.0,
                    "source_required_for_lesson_expansion": True
                },
                {
                    "claim": "The source pack supports the parent provider or subject category. Service-level deep lessons require additional per-service documentation before being marked canonical.",
                    "confidence": 1.0,
                    "source_required_for_lesson_expansion": True
                }
            ],
            "sources": pack["sources"],
            "learning_levels": {
                "L0": "starter vocabulary and purpose",
                "L1": "basic concepts and names",
                "L2": "basic use and examples",
                "L3": "professional operation and troubleshooting",
                "L4": "internals, limits, failure modes",
                "L5": "research, formal models, architecture tradeoffs"
            },
            "canonical_policy": {
                "allowed_as_canonical": False,
                "reason": "taxonomy leaf only; requires source-backed lesson nodes before canonical export"
            }
        }
        write_json(leaf / "index.json", index)
        created.append(rel(leaf / "index.json"))

report = {
    "source_pack_count": len(SOURCE_PACKS),
    "index_files_created_or_updated": len(created),
    "files": created
}

write_json(AUDITS / "source_backed_taxonomy_seed.json", report)
(AUDITS / "source_backed_taxonomy_seed.md").write_text(
    "# Source Backed Taxonomy Seed\n\n"
    + f"- source_pack_count: {len(SOURCE_PACKS)}\n"
    + f"- index_files_created_or_updated: {len(created)}\n\n"
    + "\n".join(f"- {x}" for x in created)
    + "\n",
    encoding="utf-8"
)

print(json.dumps({
    "source_pack_count": len(SOURCE_PACKS),
    "index_files_created_or_updated": len(created),
    "audit": "audits/source_backed_taxonomy_seed.md"
}, ensure_ascii=False, indent=2))
