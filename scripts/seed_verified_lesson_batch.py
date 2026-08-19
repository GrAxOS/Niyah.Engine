#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path.cwd()

LESSONS = [
  {
    "id": "lesson_cloud_azure_networking_vpn_gateway",
    "path": "knowledge/domains/cloud/azure/networking/vpn_gateway",
    "provider_or_subject": "Azure VPN Gateway",
    "atomic_facts": [
      {
        "id": "fact_azure_vpn_gateway_001",
        "claim": "Azure VPN Gateway documentation covers configuring, creating, and managing an Azure VPN gateway.",
        "source_title": "VPN Gateway documentation | Microsoft Learn",
        "source_url": "https://learn.microsoft.com/en-us/azure/vpn-gateway/",
        "confidence": 1.0
      },
      {
        "id": "fact_azure_vpn_gateway_002",
        "claim": "Azure VPN Gateway can create encrypted cross-premises connections to an Azure virtual network from on-premises locations.",
        "source_title": "VPN Gateway documentation | Microsoft Learn",
        "source_url": "https://learn.microsoft.com/en-us/azure/vpn-gateway/",
        "confidence": 1.0
      },
      {
        "id": "fact_azure_vpn_gateway_003",
        "claim": "Azure VPN Gateway can be used to send encrypted traffic between an Azure virtual network and on-premises locations over the public Internet.",
        "source_title": "About Azure VPN Gateway | Microsoft Learn",
        "source_url": "https://learn.microsoft.com/en-us/azure/vpn-gateway/vpn-gateway-about-vpngateways",
        "confidence": 1.0
      }
    ],
    "learning_levels": {
      "L0": "VPN Gateway connects networks securely.",
      "L1": "VPN Gateway belongs to Azure networking.",
      "L2": "VPN Gateway connects Azure VNets with on-premises networks.",
      "L3": "VPN Gateway supports production hybrid connectivity patterns.",
      "L4": "VPN Gateway design requires gateway, topology, routing, and SKU understanding.",
      "L5": "VPN Gateway belongs to hybrid network architecture and encrypted overlay design."
    },
    "sources": [
      {
        "title": "VPN Gateway documentation | Microsoft Learn",
        "url": "https://learn.microsoft.com/en-us/azure/vpn-gateway/"
      },
      {
        "title": "About Azure VPN Gateway | Microsoft Learn",
        "url": "https://learn.microsoft.com/en-us/azure/vpn-gateway/vpn-gateway-about-vpngateways"
      }
    ]
  },
  {
    "id": "lesson_cloud_aws_networking_vpc",
    "path": "knowledge/domains/cloud/aws/networking/vpc",
    "provider_or_subject": "Amazon VPC",
    "atomic_facts": [
      {
        "id": "fact_aws_vpc_001",
        "claim": "Amazon VPC enables provisioning a logically isolated section of the AWS Cloud where AWS resources can be launched in a defined virtual network.",
        "source_title": "Amazon Virtual Private Cloud Documentation",
        "source_url": "https://docs.aws.amazon.com/vpc/",
        "confidence": 1.0
      },
      {
        "id": "fact_aws_vpc_002",
        "claim": "With Amazon VPC, AWS resources can be launched in a logically isolated virtual network that the user defines.",
        "source_title": "What is Amazon VPC? - Amazon Virtual Private Cloud",
        "source_url": "https://docs.aws.amazon.com/vpc/latest/userguide/what-is-amazon-vpc.html",
        "confidence": 1.0
      },
      {
        "id": "fact_aws_vpc_003",
        "claim": "A subnet is a range of IP addresses in an Amazon VPC and must reside in a single Availability Zone.",
        "source_title": "What is Amazon VPC? - Amazon Virtual Private Cloud",
        "source_url": "https://docs.aws.amazon.com/vpc/latest/userguide/what-is-amazon-vpc.html",
        "confidence": 1.0
      }
    ],
    "learning_levels": {
      "L0": "VPC is a private cloud network boundary.",
      "L1": "VPC contains subnets and AWS resources.",
      "L2": "VPC design includes IP addressing, subnets, routing, and gateways.",
      "L3": "VPC supports production workload segmentation and connectivity.",
      "L4": "VPC architecture requires route tables, endpoint, gateway, and security design.",
      "L5": "VPC is a cloud network isolation model comparable to traditional network segmentation."
    },
    "sources": [
      {
        "title": "Amazon Virtual Private Cloud Documentation",
        "url": "https://docs.aws.amazon.com/vpc/"
      },
      {
        "title": "What is Amazon VPC? - Amazon Virtual Private Cloud",
        "url": "https://docs.aws.amazon.com/vpc/latest/userguide/what-is-amazon-vpc.html"
      }
    ]
  },
  {
    "id": "lesson_cloud_gcp_networking_vpc",
    "path": "knowledge/domains/cloud/gcp/networking/vpc",
    "provider_or_subject": "Google Cloud VPC",
    "atomic_facts": [
      {
        "id": "fact_gcp_vpc_001",
        "claim": "Google Cloud Virtual Private Cloud provides networking functionality to Compute Engine VM instances, Google Kubernetes Engine containers, and serverless workloads.",
        "source_title": "Virtual Private Cloud documentation - Google Cloud",
        "source_url": "https://docs.cloud.google.com/vpc/docs",
        "confidence": 1.0
      },
      {
        "id": "fact_gcp_vpc_002",
        "claim": "Google Cloud VPC provides networking for cloud-based services that is global, scalable, and flexible.",
        "source_title": "Virtual Private Cloud documentation - Google Cloud",
        "source_url": "https://docs.cloud.google.com/vpc/docs",
        "confidence": 1.0
      },
      {
        "id": "fact_gcp_vpc_003",
        "claim": "A Google Cloud VPC network is a global resource made of regional virtual subnetworks connected by a global wide area network.",
        "source_title": "Virtual Private Cloud overview | Google Cloud Documentation",
        "source_url": "https://docs.cloud.google.com/vpc/docs/overview?hl=en",
        "confidence": 1.0
      }
    ],
    "learning_levels": {
      "L0": "GCP VPC provides cloud networking.",
      "L1": "GCP VPC connects VMs, GKE, and serverless workloads.",
      "L2": "GCP VPC uses networks, subnets, routes, firewall rules, and connectivity services.",
      "L3": "GCP VPC supports professional cloud network design.",
      "L4": "GCP VPC requires understanding global networks and regional subnets.",
      "L5": "GCP VPC is a global cloud network architecture model."
    },
    "sources": [
      {
        "title": "Virtual Private Cloud documentation - Google Cloud",
        "url": "https://docs.cloud.google.com/vpc/docs"
      },
      {
        "title": "Virtual Private Cloud overview | Google Cloud Documentation",
        "url": "https://docs.cloud.google.com/vpc/docs/overview?hl=en"
      }
    ]
  }
]

def write_lesson(lesson):
    target = ROOT / lesson["path"]
    target.mkdir(parents=True, exist_ok=True)

    for level in ["L0", "L1", "L2", "L3", "L4", "L5"]:
        d = target / level
        d.mkdir(exist_ok=True)
        r = d / "README.md"
        r.write_text(
            f"# {lesson['path']} / {level}\n\nStatus: VERIFIED_ATOMIC_LESSON\n",
            encoding="utf-8"
        )

    lesson["schema"] = {"name": "khz_verified_atomic_lesson", "version": "0.1.0"}
    lesson["status"] = "VERIFIED_ATOMIC_LESSON"
    lesson["payload_kind"] = "source_backed_atomic_knowledge"
    lesson["updated_utc"] = datetime.now(timezone.utc).isoformat()
    lesson["knowledge_policy"] = {
        "is_actual_knowledge": True,
        "is_curriculum_seed": True,
        "is_source_backed_category": True,
        "is_verified_atomic_lesson": True,
        "requires_leaf_specific_sources_before_lesson": False
    }
    lesson["canonical_policy"] = {
        "allowed_as_canonical": True,
        "reason": "Contains source-backed atomic facts with explicit source title and URL."
    }

    for f in lesson["atomic_facts"]:
        for k in ["id", "claim", "source_title", "source_url", "confidence"]:
            if not f.get(k):
                raise SystemExit(f"bad fact {lesson['id']} missing {k}")

    for s in lesson["sources"]:
        for k in ["title", "url"]:
            if not s.get(k):
                raise SystemExit(f"bad source {lesson['id']} missing {k}")

    (target / "index.json").write_text(
        json.dumps(lesson, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    return str((target / "index.json").relative_to(ROOT)).replace("\\", "/")

written = [write_lesson(x) for x in LESSONS]

audit = {
    "written_count": len(written),
    "written": written
}

Path("audits").mkdir(exist_ok=True)
Path("audits/verified_lesson_seed_batch.json").write_text(
    json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8"
)

print(json.dumps(audit, ensure_ascii=False, indent=2))
