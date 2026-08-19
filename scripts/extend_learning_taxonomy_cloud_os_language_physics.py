#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path.cwd()
REG = ROOT / "knowledge" / "00_registry" / "learning_taxonomy.json"

def deep_merge(a, b):
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            deep_merge(a[k], v)
        else:
            a[k] = v
    return a

data = json.loads(REG.read_text(encoding="utf-8"))

extra = {
  "domains": {
    "cloud": {
      "azure": {
        "identity": ["microsoft_entra_id", "managed_identities", "rbac", "conditional_access", "key_vault"],
        "compute": ["virtual_machines", "vm_scale_sets", "app_service", "functions", "batch", "aks"],
        "containers": ["aks", "container_apps", "container_instances", "container_registry"],
        "networking": ["virtual_network", "subnets", "nsg", "route_tables", "vpn_gateway", "expressroute", "load_balancer", "application_gateway", "front_door", "dns", "private_link"],
        "storage": ["blob_storage", "files", "queues", "tables", "managed_disks", "archive_storage"],
        "databases": ["azure_sql", "postgresql", "mysql", "cosmos_db", "cache_for_redis"],
        "security": ["defender_for_cloud", "sentinel", "key_vault", "firewall", "ddos_protection", "private_endpoint"],
        "monitoring_governance": ["monitor", "log_analytics", "policy", "blueprints", "cost_management", "resource_graph"],
        "devops": ["azure_devops", "github_actions", "bicep", "arm_templates", "terraform"],
        "ai_data": ["azure_ai_services", "azure_machine_learning", "azure_ai_search", "fabric", "synapse", "databricks"],
        "hybrid": ["azure_arc", "azure_stack_hci", "site_recovery", "backup"]
      },
      "aws": {
        "identity": ["iam", "organizations", "identity_center", "kms", "secrets_manager"],
        "compute": ["ec2", "lambda", "lightsail", "batch", "elastic_beanstalk"],
        "containers": ["ecs", "eks", "ecr", "fargate"],
        "networking": ["vpc", "subnets", "route_tables", "security_groups", "nacl", "elb", "route53", "cloudfront", "direct_connect", "transit_gateway", "private_link"],
        "storage": ["s3", "efs", "fsx", "ebs", "glacier", "storage_gateway"],
        "databases": ["rds", "aurora", "dynamodb", "redshift", "neptune", "documentdb", "elasticache"],
        "security": ["guardduty", "security_hub", "inspector", "macie", "waf", "shield", "cloudtrail"],
        "monitoring_governance": ["cloudwatch", "config", "systems_manager", "control_tower", "service_catalog", "cost_explorer"],
        "devops": ["codecommit", "codebuild", "codedeploy", "codepipeline", "cloudformation", "cdk"],
        "ai_data": ["sagemaker", "bedrock", "athena", "emr", "glue", "kinesis", "quicksight"],
        "hybrid_edge": ["outposts", "local_zones", "wavelength", "snow"]
      },
      "gcp": {
        "identity": ["cloud_identity", "iam", "identity_platform", "cloud_kms", "secret_manager"],
        "compute": ["compute_engine", "cloud_run", "app_engine", "cloud_functions", "batch"],
        "containers": ["gke", "artifact_registry", "cloud_deploy"],
        "networking": ["vpc", "subnets", "firewall_rules", "cloud_nat", "cloud_dns", "cloud_load_balancing", "cloud_cdn", "cloud_interconnect", "private_service_connect"],
        "storage": ["cloud_storage", "persistent_disk", "filestore", "backup_dr"],
        "databases": ["cloud_sql", "alloydb", "spanner", "bigtable", "firestore", "memorystore"],
        "security": ["security_command_center", "cloud_armor", "cloud_ids", "binary_authorization", "access_context_manager"],
        "monitoring_governance": ["cloud_monitoring", "cloud_logging", "cloud_trace", "recommender", "service_catalog", "quotas"],
        "devops": ["cloud_build", "cloud_deploy", "cloud_shell", "deployment_manager", "infra_manager"],
        "ai_data": ["vertex_ai", "bigquery", "dataflow", "dataproc", "pubsub", "looker", "document_ai"],
        "hybrid_multicloud": ["google_distributed_cloud", "anthos", "bare_metal_solution", "vmware_engine"]
      },
      "huawei_cloud": {
        "identity": ["iam", "kms", "enterprise_project_management"],
        "compute": ["ecs", "flexus", "bare_metal_server", "auto_scaling"],
        "containers_middleware": ["cce", "swr", "service_stage", "functiongraph", "roma_connect"],
        "networking": ["vpc", "elastic_ip", "elb", "vpn", "direct_connect", "nat_gateway", "cdn", "dns"],
        "storage": ["obs", "evs", "sfs", "cbr"],
        "databases": ["rds_mysql", "rds_postgresql", "gaussdb", "geminidb", "dcs_redis", "dds_mongodb"],
        "security": ["waf", "hss", "anti_ddos", "dbss", "cloud_firewall"],
        "management_o_and_m": ["cloud_eye", "cts", "config", "resource_management", "cost_management"],
        "developer_services": ["codearts", "codearts_pipeline", "codearts_build", "codearts_repo"],
        "ai_data": ["modelarts", "pangu_large_models", "ges", "mrs", "dli", "dws", "dataarts_studio"],
        "hybrid_stack": ["huawei_cloud_stack", "cloudpond", "cloud_connect"]
      }
    },
    "operating_systems": {
      "linux": {
        "ubuntu": ["installation", "apt", "systemd", "netplan", "ufw", "apparmor", "snap", "server_admin", "cloud_images", "kernel_hwe"],
        "kali": ["installation", "package_management", "network_tools", "wireless_tools", "forensics_tools", "web_testing_tools", "hardening", "lab_safety"],
        "debian": ["apt", "dpkg", "systemd", "networking", "security_hardening"],
        "rhel_family": ["rhel", "rocky", "alma", "dnf", "selinux", "systemd", "firewalld"],
        "arch": ["pacman", "aur", "systemd", "rolling_release"],
        "kernel": ["boot", "syscalls", "vfs", "scheduler", "memory_manager", "network_stack", "drivers", "ebpf", "cgroups", "namespaces", "lsm"]
      },
      "windows": {
        "client": ["windows_10", "windows_11", "registry", "services", "event_viewer", "powershell"],
        "server": ["active_directory", "dns", "dhcp", "iis", "hyper_v", "failover_clustering", "gpo", "wsus"],
        "internals": ["nt_kernel", "object_manager", "io_manager", "memory_manager", "process_threads", "syscalls", "drivers", "etw"]
      }
    },
    "human_languages": {
      "arabic": {
        "beginner": ["letters", "sounds", "short_vowels", "basic_words"],
        "grammar": ["nahw", "sarf", "i3rab", "sentence_structure"],
        "morphology": ["root_pattern", "derivation", "stemming", "lemmatization"],
        "semantics": ["meaning", "synonymy", "ambiguity", "named_entities"],
        "dialects": ["gulf", "najdi", "hijazi", "levantine", "egyptian", "maghrebi"],
        "computational_arabic": ["tokenization", "normalization", "diacritization", "ner", "pos_tagging", "dependency_parsing"]
      },
      "english": {
        "beginner": ["alphabet", "sounds", "basic_vocabulary"],
        "grammar": ["parts_of_speech", "tenses", "clauses", "sentence_structure"],
        "technical_english": ["requirements", "documentation", "api_docs", "rfc_style"],
        "academic_english": ["papers", "abstracts", "citations", "argument_structure"],
        "computational_english": ["tokenization", "tagging", "parsing", "semantics"]
      }
    },
    "science": {
      "physics": {
        "classical_mechanics": ["motion", "forces", "energy", "momentum", "rotation", "oscillation"],
        "electromagnetism": ["charge", "electric_fields", "magnetic_fields", "circuits", "maxwell_equations"],
        "thermodynamics": ["temperature", "heat", "entropy", "engines", "statistical_mechanics"],
        "waves_optics": ["waves", "sound", "interference", "diffraction", "geometric_optics"],
        "relativity": ["special_relativity", "spacetime", "general_relativity", "gravity"],
        "quantum": ["wave_function", "operators", "spin", "measurement", "quantum_information"],
        "solid_state": ["crystals", "bands", "semiconductors", "superconductivity"],
        "computational_physics": ["numerical_methods", "simulation", "monte_carlo", "finite_elements"]
      }
    }
  }
}

data = deep_merge(data, extra)
REG.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("updated", REG)
