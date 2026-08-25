Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

python scripts\build_khz_realtime_pages.py --verify-only

if ($LASTEXITCODE -ne 0) {
    throw "KHZ realtime parser verification failed."
}

$required = @(
    "public\khz_realtime_bundle.json",
    "public\khz_realtime_snapshot.json"
)

foreach ($file in $required) {
    if (-not (Test-Path $file)) {
        throw "Missing: $file"
    }

    if ((Get-Item $file).Length -le 0) {
        throw "Empty: $file"
    }
}

$bundle = Get-Content "public\khz_realtime_bundle.json" -Raw | ConvertFrom-Json

Write-Output "KHZ_REALTIME_SCAN_OK"
Write-Output "State: $($bundle.state)"
Write-Output "Snapshot: $($bundle.snapshot_id)"
Write-Output "JSON files: $($bundle.files.json_files)"
Write-Output "Nodes: $($bundle.graph.unique_nodes)"
Write-Output "Edges: $($bundle.graph.edge_occurrences)"
Write-Output "Evidence: $($bundle.graph.evidence_occurrences)"
Write-Output "Parse errors: $($bundle.files.parse_errors)"
Write-Output "Duplicate node IDs: $(@($bundle.graph.duplicate_node_ids.PSObject.Properties).Count)"
Write-Output "Dangling edges: $(@($bundle.graph.dangling_edges).Count)"

if ($bundle.files.parse_errors -gt 0) {
    throw "Parse errors detected."
}

exit 0
