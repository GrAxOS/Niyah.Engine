Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root
powershell -ExecutionPolicy Bypass -File scripts\verify_khz_realtime.ps1
python scripts\build_khz_realtime_pages.py
git add .gitlab-ci.yml scripts\build_khz_realtime_pages.py scripts\run_khz_realtime_local.ps1 scripts\verify_khz_realtime.ps1 scripts\deploy-khz-realtime.ps1 public\index.html public\khz_realtime_bundle.json public\khz_realtime_snapshot.json
git diff --cached --check
git status --short
git commit -m "Add realtime KHZ GitLab Pages analysis"
git push origin HEAD
