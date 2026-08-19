Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root
python scripts\build_khz_realtime_pages.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Start-Process "http://localhost:8080/"
python -m http.server 8080 --directory public
