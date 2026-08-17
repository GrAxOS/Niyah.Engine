$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
$NativeDir = Join-Path $Root 'native'
$SearchDir = Join-Path $Root 'search'
$BuildRoot = Join-Path $Root 'build'

function Invoke-Tool {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $Name @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode"
    }
    return $exitCode
}

function Assert-Artifact {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required build artifact is missing: $Path"
    }
}

function Invoke-Smoke {
    param([Parameter(Mandatory = $true)][string]$Path)
    Assert-Artifact $Path
    & $Path
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Smoke test failed with exit code ${exitCode}: $Path"
    }
    return $exitCode
}

function Get-VcpkgToolchainArgs {
    $roots = @()
    if ($env:VCPKG_ROOT) { $roots += $env:VCPKG_ROOT }
    $roots += 'C:\vcpkg'

    foreach ($root in ($roots | Select-Object -Unique)) {
        $toolchain = Join-Path $root 'scripts\buildsystems\vcpkg.cmake'
        if (Test-Path -LiteralPath $toolchain -PathType Leaf) {
            $env:VCPKG_ROOT = $root
            return @(
                ('-DCMAKE_TOOLCHAIN_FILE={0}' -f $toolchain),
                '-DVCPKG_TARGET_TRIPLET=x64-windows'
            )
        }
    }
    return @()
}

$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    throw 'CMake is required for the integrated build. Install CMake and rerun tools/build.ps1.'
}

$ToolchainArgs = Get-VcpkgToolchainArgs
if ($ToolchainArgs.Count -gt 0) {
    Write-Host ('Using vcpkg: ' + $env:VCPKG_ROOT)
}

Write-Host '[1/3] native build + tests'
$NativeBuild = Join-Path $BuildRoot 'native'
Invoke-Tool 'cmake' (@('-S', $NativeDir, '-B', $NativeBuild) + $ToolchainArgs)
Invoke-Tool 'cmake' @('--build', $NativeBuild, '--config', 'Debug', '--parallel')
Invoke-Tool 'ctest' @('--test-dir', $NativeBuild, '--build-config', 'Debug', '--output-on-failure')

Write-Host '[2/3] native sanitizers'
$SanBuild = Join-Path $BuildRoot 'native-asan'
Invoke-Tool 'cmake' (@('-S', $NativeDir, '-B', $SanBuild, '-DNIYAH_ENABLE_ASAN=ON') + $ToolchainArgs)
Invoke-Tool 'cmake' @('--build', $SanBuild, '--config', 'Debug', '--parallel')
Invoke-Tool 'ctest' @('--test-dir', $SanBuild, '--build-config', 'Debug', '--output-on-failure')

Write-Host '[3/3] search build + smoke'
$SearchBuild = Join-Path $BuildRoot 'search'
Invoke-Tool 'cmake' (@('-S', $SearchDir, '-B', $SearchBuild) + $ToolchainArgs)
Invoke-Tool 'cmake' @('--build', $SearchBuild, '--config', 'Debug', '--parallel')

$isWindowsHost = $env:OS -eq 'Windows_NT'
$SmokeName = if ($isWindowsHost) { 'niyah_search_smoke.exe' } else { 'niyah_search_smoke' }
$SmokeCandidates = @(
    (Join-Path $SearchBuild (Join-Path 'Debug' $SmokeName)),
    (Join-Path $SearchBuild $SmokeName)
)
$SmokePath = $SmokeCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $SmokePath) {
    throw "Required search smoke executable is missing. Checked: $($SmokeCandidates -join ', ')"
}

Invoke-Smoke $SmokePath
Invoke-Tool 'ctest' @('--test-dir', $SearchBuild, '--build-config', 'Debug', '--output-on-failure')

Write-Host 'BUILD=PASS'
Write-Host 'SMOKE=PASS'
Write-Host 'TESTS=PASS'
Write-Host 'OVERALL=PASS'
