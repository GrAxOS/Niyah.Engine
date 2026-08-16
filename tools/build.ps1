$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$NativeDir = Join-Path $Root 'native'
$BuildRoot = Join-Path $Root 'build'

function Invoke-Tool {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $Name @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

$cmake = Get-Command cmake -ErrorAction SilentlyContinue
$make = Get-Command make -ErrorAction SilentlyContinue

if (-not $cmake -and -not $make) {
    throw 'No build tool found. Install CMake or GNU Make, then rerun tools/build.ps1.'
}

if ($cmake) {
    Write-Host '[1/3] native tests (CMake)'
    $NativeBuild = Join-Path $BuildRoot 'native'
    Invoke-Tool 'cmake' @('-S', $NativeDir, '-B', $NativeBuild, '-DCMAKE_BUILD_TYPE=Debug')
    Invoke-Tool 'cmake' @('--build', $NativeBuild, '--parallel')
    Invoke-Tool 'ctest' @('--test-dir', $NativeBuild, '--output-on-failure')

    Write-Host '[2/3] native sanitizers (CMake)'
    $SanBuild = Join-Path $BuildRoot 'native-asan'
    Invoke-Tool 'cmake' @('-S', $NativeDir, '-B', $SanBuild, '-DCMAKE_BUILD_TYPE=Debug', '-DNIYAH_ENABLE_ASAN=ON')
    Invoke-Tool 'cmake' @('--build', $SanBuild, '--parallel')
    Invoke-Tool 'ctest' @('--test-dir', $SanBuild, '--output-on-failure')
} else {
    Write-Host '[1/3] native tests (GNU Make)'
    & make -C $NativeDir clean test
    if ($LASTEXITCODE -ne 0) { throw "make failed with exit code $LASTEXITCODE" }

    Write-Host '[2/3] native sanitizers (GNU Make)'
    & make -C $NativeDir asan
    if ($LASTEXITCODE -ne 0) { throw "make asan failed with exit code $LASTEXITCODE" }
}

Write-Host '[3/3] search CMake tests'
$SearchDir = Join-Path $Root 'search'
$SearchBuild = Join-Path $BuildRoot 'search'
Invoke-Tool 'cmake' @('-S', $SearchDir, '-B', $SearchBuild, '-DCMAKE_BUILD_TYPE=Debug')
Invoke-Tool 'cmake' @('--build', $SearchBuild, '--parallel')
Invoke-Tool 'ctest' @('--test-dir', $SearchBuild, '--output-on-failure')

Write-Host 'local build: PASS'