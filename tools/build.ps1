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

function Get-VcpkgToolchainArgs {
    $candidates = @()

    if ($env:VCPKG_ROOT) {
        $candidates += Join-Path $env:VCPKG_ROOT 'scripts\buildsystems\vcpkg.cmake'
    }

    $candidates += 'C:\vcpkg\scripts\buildsystems\vcpkg.cmake'

    $toolchain = $candidates |
        Where-Object { Test-Path $_ } |
        Select-Object -First 1

    if ($toolchain) {
        return @('-DCMAKE_TOOLCHAIN_FILE=' + $toolchain)
    }

    return @()
}

$cmake = Get-Command cmake -ErrorAction SilentlyContinue
$make = Get-Command make -ErrorAction SilentlyContinue

if (-not $cmake -and -not $make) {
    throw 'No build tool found. Install CMake or GNU Make, then rerun tools/build.ps1.'
}

if ($cmake) {
    $ToolchainArgs = Get-VcpkgToolchainArgs

    Write-Host '[1/3] native tests (CMake)'
    $NativeBuild = Join-Path $BuildRoot 'native'
    Invoke-Tool 'cmake' (@('-S', $NativeDir, '-B', $NativeBuild) + $ToolchainArgs)
    Invoke-Tool 'cmake' @('--build', $NativeBuild, '--config', 'Debug', '--parallel')
    Invoke-Tool 'ctest' @('--test-dir', $NativeBuild, '--build-config', 'Debug', '--output-on-failure')

    Write-Host '[2/3] native sanitizers (CMake)'
    $SanBuild = Join-Path $BuildRoot 'native-asan'
    Invoke-Tool 'cmake' (@('-S', $NativeDir, '-B', $SanBuild, '-DNIYAH_ENABLE_ASAN=ON') + $ToolchainArgs)
    Invoke-Tool 'cmake' @('--build', $SanBuild, '--config', 'Debug', '--parallel')
    Invoke-Tool 'ctest' @('--test-dir', $SanBuild, '--build-config', 'Debug', '--output-on-failure')
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
Invoke-Tool 'cmake' (@('-S', $SearchDir, '-B', $SearchBuild) + $ToolchainArgs)
Invoke-Tool 'cmake' @('--build', $SearchBuild, '--config', 'Debug', '--parallel')
Invoke-Tool 'ctest' @('--test-dir', $SearchBuild, '--build-config', 'Debug', '--output-on-failure')

Write-Host 'local build: PASS'