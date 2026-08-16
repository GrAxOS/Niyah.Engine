$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot

Write-Host '[1/3] native tests'
& make -C (Join-Path $Root 'native') clean test

Write-Host '[2/3] native sanitizers'
& make -C (Join-Path $Root 'native') asan

Write-Host '[3/3] search CMake tests'
$BuildDir = Join-Path $Root 'build/search'
cmake -S (Join-Path $Root 'search') -B $BuildDir -DCMAKE_BUILD_TYPE=Debug
cmake --build $BuildDir --parallel
ctest --test-dir $BuildDir --output-on-failure

Write-Host 'local build: PASS'
