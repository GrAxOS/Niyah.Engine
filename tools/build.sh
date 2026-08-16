#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

printf '%s\n' '[1/3] native tests'
make -C "$ROOT/native" clean test

printf '%s\n' '[2/3] native sanitizers'
make -C "$ROOT/native" asan

printf '%s\n' '[3/3] search CMake tests'
BUILD_DIR="$ROOT/build/search"
cmake -S "$ROOT/search" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug
cmake --build "$BUILD_DIR" --parallel
ctest --test-dir "$BUILD_DIR" --output-on-failure

printf '%s\n' 'local build: PASS'
