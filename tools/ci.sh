#!/bin/sh
# Local check runner. Replaces the removed GitHub Actions workflow: same
# steps, executed on the developer machine.
#
#   sh tools/ci.sh            all stages
#   sh tools/ci.sh native     one stage: native | make | search | python
#
# Requires: cmake >= 3.20, ctest, a C11/C++17 toolchain, libcurl (search),
# python3 (tooling).

set -eu

STAGE="${1:-all}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

run_native() {
    cmake -S native -B build/native -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    cmake --build build/native --config "$BUILD_TYPE"
    # The suites #undef NDEBUG, so assertions run in Release too.
    ctest --test-dir build/native -C "$BUILD_TYPE" --output-on-failure
}

run_make() {
    make -C native lib
    make -C native test
}

run_search() {
    cmake -S search -B build/search -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    cmake --build build/search --config "$BUILD_TYPE"
    ctest --test-dir build/search -C "$BUILD_TYPE" --output-on-failure
}

run_python() {
    python3 -m compileall -q tools neutral
    # Writes real GGUF fixtures and converts them; see the module docstring
    # in tools/tests/test_convert_gguf.py for what is and is not covered.
    python3 tools/tests/test_convert_gguf.py
}

case "$STAGE" in
    native) run_native ;;
    make)   run_make ;;
    search) run_search ;;
    python) run_python ;;
    all)
        run_native
        run_make
        run_search
        run_python
        ;;
    *)
        echo "unknown stage: $STAGE (native|make|search|python|all)" >&2
        exit 2
        ;;
esac

echo "ok: $STAGE"
