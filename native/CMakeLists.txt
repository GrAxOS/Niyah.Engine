cmake_minimum_required(VERSION 3.16)
project(niyah_native LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Sanitizers
#
#   cmake -B build-asan -DNIYAH_SANITIZE=ON && cmake --build build-asan
#   ctest --test-dir build-asan --output-on-failure
#
# Applies to the library and every test target. The suite is small enough that
# CI should run one sanitized job unconditionally.
# ---------------------------------------------------------------------------
option(NIYAH_SANITIZE
       "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(NIYAH_SANITIZE)
    if(MSVC)
        add_compile_options(/fsanitize=address /Zi)
    else()
        add_compile_options(-fsanitize=address,undefined
                            -fno-omit-frame-pointer -g)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

# ---------------------------------------------------------------------------
# Sources
#
# niyah_core.c used to be absent from this list while being present in the
# Makefile, so the CMake build produced a library without niyah_version(),
# the truth helpers or niyah_status_to_string().
# ---------------------------------------------------------------------------
set(NIYAH_SOURCES
    niyah_core.c
    niyah_sha256.c
    niyah_identity.c
    niyah_matmul.c
    niyah_softmax.c
    niyah_rmsnorm.c
    niyah_swiglu.c
    niyah_rope.c
    niyah_attention.c
    niyah_transformer_layer.c
    niyah_sampler.c
    niyah_tokenizer.c
    niyah_runtime.c
    niyah_telemetry.c
    niyah_model.c
    niyah_llm.c
    niyah_graph.c
    niyah_embedding.c
    niyah_source.c
    niyah_storage.c
    niyah_search.c
    niyah_crawler.c
    niyah_document.c
    niyah_bridge.c
    evidence_envelope.c
    evidence_graph.c
    evidence_reasoner.c
    constraint_solver.c
)

set(NIYAH_HEADERS
    niyah.h
    niyah_sha256.h
    niyah_identity.h
    niyah_attention.h
    niyah_bridge.h
    niyah_document.h
    niyah_llm.h
    niyah_runtime.h
    niyah_telemetry.h
    niyah_tokenizer.h
    niyah_transformer_layer.h
    evidence_envelope.h
    evidence_graph.h
    evidence_reasoner.h
    constraint_solver.h
)

# ---------------------------------------------------------------------------
# Static library
# ---------------------------------------------------------------------------
add_library(niyah STATIC ${NIYAH_SOURCES} ${NIYAH_HEADERS})

target_include_directories(niyah PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#
# NIYAH_BRIDGE_EXPORTS is required, not optional.
#
# evidence_envelope.h, evidence_graph.h, evidence_reasoner.h and
# constraint_solver.h all expand their API macro to __declspec(dllimport) on
# _WIN32 unless NIYAH_BRIDGE_EXPORTS is defined. Compiling those .c files
# without it declares each function dllimport and then defines it in the same
# translation unit, which MSVC rejects outright.
#
target_compile_definitions(niyah PRIVATE NIYAH_BRIDGE_EXPORTS)

if(MSVC)
    target_compile_options(niyah PRIVATE /W4 /utf-8)
    # snprintf/fopen deprecation noise from the CRT security warnings.
    target_compile_definitions(niyah PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(niyah PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
if(WIN32)
    target_link_libraries(niyah PUBLIC wininet psapi)
endif()

# expf/sqrtf/cosf live in libm on most Unix toolchains.
if(UNIX)
    target_link_libraries(niyah PUBLIC m)
endif()

# ---------------------------------------------------------------------------
# Shared library for ui/Niyah.App P/Invoke
#
# This block used to be preceded by two statements addressing a target named
# `niyah_dll`, which is never defined by any add_library() call in the tree.
# CMake treats target_link_libraries/set_target_properties on an unknown
# target as a hard configure error, so the CMake build could not even reach
# the compiler. The intended target is this one, and it already carries
# OUTPUT_NAME niyah; the Windows link libraries are attached below.
# ---------------------------------------------------------------------------
add_library(niyah_shared SHARED ${NIYAH_SOURCES} ${NIYAH_HEADERS})

set_target_properties(niyah_shared PROPERTIES
    OUTPUT_NAME niyah
    C_VISIBILITY_PRESET hidden
)

target_include_directories(niyah_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(niyah_shared PRIVATE
    NIYAH_BRIDGE_EXPORTS
    NIYAH_BUILD_SHARED
)

if(MSVC)
    target_compile_options(niyah_shared PRIVATE /utf-8)
    target_compile_definitions(niyah_shared PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

if(WIN32)
    target_link_libraries(niyah_shared PUBLIC wininet psapi)
endif()

if(UNIX)
    target_link_libraries(niyah_shared PUBLIC m)
endif()

# niyah_mini submodule
#
# Must come AFTER the `niyah` target is defined so that
# niyah_mini/CMakeLists.txt sees `if(TARGET niyah)` as true and links
# the bridge against libniyah rather than falling back to find_library.
# ---------------------------------------------------------------------------
add_subdirectory(niyah_mini)

# ---------------------------------------------------------------------------
# Measurement tools
#
# niyah_token_tax reports what the tokeniser costs per language over a
# parallel corpus, plus a morphology probe for Arabic root families.
#
# It is an instrument, not part of the inference path, so it is a plain
# executable and is deliberately not registered with ctest: its output is a
# measurement to read, not an assertion to pass or fail.
#
#   ./niyah_token_tax --vocab <vocab.txt> \
#                     --parallel testdata/parallel_ar_en.tsv \
#                     --families testdata/root_families_ar.txt
# ---------------------------------------------------------------------------
set(NIYAH_TOOLS
    niyah_token_tax.c
)

foreach(tool_source ${NIYAH_TOOLS})
    get_filename_component(tool_name ${tool_source} NAME_WE)
    add_executable(${tool_name} ${tool_source})
    target_link_libraries(${tool_name} PRIVATE niyah)
    target_compile_definitions(${tool_name} PRIVATE NIYAH_BRIDGE_EXPORTS)
    if(MSVC)
        target_compile_options(${tool_name} PRIVATE /utf-8)
        target_compile_definitions(${tool_name} PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
endforeach()

# ---------------------------------------------------------------------------
# Niyah CLI
# ---------------------------------------------------------------------------
add_executable(niyah_cli niyah_cli.c)
target_link_libraries(niyah_cli PRIVATE niyah)
target_compile_definitions(niyah_cli PRIVATE NIYAH_BRIDGE_EXPORTS)
set_target_properties(niyah_cli PROPERTIES OUTPUT_NAME "niyah")

if(MSVC)
    target_compile_options(niyah_cli PRIVATE /W4 /utf-8)
    target_compile_definitions(niyah_cli PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(niyah_cli PRIVATE -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# Tests
#
# Previously ~20 targets were registered here, every one of them a printf
# stub with no assertions, so `ctest` reported a fully green suite while
# exercising nothing. test_niyah_document.c was not registered at all.
#
# Each test source now #undef NDEBUG before including <assert.h> so the
# assertions survive a Release build.
# ---------------------------------------------------------------------------
enable_testing()

set(NIYAH_TESTS
    test_niyah_core.c
    test_niyah_document.c
    niyah_matmul_test.c
    niyah_softmax_test.c
    niyah_rmsnorm_test.c
    niyah_swiglu_test.c
    niyah_rope_test.c
    niyah_attention_test.c
    niyah_transformer_layer_test.c
    niyah_sampler_test.c
    niyah_tokenizer_test.c
    niyah_tokenizer_roundtrip_test.c
    niyah_config_test.c
    niyah_identity_test.c
    niyah_runtime_test.c
    niyah_telemetry_test.c
    niyah_graph_test.c
    niyah_embedding_test.c
    niyah_search_test.c
    niyah_storage_test.c
    niyah_crawler_test.c
    niyah_bridge_test.c
    niyah_llm_test.c
    niyah_llm_generation_test.c
    niyah_evidence_test.c
    niyah_csp_test.c
)

foreach(test_source ${NIYAH_TESTS})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} PRIVATE niyah)
    target_compile_definitions(${test_name} PRIVATE NIYAH_BRIDGE_EXPORTS)
    if(MSVC)
        target_compile_options(${test_name} PRIVATE /utf-8)
        target_compile_definitions(${test_name} PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
