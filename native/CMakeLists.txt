cmake_minimum_required(VERSION 3.16)
project(niyah_native LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
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
    target_compile_options(niyah PRIVATE /W4)
    # snprintf/fopen deprecation noise from the CRT security warnings.
    target_compile_definitions(niyah PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(niyah PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# expf/sqrtf/cosf live in libm on most Unix toolchains.
if(UNIX)
    target_link_libraries(niyah PUBLIC m)
endif()

# ---------------------------------------------------------------------------
# Shared library for ui/Niyah.App P/Invoke
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
    target_compile_definitions(niyah_shared PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

if(UNIX)
    target_link_libraries(niyah_shared PUBLIC m)
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
        target_compile_definitions(${test_name} PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
