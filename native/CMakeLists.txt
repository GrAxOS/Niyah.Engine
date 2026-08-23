cmake_minimum_required(VERSION 3.20)
project(niyah_native LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

enable_testing()

if (MSVC)
    set(NIYAH_WARNINGS /W4 /WX /permissive- /Zi)
else()
    set(NIYAH_WARNINGS -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
endif()

# Core library objects
set(NIYAH_CORE_OBJS
    niyah_core.c
    niyah_evidence.c
    niyah_telemetry.c
)

# Math primitives
set(NIYAH_MATH_OBJS
    niyah_matmul.c
    niyah_rmsnorm.c
    niyah_rope.c
    niyah_softmax.c
)

# Attention and FFN
set(NIYAH_TRANSFORMER_OBJS
    niyah_attention.c
    niyah_attention_multihead.c
    niyah_swiglu.c
    niyah_transformer_layer.c
)

# LLM components
set(NIYAH_LLM_OBJS
    niyah_embedding.c
    niyah_sampler.c
    niyah_tokenizer.c
    niyah_llm.c
)

# Search and crawler
set(NIYAH_SEARCH_OBJS
    niyah_search.c
    niyah_crawler.c
    niyah_document.c
)

# Storage and runtime
set(NIYAH_STORAGE_OBJS
    niyah_storage.c
    niyah_runtime.c
    niyah_graph.c
)

# Bridge
set(NIYAH_BRIDGE_OBJS
    niyah_bridge.c
)

# Shared library
add_library(niyah_bridge SHARED
    ${NIYAH_CORE_OBJS}
    ${NIYAH_MATH_OBJS}
    ${NIYAH_TRANSFORMER_OBJS}
    ${NIYAH_LLM_OBJS}
    ${NIYAH_SEARCH_OBJS}
    ${NIYAH_STORAGE_OBJS}
    ${NIYAH_BRIDGE_OBJS}
)

target_include_directories(niyah_bridge PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(niyah_bridge PRIVATE ${NIYAH_WARNINGS})
if (NOT MSVC)
    target_compile_definitions(niyah_bridge PRIVATE _POSIX_C_SOURCE=200809L)
    target_link_libraries(niyah_bridge PRIVATE m)
endif()

# ============================================================================
# Tests
# ============================================================================

add_executable(test_niyah_core
    ${NIYAH_CORE_OBJS}
    test_niyah_core.c
)

add_executable(test_niyah_matmul
    niyah_matmul.c
    niyah_matmul_test.c
)

add_executable(test_niyah_rmsnorm
    niyah_rmsnorm.c
    niyah_rmsnorm_test.c
)

add_executable(test_niyah_rope
    niyah_rope.c
    niyah_rope_test.c
)

add_executable(test_niyah_softmax
    niyah_softmax.c
    niyah_softmax_test.c
)

add_executable(test_niyah_attention
    niyah_attention.c
    niyah_attention_test.c
)

add_executable(test_niyah_swiglu
    niyah_swiglu.c
    niyah_swiglu_test.c
)

add_executable(test_niyah_transformer_layer
    niyah_transformer_layer.c
    niyah_transformer_layer_test.c
)

add_executable(test_niyah_sampler
    niyah_sampler.c
    niyah_sampler_test.c
)

add_executable(test_niyah_embedding
    niyah_embedding.c
    niyah_embedding_test.c
)

add_executable(test_niyah_attention_multihead
    niyah_attention_multihead.c
    niyah_attention_multihead_test.c
)

add_executable(test_niyah_tokenizer
    niyah_tokenizer.c
    niyah_tokenizer_test.c
)

add_executable(test_niyah_llm
    niyah_llm.c
    niyah_llm_test.c
)

add_executable(test_niyah_generation
    niyah_llm.c
    niyah_tokenizer.c
    niyah_sampler.c
    niyah_matmul.c
    niyah_rmsnorm.c
    niyah_rope.c
    niyah_softmax.c
    niyah_attention.c
    niyah_swiglu.c
    niyah_embedding.c
    niyah_llm_generation_test.c
)

add_executable(test_niyah_bridge
    ${NIYAH_CORE_OBJS}
    ${NIYAH_MATH_OBJS}
    ${NIYAH_TRANSFORMER_OBJS}
    ${NIYAH_LLM_OBJS}
    ${NIYAH_SEARCH_OBJS}
    ${NIYAH_STORAGE_OBJS}
    niyah_bridge.c
    niyah_bridge_test.c
)

# All tests
set(NIYAH_TESTS
    test_niyah_core
    test_niyah_matmul
    test_niyah_rmsnorm
    test_niyah_rope
    test_niyah_softmax
    test_niyah_attention
    test_niyah_swiglu
    test_niyah_transformer_layer
    test_niyah_sampler
    test_niyah_embedding
    test_niyah_attention_multihead
    test_niyah_tokenizer
    test_niyah_llm
    test_niyah_generation
    test_niyah_bridge
)

foreach(target IN LISTS NIYAH_TESTS)
    target_compile_options(${target} PRIVATE ${NIYAH_WARNINGS})
    if (NOT MSVC)
        target_compile_definitions(${target} PRIVATE _POSIX_C_SOURCE=200809L)
        target_link_libraries(${target} PRIVATE m)
    endif()
    add_test(NAME ${target} COMMAND ${target})
endforeach()

# ============================================================================
# Sanitizers
# ============================================================================

option(NIYAH_ENABLE_ASAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if (NIYAH_ENABLE_ASAN)
    if (MSVC)
        foreach(target IN LISTS NIYAH_TESTS)
            target_compile_options(${target} PRIVATE /fsanitize=address /INCREMENTAL:NO /Zi /wd4996)
            target_link_options(${target} PRIVATE /INCREMENTAL:NO /DEBUG)
        endforeach()
        target_compile_options(niyah_bridge PRIVATE /fsanitize=address /INCREMENTAL:NO /Zi /wd4996)
        target_link_options(niyah_bridge PRIVATE /INCREMENTAL:NO /DEBUG)
        get_filename_component(NIYAH_MSVC_BIN "${CMAKE_C_COMPILER}" DIRECTORY)
        set(NIYAH_ASAN_PATH "${NIYAH_MSVC_BIN}")
        foreach(target IN LISTS NIYAH_TESTS)
            set_property(TEST ${target} APPEND PROPERTY ENVIRONMENT "PATH=${NIYAH_ASAN_PATH};$ENV{PATH}")
        endforeach()
    else()
        foreach(target IN LISTS NIYAH_TESTS)
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g)
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
        endforeach()
        target_compile_options(niyah_bridge PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g)
        target_link_options(niyah_bridge PRIVATE -fsanitize=address,undefined)
    endif()
endif()
