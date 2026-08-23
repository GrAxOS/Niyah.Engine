cmake_minimum_required(VERSION 3.10)
project(niyah C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find math library
find_library(MATH_LIBRARY m)

# Core library sources
set(NIYAH_SOURCES
    niyah_core.c
    niyah_bridge.c
    niyah_llm.c
    niyah_model.c
    niyah_runtime.c
    niyah_graph.c
    niyah_document.c
    niyah_search.c
    niyah_crawler.c
    niyah_tokenizer.c
    niyah_embedding.c
    niyah_attention.c
    niyah_transformer_layer.c
    niyah_sampler.c
    niyah_softmax.c
    niyah_rmsnorm.c
    niyah_swiglu.c
    niyah_matmul.c
    niyah_rope.c
    niyah_telemetry.c
    niyah_source.c
    niyah_storage.c
    evidence_envelope.c
    evidence_graph.c
    evidence_reasoner.c
    constraint_solver.c
)

# Main library
add_library(niyah STATIC ${NIYAH_SOURCES})
target_include_directories(niyah PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(MATH_LIBRARY)
    target_link_libraries(niyah PUBLIC ${MATH_LIBRARY})
endif()

# Tests
add_executable(test_niyah_core test_niyah_core.c)
target_link_libraries(test_niyah_core niyah)

add_executable(niyah_bridge_test niyah_bridge_test.c)
target_link_libraries(niyah_bridge_test niyah)

add_executable(niyah_llm_test niyah_llm_test.c)
target_link_libraries(niyah_llm_test niyah)

add_executable(niyah_graph_test niyah_graph_test.c)
target_link_libraries(niyah_graph_test niyah)

add_executable(niyah_search_test niyah_search_test.c)
target_link_libraries(niyah_search_test niyah)

add_executable(niyah_runtime_test niyah_runtime_test.c)
target_link_libraries(niyah_runtime_test niyah)

add_executable(niyah_embedding_test niyah_embedding_test.c)
target_link_libraries(niyah_embedding_test niyah)

add_executable(niyah_matmul_test niyah_matmul_test.c)
target_link_libraries(niyah_matmul_test niyah)

add_executable(niyah_attention_test niyah_attention_test.c)
target_link_libraries(niyah_attention_test niyah)

add_executable(niyah_rope_test niyah_rope_test.c)
target_link_libraries(niyah_rope_test niyah)

add_executable(niyah_rmsnorm_test niyah_rmsnorm_test.c)
target_link_libraries(niyah_rmsnorm_test niyah)

add_executable(niyah_swiglu_test niyah_swiglu_test.c)
target_link_libraries(niyah_swiglu_test niyah)

add_executable(niyah_softmax_test niyah_softmax_test.c)
target_link_libraries(niyah_softmax_test niyah)

add_executable(niyah_sampler_test niyah_sampler_test.c)
target_link_libraries(niyah_sampler_test niyah)

add_executable(niyah_transformer_layer_test niyah_transformer_layer_test.c)
target_link_libraries(niyah_transformer_layer_test niyah)

add_executable(niyah_llm_generation_test niyah_llm_generation_test.c)
target_link_libraries(niyah_llm_generation_test niyah)

add_executable(niyah_tokenizer_test niyah_tokenizer_test.c)
target_link_libraries(niyah_tokenizer_test niyah)

add_executable(niyah_crawler_test niyah_crawler_test.c)
target_link_libraries(niyah_crawler_test niyah)

add_executable(niyah_storage_test niyah_storage_test.c)
target_link_libraries(niyah_storage_test niyah)

add_executable(niyah_telemetry_test niyah_telemetry_test.c)
target_link_libraries(niyah_telemetry_test niyah)

enable_testing()

add_test(NAME test_niyah_core COMMAND test_niyah_core)
add_test(NAME niyah_attention_test COMMAND niyah_attention_test)
add_test(NAME niyah_bridge_test COMMAND niyah_bridge_test)
add_test(NAME niyah_embedding_test COMMAND niyah_embedding_test)
add_test(NAME niyah_graph_test COMMAND niyah_graph_test)
add_test(NAME niyah_llm_test COMMAND niyah_llm_test)
add_test(NAME niyah_llm_generation_test COMMAND niyah_llm_generation_test)
add_test(NAME niyah_matmul_test COMMAND niyah_matmul_test)
add_test(NAME niyah_rmsnorm_test COMMAND niyah_rmsnorm_test)
add_test(NAME niyah_rope_test COMMAND niyah_rope_test)
add_test(NAME niyah_runtime_test COMMAND niyah_runtime_test)
add_test(NAME niyah_sampler_test COMMAND niyah_sampler_test)
add_test(NAME niyah_softmax_test COMMAND niyah_softmax_test)
add_test(NAME niyah_storage_test COMMAND niyah_storage_test)
add_test(NAME niyah_swiglu_test COMMAND niyah_swiglu_test)
add_test(NAME niyah_telemetry_test COMMAND niyah_telemetry_test)
add_test(NAME niyah_tokenizer_test COMMAND niyah_tokenizer_test)
add_test(NAME niyah_transformer_layer_test COMMAND niyah_transformer_layer_test)
add_test(NAME niyah_search_test COMMAND niyah_search_test)
add_test(NAME niyah_crawler_test COMMAND niyah_crawler_test)
