cmake_minimum_required(VERSION 3.20)
project(niyah_native LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

enable_testing()

# FIXED: Removed /WX that broke ASAN build - now /W4 only
if (MSVC)
    set(NIYAH_WARNINGS /W4 /permissive- /Zi)
else()
    set(NIYAH_WARNINGS -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

set(NIYAH_TESTS
    test_niyah_core
    test_niyah_graph
    test_niyah_search
    test_niyah_crawler
    test_niyah_document
    test_niyah_runtime
    test_niyah_telemetry
    test_niyah_storage
)

add_executable(test_niyah_core
    niyah_core.c
    niyah_evidence.c
    niyah_telemetry.c
    test_niyah_core.c
)

add_executable(test_niyah_graph
    niyah_graph.c
    niyah_graph_test.c
)

add_executable(test_niyah_search
    niyah_search.c
    niyah_search_test.c
)

add_executable(test_niyah_crawler
    niyah_crawler.c
    niyah_crawler_test.c
)

add_executable(test_niyah_document
    niyah_document.c
    test_niyah_document.c
)

add_executable(test_niyah_runtime
    niyah_runtime.c
    niyah_core.c
    niyah_graph.c
    niyah_runtime_test.c
)

add_executable(test_niyah_telemetry
    niyah_telemetry.c
    niyah_telemetry_test.c
)

add_executable(test_niyah_storage
    niyah_storage.c
    niyah_storage_test.c
)

foreach(target IN LISTS NIYAH_TESTS)
    target_compile_options(${target} PRIVATE ${NIYAH_WARNINGS})
    if (NOT MSVC)
        target_compile_definitions(${target} PRIVATE _POSIX_C_SOURCE=200809L)
    endif()
    add_test(NAME ${target} COMMAND ${target})
endforeach()

if (NOT MSVC)
    target_link_libraries(test_niyah_core PRIVATE m)
    target_link_libraries(test_niyah_document PRIVATE m)
    target_link_libraries(test_niyah_runtime PRIVATE m)
    target_link_libraries(test_niyah_search PRIVATE m)
endif()

option(NIYAH_ENABLE_ASAN "Enable ASAN" OFF)
if (NIYAH_ENABLE_ASAN)
    if (MSVC)
        foreach(target IN LISTS NIYAH_TESTS)
            target_compile_options(${target} PRIVATE /fsanitize=address /INCREMENTAL:NO /Zi)
            target_link_options(${target} PRIVATE /INCREMENTAL:NO /DEBUG)
        endforeach()
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
    endif()
endif()
