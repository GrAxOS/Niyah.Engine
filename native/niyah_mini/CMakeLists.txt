cmake_minimum_required(VERSION 3.16)

# ---------------------------------------------------------------------------
# Guard: when included as add_subdirectory of native/ the project() call is
# already done by the parent; skip it to avoid resetting compiler detection.
# ---------------------------------------------------------------------------
if(NOT TARGET niyah)
    project(niyah_mini LANGUAGES C)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
set(NIYAH_MINI_SOURCES
    niyah_mini_config.c
    niyah_mini_vocab.c
    niyah_mini_model.c
    niyah_mini_bridge.c
)

set(NIYAH_MINI_HEADERS
    niyah_mini_config.h
    niyah_mini_vocab.h
    niyah_mini_model.h
    niyah_mini_bridge.h
)

# ---------------------------------------------------------------------------
# Static library
# ---------------------------------------------------------------------------
add_library(niyah_mini STATIC ${NIYAH_MINI_SOURCES} ${NIYAH_MINI_HEADERS})

target_include_directories(niyah_mini PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_definitions(niyah_mini PRIVATE NIYAH_BRIDGE_EXPORTS)

if(MSVC)
    target_compile_options(niyah_mini PRIVATE /W4)
    target_compile_definitions(niyah_mini PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(niyah_mini PRIVATE
        -Wall -Wextra -Wno-unused-parameter
    )
endif()

# ---------------------------------------------------------------------------
# Link niyah
#
# bridge.c references: niyah_telemetry_start/end, niyah_tokenize,
# niyah_detokenize -- all defined in libniyah.  Link PUBLIC so that test
# executables that depend on niyah_mini also inherit the full closure.
# ---------------------------------------------------------------------------
if(TARGET niyah)
    target_link_libraries(niyah_mini PUBLIC niyah)
else()
    find_library(NIYAH_LIB niyah
        PATHS ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_PREFIX_PATH}
        DOC "Path to libniyah static library"
    )
    if(NIYAH_LIB)
        target_link_libraries(niyah_mini PUBLIC ${NIYAH_LIB})
    else()
        message(WARNING
            "libniyah not found. Build from repo root via "
            "add_subdirectory(native) or install the engine and "
            "set CMAKE_PREFIX_PATH."
        )
    endif()
endif()

if(UNIX)
    target_link_libraries(niyah_mini PUBLIC m)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
    enable_testing()

    macro(niyah_mini_test target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE niyah_mini)
        target_compile_definitions(${target} PRIVATE NIYAH_BRIDGE_EXPORTS)
        if(MSVC)
            target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
        endif()
        add_test(NAME ${target} COMMAND ${target})
    endmacro()

    niyah_mini_test(test_niyah_mini_config test_niyah_mini_config.c)
    niyah_mini_test(test_niyah_mini_vocab  test_niyah_mini_vocab.c)
    niyah_mini_test(test_niyah_mini_model  test_niyah_mini_model.c)

    # Bridge test: niyah_mini already propagates niyah PUBLIC so one link
    # target is sufficient for the full symbol closure.
    add_executable(test_niyah_mini_bridge test_niyah_mini_bridge.c)
    target_link_libraries(test_niyah_mini_bridge PRIVATE niyah_mini)
    target_compile_definitions(test_niyah_mini_bridge PRIVATE NIYAH_BRIDGE_EXPORTS)
    if(MSVC)
        target_compile_definitions(test_niyah_mini_bridge PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    add_test(NAME test_niyah_mini_bridge COMMAND test_niyah_mini_bridge)
endif()
