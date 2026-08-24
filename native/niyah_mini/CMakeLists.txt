cmake_minimum_required(VERSION 3.16)
project(niyah_mini LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Sources
set(NIYAH_MINI_SOURCES
    niyah_mini_config.c
    niyah_mini_vocab.c
    niyah_mini_model.c
)

# Headers
set(NIYAH_MINI_HEADERS
    niyah_mini_config.h
    niyah_mini_vocab.h
    niyah_mini_model.h
)

# Create static library
add_library(niyah_mini STATIC ${NIYAH_MINI_SOURCES} ${NIYAH_MINI_HEADERS})

target_include_directories(niyah_mini PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(niyah_mini PRIVATE NIYAH_BRIDGE_EXPORTS)

if(MSVC)
    target_compile_options(niyah_mini PRIVATE /W4)
    target_compile_definitions(niyah_mini PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(niyah_mini PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

if(UNIX)
    target_link_libraries(niyah_mini PUBLIC m)
endif()

# Tests
if(BUILD_TESTING)
    enable_testing()
    
    add_executable(test_niyah_mini_config test_niyah_mini_config.c)
    target_link_libraries(test_niyah_mini_config PRIVATE niyah_mini)
    target_compile_definitions(test_niyah_mini_config PRIVATE NIYAH_BRIDGE_EXPORTS)
    add_test(NAME test_niyah_mini_config COMMAND test_niyah_mini_config)
    
    add_executable(test_niyah_mini_vocab test_niyah_mini_vocab.c)
    target_link_libraries(test_niyah_mini_vocab PRIVATE niyah_mini)
    target_compile_definitions(test_niyah_mini_vocab PRIVATE NIYAH_BRIDGE_EXPORTS)
    add_test(NAME test_niyah_mini_vocab COMMAND test_niyah_mini_vocab)
    
    add_executable(test_niyah_mini_model test_niyah_mini_model.c)
    target_link_libraries(test_niyah_mini_model PRIVATE niyah_mini)
    target_compile_definitions(test_niyah_mini_model PRIVATE NIYAH_BRIDGE_EXPORTS)
    add_test(NAME test_niyah_mini_model COMMAND test_niyah_mini_model)
endif()
