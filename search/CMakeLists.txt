cmake_minimum_required(VERSION 3.20)
project(niyah_search LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CURL REQUIRED)

add_library(niyah_search STATIC
    http_fetch.cpp
    html_extract.cpp
    search_engine.cpp
    niyah_crawler.c
    niyah_index.c
    niyah_url.c
    ../native/niyah_telemetry.c
)

target_include_directories(niyah_search
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../native
)

target_link_libraries(niyah_search PUBLIC CURL::libcurl m)
target_compile_definitions(niyah_search PRIVATE _POSIX_C_SOURCE=200809L)

target_compile_features(niyah_search PUBLIC cxx_std_17 c_std_11)

if (MSVC)
    target_compile_options(niyah_search PRIVATE /W4 /WX /permissive-)
else()
    target_compile_options(niyah_search PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
endif()

add_executable(niyah_search_smoke search_smoke.cpp)
target_link_libraries(niyah_search_smoke PRIVATE niyah_search)
target_compile_options(niyah_search_smoke PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow;-Werror>)

enable_testing()
add_test(NAME niyah_search_smoke COMMAND niyah_search_smoke)
