cmake_minimum_required(VERSION 3.20)
project(niyah_search LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CURL REQUIRED)

add_library(niyah_search STATIC
    http_fetch.cpp
    html_extract.cpp
)

target_include_directories(niyah_search PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(niyah_search PUBLIC CURL::libcurl)

target_compile_features(niyah_search PUBLIC cxx_std_17)
if (MSVC)
    target_compile_options(niyah_search PRIVATE /W4 /WX /permissive-)
else()
    target_compile_options(niyah_search PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
endif()

add_executable(niyah_search_smoke search_smoke.cpp)
target_link_libraries(niyah_search_smoke PRIVATE niyah_search)
