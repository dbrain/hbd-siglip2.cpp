cmake_minimum_required(VERSION 3.14)
project(siglip2-ggml VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
endif()

option(SIGLIP2_TIMING "Enable detailed timing instrumentation" OFF)
if(SIGLIP2_TIMING)
    add_compile_definitions(SIGLIP2_TIMING)
endif()

# HTTP server (gated, off until M4)
option(SIGLIP2_SERVER "Build the HTTP server reproducing kobbler-vision endpoints" OFF)
if(SIGLIP2_SERVER)
    include(FetchContent)
    FetchContent_Declare(httplib
        GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
        GIT_TAG        v0.20.0
    )
    FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
    )
    FetchContent_MakeAvailable(httplib nlohmann_json)
endif()

find_package(Threads REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Embed RPATH so binaries find ggml shared libs without LD_LIBRARY_PATH wrangling.
# - At build time: look in build/ggml/src/ and build/ggml/src/ggml-cpu/ relative to executable
# - After install: look in ../lib relative to bin/
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
add_subdirectory(${GGML_DIR})

# Vision encoder library (M1: stub; will hold the SigLIP2 ViT graph)
add_library(siglip2_vision STATIC
    src/siglip2_vision.cpp
    src/gguf_loader.cpp
)
target_include_directories(siglip2_vision PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(siglip2_vision PUBLIC
    ggml
    Threads::Threads
)

# CLI: load GGUF + image -> 1152-d embedding stdout
add_executable(siglip2-cli
    src/siglip2_cli.cpp
)
target_link_libraries(siglip2-cli PRIVATE
    siglip2_vision
)

install(TARGETS siglip2_vision siglip2-cli
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES
    src/siglip2_vision.h
    src/gguf_loader.h
    DESTINATION include
)
