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

# HTTP server reproducing kobbler-vision endpoints.
option(SIGLIP2_SERVER "Build the HTTP server" ON)
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

# Sentencepiece (used by the multilingual text tokenizer; Gemma-style .model)
include(FetchContent)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_USE_BUILTIN_PROTOBUF ON CACHE BOOL "" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG        v0.2.0
)
FetchContent_MakeAvailable(sentencepiece)

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

# Phase B (megakernel-v0): turn on CUDA graphs by default. With per-shape graph
# caching in TextEncoder/VisionEncoder (path 2 — bypass scheduler, per-entry
# gallocr), tensor pointers are stable across calls and ggml-cuda's warmup
# graph fires after 2 calls per cached shape. Drops text n=5 compute ~30 → ~19
# ms on RTX 3060. Override on the cmake line if needed.
set(GGML_CUDA_GRAPHS ON CACHE BOOL "ggml: use CUDA graphs (default-on for siglip2 path-2 graph cache)")

add_subdirectory(${GGML_DIR})

# Phase A megakernel — fused per-pattern CUDA kernels (LayerNorm-with-affine
# in v0). Built as either a real CUDA lib when GGML_CUDA=ON or a stub-only
# lib otherwise, so callers can unconditionally invoke
# siglip2_megakernel::install().
if(GGML_CUDA)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()
    add_library(siglip2_megakernel STATIC
        src/cuda/siglip2_megakernel.cu
    )
    set_target_properties(siglip2_megakernel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE  ON
    )
    target_include_directories(siglip2_megakernel PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
        ${GGML_DIR}/include
    )
    target_link_libraries(siglip2_megakernel PUBLIC ggml)
else()
    add_library(siglip2_megakernel STATIC
        src/cuda/siglip2_megakernel_stub.cpp
    )
    target_include_directories(siglip2_megakernel PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
    )
endif()

# Image preprocessor library (NaFlex resize + patchify; stb_image for I/O)
add_library(siglip2_preproc STATIC
    src/siglip2_preproc.cpp
)
target_include_directories(siglip2_preproc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor
)
target_link_libraries(siglip2_preproc PUBLIC
    Threads::Threads
)

# Vision encoder library (graph + ggml backend)
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

# Text encoder library
add_library(siglip2_text STATIC
    src/siglip2_text.cpp
    src/gguf_loader.cpp
)
target_include_directories(siglip2_text PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(siglip2_text PUBLIC
    ggml
    Threads::Threads
)

# Image-text sigmoid scoring (CPU math; reads logit_scale/bias from GGUF)
add_library(siglip2_score STATIC
    src/siglip2_score.cpp
    src/gguf_loader.cpp
)
target_include_directories(siglip2_score PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(siglip2_score PUBLIC
    ggml
    Threads::Threads
)

# Multilingual tokenizer (sentencepiece wrapper)
add_library(siglip2_tokenizer STATIC
    src/siglip2_tokenizer.cpp
)
target_include_directories(siglip2_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${sentencepiece_SOURCE_DIR}/src
    ${sentencepiece_BINARY_DIR}
)
target_link_libraries(siglip2_tokenizer PUBLIC
    sentencepiece-static
    Threads::Threads
)

# CLI: load GGUF + (image|pixel-values) -> embedding
add_executable(siglip2-cli
    src/siglip2_cli.cpp
)
target_link_libraries(siglip2-cli PRIVATE
    siglip2_vision
    siglip2_preproc
    siglip2_megakernel
)

# Text-side parity tool
add_executable(siglip2-text-cli
    src/siglip2_text_cli.cpp
)
target_link_libraries(siglip2-text-cli PRIVATE
    siglip2_text
    siglip2_tokenizer
    siglip2_megakernel
)

if(SIGLIP2_SERVER)
    add_executable(siglip2-server
        src/siglip2_server.cpp
    )
    target_include_directories(siglip2-server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor
    )
    target_link_libraries(siglip2-server PRIVATE
        siglip2_vision
        siglip2_text
        siglip2_tokenizer
        siglip2_score
        siglip2_preproc
        siglip2_megakernel
        httplib::httplib
        nlohmann_json::nlohmann_json
    )
endif()

install(TARGETS siglip2_vision siglip2_text siglip2_preproc siglip2_tokenizer siglip2_score siglip2_megakernel siglip2-cli siglip2-text-cli
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES
    src/siglip2_vision.h
    src/siglip2_text.h
    src/siglip2_preproc.h
    src/gguf_loader.h
    DESTINATION include
)
