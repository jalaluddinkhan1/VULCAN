cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(vulcan LANGUAGES CXX CUDA)

# ─── Standards ───────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ─── CUDA Architecture ──────────────────────────────────────────────────────
# Override with -DCMAKE_CUDA_ARCHITECTURES=native for auto-detect
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

# ─── Compiler Flags ─────────────────────────────────────────────────────────
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Use static CUDA runtime for hermetic builds
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
endif()

# ─── Include Directories ────────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)

# ─── Collect Source Files ────────────────────────────────────────────────────
file(GLOB_RECURSE VULCAN_CPP_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE VULCAN_CUDA_SOURCES "${CMAKE_SOURCE_DIR}/cuda/*.cu")

# ─── Vulcan Library ─────────────────────────────────────────────────────────
add_library(vulcan STATIC
    ${VULCAN_CPP_SOURCES}
    ${VULCAN_CUDA_SOURCES}
)

target_include_directories(vulcan PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(vulcan PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ─── GoogleTest (fetched at configure time) ──────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
# Prevent GoogleTest from overriding compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# ─── pybind11 Python Module (Phase 6) ────────────────────────────────────────
# Optional: builds `vulcan` Python module if pybind11 is available
option(VULCAN_BUILD_PYTHON "Build Python bindings" OFF)

if(VULCAN_BUILD_PYTHON)
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.12.0
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    pybind11_add_module(pyvulcan src/pybind_vulcan.cpp)
    target_link_libraries(pyvulcan PRIVATE vulcan)
    set_target_properties(pyvulcan PROPERTIES
        OUTPUT_NAME "vulcan"
        CUDA_SEPARABLE_COMPILATION ON
    )
    message(STATUS "║  Python Module:  ENABLED")
else()
    message(STATUS "║  Python Module:  DISABLED (use -DVULCAN_BUILD_PYTHON=ON)")
endif()

# ─── Subdirectories ──────────────────────────────────────────────────────────
add_subdirectory(tests)
add_subdirectory(benchmarks)

# ─── Summary ─────────────────────────────────────────────────────────────────
message(STATUS "╔══════════════════════════════════════════════════╗")
message(STATUS "║  VULCAN Inference Engine — Build Configuration  ║")
message(STATUS "╠══════════════════════════════════════════════════╣")
message(STATUS "║  C++ Standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "║  CUDA Archs:       ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "║  Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "╚══════════════════════════════════════════════════╝")
