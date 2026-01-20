cmake_minimum_required(VERSION 3.14)
project(flatnav_c VERSION 1.0.0 LANGUAGES C)

# C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(FN_BUILD_SHARED "Build shared library" OFF)
option(FN_BUILD_TESTS "Build tests" ON)
option(FN_DPU_TARGET "Build for UPMEM DPU" OFF)
option(FN_DISABLE_SIMD "Disable all SIMD optimizations" OFF)
option(FN_DISABLE_THREADS "Disable threading support" OFF)

# Compiler-specific settings
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(NOT FN_DPU_TARGET)
        add_compile_options(-fvisibility=hidden)
    endif()

    # Release optimizations
    set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")

    # Enable native architecture optimizations for Release
    if(NOT FN_DPU_TARGET AND NOT FN_DISABLE_SIMD)
        option(FN_NATIVE_ARCH "Use -march=native" OFF)
        if(FN_NATIVE_ARCH)
            add_compile_options(-march=native)
        endif()
    endif()
endif()

if(MSVC)
    add_compile_options(/W4)
endif()

# DPU-specific configuration
if(FN_DPU_TARGET)
    add_definitions(-DFN_DPU_TARGET=1)
    add_definitions(-DFN_NO_FLOAT=1)
    add_definitions(-DFN_NO_THREADS=1)
    add_definitions(-DFN_STATIC_ALLOC=1)
    set(FN_DISABLE_SIMD ON)
    set(FN_DISABLE_THREADS ON)
endif()

# SIMD detection
if(NOT FN_DISABLE_SIMD AND NOT FN_DPU_TARGET)
    include(CheckCCompilerFlag)

    # Check for SSE
    check_c_compiler_flag("-msse" HAS_SSE)
    if(HAS_SSE)
        add_definitions(-DFN_USE_SSE=1)
    endif()

    # Check for SSE4.1
    check_c_compiler_flag("-msse4.1" HAS_SSE4)
    if(HAS_SSE4)
        add_definitions(-DFN_USE_SSE4=1)
    endif()

    # Check for AVX2
    check_c_compiler_flag("-mavx2" HAS_AVX2)
    if(HAS_AVX2)
        add_definitions(-DFN_USE_AVX2=1)
    endif()

    # Check for AVX-512
    check_c_compiler_flag("-mavx512f" HAS_AVX512)
    if(HAS_AVX512)
        add_definitions(-DFN_USE_AVX512=1)
    endif()
endif()

# Threading
if(FN_DISABLE_THREADS)
    add_definitions(-DFN_NO_THREADS=1)
else()
    find_package(Threads REQUIRED)
endif()

# Source files
set(FLATNAV_SOURCES
    src/flatnav.c
    src/distance_l2.c
    src/distance_ip.c
    src/index.c
    src/search.c
    src/visited_set.c
    src/priority_queue.c
    src/serialization.c
    src/npy_reader.c
)

# Add SIMD source files conditionally
if(NOT FN_DISABLE_SIMD AND NOT FN_DPU_TARGET)
    list(APPEND FLATNAV_SOURCES src/simd_dispatch.c)
endif()

# Add fixed-point source for DPU
if(FN_DPU_TARGET)
    list(APPEND FLATNAV_SOURCES src/distance_fixed.c)
endif()

# Library target
if(FN_BUILD_SHARED)
    add_library(flatnav_c SHARED ${FLATNAV_SOURCES})
    target_compile_definitions(flatnav_c PRIVATE FN_BUILD_SHARED)
else()
    add_library(flatnav_c STATIC ${FLATNAV_SOURCES})
    target_compile_definitions(flatnav_c PRIVATE FN_BUILD_STATIC)
endif()

target_include_directories(flatnav_c
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link threading library
if(NOT FN_DISABLE_THREADS)
    target_link_libraries(flatnav_c PRIVATE Threads::Threads)
endif()

# Link math library on Unix
if(UNIX AND NOT FN_DPU_TARGET)
    target_link_libraries(flatnav_c PRIVATE m)
endif()

# Note: SIMD functions use __attribute__((target(...))) for per-function
# instruction set enabling, so no global SIMD flags are needed here.

# Tests
if(FN_BUILD_TESTS AND NOT FN_DPU_TARGET)
    enable_testing()

    add_executable(test_distance tests/test_distance.c)
    target_link_libraries(test_distance PRIVATE flatnav_c)
    add_test(NAME test_distance COMMAND test_distance)

    add_executable(test_index tests/test_index.c)
    target_link_libraries(test_index PRIVATE flatnav_c)
    add_test(NAME test_index COMMAND test_index)

    add_executable(test_serialization tests/test_serialization.c)
    target_link_libraries(test_serialization PRIVATE flatnav_c)
    add_test(NAME test_serialization COMMAND test_serialization)

    # Benchmark test (requires data files)
    add_executable(test_benchmark tests/test_benchmark.c)
    target_link_libraries(test_benchmark PRIVATE flatnav_c)
    if(NOT FN_DISABLE_THREADS)
        target_link_libraries(test_benchmark PRIVATE Threads::Threads)
    endif()
    # Don't add to ctest by default since it needs data files
    # add_test(NAME test_benchmark COMMAND test_benchmark)
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS flatnav_c
    EXPORT flatnav_c-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/flatnav
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT flatnav_c-targets
    FILE flatnav_c-targets.cmake
    NAMESPACE flatnav::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/flatnav_c
)

# Package config
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/flatnav_c-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_file(cmake/flatnav_c-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/flatnav_c-config.cmake"
    @ONLY
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/flatnav_c-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/flatnav_c-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/flatnav_c
)

# Print configuration summary
message(STATUS "")
message(STATUS "FlatNav C API Configuration:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Shared lib:     ${FN_BUILD_SHARED}")
message(STATUS "  DPU target:     ${FN_DPU_TARGET}")
message(STATUS "  SIMD disabled:  ${FN_DISABLE_SIMD}")
message(STATUS "  Threading:      ${FN_DISABLE_THREADS}")
if(NOT FN_DISABLE_SIMD AND NOT FN_DPU_TARGET)
    message(STATUS "  SSE:            ${HAS_SSE}")
    message(STATUS "  SSE4.1:         ${HAS_SSE4}")
    message(STATUS "  AVX2:           ${HAS_AVX2}")
    message(STATUS "  AVX-512:        ${HAS_AVX512}")
endif()
message(STATUS "")
