cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(FlatNav CXX)

# print ${PROJECT_SOURCE_DIR} and ${PROJECT_BINARY_DIR}
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")
message(STATUS "CMake CXX Compiler: ${CMAKE_CXX_COMPILER}")

# Add the parent directory to the include path This allows us to write #include
# <flatnav/...> instead of #include "../flatnav/..."
include_directories(${CMAKE_SOURCE_DIR})

# Include cereal
include_directories(${CMAKE_SOURCE_DIR}/external/cereal/include/)

# set standard to c++17 in order to use std::unique_ptr(c++14) and nested
# namespaces (c++17)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} \
    -Ofast \
    -DHAVE_CXX0X \
    -DNDEBUG \
    -fopenmp \
    -fpic \
    -w \
    -ffast-math \
    -funroll-loops")

# Code adapted from PyTorch:
# https://github.com/pytorch/pytorch/blob/main/cmake/Modules/FindAVX.cmake
include(CheckCXXSourceRuns)
set(AVX_CODE
    "
      #include <immintrin.h>
      int main() {
        __m256 a;
        a = _mm256_set1_ps(0);
        return 0;
      }
    ")

set(AVX512_CODE
    "
    #include <immintrin.h>
    int main() {
      __m512 a = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0);
      __m512i b = a;
      __mmask64 equality_mask = _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_EQ);
      return 0;
      )
    }
    ")

include(CheckCXXCompilerFlag)

# Function to check compiler support and hardware capability for a given flag
function(check_compiler_and_hardware_support FLAG CODE_VAR EXTENSION_NAME)
  check_cxx_compiler_flag(${FLAG} COMPILER_SUPPORTS_${EXTENSION_NAME})
  if(COMPILER_SUPPORTS_${EXTENSION_NAME})
    set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${FLAG}")

    check_cxx_source_runs("${${CODE_VAR}}"
                          SYSTEM_SUPPORTS_${EXTENSION_NAME}_EXTENSIONS)
    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

    if(SYSTEM_SUPPORTS_${EXTENSION_NAME}_EXTENSIONS)
      set(CMAKE_CXX_FLAGS
          "${CMAKE_CXX_FLAGS} ${FLAG}"
          PARENT_SCOPE)
      message(STATUS "Building with ${EXTENSION_NAME}")
    else()
      message(
        STATUS "Compiler supports ${FLAG} flag but the target machine does not "
               "support ${EXTENSION_NAME} instructions")
    endif()
  endif()
endfunction()

# Build SSE/AVX/AVX512 code only on x86-64 processors.
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
  check_compiler_and_hardware_support("-mavx512f" "AVX512_CODE" "AVX512")
  check_compiler_and_hardware_support("-mavx" "AVX_CODE" "AVX")

  check_cxx_compiler_flag("-msse" CXX_SSE)
  if(CXX_SSE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse")
    message(STATUS "Building with SSE")
  endif()
endif()

option(CMAKE_BUILD_TYPE "Build type" Release)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  # Add debug compile flags
  message(STATUS "Building in Debug mode")
  # For more on address sanitizer:
  # https://clang.llvm.org/docs/AddressSanitizer.html ASan is supposed to be
  # very fast, but if we find it slow, we can remove it or use compiler
  # directives to skip analyzing functions: __attribute__((no_sanitize_address))
  # add_compile_options(-g -Wall -fsanitize=address)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -fsanitize=address")
endif()

include(ExternalProject)
include(FeatureSummary)
include(FetchContent)

find_package(Git REQUIRED)

option(USE_GIT_PROTOCOL
       "If behind a firewall turn this off to use HTTPS instead." OFF)

# All options summary
feature_summary(WHAT ALL)

# Configure options
configure_file("${PROJECT_SOURCE_DIR}/quantization/Config.h.in"
               "${PROJECT_BINARY_DIR}/Config.h")
include_directories("${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}")

function(functionInstallExternalCMakeProject ep_name)
  ExternalProject_Get_Property(${ep_name} binary_dir)
  install(SCRIPT ${binary_dir}/cmake_install.cmake)
endfunction()

# TODO(blaise): Remove this old school way of including external libraries via
# ExternalProject_Add. Use FetchContent instead, which is much cleaner
ExternalProject_Add(
  ZLIB
  DEPENDS ""
  GIT_REPOSITORY https://github.com/madler/zlib.git
  GIT_TAG v1.2.11
  SOURCE_DIR ZLIB-source
  BINARY_DIR ZLIB-build
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  # INSTALL_COMMAND ""
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX:STRING=${PROJECT_BINARY_DIR}/ep
    -DINSTALL_BIN_DIR:STRING=${PROJECT_BINARY_DIR}/ep/bin
    -DINSTALL_INC_DIR:STRING=${PROJECT_BINARY_DIR}/ep/include
    -DINSTALL_LIB_DIR:STRING=${PROJECT_BINARY_DIR}/ep/lib
    -DINSTALL_MAN_DIR:STRING=${PROJECT_BINARY_DIR}/ep/share/man
    -DINSTALL_PKGCONFIG_DIR:STRING=${PROJECT_BINARY_DIR}/ep/share/pkgconfig
    -DCMAKE_BUILD_TYPE:STRING=Release)
functioninstallexternalcmakeproject(ZLIB)

set(ZLIB_LIB_DEBUG ${PROJECT_BINARY_DIR}/ep/lib/libz.a)
set(ZLIB_LIB_RELEASE ${PROJECT_BINARY_DIR}/ep/lib/libz.a)

ExternalProject_Add(
  CNPY
  DEPENDS ZLIB
  GIT_REPOSITORY https://github.com/sarthakpati/cnpy.git
  # GIT_TAG v1.2.11
  SOURCE_DIR CNPY-source
  BINARY_DIR CNPY-build
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  # INSTALL_COMMAND ""
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS -DZLIB_INCLUDE_DIR:STRING=${PROJECT_BINARY_DIR}/ep/include
             -DZLIB_LIBRARY_DEBUG:STRING=${ZLIB_LIB_DEBUG}
             -DZLIB_LIBRARY_RELEASE:STRING=${ZLIB_LIB_RELEASE}
             -DCMAKE_INSTALL_PREFIX:STRING=${PROJECT_BINARY_DIR}/ep
             -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
             -DCMAKE_BUILD_TYPE:STRING=Release)
functioninstallexternalcmakeproject(CNPY)

include_directories(${PROJECT_BINARY_DIR}/ep/include)

set(CNPY_LIB ${PROJECT_BINARY_DIR}/ep/lib/libcnpy.a)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  message(STATUS "OpenMP Found. Building the Package using the system OpenMP.")
else()
  message(
    FATAL_ERROR
      "OpenMP Not Found. Building the Package using LLVM's OpenMP. This is slower than the system OpenMP."
  )
endif(OpenMP_FOUND)

option(BUILD_TESTS "Build all tests")
option(BUIL_EXAMPLES "Build examples")
option(BUILD_BENCHMARKS "Build ANNS benchmarks")
option(NO_MANUAL_VECTORIZATION "Disable manual vectorization (SIMD)")
message(STATUS "Building tests: ${BUILD_TESTS}")
message(STATUS "Building examples: ${BUILD_EXAMPLES}")
message(STATUS "Building benchmarks: ${BUILD_BENCHMARKS}")

# Enable auto-vectorization if we are not using SIMD.
if(NO_MANUAL_VECTORIZATION)
  message(STATUS "Disabling manual vectorization (SIMD)")
  add_definitions(-DNO_MANUAL_VECTORIZATION)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize")
endif()

if(BUILD_BENCHMARKS)
  enable_testing()

  message(STATUS "Running a suite of benchmarks using Google Benchmark")
  set(GOOGLE_BENCHMARK_DIR "${PROJECT_BINARY_DIR}/include/google-benchmark")

  if(NOT EXISTS ${GOOGLE_BENCHMARK_DIR})
    message(STATUS "Downloading google-benchmark to ${GOOGLE_BENCHMARK_DIR}")

    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG main)
    FetchContent_Declare(
      googlebenchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG main) # need main for benchmark::benchmark

    FetchContent_MakeAvailable(googletest googlebenchmark)

  else()
    add_subdirectory(${GOOGLE_BENCHMARK_DIR})
  endif()

  set(YAML_CPP_DIR "${PROJECT_SOURCE_DIR}/external/yaml-cpp")

  # Disable building tests for YAML-CPP
  set(YAML_CPP_BUILD_TESTS
      OFF
      CACHE BOOL "Enable testing" FORCE)
  # Fetch the YAML submodule if it doesn't exist
  if(NOT EXISTS ${YAML_CPP_DIR})
    message(STATUS "Fetching yaml-cpp submodule")
    FetchContent_Declare(
      yamlcpp
      GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
      GIT_TAG yaml-cpp-0.6.3)
    FetchContent_MakeAvailable(yamlcpp)
  else()
    add_subdirectory(${YAML_CPP_DIR})
  endif()

endif()

# TODO: Using globbing or some other command that does not require writing down
# every header file
set(HEADERS
    ${PROJECT_SOURCE_DIR}/flatnav/distances/InnerProductDistance.h
    ${PROJECT_SOURCE_DIR}/flatnav/distances/SquaredL2Distance.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/VisitedSetPool.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/GorderPriorityQueue.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/Reordering.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/SIMDDistanceSpecializations.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/ParallelConstructs.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/PreprocessorUtils.h
    ${PROJECT_SOURCE_DIR}/flatnav/DistanceInterface.h
    ${PROJECT_SOURCE_DIR}/flatnav/Index.h
    ${PROJECT_SOURCE_DIR}/quantization/ProductQuantization.h
    ${PROJECT_SOURCE_DIR}/quantization/CentroidsGenerator.h
    ${PROJECT_SOURCE_DIR}/quantization/Utils.h)

add_library(FLAT_NAV_LIB STATIC ${PROJECT_SOURCE_DIR}/tools/construct_npy.cpp
                                ${HEADERS})
add_dependencies(FLAT_NAV_LIB CNPY)

target_link_libraries(FLAT_NAV_LIB PUBLIC ${CNPY_LIB} OpenMP::OpenMP_CXX)
set_target_properties(FLAT_NAV_LIB PROPERTIES LINKER_LANGUAGE CXX)

if(BUILD_EXAMPLES)
  message(STATUS "Building examples for Flatnav")
  foreach(CONSTRUCT_EXEC construct_npy query_npy cereal_tests)
    add_executable(${CONSTRUCT_EXEC}
                   ${PROJECT_SOURCE_DIR}/tools/${CONSTRUCT_EXEC}.cpp ${HEADERS})
    add_dependencies(${CONSTRUCT_EXEC} FLAT_NAV_LIB)
    target_link_libraries(${CONSTRUCT_EXEC} FLAT_NAV_LIB ${CNPY_LIB}
                          ${ZLIB_LIB_RELEASE})
    install(TARGETS ${CONSTRUCT_EXEC} DESTINATION bin)
  endforeach(CONSTRUCT_EXEC)

endif()

if(BUILD_TESTS)
  message(STATUS "Building flatnav + quantization unit tests using gtest")
  set(GOOGLE_TEST_DIR "${PROJECT_BINARY_DIR}/_deps/googletest-src")

  if(NOT EXISTS ${GOOGLE_TEST_DIR})
    message(
      STATUS
        "Downloading googletest to ${PROJECT_BINARY_DIR}/_deps/googletest-src")
  endif()
  # This does not download googletest again if its already available in the
  # CMakeCache file
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main)

  FetchContent_MakeAvailable(googletest)

  FetchContent_GetProperties(googletest)
  if(NOT googletest_POPULATED)
    message(STATUS "GoogleTest not populated")
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
  else()
    message(STATUS "GoogleTest already populated")
  endif()

  message(STATUS "googletest_BINARY_DIR: ${googletest_BINARY_DIR}")
  message(STATUS "googletest_SOURCE_DIR: ${googletest_SOURCE_DIR}")

  target_include_directories(FLAT_NAV_LIB
                             PUBLIC ${googletest_SOURCE_DIR}/googletest/include)
  add_subdirectory(${PROJECT_SOURCE_DIR}/flatnav/tests)
  add_subdirectory(${PROJECT_SOURCE_DIR}/quantization/tests)

endif()

if(BUILD_BENCHMARKS)
  target_link_libraries(FLAT_NAV_LIB benchmark::benchmark)
  add_executable(run_benchmark ${PROJECT_SOURCE_DIR}/benchmarks/runner.cpp)
  target_link_libraries(run_benchmark benchmark::benchmark yaml-cpp
                        FLAT_NAV_LIB ${CNPY_LIB} ${ZLIB_LIB_RELEASE})
  install(TARGETS run_benchmark DESTINATION BIN)
endif()
