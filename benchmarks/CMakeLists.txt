cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
enable_testing()

include(FetchContent)
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

target_link_libraries(FLAT_NAV_LIB INTERFACE benchmark::benchmark)
add_executable(run_benchmark ${PROJECT_SOURCE_DIR}/benchmarks/runner.cpp)
target_link_libraries(run_benchmark benchmark::benchmark yaml-cpp
                      FLAT_NAV_LIB ${CNPY_LIB} ${ZLIB_LIB_RELEASE})
install(TARGETS run_benchmark DESTINATION BIN)