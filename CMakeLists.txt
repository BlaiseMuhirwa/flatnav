cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(FlatNav CXX)

# print ${PROJECT_SOURCE_DIR} and ${PROJECT_BINARY_DIR}
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")
message(STATUS "CMake CXX Compiler: ${CMAKE_CXX_COMPILER}")

# Add the parent directory to the include path This allows us to write #include
# <flatnav/...> instead of #include "../flatnav/..."
include_directories(${CMAKE_SOURCE_DIR}/include)

# Include cereal
include_directories(${CMAKE_SOURCE_DIR}/external/cereal/include/)

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

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(cmake/FindAVX.cmake)

option(CMAKE_BUILD_TYPE "Build type" Release)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  # Add debug compile flags
  message(STATUS "Building in Debug mode")
  # Address sanitizer: https://clang.llvm.org/docs/AddressSanitizer.html
  # For some reason, using -fsanitize=address,thread,undefined doesn't workt since it appears
  # that Asan and thread sanitizer have conflicting behavior. To use thread sanitizer, remove 
  # Asan first and vice-versa
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -fsanitize=address,undefined")
endif()


include(FeatureSummary)
# All options summary
feature_summary(WHAT ALL)

include_directories("${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}")

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
option(NO_SIMD_VECTORIZATION "Disable using SIMD instructions")
message(STATUS "Building tests: ${BUILD_TESTS}")
message(STATUS "Building examples: ${BUILD_EXAMPLES}")

# Enable auto-vectorization if we are not using SIMD.
if(NO_SIMD_VECTORIZATION)
  message(STATUS "Disabling using SIMD instructions")
  add_definitions(-DNO_SIMD_VECTORIZATION)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize")
endif()

# TODO: Using globbing or some other command that does not require writing down
# every header file
set(HEADERS
    ${PROJECT_SOURCE_DIR}/include/flatnav/distances/InnerProductDistance.h
    ${PROJECT_SOURCE_DIR}/include/flatnav/distances/SquaredL2Distance.h
    ${PROJECT_SOURCE_DIR}/include/flatnav/distances/L2DistanceDispatcher.h
    ${PROJECT_SOURCE_DIR}/include/flatnav/distances/IPDistanceDispatcher.h
    ${PROJECT_SOURCE_DIR}/include/flatnav/util/SquaredL2SimdExtensions.h
    ${PROJECT_SOURCE_DIR}/include/flatnav/util/InnerProductSimdExtensions.h
    ${PROJECT_SOURCE_DIR}/include/flatnav/util/VisitedSetPool.h
    ${PROJECT_SOURCE_DIR}/include/flatnav/util/GorderPriorityQueue.h
    ${PROJECT_SOURCE_DIR}/include/flatnav/util/Reordering.h
    ${PROJECT_SOURCE_DIR}/include/flatnav/util/Multithreading.h
    ${PROJECT_SOURCE_DIR}/include/flatnav/util/Macros.h
    ${PROJECT_SOURCE_DIR}/include/flatnav/util/Datatype.h
    ${PROJECT_SOURCE_DIR}/include/flatnav/util/SimdUtils.h
    ${PROJECT_SOURCE_DIR}/include/flatnav/distances/DistanceInterface.h
    ${PROJECT_SOURCE_DIR}/include/flatnav/index/Index.h
    ${PROJECT_SOURCE_DIR}/developmental-features/quantization/ProductQuantization.h
    ${PROJECT_SOURCE_DIR}/developmental-features/quantization/CentroidsGenerator.h
    ${PROJECT_SOURCE_DIR}/developmental-features/quantization/Utils.h)

# Interface here means that the library is header only
add_library(FLAT_NAV_LIB INTERFACE)
target_sources(FLAT_NAV_LIB INTERFACE ${HEADERS})
target_include_directories(FLAT_NAV_LIB INTERFACE ${PROJECT_SOURCE_DIR})


target_link_libraries(FLAT_NAV_LIB INTERFACE OpenMP::OpenMP_CXX)

if(BUILD_EXAMPLES)
  message(STATUS "Building examples")
  list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
  include(cmake/FindCNPYAndZLIB.cmake)
  add_subdirectory(${PROJECT_SOURCE_DIR}/tools)
endif()

if(BUILD_TESTS)
  message(STATUS "Building flatnav + quantization unit tests using gtest")
  include(cmake/FindGoogleTest.cmake)
  add_subdirectory(${PROJECT_SOURCE_DIR}/include/flatnav/tests)
endif()
