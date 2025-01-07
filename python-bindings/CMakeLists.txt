cmake_minimum_required(VERSION 3.21)

project(flatnav LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src/flatnav)
# Add include directories for the target
# 1. Where to find header files for flatnav CPP
# 2. Where to find header files for cereal
include_directories(${CMAKE_SOURCE_DIR}/../include)
include_directories(${CMAKE_SOURCE_DIR}/../external/cereal/include)


# Fetch Pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.10.4
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_MakeAvailable(pybind11)
endif()

# Find Python executable
if(DEFINED PYTHON_EXECUTABLE)
    set(FLATNAV_PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}")
else()
    set(FLATNAV_PYTHON_EXECUTABLE "python")
endif()

# Add source files
set(CPP_SOURCE
    src/flatnav/bindings.cpp
)

# Add library for the Python extension
pybind11_add_module(flatnav MODULE ${CPP_SOURCE})
# Print include directories for the target
get_property(INCLUDE_DIRS TARGET flatnav PROPERTY INCLUDE_DIRECTORIES)
message(STATUS "Include directories for flatnav: ${INCLUDE_DIRS}")


# Add debug and release flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(flatnav PUBLIC -g)
else()
    target_compile_options(flatnav PUBLIC -O3 -ffast-math -funroll-loops)
endif()

# Install the library
# install(TARGETS flatnav LIBRARY DESTINATION flatnav)

# Place the shared object file (.so) in the destination indicated by setup.py
install(TARGETS flatnav LIBRARY DESTINATION flatnav)
