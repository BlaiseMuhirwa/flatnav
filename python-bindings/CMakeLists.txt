cmake_minimum_required(VERSION 3.21)

project(flatnav LANGUAGES CXX)

# Keep this in sync with setup.py
set(VERSION_INFO "0.1.2")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src/flatnav)
# Add include directories for the target
# 1. Where to find header files for flatnav CPP
# 2. Where to find header files for cereal
include_directories(${CMAKE_SOURCE_DIR}/../include)
include_directories(${CMAKE_SOURCE_DIR}/../external/cereal/include)
include_directories(${CMAKE_SOURCE_DIR}/../external/eigen)

# Option 1: Use pybind11 from Python environment
# This requires setting the PYBIND11_INCLUDE_DIR variable before calling this
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${PYBIND11_INCLUDE_DIR})

# Get Python include path
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils import sysconfig; print(sysconfig.get_python_inc())"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${PYTHON_INCLUDE_DIR})

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
add_library(_core MODULE ${CPP_SOURCE})
set_target_properties(_core PROPERTIES PREFIX "")

target_compile_definitions(_core PRIVATE VERSION_INFO=${VERSION_INFO})

# Print include directories for the target
get_property(INCLUDE_DIRS TARGET _core PROPERTY INCLUDE_DIRECTORIES)
message(STATUS "Include directories for core library: ${INCLUDE_DIRS}")

# Add debug and release flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(_core PUBLIC -g -Wall -fsanitize=address,undefined)
endif()

# Place the shared object file (.so) in the destination indicated by setup.py
install(TARGETS _core DESTINATION .)