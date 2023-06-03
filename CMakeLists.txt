cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(FlatNav CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS}  -std=c++11 -Ofast -DHAVE_CXX0X -DNDEBUG -openmp -fpic -w -funroll-loops -ftree-vectorize -g -c"
)

# Set the install directory
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/bin)

include(ExternalProject)

find_package(Git REQUIRED)

option(USE_GIT_PROTOCOL
       "If behind a firewall turn this off to use HTTPS instead." OFF)

function(functionInstallExternalCMakeProject ep_name)
  ExternalProject_Get_Property(${ep_name} binary_dir)
  install(SCRIPT ${binary_dir}/cmake_install.cmake)
endfunction()

ExternalProject_Add(
  ZLIB
  DEPENDS ""
  GIT_REPOSITORY https://github.com/madler/zlib.git
  GIT_TAG v1.2.11
  SOURCE_DIR ZLIB-source
  BINARY_DIR ZLIB-build
  UPDATE_COMMAND ""
  PATCH_COMMAND "" # INSTALL_COMMAND ""
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

file(
  GLOB_RECURSE
  HEADERS
  "${PROJECT_SOURCE_DIR}/flatnav/distances/*.h"
  "${PROJECT_SOURCE_DIR}/flatnav/util/*.h"
  "${PROJECT_SOURCE_DIR}/flatnav/DistanceInterface.h"
  "${PROJECT_SOURCE_DIR}/flatnav/Index.h")

add_library(FLAT_NAV_LIB ${HEADERS} ${PROJECT_SOURCE_DIR}/flatnav/lib.cpp)
add_dependencies(FLAT_NAV_LIB CNPY)

target_link_libraries(FLAT_NAV_LIB ${CNPY_LIB})
set_target_properties(FLAT_NAV_LIB PROPERTIES LINKER_LANGUAGE CXX)

foreach(CONSTRUCT_EXEC index_construction_float32)
  add_executable(${CONSTRUCT_EXEC}
                 ${PROJECT_SOURCE_DIR}/examples/${CONSTRUCT_EXEC}.cpp)
  add_dependencies(${CONSTRUCT_EXEC} FLAT_NAV_LIB)
  target_link_libraries(${CONSTRUCT_EXEC} FLAT_NAV_LIB ${CNPY_LIB}
                        ${ZLIB_LIB_RELEASE})
  install(TARGETS ${CONSTRUCT_EXEC} RUNTIME DESTINATION ${PROJECT_BINARY_DIR}/bin)
endforeach(CONSTRUCT_EXEC)
