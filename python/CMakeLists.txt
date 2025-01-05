cmake_minimum_required(VERSION 3.14)

project(flatnav LANGUAGES CXX)

find_package(PythonExtensions REQUIRED)

add_library(_flatnav MODULE src/bindings.cpp)
python_extension_module(_flatnav)
install(TARGETS _flatnav LIBRARY DESTINATION flatnav)