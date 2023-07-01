#!/bin/bash 

# Default is OFF if no argument is passed
BUILD_TESTS=${1:-OFF}


mkdir -p build 
cd build && cmake -DBUILD_TESTS=${BUILD_TESTS} .. 
make -j 