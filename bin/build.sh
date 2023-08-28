#!/bin/bash 

BUILD_TESTS=OFF
BUILD_EXAMPLES=OFF 

while getopts "te" opt; do
    case $opt in 
        t) BUILD_TESTS=ON ;;
        e) BUILD_EXAMPLES=ON ;;
        *) echo "Usage ./build.sh [-t] [-e]" && exit 1 ;;
    esac 
done

mkdir -p build 
cd build && cmake -DBUILD_TESTS=${BUILD_TESTS} -DBUILD_EXAMPLES=${BUILD_EXAMPLES} .. 
make -j 