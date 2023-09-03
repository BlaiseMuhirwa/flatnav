#!/bin/bash 

BUILD_TESTS=OFF
BUILD_EXAMPLES=OFF 
MAKE_VERBOSE=0

while getopts "te" opt; do
    case $opt in 
        t) BUILD_TESTS=ON ;;
        e) BUILD_EXAMPLES=ON ;; 
        v) MAKE_VERBOSE=1 ;;
        *) echo "Usage ./build.sh [-t] [-e] [-v]" && exit 1 ;;
    esac 
done

export CC=/usr/bin/clang
export CXX=/usr/bin/clang++

if [[ "$(uname)" == "Darwin" ]]; then
    echo "Using LLVM clang"
    export CC=/usr/local/opt/llvm/bin/clang
    export CXX=/usr/local/opt/llvm/bin/clang++
elif [[ "$(uname)" == "Linux" ]]; then
    echo "Using system clang"
else
    echo "Unsupported Operating System. Exiting..."
    exit 1
fi

mkdir -p build 
cd build && cmake -DBUILD_TESTS=${BUILD_TESTS} -DBUILD_EXAMPLES=${BUILD_EXAMPLES} .. 
make -j VERBOSE=${MAKE_VERBOSE}