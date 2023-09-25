#!/bin/bash 

BUILD_TESTS=OFF
BUILD_EXAMPLES=OFF 
BUILD_BENCHMARKS=OFF
MAKE_VERBOSE=0

function print_usage() {
    echo "Usage ./build.sh [OPTIONS]"
    echo ""
    echo "Available Options:"
    echo "  -t, --tests:        Build tests"
    echo "  -e, --examples:     Build examples"
    echo "  -v, --verbose:      Make verbose"
    echo "  -b, --benchmark:    Build benchmarks"
    echo "  -h, --help:         Print this help message"
    echo ""
    echo "Example Usage:"
    echo "  ./build.sh -t -e -v"
    exit 1
}

function check_clang_installed() {
    if [[ ! -x "$(command -v clang)" ]]; then
        echo "clang is not installed. You should have clang installed first.Exiting..."
        exit 1
    fi
}

# Process the options and arguments 
while [[ "$#" -gt 0 ]]; do
    case $1 in 
        -t|--tests) BUILD_TESTS=ON; shift ;;
        -e|--examples) BUILD_EXAMPLES=ON; shift ;; 
        -v|--verbose) MAKE_VERBOSE=1; shift ;;
        -b|--benchmark) BUILD_BENCHMARKS=ON; shift ;;
        *) print_usage ;;
    esac 
done



check_clang_installed

export CC=/usr/bin/clang
export CXX=/usr/bin/clang++

if [[ "$(uname)" == "Darwin" ]]; then
    echo "Using LLVM clang"
    export CC=/opt/homebrew/opt/llvm/bin/clang
    export CXX=/opt/homebrew/opt/llvm/bin/clang++
elif [[ "$(uname)" == "Linux" ]]; then
    echo "Using system clang"
else
    echo "Unsupported Operating System. Exiting..."
    exit 1
fi

echo "Using CC=${CC} and CXX=${CXX} compilers for building."

mkdir -p build 
cd build && cmake \
                -DBUILD_TESTS=${BUILD_TESTS} \
                -DBUILD_EXAMPLES=${BUILD_EXAMPLES} \
                -DBUILD_BENCHMARKS=${BUILD_BENCHMARKS} ..  
make -j VERBOSE=${MAKE_VERBOSE}