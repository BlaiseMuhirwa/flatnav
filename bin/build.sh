#!/bin/bash 

# Make sure we are at the root directory
cd "$(dirname "$0")/.."

BUILD_TESTS=OFF
BUILD_EXAMPLES=OFF 
BUILD_BENCHMARKS=OFF
NO_MANUAL_VECTORIZATION=OFF
MAKE_VERBOSE=0
CMAKE_BUILD_TYPE=Release

function print_usage() {
    echo "Usage ./build.sh [OPTIONS]"
    echo ""
    echo "Available Options:"
    echo "  -t, --tests:                    Build tests"
    echo "  -e, --examples:                 Build examples"
    echo "  -v, --verbose:                  Make verbose"
    echo "  -b, --benchmark:                Build benchmarks"
    echo "  -bt, --build_type:              Build type (Debug, Release, RelWithDebInfo, MinSizeRel)"
    echo "  -nmv, --no_manual_vectorization:Disable manual vectorization (SIMD)"
    echo "  -h, --help:                     Print this help message"
    echo ""
    echo "Example Usage:"
    echo "  ./build.sh -t -e -v"
    exit 1
}

function check_clang_installed() {
    if [[ ! -x "$(command -v clang)" ]]; then
        echo "clang is not installed. Installing it..."
        ./bin/install_clang.sh
    fi
}

# Process the options and arguments     
while [[ "$#" -gt 0 ]]; do
    case $1 in 
        -t|--tests) BUILD_TESTS=ON; shift ;;
        -e|--examples) BUILD_EXAMPLES=ON; shift ;; 
        -v|--verbose) MAKE_VERBOSE=1; shift ;;
        -b|--benchmark) BUILD_BENCHMARKS=ON; shift ;;
        -nmv|--no_manual_vectorization) NO_MANUAL_VECTORIZATION=ON; shift ;;
        -bt|--build_type) CMAKE_BUILD_TYPE=$2; shift; shift ;;
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
    export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
    export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
elif [[ "$(uname)" == "Linux" ]]; then
    echo "Using system clang"
else
    echo "Unsupported Operating System. Exiting..."
    exit 1
fi

echo "Using CC=${CC} and CXX=${CXX} compilers for building."

mkdir -p build 
cd build && cmake \
                -DCMAKE_C_COMPILER=${CC} \
                -DCMAKE_CXX_COMPILER=${CXX} \
                -DNO_MANUAL_VECTORIZATION=${NO_MANUAL_VECTORIZATION} \
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
                -DBUILD_TESTS=${BUILD_TESTS} \
                -DBUILD_EXAMPLES=${BUILD_EXAMPLES} \
                -DBUILD_BENCHMARKS=${BUILD_BENCHMARKS} .. 
make -j VERBOSE=${MAKE_VERBOSE}