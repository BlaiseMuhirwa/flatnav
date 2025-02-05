#!/bin/bash 

# Make sure we are at the root directory
cd "$(dirname "$0")/.."

BUILD_TESTS=OFF
BUILD_EXAMPLES=OFF
NO_SIMD_VECTORIZATION=OFF
MAKE_VERBOSE=0
CMAKE_BUILD_TYPE=Release

function print_usage() {
    echo "Usage ./build.sh [OPTIONS]"
    echo ""
    echo "Available Options:"
    echo "  -t, --tests:                    Build tests"
    echo "  -e, --examples:                 Build examples"
    echo "  -v, --verbose:                  Make verbose"
    echo "  -bt, --build_type:              Build type (Debug, Release, RelWithDebInfo, MinSizeRel)"
    echo "  -nsv, --no_simd_vectorization:  Disable SIMD vectorization"
    echo "  -h, --help:                     Print this help message"
    echo ""
    echo "Example Usage:"
    echo "  ./build.sh -t -e -v"
    exit 1
}

function set_compilers() {
    if command -v clang &> /dev/null; then 
        echo "Building with clang/clang++ compilers"
        export CC=$(command -v clang)
        export CXX=$(command -v clang++)
    elif command -v gcc &> /dev/null; then
        echo "Building with gcc/g++ compilers"
        export CC=$(command -v gcc)
        export CXX=$(command -v g++)
    else
        echo "Please install either clang or gcc. Exiting..."
        exit 1
    fi
}

# Process the options and arguments     
while [[ "$#" -gt 0 ]]; do
    case $1 in 
        -t|--tests) BUILD_TESTS=ON; shift ;;
        -e|--examples) BUILD_EXAMPLES=ON; shift ;; 
        -v|--verbose) MAKE_VERBOSE=1; shift ;;
        -nsv|--NO_SIMD_VECTORIZATION) NO_SIMD_VECTORIZATION=ON; shift ;;
        -bt|--build_type) CMAKE_BUILD_TYPE=$2; shift; shift ;;
        *) print_usage ;;
    esac 
done

set_compilers

if [[ "$(uname)" == "Darwin" ]]; then
    if [[ -x "/opt/homebrew/opt/llvm/bin/clang" ]]; then
        echo "Using LLVM clang"
        export CC=/opt/homebrew/opt/llvm/bin/clang
        export CXX=/opt/homebrew/opt/llvm/bin/clang++
        export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
        export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
        export CMAKE_PREFIX_PATH="/opt/homebrew/opt/libomp"
        export CMAKE_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
        export CMAKE_EXE_LINKER_FLAGS="-L/opt/homebrew/opt/libomp/lib -lomp"
    fi
elif [[ "$(uname)" == "Linux" ]]; then
    echo "Using system compiler: ${CC} and ${CXX}"
else
    echo "Unsupported Operating System. Exiting..."
    exit 1
fi

mkdir -p build 
cd build && cmake \
                -DCMAKE_C_COMPILER=${CC} \
                -DCMAKE_CXX_COMPILER=${CXX} \
                -DNO_SIMD_VECTORIZATION=${NO_SIMD_VECTORIZATION} \
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
                -DBUILD_TESTS=${BUILD_TESTS} \
                -DBUILD_EXAMPLES=${BUILD_EXAMPLES} \
                -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
                -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
                -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS}" \
                -DCMAKE_VERBOSE_MAKEFILE=ON ..
make -j VERBOSE=${MAKE_VERBOSE}
