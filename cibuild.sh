#!/bin/bash

set -e

show_usage() {
    echo "Usage: $0 [--current-version PYTHON_VERSION]"
    echo "  --current-version PYTHON_VERSION    Build wheel only for specified Python version (e.g., 3.8)"
    echo "  Without arguments, builds wheels for all supported Python versions"
    exit 1
}

BUILD_CURRENT=0
PYTHON_VERSION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --current-version)
            BUILD_CURRENT=1
            if [ -n "$2" ]; then
                PYTHON_VERSION=$2
                shift
            else
                echo "Error: Python version argument missing"
                show_usage
            fi
            shift
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

if [ $BUILD_CURRENT -eq 1 ]; then
    # Convert version format (e.g., 3.8 -> 38)
    PYVER=$(echo $PYTHON_VERSION | tr -d '.')
    
    # Determine platform
    if [[ "$OSTYPE" == "darwin"* ]]; then
        PLATFORM_TAG="macosx_x86_64"
    else
        PLATFORM_TAG="manylinux_x86_64"
    fi
    
    WHEEL_KEY="cp${PYVER}-${PLATFORM_TAG}"
    echo "Building wheel for Python version: $WHEEL_KEY"
    export CIBW_BUILD="$WHEEL_KEY"
else
    echo "Building wheels for all supported Python versions"
    export CIBW_BUILD="cp{38,39,310,311,312}-{manylinux_x86_64,macosx_x86_64}"
fi

# Skip Windows builds
export CIBW_SKIP="*win*"

# Run cibuildwheel with appropriate configuration
echo "Starting wheel build..."
cibuildwheel python-bindings --output-dir wheelhouse

echo "Build complete! Wheels are in the wheelhouse directory."