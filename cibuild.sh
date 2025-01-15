#!/bin/bash

set -e

show_usage() {
    echo "Usage: $0 [--current-version]"
    echo "  --current-version    Build wheel only for current Python version"
    echo "  Without arguments, builds wheels for all supported Python versions"
    exit 1
}


BUILD_CURRENT=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --current-version)
            BUILD_CURRENT=1
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
    WHEEL_KEY=$(python python-bindings/get-wheel-key.py)
    echo "Building wheel for current Python version: $WHEEL_KEY"
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