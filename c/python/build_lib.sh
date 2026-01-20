#!/bin/bash
# Build the FlatNav C shared library for Python bindings

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$C_DIR/build"

echo "Building FlatNav C shared library..."
echo "  Source: $C_DIR"
echo "  Build:  $BUILD_DIR"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
cmake "$C_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DFN_BUILD_SHARED=ON \
    -DFN_BUILD_TESTS=ON \
    -DFN_NATIVE_ARCH=ON

# Build
cmake --build . --config Release -j$(nproc)

# Check if library was built
if [ -f "libflatnav_c.so" ]; then
    echo ""
    echo "Library built successfully: $BUILD_DIR/libflatnav_c.so"
    echo ""
    echo "To use the Python bindings, either:"
    echo "  1. Set LD_LIBRARY_PATH=$BUILD_DIR"
    echo "  2. Copy libflatnav_c.so to $SCRIPT_DIR/flatnav_c/"

    # Copy to package directory for convenience
    cp libflatnav_c.so "$SCRIPT_DIR/flatnav_c/"
    echo ""
    echo "Library copied to $SCRIPT_DIR/flatnav_c/libflatnav_c.so"
else
    echo "ERROR: Library not found after build"
    exit 1
fi
