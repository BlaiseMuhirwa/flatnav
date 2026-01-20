#!/usr/bin/env python3
"""
Setup script for FlatNav C API Python bindings.

This script builds the C shared library and installs the Python package.
"""

import os
import subprocess
import sys
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    """Build the C library using CMake."""

    def run(self):
        # Get paths
        source_dir = Path(__file__).parent.parent.resolve()  # c/
        build_dir = source_dir / "build_python"
        package_dir = Path(__file__).parent / "flatnav_c"

        # Create build directory
        build_dir.mkdir(exist_ok=True)

        # Determine library name
        if sys.platform == "darwin":
            lib_name = "libflatnav_c.dylib"
        elif sys.platform == "win32":
            lib_name = "flatnav_c.dll"
        else:
            lib_name = "libflatnav_c.so"

        # Run CMake configure
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DFN_BUILD_SHARED=ON",
            f"-DFN_BUILD_TESTS=OFF",
            f"-DFN_NATIVE_ARCH=ON",
        ]

        subprocess.check_call(
            ["cmake", str(source_dir)] + cmake_args,
            cwd=build_dir,
        )

        # Run CMake build
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release", "-j"],
            cwd=build_dir,
        )

        # Copy library to package directory
        lib_src = build_dir / lib_name
        lib_dst = package_dir / lib_name

        if lib_src.exists():
            import shutil
            shutil.copy2(lib_src, lib_dst)
            print(f"Copied {lib_src} to {lib_dst}")
        else:
            print(f"Warning: Library not found at {lib_src}")
            # Try to find it
            for f in build_dir.glob("*flatnav*"):
                print(f"  Found: {f}")


setup(
    name="flatnav-c",
    version="1.0.0",
    description="Python bindings for FlatNav C API",
    author="FlatNav Team",
    packages=find_packages(),
    package_data={
        "flatnav_c": ["*.so", "*.dylib", "*.dll"],
    },
    include_package_data=True,
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": ["pytest", "scipy"],
    },
    python_requires=">=3.8",
    cmdclass={
        "build_ext": CMakeBuild,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
