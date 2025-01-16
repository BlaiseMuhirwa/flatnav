import os
from skbuild import setup
import sys, subprocess
from typing import List
__version__ = "0.1.2"

package_description_path = os.path.join(
    os.path.join(os.path.dirname(__file__), "..", "README.md")
)
with open(package_description_path, "r") as f:
    package_description = f.read()

class UnsupportedPlatformError(Exception):
    pass


def simd_extension_supported(extension: str) -> bool:
    """
    Check if the CPU supports a given SIMD extension.
    """
    if sys.platform in ["linux", "linux2"]:
        command = "cat /proc/cpuinfo"
    elif sys.platform == "darwin":
        command = "/usr/sbin/sysctl -n machdep.cpu.features"
    else:
        raise UnsupportedPlatformError(f"Unsupported platform: {sys.platform}")

    try:
        output = subprocess.check_output(command, shell=True)
        # Using .lower() to make the output case-insensitive since on MacOS the output
        # is in uppercase
        decoded_output = output.decode().lower()
        return extension in decoded_output
    except Exception:
        return False


def construct_cmake_args() -> List[str]:
    """
    Construct CMake arguments based on the platform.
    """
    current_wdir = os.getcwd()
    cmake_include_directories = [
        os.path.join(current_wdir, "..", "include"),
        os.path.join(current_wdir, "..", "external", "cereal", "include"),
    ]
    cmake_linker_args = []

    if sys.platform == "darwin":
        omp_flag = "-Xpreprocessor -fopenmp"
        cmake_include_directories.extend(["/opt/homebrew/opt/libomp/include"])
        cmake_linker_args.extend(["-lomp", "-L/opt/homebrew/opt/libomp/lib"])
        
        # Base compile args for all platforms
        compile_args = [
            omp_flag,
            "-Ofast",
            "-fpic",
            "-w",
            "-ffast-math",
            "-funroll-loops",
        ]
        
        # Add x86_64 specific flags for macOS
        compile_args.extend([
            "-arch", "x86_64",
            "-mmacosx-version-min=10.14",
            "-stdlib=libc++"
        ])
        
    elif sys.platform in ["linux", "linux2"]:
        omp_flag = "-fopenmp"
        cmake_linker_args.extend(["-fopenmp"])
        
        compile_args = [
            omp_flag,
            "-Ofast",
            "-fpic",
            "-w",
            "-ffast-math",
            "-funroll-loops",
            # Keep native architecture only for Linux
            "-march=native"  
        ]

    # Add SIMD flags if SIMD vectorization is enabled
    if not int(os.environ.get("NO_SIMD_VECTORIZATION", "0")):
        if sys.platform in ["linux", "linux2"]:
            # Only add SIMD flags on Linux
            supported_simd_flags = [
                "sse",
                "sse2",
                "sse3",
                "avx",
                "avx512f",
                "avx512bw",
            ]
            simd_flags = [
                f"-m{flag}"
                for flag in supported_simd_flags
                if simd_extension_supported(extension=flag)
            ]
            if simd_flags:
                compile_args.extend(simd_flags)
            else:
                compile_args.append("-ftree-vectorize")

    cmake_args = []
    cmake_args.append(f"-DCMAKE_INCLUDE_PATH={';'.join(cmake_include_directories)}")
    cmake_args.append(f"-DCMAKE_CXX_FLAGS={' '.join(compile_args)}")

    if cmake_linker_args:
        cmake_args.append(f"-DCMAKE_EXE_LINKER_FLAGS={' '.join(cmake_linker_args)}")

    return cmake_args

setup(
    name="flatnav",
    version=__version__,
    author="Benjamin Coleman, Blaise Munyampirwa, Vihan Lakshman",
    author_email="benjamin.ray.coleman@gmail.com, blaisemunyampirwa@gmail.com, vihan@mit.edu",
    maintainer_email="blaisemunyampirwa@gmail.com",
    url="https://flatnav.net",
    project_urls={
        "Source Code": "https://github.com/BlaiseMuhirwa/flatnav",
        "Documentation": "https://blaisemuhirwa.github.io/flatnav",
        "Bug Tracker": "https://github.com/BlaiseMuhirwa/flatnav/issues",
    },
    description="A performant graph-based kNN search library with re-ordering.",
    long_description=package_description,
    long_description_content_type="text/markdown",
    packages=["flatnav"],
    package_dir={"": "src"},
    cmake_install_dir="src/flatnav",
    cmake_args=construct_cmake_args(),
    install_requires=[
        # The following need to be synced with pyproject.toml
        "numpy>=1.21.0,<2",
        "h5py==3.11.0"
    ],
    license="Apache License, Version 2.0",
    keywords=["similarity search", "vector databases", "machine learning"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Other Audience",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: System",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    include_package_data=True,
)

