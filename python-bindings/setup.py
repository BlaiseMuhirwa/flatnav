import os
# from pybind11.setup_helpers import Pybind11Extension, build_ext

# from setuptools import setup
from skbuild import setup
import sys, subprocess
from typing import List
__version__ = "0.0.1"

CURRENT_DIR = os.getcwd()
SOURCE_PATH = os.path.join(CURRENT_DIR, "python_bindings.cpp")

long_description_path = os.path.join(
    os.path.join(os.path.dirname(__file__), "..", "README.md")
)
with open(long_description_path, "r") as f:
    long_description = f.read()


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


# INCLUDE_DIRS = [
#     os.path.join(CURRENT_DIR, ".."),
#     os.path.join(CURRENT_DIR, "..", "external", "cereal", "include"),
# ]
# EXTRA_LINK_ARGS = []

# if sys.platform == "darwin":
#     omp_flag = "-Xclang -fopenmp"
#     INCLUDE_DIRS.extend(["/opt/homebrew/opt/libomp/include"])
#     EXTRA_LINK_ARGS.extend(["-lomp", "-L/opt/homebrew/opt/libomp/lib"])
# elif sys.platform in ["linux", "linux2"]:
#     omp_flag = "-fopenmp"
#     EXTRA_LINK_ARGS.extend(["-fopenmp"])


# EXTRA_COMPILE_ARGS = [
#     omp_flag,  # Enable OpenMP
#     "-Ofast",  # Use the fastest optimization
#     "-fpic",  # Position-independent code
#     "-w",  # Suppress all warnings (note: this overrides -Wall)
#     "-ffast-math",  # Enable fast math optimizations
#     "-funroll-loops",  # Unroll loops
#     "-march=native",  # Use the native architecture
# ]

# # We don't include SIMD flags if the NO_SIMD_VECTORIZATION variable is set to 1
# no_simd_vectorization = int(os.environ.get("NO_SIMD_VECTORIZATION", "0"))

# if not no_simd_vectorization:
#     SIMD_EXTENSIONS = ["sse", "sse3", "sse4", "avx", "avx512f", "avx512bw"]
#     found_single_extension = False
#     for extension in SIMD_EXTENSIONS:
#         if simd_extension_supported(extension=extension):
#             EXTRA_COMPILE_ARGS.append(f"-m{extension}")
#             found_single_extension = True

#     if not found_single_extension:
#         # We've found that ftree-vectorize (auto-vectorization) is good in general
#         # but it does slow down SIMD extensions considerably. So, we only enable it
#         # if we haven't found any SIMD extensions.
#         # Reference: https://llvm.org/docs/Vectorizers.html
#         EXTRA_COMPILE_ARGS.append("-ftree-vectorize")


# ext_modules = [
#     Pybind11Extension(
#         "flatnav",
#         [SOURCE_PATH],
#         define_macros=[("VERSION_INFO", __version__)],
#         cxx_std=17,
#         include_dirs=INCLUDE_DIRS,
#         extra_compile_args=EXTRA_COMPILE_ARGS,
#         extra_link_args=EXTRA_LINK_ARGS,  # Link OpenMP when linking the extension
#     )
# ]


def construct_cmake_args() -> List[str]:
    """
    Construct CMake arguments based on the platform.
    Env vars to set up:
    1. -DCMAKE_BUILD_TYPE:STRING=Release
    """
    current_wdir = os.getcwd()
    cmake_include_directories = [
        os.path.join(current_wdir, "..", "include"),
        os.path.join(current_wdir, "..", "external", "cereal", "include"),
    ]
    cmake_linker_args = []

    if sys.platform == "darwin":
        omp_flag = "-Xclang -fopenmp"
        cmake_include_directories.extend(["/opt/homebrew/opt/libomp/include"])
        cmake_linker_args.extend(["-lomp", "-L/opt/homebrew/opt/libomp/lib"])

    elif sys.platform in ["linux", "linux2"]:
        omp_flag = "-fopenmp"
        cmake_linker_args.extend(["-fopenmp"])

    cmake_args = []
    cmake_args.append(f"-DCMAKE_INCLUDE_PATH={';'.join(cmake_include_directories)}")

    # Here is what each of the following flags do:
    # - OpenMP: Enable OpenMP
    # - Ofast: Use the fastest optimization
    # - fpic: Position-independent code
    # - w: Suppress all warnings (note: this overrides -Wall)
    # - ffast-math: Enable fast math optimizations
    # - funroll-loops: Unroll loops
    # - march=native: Use the native architecture
    compile_args = [
        omp_flag,
        "-Ofast",
        "-fpic",
        "-w",
        "-ffast-math",
        "-funroll-loops",
        "-march=native",
    ]

    # Add SIMD flags if SIMD vectorization is enabled
    if not int(os.environ.get("NO_SIMD_VECTORIZATION", "0")):
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

    # Add compile arguments to cmake_args
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
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["flatnav"],
    package_dir={"": "src"},
    cmake_install_dir="src/flatnav",
    cmake_args=construct_cmake_args(),
    install_reqquires=[
        # The following need to be synced with pyproject.toml
        "numpy>=1.21.0,<2",
        "archspec>=0.2.0",
        "toml>=0.10.2",
    ],
    license="Apache License, Version 2.0",
    # license_files=["../LICENSE.md"],
    keywords=["similarity search", "vector databases", "machine learning"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Other Audience",
        "License :: OSI Approved :: Apache License, Version 2.0",
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


# ext_modules=ext_modules,
# cmdclass={"build_ext": build_ext},
# zip_safe=False,
# python_requires=">=3.9",
