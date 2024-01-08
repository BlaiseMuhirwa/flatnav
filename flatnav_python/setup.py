import os
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import sys, subprocess

__version__ = "0.0.1"

CURRENT_DIR = os.getcwd()
SOURCE_PATH = os.path.join(CURRENT_DIR, "python_bindings.cpp")


def platform_has_avx_support():
    """
    TODO: Make this check more robust across compilers and supported platforms.
    """
    if sys.platform in ["linux", "linux2", "darwin"]:
        try:
            output = subprocess.check_output("cat /proc/cpuinfo", shell=True)
            decoded_output = output.decode()
            return "avx" in decoded_output or "avx512" in decoded_output
        except Exception:
            return False

    else:
        return False


INCLUDE_DIRS = [
    os.path.join(CURRENT_DIR, ".."),
    os.path.join(CURRENT_DIR, "..", "external", "cereal", "include"),
]
EXTRA_LINK_ARGS = []

if sys.platform == "darwin":
    omp_flag = "-Xclang -fopenmp"
    INCLUDE_DIRS.extend(["/opt/homebrew/opt/libomp/include"])
    EXTRA_LINK_ARGS.extend(["-lomp", "-L/opt/homebrew/opt/libomp/lib"])
elif sys.platform == "linux":
    omp_flag = "-fopenmp"
    EXTRA_LINK_ARGS.extend(["-fopenmp"])


EXTRA_COMPILE_ARGS = [
    omp_flag,  # Enable OpenMP
    "-Ofast",  # Use the fastest optimization
    "-fpic",  # Position-independent code
    "-w",  # Suppress all warnings (note: this overrides -Wall)
    "-ffast-math",  # Enable fast math optimizations
    "-funroll-loops",  # Unroll loops
    "-mavx",  # Enable AVX instructions
    # "-mavx512f",  # Enable AVX-512 instructions
]

if not platform_has_avx_support():
    EXTRA_COMPILE_ARGS.append("-ftree-vectorize")


ext_modules = [
    Pybind11Extension(
        "flatnav",
        [SOURCE_PATH],
        define_macros=[("VERSION_INFO", __version__)],
        cxx_std=17,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,  # Link OpenMP when linking the extension
    )
]


setup(
    name="flatnav",
    version=__version__,
    author="Benjamin Coleman",
    author_email="benjamin.ray.coleman@gmail.com",
    url="https://randorithms.com",
    description="Graph kNN with reordering.",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
