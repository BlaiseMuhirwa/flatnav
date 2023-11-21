import os
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

CURRENT_DIR = os.getcwd()
SOURCE_PATH = os.path.join(CURRENT_DIR, "python_bindings.cpp")


ext_modules = [
    Pybind11Extension(
        "flatnav",
        [SOURCE_PATH],
        define_macros=[("VERSION_INFO", __version__)],
        cxx_std=17,
        include_dirs=[
            os.path.join(CURRENT_DIR, ".."),
            os.path.join(CURRENT_DIR, "..", "external", "cereal", "include"),
        ],
        extra_compile_args=[
            "-fopenmp",  # Enable OpenMP
            "-Ofast",  # Use the fastest optimization
            "-fpic",  # Position-independent code
            "-w",  # Suppress all warnings (note: this overrides -Wall)
            "-ffast-math",  # Enable fast math optimizations
            "-funroll-loops",  # Unroll loops
            "-ftree-vectorize",  # Vectorize where possible
        ],
        extra_link_args=["-fopenmp"],  # Link OpenMP when linking the extension
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
