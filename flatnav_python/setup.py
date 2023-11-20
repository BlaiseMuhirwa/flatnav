# import toml
import os

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# def parse_version_from_pyproject() -> str:
#     with open("pyproject.toml") as f:
#         pyproject = toml.load(f)
#         return pyproject["tool"]["poetry"]["version"]

#     raise RuntimeError("Unable to find version string.")

__version__ = "0.0.1"

CURRENT_DIR = os.getcwd()
SOURCE_PATH = os.path.join(CURRENT_DIR, "bindings.cpp")


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
        # Ignoring the `Wno-sign-compare` which warns you when you compare int with something like
        # uint64_t. 
        extra_compile_args=["-Wno-sign-compare", "-fopenmp"],
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
    # extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
