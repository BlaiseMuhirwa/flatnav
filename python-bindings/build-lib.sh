#!/bin/bash 

# python -m pip install --upgrade pip setuptools wheel scikit-build cmake ninja

# python -m pip install build
# python -m build

python setup.py bdist_wheel

pip install dist/flatnav-0.0.1-cp38-cp38-linux_x86_64.whl --force-reinstall