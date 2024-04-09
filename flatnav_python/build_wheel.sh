#!/bin/bash 

set -ex 


# Make sure we are in this directory 
cd "$(dirname "$0")"

# Clear old build artifacts
rm -rf build dist *.egg-info

poetry lock && poetry install --no-root

# Generate wheel file
poetry run python setup.py bdist_wheel

# Assuming the build only produces one wheel file in the dist directory
WHEEL_FILE=$(ls dist/*.whl)
