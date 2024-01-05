#!/bin/bash 

set -ex 


# Make sure we are in this directory 
cd "$(dirname "$0")"

# Clear old build artifacts
rm -rf build dist *.egg-info

poetry lock && poetry install --no-root

# Activate the poetry environment 
POETRY_ENV=$(poetry env info --path)

# Generate wheel file
poetry run python setup.py bdist_wheel

# Assuming the build only produces one wheel file in the dist directory
WHEEL_FILE=$(ls dist/*.whl)


# Install the wheel using pip 
# This is not optimal because we are not caching the wheel file, but it's good 
# because it avoids headaches with pip refusing to install a new wheel without 
# changing the version number
poetry run pip install $WHEEL_FILE --force-reinstall --no-cache-dir

echo "Installation of wheel completed"

#Testing the wheel 
poetry run python -c "import flatnav"

