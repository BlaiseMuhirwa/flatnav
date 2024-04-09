#!/bin/bash 


set -ex 

# Make sure we are in this directory
cd "$(dirname "$0")"

cd ../flatnav_python 
./build_wheel.sh
cd ../docs

WHEEL_FILE=$(ls ../flatnav_python/dist/*.whl)

poetry run pip install ${WHEEL_FILE} 

# Build the doxygen documentation for the C++ code. 
# First, we need to install doxygen if it is not already installed
if ! [ -x "$(command -v doxygen)" ]; then
    echo "doxygen is not installed. Installing doxygen"
    sudo apt-get update
    sudo apt-get install doxygen
fi

# Now we can build the documentation
doxygen Doxyfile 

# Now onto the sphinx documentation
poetry run make clean 
poetry run make html 
