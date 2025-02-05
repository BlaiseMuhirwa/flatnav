#!/bin/bash 

set -ex 

# Make sure we are in this directory
cd "$(dirname "$0")"

# Just another sanity check to make sure every dependency is installed
poetry install

# Determine the operating system
OS="$(uname -s)"

# Install doxygen if not installed
if ! [ -x "$(command -v doxygen)" ]; then
    echo "doxygen is not installed. Installing doxygen"

    if [ "$OS" == "Linux" ]; then
        sudo apt-get update
        sudo apt-get install -y doxygen
    elif [ "$OS" == "Darwin" ]; then
        brew install doxygen
    else
        echo "Unsupported OS: $OS"
        exit 1
    fi
fi

# Now we can build the documentation
doxygen Doxyfile 

# Now onto the sphinx documentation
poetry run make clean 
poetry run make html 
