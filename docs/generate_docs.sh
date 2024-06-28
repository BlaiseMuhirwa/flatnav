#!/bin/bash 

set -ex 

# Make sure we are in this directory
cd "$(dirname "$0")"

FORCE_BUILD=0

# Check if a -f or --force flag was passed. If yes, then force the build
# of the wheel. Otherwise, check if flatnav exists in the current environment
# If it does, then skip the build process. 
if [ "$1" == "-f" ] || [ "$1" == "--force" ]; then
    FORCE_BUILD=1
else
    if poetry run python -c "import flatnav" &> /dev/null; then
        echo "flatnav already installed. Skipping build process"
    fi
fi

# Build the wheel file
if [ $FORCE_BUILD -eq 1 ]; then
    echo "Forcing the build of the wheel file"
    cd ../flatnav_python
    ./build_wheel.sh
    cd ../docs
    WHEEL_FILE=$(ls ../flatnav_python/dist/*.whl)
    poetry run pip install ${WHEEL_FILE}
fi

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
