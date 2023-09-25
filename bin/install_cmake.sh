#!/bin/bash

# Function to install CMake on Ubuntu
install_cmake_ubuntu() {
    sudo apt update
    sudo apt install -y cmake
}

# Function to install CMake on macOS
install_cmake_mac() {
    # Ensure Homebrew is installed
    if ! command -v brew &>/dev/null; then
        echo "Error: Homebrew is not installed. Please install it from https://brew.sh/"
        exit 1
    fi
    brew install cmake
}

# Main part of the script
if command -v cmake &>/dev/null; then
    echo "CMake is already installed."
    cmake --version
else
    OS=$(uname)
    echo "CMake is not installed. Installing..."
    if [ "$OS" == "Darwin" ]; then
        install_cmake_mac
    elif [ "$OS" == "Linux" ]; then
        # We assume it's Ubuntu/Debian for simplicity.
        install_cmake_ubuntu
    else
        echo "Unsupported operating system: $OS"
        exit 1
    fi

    # Verify installation
    if command -v cmake &>/dev/null; then
        echo "CMake has been installed successfully."
        cmake --version
    else
        echo "Error installing CMake."
    fi
fi
