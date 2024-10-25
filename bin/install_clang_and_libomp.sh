#!/bin/bash

# Function to check if a command exists
command_exists() {
    type "$1" &> /dev/null ;
}

function install_clang_mac() {
    # Install clang and clang-format on Darwin
    if ! command_exists brew; then
        echo "Homebrew not found. Homebrew should be installed first."
        exit 1
    fi 
    brew install llvm 
}

function install_clang_linux() {
    # Install clang and clang-format on Linux
    if ! command_exists apt; then
        echo "apt not found. apt should be installed first."
        exit 1
    fi 
    echo "Installing clang and clang-format..."
    sudo apt update
    sudo apt install -y clang clang-format 
}

OS_NAME=$(uname)

# Check for clang
if ! command_exists clang++; then
    echo "clang++ not found. Installing..."

    if [[ "$OS_NAME" == "Darwin" ]]; then
        install_clang_mac
    elif [[ "$OS_NAME" == "Linux" ]]; then
        install_clang_linux
    else
        echo "Unsupported OS."
        exit 1
    fi
else
    echo "clang/clang++ already installed."
fi


# Check for libomp. This is required for OpenMP support.
if [[ "$OS_NAME" == "Darwin" ]]; then
    if ! brew ls --versions libomp > /dev/null; then
        echo "libomp not found. Installing..."
        brew install libomp

        # For compilers to find libomp we export the following variables
        export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
        export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
    else
        echo "libomp already installed."
    fi
elif [[ "$OS_NAME" == "Linux" ]]; then
    PKG_STATUS=$(dpkg-query -W --showformat='${Status}\n' libomp-dev | grep "install ok installed")
    if [ "" == "$PKG_STATUS" ]; then
        echo "libomp-dev not found. Installing..."
        sudo apt update
        sudo apt install -y libomp-dev
    else
        echo "libomp-dev already installed."
    fi
else
    echo "Unsupported OS."
    exit 1
fi
