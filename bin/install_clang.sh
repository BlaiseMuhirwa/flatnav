#!/bin/bash

# Function to check if a command exists
command_exists() {
    type "$1" &> /dev/null ;
}

# Check for clang
if ! command_exists clang++; then
    echo "clang++ not found. Installing..."
    sudo apt update
    sudo apt install -y clang
else
    echo "clang++ already installed."
fi

if ! command_exists clang-format; then
    echo "clang-format not found. Installing..."
    sudo apt update
    sudo apt install -y clang-format
else
    echo "clang-format already installed."
fi

# Check for libomp-dev
PKG_STATUS=$(dpkg-query -W --showformat='${Status}\n' libomp-dev | grep "install ok installed")
if [ "" == "$PKG_STATUS" ]; then
    echo "libomp-dev not found. Installing..."
    sudo apt update
    sudo apt install -y libomp-dev
else
    echo "libomp-dev already installed."
fi
