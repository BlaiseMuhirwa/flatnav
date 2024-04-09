#!/bin/bash 


set -ex 

# Make sure we are in this directory
cd "$(dirname "$0")"

cd ../flatnav_python 
./build_wheel.sh
cd ../docs

WHEEL_FILE=$(ls ../flatnav_python/dist/*.whl)

poetry run pip install ${WHEEL_FILE} 
poetry run make clean 
poetry run make html 
