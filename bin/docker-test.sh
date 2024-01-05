#!/bin/bash 

# Print commands and exit on errors
set -ex 

# Make sure we are one level above this directory
cd "$(dirname "$0")/.."

# Build docker container 
docker build --tag flatnav:latest -f Dockerfile . 


# Check if the first argument is set. If it is, then run docker container with the 
# first argument as the make target. If not, then run the container with the default
# make target
if [ -z "$1" ]
then
    # This will build the image and run the container with the default make target
    # (i.e., print help message)
    CONTAINER_ID=$(docker run -it \
        --volume $(pwd)/data:/root/data \
        --rm flatnav:latest \
        make help)
    echo $CONTAINER_ID
    exit 0
fi


# Run the container and mount the data/ directory as volume to /root/data
# Pass the make target as argument to the container. 
CONTAINER_ID=$(docker run -it \
    --volume $(pwd)/data:/root/data \
    --rm flatnav:latest \
    make $1)
echo $CONTAINER_ID