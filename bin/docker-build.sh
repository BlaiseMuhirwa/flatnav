#!/bin/bash 


set -ex

# Make sure we are one level above this directory
cd "$(dirname "$0")/.."

function get_tag_name() {
    # Returns a string to be used as a docker tag revision.
    # If it's in a clean git repo, it returns the commit's short hash with branch name like ce37fd7-main
    # If the working tree is dirty, it returns something like main-ce37fd7-dirty-e52e78f86e575bd
    #     including the branch name, and a consistent hash of the uncommitted changes

    fail() {
        echo $1
        exit 1
    }

    if [[ ! -z "${OVERRIDE_GIT_TAG_NAME}" ]]; then
        echo $OVERRIDE_GIT_TAG_NAME
        exit 0
    fi

    # Figure out which SHA utility exists on this machine.
    HASH_FUNCTION=sha1sum
    which $HASH_FUNCTION > /dev/null || HASH_FUNCTION=shasum
    which $HASH_FUNCTION > /dev/null || fail "Can't find SHA utility"

    # Try to get current branch out of GITHUB_REF for CI
    # The ##*/ deletes everything up to /
    CURRENT_BRANCH=${GITHUB_REF##*/}
    # Now generate the short commit
    CURRENT_COMMIT=$(echo $GITHUB_SHA | cut -c -9)

    # If we're not running in CI, GITHUB_REF and GITHUB_SHA won't be set.
    # In this case, figure them out from our git repository
    # (If we do this during github CI, we get a useless unique commit on the "merge" branch.)
    # When infering CURRENT_BRANCH, convert '/'s to '-'s, since '/' is not allowed in docker tags but
    # is part of common git branch naming formats e.g. "feature/branch-name" or "user/branch-name"
    CURRENT_BRANCH=${CURRENT_BRANCH:-$(git rev-parse --abbrev-ref HEAD | sed -e 's/\//-/g')}
    CURRENT_COMMIT=${CURRENT_COMMIT:-$(git rev-parse --short=9 HEAD)}

    if [[ -z "$(git status --porcelain)" ]] || [[ "${CI}" = true ]]; then
        # Working tree is clean
        echo "${CURRENT_COMMIT}-${CURRENT_BRANCH}"
    else
        # Working tree is dirty.
        HASH=$(echo $(git diff && git status) | ${HASH_FUNCTION} | cut -c -15)
        echo "${CURRENT_BRANCH}-${CURRENT_COMMIT}-dirty-${HASH}"
    fi
}


# We use docker buildx to build the image for multiple platforms. buildx comes
# installed with Docker Engine when installed via Docker Desktop. If you're
# on a Linux machine with an old version of Docker Engine, you may need to
# install buildx manually. Follow these instructions to install docker-buildx-plugin:
# https://docs.docker.com/engine/install/ubuntu/

# Install QEMU, a generic and open-source machine emulator and virtualizer
docker run --rm --privileged linuxkit/binfmt:af88a591f9cc896a52ce596b9cf7ca26a061ef97

# Check if builder already exists
if ! docker buildx ls | grep -q flatnavbuilder; then
  # Prep for multiplatform build - the build is done INSIDE a docker container
  docker buildx create --name flatnavbuilder --use
else
  # If builder exists, set it as the current builder
  docker buildx use flatnavbuilder
fi

# Ensure that the builder container is running
docker buildx inspect flatnavbuilder --bootstrap

# Get the tag name
TAG_NAME=$(get_tag_name)

echo "Building docker image with tag name: $TAG_NAME"

# Build the image for multiple platforms (i.e., x86 intel and amd chips, and ARM chips)
# Use --load to load the image into the docker cache (by default, the image is not loaded.
# It is only available in the buildx cache)
docker buildx build --platform linux/amd64 -t flatnav:$TAG_NAME -f Dockerfile . --load 

if [ -z "$1" ]
then
    # This will build the image and run the container with the default make target
    # (i.e., print help message)
    docker run -it --volume $(pwd)/data:/root/data --rm flatnav:$TAG_NAME make help
    exit 0
fi


# Run the container and mount the data/ directory as volume to /root/data
# Pass the make target as argument to the container. 
docker run -it --volume $(pwd)/data:/root/data --rm flatnav:$TAG_NAME make $1

