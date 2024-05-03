# Build arguments 
# This is a relatively large image, so we might want to use a smaller base image, such as
# alpine in the future if image size becomes an issue.
ARG BASE_IMAGE=ubuntu:22.04

# Base image
FROM ${BASE_IMAGE} AS base

ARG POETRY_VERSION=1.7.1
ARG PYTHON_VERSION=3.11.6
ARG POETRY_HOME="/opt/poetry"
ARG ROOT_DIR="/root"
ARG FLATNAV_PATH="${ROOT_DIR}/flatnavlib"

# Install base system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        # Need for python installation: 
        # https://github.com/pyenv/pyenv/wiki#suggested-build-environment
        make \
        build-essential \
        ca-certificates \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        # Installation necessary for running a cronjob inside the container
        cron \
        # Install the rest
        git \
        gcc \
        g++ \
        apt-utils \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

# Install python 
# We use pyenv to manage python versions 
ENV PYENV_ROOT=$HOME/.pyenv

# Shims are small proxy executables that intercept calls to Python commands. 
# Putting $PYENV_ROOT/shims at the beginning of PATH ensures that the shimmed 
# Python commands are found and used before any system-wide Python installations.
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH 

ENV PYTHON_VERSION=${PYTHON_VERSION}


RUN set -ex \
    && curl -L https://pyenv.run | /bin/sh \
    && pyenv update \
    && pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && pyenv rehash 

# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_HOME=${POETRY_HOME} \
    POETRY_VERSION=${POETRY_VERSION} 

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 - 

# Add poetry to PATH
ENV PATH="${POETRY_HOME}/bin:${PATH}"

WORKDIR ${FLATNAV_PATH}

# Copy source code
COPY flatnav/ ./flatnav/
COPY flatnav_python/ ./flatnav_python/
COPY experiments/ ./experiments/
# We don't need quantization for now, but it doesn't hurt to have it.
COPY quantization/ ./quantization/
# Copy external dependencies (for now only cereal)
COPY external/ ./external/

# Copy and set up the cron file 
# Set up a cron job to push snapshots to s3 every minute
# * * * * * * means every minute
RUN * * * * * poetry run python \
    ${ROOT_DIR}/experiments/push_snapshot_to_s3.py >> /var/log/cron.log 2>&1

COPY cronjob /etc/cron.d/upload-cron

# Give the correct permissions to the cron job 
RUN chmod 0644 /etc/cron.d/upload-cron \
    && crontab /etc/cron.d/upload-cron
RUN touch /var/log/cron.log

# Start the cron and keep the container running by tailing the log. 
CMD cron && tail -f /var/log/cron.log 


# Install needed dependencies including flatnav. 
# This installs numpy as well, which is a large dependency. 
WORKDIR ${FLATNAV_PATH}/flatnav_python
RUN ./install_flatnav.sh 

# Install hnwlib (from a forked repo that has extensions we need)
WORKDIR ${FLATNAV_PATH}
RUN git clone https://github.com/BlaiseMuhirwa/hnswlib-original.git \
    && cd hnswlib-original/python_bindings \
    && poetry install --no-root \
    && poetry run python setup.py bdist_wheel  

# Get the wheel as an environment variable 
# NOTE: This is not robust and will break if there are multiple wheels in the dist folder
ENV FLATNAV_WHEEL=${FLATNAV_PATH}/flatnav_python/dist/*.whl
ENV HNSWLIB_WHEEL=${FLATNAV_PATH}/hnswlib-original/python_bindings/dist/*.whl

# Add flatnav and hnswlib to the experiment runner 
WORKDIR ${FLATNAV_PATH}/experiments
RUN poetry add ${FLATNAV_WHEEL} \
    && poetry add ${HNSWLIB_WHEEL} \
    && poetry install --no-root

