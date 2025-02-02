# Build arguments 
# debian:buster-slim is much smaller than ubuntu 22
ARG BASE_IMAGE=debian:buster-slim

FROM ${BASE_IMAGE} AS base

ARG POETRY_VERSION=1.8.2
ARG PYTHON_VERSION=3.11.6
ARG POETRY_HOME="/opt/poetry"
ARG ROOT_DIR="/root"
ARG FLATNAV_PATH="${ROOT_DIR}/flatnavlib"

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
        # Install the rest
        git \
        gcc \
        g++ \
        apt-utils \
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
COPY include/flatnav/ ./flatnav/
COPY python-bindings/ ./python-bindings/
COPY experiments/ ./experiments/

# Copy external dependencies (for now only cereal)
COPY external/ ./external/

# Install needed dependencies including flatnav. 
# Install hnwlib (from a forked repo that has extensions we need)
WORKDIR ${FLATNAV_PATH}
RUN git clone https://github.com/BlaiseMuhirwa/hnswlib-original.git \
    && cd hnswlib-original/python_bindings \
    && poetry install --no-root \
    && poetry run python setup.py bdist_wheel  

# Get the wheel as an environment variable 
ENV HNSWLIB_WHEEL=${FLATNAV_PATH}/hnswlib-original/python_bindings/dist/*.whl

# Add hnswlib to the experiment runner 
WORKDIR ${FLATNAV_PATH}/experiments
RUN poetry add ${HNSWLIB_WHEEL} && poetry install --no-root