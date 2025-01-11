#!/bin/bash 

# Use cibuildwheel to build wheels. 
# Let's only use python 3.11 for now for linux builds. We will use CI_SKIP for other platforms and python versions.

export CIBW_BUILD="cp311-macosx_universal2"

cibuildwheel python-bindings