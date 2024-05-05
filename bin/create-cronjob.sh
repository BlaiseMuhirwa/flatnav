#!/bin/bash 

export BUCKET_PREFIX=$1


cd /root/flatnavlib/experiments

# Execute the python script to save indexes to s3
poetry run python push-snapshot-to-s3.py

