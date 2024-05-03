#!/bin/bash 

# Create a cronjob that runs every 30 seconds and executes a file called push_snapshot_to_s3.py
# This file is responsible for pushing the snapshot to s3

# Write out current crontab
*/30 * * * * poetry run python push_snapshot_to_s3.py >> /var/log/cron.log 2>&1