#!/bin/bash 

# Use this script to collect statistics (memory, CPU, etc.) from the host machine including 
# the Docker container(s) running on it. 
# Run cAdvisor to collect container statistics, prometheus to store the statistics, and 
# Grafana to visualize the statistics.
# cAdvisor exposes port 8080 that prometheus scrapes to collect the statistics.


set -ex 

# Make sure we're at the root directory 
cd "$(dirname "$0")/.."

PROMETHEUS_CONFIG_FILE=bin/memory-profiling/prometheus.yml

# Start cAdvisor
docker run \
    --volume=/:/rootfs:ro \
    --volume=/var/run:/var/run:ro \
    --volume=/sys:/sys:ro \
    --volume=/var/lib/docker/:/var/lib/docker:ro \
    --publish=8080:8080 \
    --detach=true \
    --name=cadvisor \
    google/cadvisor:latest

# Start Prometheus
docker run \
    -d \
    -p 9090:9090 \
    -v ${PROMETHEUS_CONFIG_FILE}:/etc/prometheus/prometheus.yml \
    --name=prometheus \
    prom/prometheus

# Start Grafana
docker run \
    -d \
    -p 3000:3000 \
    --name=grafana \
    grafana/grafana