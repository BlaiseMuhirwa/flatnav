#!/bin/bash 

# Use this script to collect statistics (memory, CPU, etc.) from the host machine including 
# the Docker container(s) running on it. 
# Run cAdvisor to collect container statistics, prometheus to store the statistics, and 
# Grafana to visualize the statistics.
# cAdvisor exposes port 8080 that prometheus scrapes to collect the statistics.



set -e

# Move to the root directory
cd "$(dirname "$0")/.."

PROMETHEUS_CONFIG_FILE="$(pwd)/bin/memory-profiling/prometheus.yml"

# Parse --force or -f flag
FORCE=false
if [[ "$1" == "--force" || "$1" == "-f" ]]; then
    FORCE=true
fi

# Function to check if a container exists and start or restart it
start_container() {
    local container_name=$1
    local image=$2
    shift 2
    local -a run_command=("$@")

    if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        if [ "$FORCE" = true ]; then
            echo "Force mode enabled. Removing existing container: $container_name"
            docker rm -f "$container_name"
        else
            echo "Container $container_name already exists. Skipping..."
            return
        fi
    fi

    echo "Starting container: $container_name"
    docker run --name="$container_name" "${run_command[@]}" "$image"
}

# Start cAdvisor
start_container "cadvisor" "google/cadvisor:latest" \
    --volume=/:/rootfs:ro \
    --volume=/var/run:/var/run:ro \
    --volume=/sys:/sys:ro \
    --volume=/var/lib/docker/:/var/lib/docker:ro \
    --publish=8080:8080 \
    --detach=true

# Start Prometheus
start_container "prometheus" "prom/prometheus" \
    -d \
    -p 9091:9090 \
    -v "${PROMETHEUS_CONFIG_FILE}:/etc/prometheus/prometheus.yml"

# Start Grafana
start_container "grafana" "grafana/grafana" \
    -d \
    -p 3000:3000
