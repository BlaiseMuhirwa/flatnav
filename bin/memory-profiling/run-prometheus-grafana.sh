#!/bin/bash 

# Use this script to collect statistics (memory, CPU, etc.) from the host machine including 
# the Docker container(s) running on it. 
# Run cAdvisor to collect container statistics, prometheus to scrape metrics, and 
# Grafana for visualization.
# cAdvisor exposes port 8080 that prometheus scrapes to collect the statistics.


set -e

# Move to the root directory
cd "$(dirname "$0")/../.."


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
    docker run "${run_command[@]}" --name="$container_name" "$image"
}

# Start cAdvisor
start_container "cadvisor" "gcr.io/cadvisor/cadvisor:latest" \
    --volume=/:/rootfs:ro \
    --volume=/var/run:/var/run:ro \
    --volume=/sys:/sys:ro \
    --volume=/var/lib/docker/:/var/lib/docker:ro \
    --publish=8080:8080 \
    --detach=true

# Start Prometheus
PROMETHEUS_CONFIG_TEMPLATE="$(pwd)/bin/memory-profiling/prometheus-template.yml"
PROMETHEUS_CONFIG_FILE="prometheus.yml"
CADVISOR_IP=$(docker inspect cadvisor --format '{{.NetworkSettings.Networks.bridge.IPAddress}}')
CADVISOR_PORT=8080

# Use sed to replace the IP and port in the template file
sed "s/\${CADVISOR_IP}/${CADVISOR_IP}/g; s/\${CADVISOR_PORT}/${CADVISOR_PORT}/g" "$PROMETHEUS_CONFIG_TEMPLATE" > "$PROMETHEUS_CONFIG_FILE"

start_container "prometheus" "prom/prometheus" -p 5000:9090 -d \
    -v "${PROMETHEUS_CONFIG_FILE}:/etc/prometheus/prometheus.yml"

# Start Grafana
start_container "grafana" "grafana/grafana" -d -p 3000:3000


