#!/bin/bash 

docker run \
    -d \
    -p 9090:9090 \
    -v /home/brc7/flatnav/bin/prometheus.yml:/etc/prometheus/prometheus.yml \
    prom/prometheus