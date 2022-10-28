#!/bin/bash

docker build -t tononi_phi_service \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    -f Dockerfile.service .