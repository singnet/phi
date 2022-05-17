#!/bin/bash

docker build -t phi --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
