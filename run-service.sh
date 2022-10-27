#!/bin/bash

docker run -p 8887:8887 \
    --rm -ti \
    --mount "type=bind,src=$(pwd),dst=/opt/singnet/phi" \
    --name TONONI_PHI_SERVICE \
    tononi_phi_service bash