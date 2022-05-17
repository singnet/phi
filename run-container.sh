#!/bin/bash

docker run -p 8887:8887 --name phi_container --mount "type=bind,src=$(pwd),dst=/opt/singnet/phi" -v ~/Shared:/home/user/Shared -ti phi bash
