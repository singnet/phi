#!/bin/sh

if [ -z "$1" ]
  then
    echo "Invalid PAYMENT_ADDRESS: ./publishAndStartService.sh PAYMENT_ADDRESS"
    exit 1
fi

TMP_FILE=/tmp/__SNET_SERVICE_PUBLISH_LOG.txt
rm -f $TMP_FILE

if [ "$(snet organization info snet 2>&1 | tee $TMP_FILE)" != 0 ]
  then
    echo "Creating the snet organization. Please wait..."
    snet organization metadata-init "snet" snet individual
    snet organization add-group default_group "$1" http://127.0.0.1:2379
    snet organization create snet -y 2>&1 | tee $TMP_FILE
fi

echo "Publishing your service. Please wait..."

snet service \
    metadata-init \
    service_spec \
    "phi-service" \
    --group-name default_group \
    --fixed-price 0.00000001 \
    --endpoints http://localhost:7000
snet service publish snet phi-service -y 2>&1 | tee $TMP_FILE

python3 server.py &
snetd --config config/snetd.config.json &
sleep 3
