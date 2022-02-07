#!/bin/sh

MEASUREMENT_FILE="measurements/mmf-measurement.mat"
DATA_MEASUREMENT="measurements/data-measurement.mat"

echo "---------------------------------------------------"
echo "Calculate the secrecy capacity for the BPSK example with SVD precoding"
echo "> Perfect SVD (upper bound with measured channel)"
python3 svd_precoding.py --matrix="$MEASUREMENT_FILE" -k 15 -s 10 --plot
echo "> Optical SVD (actual measured channel)"
python3 svd_precoding.py --matrix="$MEASUREMENT_FILE" -k 15 -s 10 --precoded --plot
echo "> No Precoding (based on measured channel)"
python3 no_precoding.py --matrix="$MEASUREMENT_FILE" -k 15 -s 10 --plot


echo "---------------------------------------------------"
echo "Show data transmission"
python3 data_transmission.py "$MEASUREMENT_FILE" "$DATA_MEASUREMENT" --export
