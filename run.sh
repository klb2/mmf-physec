#!/bin/sh

# This repository is accompanying the paper "Securing Data in Multimode Fibers
# by Exploiting Mode-Dependent Light Propagation Effects" (S. Rothe, K.-L.
# Besser, D. Krause, R. Kuschmierz, N. Koukourakis, E. Jorswieck, J. Czarske.
# Research, vol. 6: 0065, Jan. 2023. DOI:10.34133/research.0065).


MEASUREMENT_FILE="measurements/mmf-measurement.mat"
DATA_MEASUREMENT="measurements/data-measurement.mat"
LOGO_FILE="logo/tud_logo_60.txt"

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
python3 data_transmission.py "${DATA_MEASUREMENT}" "${LOGO_FILE}"

echo "---------------------------------------------------"
echo "Run coded image transmission"
sh run_coded_image.sh
