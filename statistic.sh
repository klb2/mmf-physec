#!/bin/sh

MEASUREMENT_FILE="measurements/statistic/measurement"

for counter in {1..3}
do
	echo "---------------------------------------------------"
	echo "Calculate the secrecy capacity for the BPSK example with SVD precoding"
	echo "> Perfect SVD (upper bound)"
	python3 svd_precoding.py --matrix="$MEASUREMENT_FILE$counter.mat" -k 15 -s 10
	echo "> Optical SVD (actual measured channel)"
	python3 svd_precoding.py --matrix="$MEASUREMENT_FILE$counter.mat" -k 15 -s 10 --precoded
	echo "> No Precoding (based on measured channel)"
	python3 no_precoding.py --matrix="$MEASUREMENT_FILE$counter.mat" -k 15 -s 10
done
python show_capacity_statistics.py
