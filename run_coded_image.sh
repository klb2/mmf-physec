#!/bin/sh

PATH_MEAS="measurements/coded_image"
PATH_RESULTS="results"
PATH_LOGO="logo"

for i in {1..6}; do
	echo "Running measurement $i/6"
	OUTF="ber-tud-logo_60-$i.dat"
	python3 coded_image_transmission.py "${PATH_MEAS}/final60-${i}" "${PATH_LOGO}" -o "$OUTF" --export
done
python3 create_ber_curve.py ${PATH_RESULTS}/ber-tud-logo_60-{1..6}.dat --plot --export
