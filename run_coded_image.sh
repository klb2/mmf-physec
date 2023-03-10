#!/bin/sh

# This repository is accompanying the paper "Securing Data in Multimode Fibers
# by Exploiting Mode-Dependent Light Propagation Effects" (S. Rothe, K.-L.
# Besser, D. Krause, R. Kuschmierz, N. Koukourakis, E. Jorswieck, J. Czarske.
# Research, vol. 6: 0065, Jan. 2023. DOI:10.34133/research.0065).


PATH_MEAS="measurements/coded_image"
PATH_RESULTS="results"
PATH_LOGO="logo"

for i in {1..6}; do
	echo "Running measurement $i/6"
	OUTF="ber-tud-logo_60-$i.dat"
	python3 coded_image_transmission.py "${PATH_MEAS}/final60-${i}" "${PATH_LOGO}" -o "$OUTF" --export
done
python3 create_ber_curve.py ${PATH_RESULTS}/ber-tud-logo_60-{1..6}.dat --plot --export
