# Physical Layer Security on Multi-Mode Fibers

This repository is accompanying the paper "Programmable Optical Data
Transmission Through Multimode Fibres Enabling Confidentiality by Physical
Layer Security" (S. Rothe, K.-L. Besser, D. Krause, R. Kuschmierz, N.
Koukourakis, E. Jorswieck, J. Czarske. 2021. [arXiv:XXX]()).

The idea is to make all calculations, simulations, and presented results
publicly available to the reader and, therefore, reproducible.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/klb2/mmf-physec/HEAD)


## File List
The following files are provided in this repository:

- [BPSK Secrecy
  Rate.ipynb](https://mybinder.org/v2/gh/klb2/mmf-physec/HEAD?labpath=BPSK%20Secrecy%20Rate.ipynb):
  Jupyter notebook that contains an interactive version of the simulations.
- `run.sh`: Bash script that reproduces the figures presented in the paper.
- `statistic.sh`: Bash script that reproduces the results which are averaged
  over multiple measurements.
- `svd_precoding.py`: Python module that calculates the secrecy rate with SVD
  precoding.
- `no_precoding.py`: Python module that calculates the secrecy rate when no
  precoding at the transmitter is performed.
- `data_transmission.py`: Python module that shows the measured data
  transmission of the TU Dresden logo and performs the MC simulation with an
  additional polar wiretap code.
- `waterfilling.py`: Python module that solves the optimization problem of
  optimal power allocation for parallel BSCs.
- `show_capacity_statistics.py`: Python module that averages the results of
  multiple MMF measurements and plots them. (Automatically called at the end of
  `statistic.sh`.)
- `read_matrices.py`: Python module that contains functions for reading the
  channel matrices from the measurement files.
- `util.py`: Some utility functions like setting up the logging etc.

### Measurements
In the `measurements/` directory, one can find the measured MMF channel
matrices that where used to obtain the results.
All measurements were performed on a 55-mode MMF of 10m length, where an
eavesdropper is physically coupled (50:50 coupling).
A detailed description can be found in the paper.

- `mmf-measurement.mat`: Channel matrix measurements of Alice to Bob and Alice
  to Eve, both with and without precoding at the transmitter
- `data-measurement.mat`: Measurements of the transmission of the TU Dresden
  logo
- `statistic/`: Folder containing multiple measurements of the MMF for
  averaging purposes


## Usage
### Running it online
The easiest way is to use services like [Binder](https://mybinder.org/) to run
the notebook online. Simply navigate to
[https://mybinder.org/v2/gh/klb2/mmf-physec/HEAD](https://mybinder.org/v2/gh/klb2/mmf-physec/HEAD)
to run the notebooks in your browser without setting everything up locally.

### Local Installation
If you want to run it locally on your machine, Python3 and Jupyter are needed.
The present code was developed and tested with the following versions:
- Python 3.10
- Jupyter 1.0
- notebook 6.4
- numpy 1.22
- scipy 1.8
- Pandas 1.4.1
- h5py 3.6
- digcommpy 0.8

Make sure you have [Python3](https://www.python.org/downloads/) installed on
your computer.
You can then install the required packages (including Jupyter) by running
```bash
pip3 install -r requirements.txt
```
This will install all the needed packages which are listed in the requirements 
file. 

Finally, you can run the Jupyter notebook with
```bash
jupyter notebook 'BPSK Secrecy Rate.ipynb'
```

You can also recreate the figures from the paper by running
```bash
bash run.sh
```


## Acknowledgements
This research was supported in part by the Deutsche Forschungsgemeinschaft
(DFG) under grants JO 801/25-1 and CZ 55/42-1.


## License and Referencing
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

```bibtex
@article{Rothe2022mmfphysec,
  author = {Rothe, Stefan and Besser, Karl-Ludwig and Krause, David and Kuschmierz, Robert and Koukourakis, Nektarios and Jorswieck, Eduard and Czarske, J\"urgen W.},
  title = {Programmable Optical Data Transmission Through Multimode Fibres Enabling Confidentiality by Physical Layer Security},
  year = {2022},
  month = {3},
  journal = {Submitted for publication},
  archiveprefix = {arXiv},
  eprint = {},
  primaryClass = {physics.optics},
}
```
