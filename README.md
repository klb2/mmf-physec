# Physical Layer Security on Multi-Mode Fibers

This repository is accompanying the paper "Programmable Optical Data
Transmission Through Multimode Fibres Enabling Confidentiality by Physical
Layer Security" (S. Rothe, K.-L. Besser, N. Koukourakis, E. Jorswieck, J.
Czarske. 2021. [arXiv:XXX]()).

The idea is to make all calculations, simulations, and presented results
publicly available to the reader and, therefore, reproducible.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/klb2/mmf-physec/HEAD)


## File List
The following files are provided in this repository:

- [TODO.ipynb](): Jupyter notebook that contains an interactive version of the
  simulations.
- `run.sh`: Bash script that reproduces the figures presented in the paper.
- TODO

### Measurements
In the `measurements/` directory, one can find the measured MMF channel
matrices that where used to obtain the results.

TODO: Description of measurement files


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

Finally, you can run the Jupyter notebooks with
```bash
jupyter notebook
```

You can also recreate the figures from the paper by running
```bash
bash run.sh
```


## Acknowledgements
This research was supported in part by the Deutsche Forschungsgemeinschaft
(DFG) under grant JO 801/23-1.


## License and Referencing
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

The implementation of the waterfilling algorithm is based on the Matlab
implementation by Kenneth Shum (2010,
[https://www.mathworks.com/matlabcentral/fileexchange/28022-waterfilling-algorithm](https://www.mathworks.com/matlabcentral/fileexchange/28022-waterfilling-algorithm)).
