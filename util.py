"""
Copyright (C) 2023 Karl-Ludwig Besser

This program is used in the article:
"Securing Data in Multimode Fibers by Exploiting Mode-Dependent Light
Propagation Effects" (S. Rothe, K.-L. Besser, D. Krause, R. Kuschmierz, N.
Koukourakis, E. Jorswieck, J. Czarske. Research, vol. 6: 0065, Jan. 2023.
DOI:10.34133/research.0065).


License:
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.
See the GNU General Public License for more details.

Author: Karl-Ludwig Besser, Technische Universität Braunschweig
"""


import logging
from logging.config import dictConfig
import os

import numpy as np
import pandas as pd


RESULTS_DIR = "results"
RESULTS_DATA = "capac_results-snr{}.dat"
RESULTS_IMG = "results-snr{}.png"


def setup_logging_config(dirname, level=logging.DEBUG):
    logging_config = dict(
        version = 1,
        formatters = {
                      'f': {'format': "%(asctime)s - [%(levelname)8s]: %(message)s"}
                     },
        handlers = {'console': {'class': 'logging.StreamHandler',
                                'formatter': 'f',
                                'level': level,
                               },
                    'file': {'class': 'logging.FileHandler',
                             'formatter': 'f',
                             'level': logging.DEBUG,
                             'filename': os.path.join(dirname, "main.log")
                            },
                   },
        loggers = {"main": {'handlers': ['console', 'file'],
                            'level': logging.DEBUG,
                           },
                   "algorithm": {'handlers': ['console', 'file'],
                                 'level': logging.DEBUG,
                                },
                  }
        )
    dictConfig(logging_config)

def generate_data(num_streams, num_samples, mod="bpsk"):
    mod = mod.lower()
    if mod == "bpsk":
        data = np.random.choice([-1, 1], size=(num_samples, num_streams))
    else:
        raise NotImplementedError
    return data

def save_results(dirname, results, snr, filename=None):
    if filename is None:
        dat_file = os.path.join(dirname, RESULTS_DATA.format(snr))
    else:
        dat_file = os.path.join(dirname, filename)
    pd.DataFrame.from_dict(results).to_csv(dat_file, sep='\t', index=False)
