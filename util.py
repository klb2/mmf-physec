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
