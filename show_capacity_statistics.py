import os.path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import RESULTS_DIR, RESULTS_DATA

LOGGER = logging.getLogger(__name__)

def main(snr, option=0):# precoded=False):
    """
    Parameters
    ----------
    option : int
        Select one of the following:
        0... SVD precoding with perfect SVD
        1... Optical SVD precoding
        2... No precoding
    """
    results = []
    if option in [0, 1]:
        dirname = "SVD-measurement{}"
        if option == 1:
            dirname = dirname + "-precoded"
    elif option == 2:
        dirname = "NOP-measurement{}"
    for i in range(1, 4):
        data_file = os.path.join(RESULTS_DIR, dirname.format(i), RESULTS_DATA.format(snr))
        data = pd.read_csv(data_file, sep='\t', index_col=False)
        #data = np.array(data["secCapacMC"])
        data = np.array(data.get("secCapacMC", data.get("secCapac")))
        results.append(data)
    results = np.dstack(results)
    average = np.ravel(np.mean(results, axis=-1))
    std = np.ravel(np.std(results, axis=-1, ddof=1))
    quant_low = np.ravel(np.quantile(results, .00, axis=-1))
    quant_high = np.ravel(np.quantile(results, 1., axis=-1))
    LOGGER.info("Average secrecy rate (for the first 5 modes):\t{}".format(average[:5]))
    LOGGER.info("Minimum secrecy rate:\t{}".format(quant_low[:5]))
    LOGGER.info("Maximum secrecy rate:\t{}".format(quant_high[:5]))
    #plt.errorbar(range(len(average)), average, yerr=std)
    k = range(len(std))
    plt.figure()
    plt.plot(k, average, 'o-')
    #plt.fill_between(k, average-std, average+std, color='gray', alpha=0.2)
    plt.fill_between(k, quant_low, quant_high, color='gray', alpha=0.2)
    return (average, std, quant_low, quant_high)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - [%(levelname)8s]: %(message)s",
                        handlers=[
                            logging.FileHandler("main.log", encoding="utf-8"),
                            logging.StreamHandler()
                        ])
    LOGGER.setLevel(logging.INFO)
    snr = 10.
    LOGGER.debug(f"Using SNR: {snr} dB")

    LOGGER.info("Upper Bound (Perfect SVD at Alice)")
    avg_main, std_main, low_main, high_main = main(snr, option=0)

    LOGGER.info("Real Implemented SVD at Alice")
    avg_prec, std_prec, low_prec, high_prec = main(snr, option=1)

    LOGGER.info("No Precoding at Alice")
    avg_nop, std_nop, low_nop, high_nop = main(snr, option=2)

    LOGGER.info("Gain of precoding (real system vs no precoding: {:.1f}".format(max(avg_prec)/max(avg_nop)))
    results_file = os.path.join(RESULTS_DIR, "capac-statistic-{}.dat".format(snr))
    pd.DataFrame.from_dict({"k": np.arange(1, len(avg_main)+1),
                            "avg": avg_main, "std": std_main,
                            "low": avg_main-low_main, "high": high_main-avg_main,
                            "avgPrec": avg_prec, "stdPrec": std_prec,
                            "lowPrec": avg_prec-low_prec, "highPrec": high_prec-avg_prec,
                            "avgNop": avg_nop, "stdNop": std_nop,
                            "lowNop": avg_nop-low_nop, "highNop": high_nop-avg_nop,
                           }).to_csv(
                                    results_file, sep="\t", index=False)
    plt.show()
