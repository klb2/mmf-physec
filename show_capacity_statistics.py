import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import RESULTS_DIR, RESULTS_DATA


def main(snr, precoded=False):
    results = []
    for i in range(1, 4):
        dirname = "SVD-measurement{}"
        if precoded:
            dirname = dirname + "-precoded"
        data_file = os.path.join(RESULTS_DIR, dirname.format(i), RESULTS_DATA.format(snr))
        data = pd.read_csv(data_file, sep='\t', index_col=False)
        data = np.array(data["secCapacMC"])
        results.append(data)
    results = np.dstack(results)
    print(results.shape)
    average = np.ravel(np.mean(results, axis=-1))
    std = np.ravel(np.std(results, axis=-1, ddof=1))
    print(average)
    print(std)
    #plt.errorbar(range(len(average)), average, yerr=std)
    k = range(len(std))
    plt.figure()
    plt.plot(k, average, 'o-')
    plt.fill_between(k, average-std, average+std, color='gray', alpha=0.2)
    return (average, std)


if __name__ == "__main__":
    snr = 10.
    avg_main, std_main = main(snr)
    avg_prec, std_prec = main(snr, precoded=True)
    results_file = os.path.join(RESULTS_DIR, "capac-statistic-{}.dat".format(snr))
    pd.DataFrame.from_dict({"k": np.arange(1, len(avg_main)+1),
                            "avg": avg_main, "std": std_main,
                            "avgPrec": avg_prec, "stdPrec": std_prec}).to_csv(
                                    results_file, sep="\t", index=False)
    plt.show()
