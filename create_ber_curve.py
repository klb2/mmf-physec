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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import save_results, RESULTS_DIR

def main(ber_files, plot=False, export=False):
    data = []
    for _ber_file in ber_files:
        _df = pd.read_csv(_ber_file, sep="\t")
        data.append(_df)
    data = pd.concat(data, ignore_index=True)
    results = {"bob": [],
               "eve": [],
              }
    for k, _data_k in data.groupby("secrate"):
        for _name in results:
            _data_user = _data_k[_name]
            _mean = np.mean(_data_user)
            _std = np.std(_data_user, ddof=1)
            if not np.isfinite(_std):
                _std = 0
            _min = _mean-_std
            _max = _mean+_std
            results[_name].append((k, _mean, _min, _max))
    results = {k: np.array(v) for k, v in results.items()}

    if export:
        for _name, _results in results.items():
            outfile = f"ber-statistic-{_name}.dat"
            _results = pd.DataFrame(_results)
            _results.columns = ["secrate", "mean", "min", "max"]
            save_results(RESULTS_DIR, _results, snr=None, filename=outfile)

    if plot:
        fig, axs = plt.subplots()
        for _name, _results in results.items():
            _rate, _mean, _min, _max = _results.T
            axs.fill_between(_rate, _min, _max, alpha=.5)
            axs.plot(_rate, _mean, label=_name)
        axs.legend()
        axs.set_xlabel("Secrecy Rate")
        axs.set_ylabel("BER")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ber_files", nargs="+")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--export", action="store_true")
    args = vars(parser.parse_args())
    main(**args)
    plt.show()
