"""Show transmitted and received images.

This module shows the results of the image transmission measurements.
It shows both the transmitted image and the received images at Bob and Eve.


Copyright (C) 2022 Karl-Ludwig Besser

This program is used in the article:
S. Rothe, K.-L. Besser, N. Koukourakis, E. Jorswieck, and J. Czarske,
"Programmable Optical Data Transmission Through Multimode Fibres Enabling
Confidentiality by Physical Layer Security", 2021.


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
__author__ = "Karl-Ludwig Besser"
__copyright__ = "Copyright (C) 2022 Karl-Ludwig Besser"
__credits__ = ["Karl-Ludwig Besser"]
__license__ = "GPLv3"
__version__ = "1.0"

import logging
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy import io
from digcommpy import encoders, decoders, channels

from read_matrices import read_measurement_file, EVE, BOB, read_mat_file
from util import RESULTS_DIR, save_results
from svd_precoding import monte_carlo_simulation, calc_power_waterfilling

logging.basicConfig(format="%(asctime)s - [%(levelname)8s]: %(message)s")

CMAP = "hot"
CMAP = plt.get_cmap(CMAP)


def read_data_file(data_file):
    mat_dict = read_mat_file(data_file)
    return mat_dict

def process_single_measurement(data_file, codewords, pos_lookup,
                               image_file=None, export=False, plot=False,
                               loglevel=logging.INFO):
    logger = logging.getLogger('main')
    logger.setLevel(loglevel)

    mat_data = read_data_file(data_file)

    pos_lookup = np.loadtxt(pos_lookup)
    code_length = len(pos_lookup)
    info_length = np.count_nonzero(pos_lookup == -1)
    random_length = np.count_nonzero(pos_lookup == -2)
    logger.info(f"Code parameters: n={code_length:d}, k={info_length:d}, r={random_length:d}")

    logo_coded = np.loadtxt(codewords)
    logo_expected = np.loadtxt(image_file, delimiter=",")

    wtc_decoder = decoders.PolarWiretapDecoder(code_length, "BAWGN",
                                               pos_lookup=pos_lookup)

    rec_bob_post = mat_data["y_Bob_after_decoding"]
    rec_data_bob = rec_bob_post[1:3, :] 
    rec_data_bob = np.ravel(rec_data_bob, order="F")
    assert len(rec_data_bob) == len(logo_coded)

    offset = 0.5
    image_bob = _decode_symbols(rec_data_bob-offset,
                                code_length, wtc_decoder,
                                shape=np.shape(logo_expected))
    ber_bob = np.count_nonzero(image_bob != logo_expected)/np.prod(np.shape(logo_expected))
    logger.info(f"BER Bob: {ber_bob:E}")

    _mat_prec = read_measurement_file(data_file, precoded=True)
    eff_mat_eve = _mat_prec[EVE]
    inv_eff_mat_eve = np.linalg.inv(eff_mat_eve)
    rec_eve = mat_data["y_Eve"]
    rec_eve = rec_eve.T @ inv_eff_mat_eve
    rec_eve = rec_eve.T
    #rec_eve = inv_eff_mat_eve @ rec_eve
    rec_eve = rec_eve[1:3, :]
    est_eve = np.real(rec_eve)
    est_eve = est_eve/np.max(est_eve)
    image_eve = _decode_symbols(est_eve, code_length, wtc_decoder,
                                shape=np.shape(logo_expected))
    ber_eve = np.count_nonzero(image_eve != logo_expected)/np.prod(np.shape(logo_expected))
    logger.info(f"BER Eve: {ber_eve:E}")
    
    if plot:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(image_bob)
        axs[0].set_title("Bob")
        axs[1].imshow(image_eve)
        axs[1].set_title("Eve")
        fig.suptitle(f"K={info_length:d}, r={random_length:d}")
    if export:
        plt.imsave(os.path.join(os.path.dirname(data_file),
                                f"decoded_60-bob-n{code_length:d}-k{info_length:d}-r{random_length:d}.pdf"),
                   image_bob,
                   cmap=CMAP)
        plt.imsave(os.path.join(os.path.dirname(data_file),
                                f"decoded_60-eve-n{code_length:d}-k{info_length:d}-r{random_length:d}.pdf"),
                   image_eve,
                   cmap=CMAP)
    return ber_bob, ber_eve

def _decode_symbols(received_data, code_length, decoder, shape):
    rec_cw = np.reshape(received_data, (-1, code_length))
    dec_cw = decoder.decode_messages(rec_cw)
    dec_cw = np.ravel(dec_cw)
    len_logo = int(np.prod(shape))
    dec_logo_bits = dec_cw[:len_logo]
    dec_image = np.reshape(dec_logo_bits, shape)
    return dec_image


def main(measurement_path, logo_path, info_length, random_length: int = 2,
         code_length: int = 8192,
         output: str = "ber-tud-logo_60.dat",
         plot: bool = False, export: bool = False):
    IMAGE_FILE = "tud_logo_60.txt"
    IMAGE_FILE = os.path.join(logo_path, IMAGE_FILE)

    if not info_length:
        _files = os.listdir(measurement_path)
        re_meas = r'n(\d+)-k(\d+)-r(\d+).mat'
        matches = [re.findall(re_meas, _f) for _f in _files]
        info_length = [int(l[0][1]) for l in matches if l]

    info_length = sorted(info_length)
    results = []
    for k in info_length:
        _meas_file = f"n{code_length:d}-k{k:d}-r{random_length:d}.mat"
        _meas_file = os.path.join(measurement_path, _meas_file)
        _data_file = f"logo-coded_60-n{code_length:d}-k{k:d}-r{random_length:d}.txt"
        _data_file = os.path.join(logo_path, _data_file)
        _pos_lookup_file = f"logo-coded_60-n{code_length:d}-k{k:d}-r{random_length:d}.enc"
        _pos_lookup_file = os.path.join(logo_path, _pos_lookup_file)
        ber_bob, ber_eve = process_single_measurement(_meas_file, _data_file,
                                                      _pos_lookup_file,
                                                      image_file=IMAGE_FILE,
                                                      plot=plot,
                                                      export=export)
        results.append((k, ber_bob, ber_eve))
    results = np.array(results)
    if plot:
        fig, axs = plt.subplots()
        axs.plot(results[:, 0]/code_length, results[:, 1], 'o-', label="Bob")
        axs.plot(results[:, 0]/code_length, results[:, 2], 'o-', label="Eve")
        axs.set_xlabel("Secrecy Rate")
        axs.set_ylabel("BER")
        axs.legend()
    if export:
        results_dict = {
                        "k": info_length,
                        "secrate": results[:, 0]/code_length,
                        "bob": results[:, 1],
                        "eve": results[:, 2],
                       }
        save_results(RESULTS_DIR, results_dict, snr=None,
                     filename=output)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("measurement_path")
    parser.add_argument("logo_path")
    parser.add_argument("-k", "--info_length", nargs="+", type=int)
    parser.add_argument("-r", "--random_length", default=2, type=int)
    parser.add_argument("-o", "--output", default="ber-tud-logo_60.dat")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = vars(parser.parse_args())
    main(**args)
    plt.show()
