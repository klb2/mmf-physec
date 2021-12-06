"""Calculations and simulations of secrecy rate for the BPSK example

This module contains the calculations and simulations of the secrecy rate of a
BPSK tranmission over a MMF channel with SVD precoding.
Both perfect SVD precoding (upper bound) and real measurements of optical SVD
precoding are supported.


Copyright (C) 2021 Karl-Ludwig Besser

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
__copyright__ = "Copyright (C) 2021 Karl-Ludwig Besser"
__credits__ = ["Karl-Ludwig Besser"]
__license__ = "GPLv3"
__version__ = "1.0"


import logging
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import io

from read_matrices import read_measurement_file, EVE, BOB
from util import setup_logging_config, generate_data, save_results, RESULTS_DIR, RESULTS_IMG
from waterfilling import water_filling

# Tse 2005 page 294

#logging.basicConfig(format="%(asctime)s - [%(levelname)8s]: %(message)s")
np.random.seed(20210716)

def capac_bsc(bit_flip_prob):
    return 1.-stats.bernoulli.entropy(p=bit_flip_prob)/np.log(2)

def calc_power_waterfilling(sing_values, power):
    alpha_i = 1/sing_values**2
    power_vector = water_filling(alpha_i, power)
    return power_vector

def calc_bit_flip_prob(noise_trans_matrix, power_vec):
    noise_var = np.real(noise_trans_matrix @ np.conj(noise_trans_matrix).T)
    q_val = stats.norm.sf(power_vec/(np.sqrt(np.diag(noise_var))))
    return q_val


def monte_carlo_simulation(messages, power, eff_mat_bob, reception_matrix,
                           eff_mat_eve, inv_eff_mat_eve=None):
    if inv_eff_mat_eve is None:
        inv_eff_mat_eve = np.linalg.inv(eff_mat_eve)

    num_samples, num_streams = np.shape(messages)
    n = len(eff_mat_bob)

    tx_symbols = np.zeros((num_samples, n))
    tx_symbols[:, :num_streams] = power * messages

    # Monte Carlo Simulations
    noise_bob = np.random.randn(num_samples, n) + 1j*np.random.randn(num_samples, n)
    noise_eve = np.random.randn(num_samples, n) + 1j*np.random.randn(num_samples, n)
    rec_bob = tx_symbols @ eff_mat_bob + noise_bob
    rec_eve = tx_symbols @ eff_mat_eve + noise_eve
    est_bob = rec_bob @ reception_matrix
    est_eve = rec_eve @ inv_eff_mat_eve
    est_mess_bob = np.real(np.sign(est_bob[:, :num_streams]))
    est_mess_eve = np.real(np.sign(est_eve[:, :num_streams]))
    return est_mess_bob, est_mess_eve


def main(snr, n=3, k=1, matrix=None, precoded=False, num_samples=100000,
         loglevel=logging.INFO):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _prefix = "SVD"
    if matrix is not None:
        _matrices = read_measurement_file(matrix, precoded=False)
        _basename = os.path.splitext(os.path.basename(matrix))[0]
        if precoded:
            dirname = "{}-{}-precoded".format(_prefix, _basename)
        else:
            dirname = "{}-{}".format(_prefix, _basename)
        mat_bob = _matrices[BOB]
        mat_eve = _matrices[EVE]
        n = len(mat_bob)
    else:
        np.random.seed(100)
        mat_bob = np.random.randn(n, n) + 1j*np.random.randn(n, n)
        mat_eve = np.random.randn(n, n) + 1j*np.random.randn(n, n)
        dirname = "{0}-{1}x{1}".format(_prefix, n)

    dirname = os.path.join(RESULTS_DIR, dirname)
    os.makedirs(dirname, exist_ok=True)
    setup_logging_config(dirname)
    logger = logging.getLogger('main')
    logger.setLevel(loglevel)

    U, S, Vh = np.linalg.svd(mat_bob)
    _normalization_factor = 1./np.max(S)

    mat_bob = _normalization_factor * mat_bob
    U, S, Vh = np.linalg.svd(mat_bob)

    S = np.diag(S)
    reception_matrix = np.conj(Vh).T
    if precoded:
        _mat_prec = read_measurement_file(matrix, precoded=True)
        eff_mat_bob = _mat_prec[BOB]
        eff_mat_eve = _mat_prec[EVE]
        plt.matshow(np.abs(eff_mat_bob))
        eff_mat_bob = eff_mat_bob @ Vh #np.conj(Vh).T  # saved matrix is U^H @ Bob @ V
    else:
        precoding_matrix = np.conj(U).T
        eff_mat_bob = precoding_matrix @ mat_bob
        eff_mat_eve = precoding_matrix @ mat_eve
        plt.matshow(np.abs(precoding_matrix @ mat_bob @ reception_matrix))
    inv_eff_mat_eve = np.linalg.inv(eff_mat_eve)

    plt.colorbar()
    plt.show()

    power = 10**(snr/10.)
    logger.info("SNR: %f dB", snr)
    logger.debug("Power constraint: %f", power)

    results = {"k": [], "secCapacMC": [], "secCapacTheo": []}
    for _k in range(1, k+1):
        logger.info("Number of modes: %d", _k)
        sing_val = np.diag(S)[:_k]
        logger.debug("Singular values: %s", sing_val)
        _power_vec_wf = calc_power_waterfilling(sing_val, power)
        power_vec_wf = np.zeros(n)#np.zeros_like(S)
        power_vec_wf[:_k] = _power_vec_wf
        #logger.debug(power_vec_wf)
        logger.debug("Sum power allocated vs available power: {:.3f}/{:.3f}".format(sum(power_vec_wf), power))
    
        messages = generate_data(_k, num_samples, mod="bpsk")
        est_mess_bob, est_mess_eve = monte_carlo_simulation(messages,
                _power_vec_wf, eff_mat_bob, reception_matrix, eff_mat_eve,
                inv_eff_mat_eve)

        bit_flip_bob = np.count_nonzero(est_mess_bob != messages, axis=0)/num_samples
        bit_flip_eve = np.count_nonzero(est_mess_eve != messages, axis=0)/num_samples
        capac_bob_mc = capac_bsc(bit_flip_bob)
        capac_eve_mc = capac_bsc(bit_flip_eve)
        _sec_capac_mc = np.sum(np.maximum(capac_bob_mc-capac_eve_mc, 0))
        logger.debug("MC Bit Flip Bob: %s", bit_flip_bob)
        logger.debug("MC Bit Flip Eve: %s", bit_flip_eve)
        logger.info("Secrecy Capacity (MC): %f", _sec_capac_mc)
        results["secCapacMC"].append(_sec_capac_mc)

        # Theoretical
        bit_flip_prob_bob = calc_bit_flip_prob(np.linalg.inv(eff_mat_bob @ reception_matrix), power_vec_wf)
        bit_flip_prob_eve = calc_bit_flip_prob(inv_eff_mat_eve, power_vec_wf)

        capac_bob = capac_bsc(bit_flip_prob_bob)
        capac_eve = capac_bsc(bit_flip_prob_eve)
        _sec_capac = np.maximum(capac_bob - capac_eve, 0)
        sec_capac = sum(_sec_capac)
        logger.debug("Theo Bit Flip Bob: %s", bit_flip_prob_bob)
        logger.debug("Theo Bit Flip Eve: %s", bit_flip_prob_eve)
        logger.debug("Secrecy capacity (TH): %f", sec_capac)
        results["k"].append(_k)
        results['secCapacTheo'].append(sec_capac)

    save_results(dirname, results, snr)

    plt.figure()
    if not precoded:
        plt.plot(results['k'], results['secCapacTheo'], 'o-', label="Theoretical Values")
    plt.plot(results['k'], results['secCapacMC'], '^-', label="Monte Carlo Simulation")
    plt.xlabel("Number of used modes $|\mathcal{K}|$")
    plt.ylabel("Secrecy Capacity [bit]")
    plt.legend()
    plt.tight_layout()
    _plot_fn = RESULTS_IMG.format(snr)
    plt.savefig(os.path.join(dirname, _plot_fn), dpi=200)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, help="Number of modes", default=4)
    parser.add_argument("-k", type=int, help="Max number of used modes", default=2)
    parser.add_argument("-s", "--snr", type=float, help="SNR", default=10)
    parser.add_argument("--matrix", help="Mat-file with matrices")
    parser.add_argument("--num_samples", type=int, default=100000, help="Number of Monte Carlo samples")
    parser.add_argument("--precoded", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = vars(parser.parse_args())
    plot = args.pop("plot")
    main(**args)
    if plot:
        plt.show()
