import logging
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import io

from read_matrices import read_measurement_file, EVE, BOB
from util import setup_logging_config, generate_data, save_results, RESULTS_DIR, RESULTS_IMG
from waterfilling import water_filling_bsc

# Tse 2005 page 294

#logging.basicConfig(format="%(asctime)s - [%(levelname)8s]: %(message)s")
np.random.seed(20210716)

def capac_bsc(bit_flip_prob):
    return 1.-stats.bernoulli.entropy(p=bit_flip_prob)/np.log(2)

def calc_bit_flip_prob(noise_trans_matrix, power_vec):
    noise_var = np.real(noise_trans_matrix @ np.conj(noise_trans_matrix).T)
    q_val = stats.norm.sf(power_vec/(np.sqrt(np.diag(noise_var))))
    return q_val

def main(snr, n=3, k=1, matrix=None, num_samples=100000, loglevel=logging.INFO):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _prefix = "NOP"
    if matrix is not None:
        _matrices = read_measurement_file(matrix, precoded=False)
        _basename = os.path.splitext(os.path.basename(matrix))[0]
        dirname = "{}-{}".format(_prefix, _basename)
        mat_bob = _matrices[BOB]
        mat_eve = _matrices[EVE]
        n = len(mat_bob)
    else:
        mat_bob = np.random.randn(n, n) + 1j*np.random.randn(n, n)
        mat_eve = np.random.randn(n, n) + 1j*np.random.randn(n, n)
        dirname = "{0}-{1}x{1}".format(_prefix, n)

    dirname = os.path.join(RESULTS_DIR, dirname)
    os.makedirs(dirname, exist_ok=True)
    setup_logging_config(dirname)
    logger = logging.getLogger('main')
    logger.setLevel(loglevel)

    _S_eve = np.linalg.svd(mat_eve)[1]
    mat_eve = mat_eve/np.max(_S_eve)
    U, S, Vh = np.linalg.svd(mat_bob)
    _normalization_factor = 1./np.max(S)
    mat_bob = _normalization_factor * mat_bob

    inv_mat_bob = np.linalg.inv(mat_bob)
    inv_mat_eve = np.linalg.inv(mat_eve)

    power = 10**(snr/10.)
    logger.info("SNR: %f dB", snr)
    logger.debug("Power constraint: %f", power)

    results = {"k": [], "secCapac": []}
    for _k in range(1, k+1):
        logger.info("Number of modes: %d", _k)
        _power_vec_wf = power/_k*np.ones(_k)
        _main_diag = np.abs(np.diag(mat_bob)[:_k])
        #_power_vec_wf = _main_diag*power/sum(_main_diag)
        power_vec_wf = np.zeros(n)#np.zeros_like(S)
        power_vec_wf[:_k] = _power_vec_wf
        logger.debug(power_vec_wf)
        logger.debug("Sum power allocated vs available power: {:.3f}/{:.3f}".format(sum(power_vec_wf), power))

        bit_flip_prob_bob = calc_bit_flip_prob(inv_mat_bob, power_vec_wf)
        bit_flip_prob_eve = calc_bit_flip_prob(inv_mat_eve, power_vec_wf)

        capac_bob = capac_bsc(bit_flip_prob_bob)
        capac_eve = capac_bsc(bit_flip_prob_eve)
        _sec_capac = np.maximum(capac_bob - capac_eve, 0)
        sec_capac = sum(_sec_capac)
        logger.debug("Theo Bit Flip Bob: %s", bit_flip_prob_bob)
        logger.debug("Theo Bit Flip Eve: %s", bit_flip_prob_eve)
        logger.info("Secrecy capacity: %f", sec_capac)
        results["k"].append(_k)
        results['secCapac'].append(sec_capac)

    save_results(dirname, results, snr)

    plt.figure()
    plt.plot(results['k'], results['secCapac'], 'o-')
    plt.xlabel("Number of used modes $|\mathcal{K}|$")
    plt.ylabel("Secrecy Capacity [bit]")
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
    parser.add_argument("--plot", action="store_true")
    args = vars(parser.parse_args())
    plot = args.pop("plot")
    main(**args)
    if plot:
        plt.show()
