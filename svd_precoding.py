import logging
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import io

from read_matrices import read_measurement_file, EVE, BOB
from util import setup_logging_config, generate_data, save_results, RESULTS_DIR
from waterfilling import water_filling

# Tse 2005 page 294

#logging.basicConfig(format="%(asctime)s - [%(levelname)8s]: %(message)s")
np.random.seed(20210716)


def capac_bsc(bit_flip_prob):
    return 1.-stats.bernoulli.entropy(p=bit_flip_prob)/np.log(2)

def calc_power_waterfilling(sing_values, power):
    #inv_sing = 1./sing_values**2
    #mu = (power + sum(inv_sing))/len(inv_sing)
    #print(mu)
    #power_vector = np.maximum(mu-inv_sing, 0)
    alpha_i = 1/sing_values**2
    #_status, _value, power_vector = water_filling(len(alpha_i), alpha_i, sum_x=power)
    power_vector = water_filling(alpha_i, power)
    return power_vector


#def calc_bit_flip_prob(sing_values, power_vec):
def calc_bit_flip_prob(noise_trans_matrix, power_vec):
    noise_var = np.real(noise_trans_matrix @ np.conj(noise_trans_matrix).T)
    q_val = stats.norm.sf(power_vec/(np.sqrt(np.diag(noise_var))))
    return q_val


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
        #matrices = {BOB: np.array([[.77, -.3], [-.32, -.64]]),
        #            EVE: np.array([[.54, -.11], [-.93, -1.71]])}
        #mat_bob = stats.expon.rvs(size=(n, n), scale=.1)
        mat_bob = np.random.randn(n, n) + 1j*np.random.randn(n, n)
        #mat_bob = np.eye(n)
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
        #reception_matrix = np.identity(n)
        _mat_prec = read_measurement_file(matrix, precoded=True)
        eff_mat_bob = _mat_prec[BOB]
        eff_mat_eve = _mat_prec[EVE]
        plt.matshow(np.abs(eff_mat_bob))
        eff_mat_bob = eff_mat_bob @ Vh #np.conj(Vh).T  # saved matrix is U^H @ Bob @ V
    else:
        #_svd_noise = stats.uniform.rvs(loc=0,
        #                            #scale=3/(0.5*np.arange(n)+1),
        #                            scale=np.linspace(1., .01, n)*2,
        #                            size=np.shape(mat_bob))
        #_svd_noise[np.diag_indices_from(_svd_noise)] = 0.
        #_mat_bob = np.copy(mat_bob)
        #mat_bob = mat_bob + U @ _svd_noise @ Vh
        #io.savemat(os.path.join(dirname, "rolle01.mat"), {"clean": _mat_bob, "noisy": mat_bob})
        precoding_matrix = np.conj(U).T
        #precoding_matrix = _normalization_factor * precoding_matrix
        #reception_matrix = np.conj(Vh).T
        #reception_matrix = _normalization_factor * reception_matrix
        eff_mat_bob = precoding_matrix @ mat_bob
        eff_mat_eve = precoding_matrix @ mat_eve
        plt.matshow(np.abs(precoding_matrix @ mat_bob @ reception_matrix))
    #inv_eff_mat_bob = np.linalg.inv(eff_mat_bob)
    #reception_matrix = inv_eff_mat_bob
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

        # Monte Carlo Simulations
        messages = generate_data(_k, num_samples, mod="bpsk")
        tx_symbols = np.zeros((num_samples, n))
        tx_symbols[:, :_k] = _power_vec_wf * messages
        #tx_symbols[:, :_k] = np.expand_dims(_power_vec_wf, 1) * messages
        noise_bob = np.random.randn(num_samples, n) + 1j*np.random.randn(num_samples, n)
        noise_eve = np.random.randn(num_samples, n) + 1j*np.random.randn(num_samples, n)
        rec_bob = tx_symbols @ eff_mat_bob + noise_bob
        rec_eve = tx_symbols @ eff_mat_eve + noise_eve
        est_bob = rec_bob @ reception_matrix
        est_eve = rec_eve @ inv_eff_mat_eve
        est_mess_bob = np.real(np.sign(est_bob[:, :_k]))
        est_mess_eve = np.real(np.sign(est_eve[:, :_k]))

        #bit_flip_bob = np.count_nonzero(est_mess_bob != messages, axis=1)/num_samples
        bit_flip_bob = np.count_nonzero(est_mess_bob != messages, axis=0)/num_samples
        bit_flip_eve = np.count_nonzero(est_mess_eve != messages, axis=0)/num_samples
        capac_bob_mc = capac_bsc(bit_flip_bob)
        capac_eve_mc = capac_bsc(bit_flip_eve)
        _sec_capac_mc = np.sum(np.maximum(capac_bob_mc-capac_eve_mc, 0))
        logger.debug("MC Bit Flip Bob: %s", bit_flip_bob)
        logger.debug("MC Bit Flip Eve: %s", bit_flip_eve)
        #logger.debug("BSC Capac Bob: %s", capac_bob_mc)
        #logger.debug("BSC Capac Eve: %s", capac_eve_mc)
        #logger.debug("Differences: %s", np.maximum(capac_bob_mc-capac_eve_mc, 0))
        logger.info("Secrecy Capacity: %f", _sec_capac_mc)
        results["secCapacMC"].append(_sec_capac_mc)

        # Theoretical
        #bit_flip_prob_bob = calc_bit_flip_prob(np.diag(1./S), power_vec_wf)
        bit_flip_prob_bob = calc_bit_flip_prob(np.linalg.inv(eff_mat_bob @ reception_matrix), power_vec_wf)
        bit_flip_prob_eve = calc_bit_flip_prob(inv_eff_mat_eve, power_vec_wf)
        #logger.info(bit_flip_prob_bob)
        #logger.info(bit_flip_prob_eve)

        capac_bob = capac_bsc(bit_flip_prob_bob)
        capac_eve = capac_bsc(bit_flip_prob_eve)
        _sec_capac = np.maximum(capac_bob - capac_eve, 0)
        sec_capac = sum(_sec_capac)
        logger.debug("Theo Bit Flip Bob: %s", bit_flip_prob_bob)
        logger.debug("Theo Bit Flip Eve: %s", bit_flip_prob_eve)
        logger.info("Secrecy capacity: %f", sec_capac)
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
    _plot_fn = "results-snr{}.png".format(snr)
    plt.savefig(os.path.join(dirname, _plot_fn), dpi=200)
    #plt.matshow(np.abs(np.conj(Vh).T))
    #logger.info("Maximum absolute value in precoding matrix: %f", np.max(np.abs(np.conj(Vh).T)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, help="Number of modes", default=4)
    parser.add_argument("-k", type=int, help="Max number of used modes", default=2)
    parser.add_argument("-s", "--snr", type=float, help="SNR", default=10)
    parser.add_argument("--matrix", help="Mat-file with matrices")
    parser.add_argument("--num_samples", type=int, default=100000, help="Number of Monte Carlo samples")
    parser.add_argument("--precoded", action="store_true")
    args = vars(parser.parse_args())
    main(**args)
    plt.show()
