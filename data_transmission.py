import logging
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy import io

from read_matrices import read_measurement_file, EVE, BOB, read_mat_file
from util import RESULTS_DIR


logging.basicConfig(format="%(asctime)s - [%(levelname)8s]: %(message)s")

CMAP = "hot"
CMAP = plt.get_cmap(CMAP)


def read_data_file(data_file):
    mat_dict = read_mat_file(data_file)
    return mat_dict

def expected_received_data(eff_channel, reception_matrix, data):
    n = len(reception_matrix)
    data = pack_data(data, n)
    noise = np.random.rand(n) + 1j*np.random.rand(n)
    #data = 10*data
    #noise = 10*noise
    received = data @ eff_channel + noise
    received = received @ reception_matrix
    received = unpack_data(received, n)
    return received

def pack_data(data, n=55, _packed_bits=2):
    _data = np.reshape(data, (-1, _packed_bits))
    _data = np.hstack((np.ones((len(_data), 1)), _data, np.zeros((len(_data), n-_packed_bits-1))))
    assert np.shape(_data)[1] == n
    return _data

def unpack_data(received, n=55, _packed_bits=2):
    if len(received) != n:
        received = received.T
    _received = received[1:_packed_bits+1, :] 
    _received = np.ravel(_received, order="F")
    _received = np.real(_received)
    _received = np.abs(_received)
    return _received

def main(mat_file, data_file, export=False, loglevel=logging.INFO):
    logger = logging.getLogger('main')
    logger.setLevel(loglevel)

    _matrices = read_measurement_file(mat_file, precoded=False)
    mat_bob = _matrices[BOB]
    mat_eve = _matrices[EVE]
    n = len(mat_bob)

    U, S, Vh = np.linalg.svd(mat_bob)
    _normalization_factor = 1./np.max(S)
    mat_bob = _normalization_factor * mat_bob
    U, S, Vh = np.linalg.svd(mat_bob)

    S = np.diag(S)
    reception_matrix = np.conj(Vh).T
    _mat_prec = read_measurement_file(mat_file, precoded=True)
    eff_mat_bob = _mat_prec[BOB]
    eff_mat_eve = _mat_prec[EVE]
    eff_mat_bob = eff_mat_bob @ Vh #np.conj(Vh).T  # saved matrix is U^H @ Bob @ V
    inv_eff_mat_eve = np.linalg.inv(eff_mat_eve)

    mat_data = read_data_file(data_file)
    #plt.matshow(mat_data["transmitted_image"])
    rec_bob_post = mat_data["y_Bob_after_decoding"]
    _rec_data_bob = rec_bob_post[1:3, :] 
    _rec_data_bob = np.ravel(_rec_data_bob, order="F")
    assert np.all(_rec_data_bob == np.ravel(mat_data["data_received"]))
    #_rec_data_bob = np.round(_rec_data_bob)
    #_rec_data_bob = np.where(_rec_data_bob > .4, 1, 0)
    _rec_image_bob = np.reshape(_rec_data_bob, (30, 30), order="F")

    rec_eve = mat_data["y_Eve"]
    rec_eve = rec_eve.T @ inv_eff_mat_eve
    rec_eve = rec_eve.T
    #rec_eve = inv_eff_mat_eve @ rec_eve
    #_rec_data_eve = rec_eve[1:3, :] 
    #_rec_data_eve = np.ravel(_rec_data_eve, order="F")
    #_rec_data_eve = np.real(_rec_data_eve)
    _rec_data_eve = unpack_data(rec_eve)
    _rec_image_eve = np.reshape(_rec_data_eve, (30, 30), order="F")

    #expected = expected_received_data(np.eye(55), np.eye(55), mat_data["data"])
    #expected = expected_received_data(eff_mat_bob, reception_matrix, mat_data["data"])
    expected = expected_received_data(eff_mat_eve, inv_eff_mat_eve, mat_data["data"])
    expected_image = np.reshape(expected, (30, 30), order="F")

    #fig, axs = plt.subplots(2, 2)
    fig = plt.figure() # constrained_layout=True
    gs = GridSpec(2, 2, figure=fig)
    _im_orig = fig.add_subplot(gs[0, :]).matshow(mat_data["transmitted_image"], cmap=CMAP)
    _im_bob = fig.add_subplot(gs[1, 0]).matshow(_rec_image_bob, cmap=CMAP)
    _im_eve = fig.add_subplot(gs[1, 1]).matshow(_rec_image_eve, cmap=CMAP)

    if export:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.imsave(os.path.join(RESULTS_DIR, "image_original.pdf"),
                   mat_data["transmitted_image"], cmap=CMAP)
        plt.imsave(os.path.join(RESULTS_DIR, "image_bob.pdf"), _rec_image_bob,
                   cmap=CMAP)
        plt.imsave(os.path.join(RESULTS_DIR, "image_eve.pdf"), _rec_image_eve,
                   cmap=CMAP)
    #return mat_data



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument("-k", type=int, help="Max number of used modes", default=2)
    parser.add_argument("mat_file", help="Mat-file with matrices")
    parser.add_argument("data_file", help="Mat-file with transmitted data")
    parser.add_argument("--export", action="store_true")
    args = vars(parser.parse_args())
    main(**args)
    plt.show()
