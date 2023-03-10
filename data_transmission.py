"""Show transmitted and received images.

This module shows the results of the image transmission measurements.
It shows both the transmitted image and the received images at Bob and Eve.


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
__author__ = "Karl-Ludwig Besser"
__copyright__ = "Copyright (C) 2021 Karl-Ludwig Besser"
__credits__ = ["Karl-Ludwig Besser"]
__license__ = "GPLv3"
__version__ = "1.0"

import logging
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from scipy import io
from digcommpy import encoders, decoders, channels

from read_matrices import read_measurement_file, EVE, BOB, read_mat_file
from util import RESULTS_DIR
from svd_precoding import monte_carlo_simulation, calc_power_waterfilling

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

def main(data_file, image_file, export=False, loglevel=logging.INFO):
    logger = logging.getLogger('main')
    logger.setLevel(loglevel)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    _matrices = read_measurement_file(data_file, precoded=False)
    mat_bob = _matrices[BOB]
    mat_eve = _matrices[EVE]
    n = len(mat_bob)

    U, S, Vh = np.linalg.svd(mat_bob)
    _normalization_factor = 1./np.max(S)
    mat_bob = _normalization_factor * mat_bob
    U, S, Vh = np.linalg.svd(mat_bob)

    S = np.diag(S)
    reception_matrix = np.conj(Vh).T
    _mat_prec = read_measurement_file(data_file, precoded=True)
    eff_mat_bob = _mat_prec[BOB]
    eff_mat_eve = _mat_prec[EVE]
    eff_mat_bob = eff_mat_bob @ Vh #np.conj(Vh).T  # saved matrix is U^H @ Bob @ V
    inv_eff_mat_eve = np.linalg.inv(eff_mat_eve)

    mat_data = read_data_file(data_file)
    image = np.loadtxt(image_file, delimiter=',')
    image_shape = np.shape(image)
    image_vec = np.reshape(image, (-1, 1), order="F")
    mat_data["transmitted_image"] = image
    mat_data["data"] = image_vec
    rec_image_bob, rec_image_eve = uncoded_image(mat_data, eff_mat_eve,
                                                 inv_eff_mat_eve,
                                                 size=image_shape,
                                                 export=export)

    _rec_image_bob = rec_image_bob-np.mean(rec_image_bob)
    _rec_image_bob_binary = np.where(_rec_image_bob<0, 0, 1)
    _rec_image_eve = rec_image_eve-np.mean(rec_image_eve)
    _rec_image_eve = _rec_image_eve/np.std(_rec_image_eve)
    _rec_image_eve_binary = np.where(_rec_image_eve<0.2, 1, 0)
    ber_bob = np.count_nonzero(_rec_image_bob_binary!=image)/np.size(image)
    ber_eve = np.count_nonzero(_rec_image_eve_binary!=image)/np.size(image)
    logger.info(f"BER (Bob): {ber_bob:.4f}")
    logger.info(f"BER (Eve): {ber_eve:.4f}")
    fig, axs = plt.subplots(1, 2)
    axs[0].matshow(_rec_image_bob_binary, cmap=CMAP)
    axs[1].matshow(_rec_image_eve_binary, cmap=CMAP)

    if export:
        plt.imsave(os.path.join(RESULTS_DIR, "logo_60-uncoded-binary-bob.pdf"),
                   _rec_image_bob_binary, cmap=CMAP)
        plt.imsave(os.path.join(RESULTS_DIR, "logo_60-uncoded-binary-eve.pdf"),
                   _rec_image_eve_binary, cmap=CMAP)

def uncoded_image(mat_data, eff_mat_eve, inv_eff_mat_eve, export, size=(30, 30)):
    rec_bob_post = mat_data["y_Bob_after_decoding"]
    _rec_data_bob = rec_bob_post[1:3, :] 
    _rec_data_bob = np.ravel(_rec_data_bob, order="F")
    _rec_image_bob = np.reshape(_rec_data_bob, size, order="F")

    rec_eve = mat_data["y_Eve"]
    rec_eve = rec_eve.T @ inv_eff_mat_eve
    rec_eve = rec_eve.T
    _rec_data_eve = unpack_data(rec_eve)
    _rec_image_eve = np.reshape(_rec_data_eve, size, order="F")

    expected = expected_received_data(eff_mat_eve, inv_eff_mat_eve,
                                      mat_data["data"])
    expected_image = np.reshape(expected, size, order="F")

    _rec_image_bob = _rec_image_bob-np.min(_rec_image_bob)
    _rec_image_bob = _rec_image_bob/np.max(_rec_image_bob)
    _rec_image_eve = _rec_image_eve-np.min(_rec_image_eve)
    _rec_image_eve = _rec_image_eve/np.max(_rec_image_eve)
    _rec_image_eve = 1-_rec_image_eve

    for _name, _image in (("bob", _rec_image_bob), ("eve", _rec_image_eve)):
        fig, axs = plt.subplots()
        im = axs.matshow(_image, cmap=CMAP, vmin=0, vmax=1)
        axs.axis('off')
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=20)
        _name = f"image_{_name}-uncoded_60-colorbar.png"
        if export:
            plt.savefig(os.path.join(RESULTS_DIR, _name), bbox_inches="tight",
                        transparent=True)



    if export:
        plt.imsave(os.path.join(RESULTS_DIR, "image_original_60.pdf"),
                   mat_data["transmitted_image"], cmap=CMAP)
        plt.imsave(os.path.join(RESULTS_DIR, "image_bob-uncoded_60.pdf"),
                   _rec_image_bob,
                   cmap=CMAP)
        plt.imsave(os.path.join(RESULTS_DIR, "image_eve-uncoded_60.pdf"),
                   _rec_image_eve,
                   cmap=CMAP)
    return _rec_image_bob, _rec_image_eve
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="Mat-file with transmitted data")
    parser.add_argument("image_file")
    parser.add_argument("--export", action="store_true")
    args = vars(parser.parse_args())
    main(**args)
    plt.show()
