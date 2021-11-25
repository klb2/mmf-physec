"""
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


import numpy as np
import scipy.io as sio
import h5py

BOB = "bob"
EVE = "eve"
USERS = [BOB, EVE]


def get_mat_key(user, precoded=True):
    PATTERN = "matrix_{}{}"
    if precoded:
        _pre = "_precoding"
    else:
        _pre = ""
    #return PATTERN.format(user.capitalize(), _pre)
    return PATTERN.format(user, _pre)

def read_mat_file(mat_file):
    try:
        mat_dict = sio.loadmat(mat_file)
        mat_dict = {k: v for k, v in mat_dict.items() if not k.startswith("__")}
    except NotImplementedError:
        mat_dict = {}
        with h5py.File(mat_file, 'r') as h5f:
            for k, v in h5f.items():
                _struct_mat = np.array(v)
                mat_dict[k] = np.matrix(v['real'] + 1j*v['imag'])
    return mat_dict

def read_measurement_file(mat_file, precoded=True):
    mat_dict = read_mat_file(mat_file)
    mat_orig, mat_prec = parse_mat_dict(mat_dict)
    if precoded:
        return mat_prec
    else:
        return mat_orig

def parse_mat_dict(mat_dict):
    mat_orig = {k: mat_dict[get_mat_key(k, precoded=False)] for k in USERS}
    try:
        mat_prec = {k: mat_dict[get_mat_key(k, precoded=True)] for k in USERS}
    except KeyError:
        mat_prec = None
    return mat_orig, mat_prec

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument("mat_file")
    args = vars(parser.parse_args())
    mat_file = args['mat_file']
    mat_dict = read_mat_file(mat_file)
    print(mat_dict.keys())
    mat_orig, mat_prec = parse_mat_dict(mat_dict)
    U, S, Vh = np.linalg.svd(mat_orig[BOB])
    #print(S)
    #print(np.abs(np.diag(mat_prec[BOB])))
    plt.matshow(np.abs(mat_prec[BOB]))
    plt.matshow(np.abs(np.diag(S)-mat_prec[BOB]))
    plt.show()
