import logging
import os

import numpy as np
import matplotlib.pyplot as plt
from digcommpy import encoders, decoders, channels

from read_matrices import BOB, EVE


logging.basicConfig(format="%(asctime)s - [%(levelname)8s]: %(message)s")


def main(data_file, output, code_length: int,
         bf_bob: float, bf_eve: float,
         loglevel=logging.INFO):
    logger = logging.getLogger('main')
    logger.setLevel(loglevel)

    image = np.loadtxt(data_file, delimiter=',')
    shape_img = np.shape(image)
    vec_image = np.reshape(image, (1, -1))

    bit_flip = {BOB: bf_bob, EVE: bf_eve}
    _channels = {k: channels.BscChannel(v) for k, v in bit_flip.items()}
    _encoder = encoders.PolarWiretapEncoder(code_length, "BSC", "BSC", bit_flip[BOB], bit_flip[EVE])#, info_length_bob=1)
    info_length = _encoder.info_length
    random_length = _encoder.random_length
    logger.info(f"Code parameters: n={code_length:d}, k={info_length:d}, r={random_length:d}")

    pad_width = info_length - (np.shape(vec_image)[1] % info_length)
    logger.info(f"Padding width: {pad_width:d}")
    vec_image = np.pad(vec_image, [[0, 0], [0, pad_width]])
    vec_image = np.reshape(vec_image, (-1, info_length))
    enc_image = _encoder.encode_messages(vec_image)
    logger.info(f"Encoded image: {enc_image}")
    logger.info(f"Shape of data: {np.shape(enc_image)}")
    enc_image_stream = np.ravel(enc_image)
    logger.info(f"Shape of data stream: {np.shape(enc_image_stream)}")

    out_fname = f"logo-coded_{shape_img[0]:d}-n{code_length:d}-k{info_length:d}-r{random_length:d}.txt"
    np.savetxt(out_fname, enc_image_stream)
    out_fname_coder = f"logo-coded_{shape_img[0]:d}-n{code_length:d}-k{info_length:d}-r{random_length:d}.enc"
    np.savetxt(out_fname_coder, _encoder.pos_lookup)


    _decoder = decoders.PolarWiretapDecoder(code_length, "BAWGN", 
                                            pos_lookup=_encoder.pos_lookup)
    rec_codewords = np.reshape(2*enc_image_stream-1, (-1, code_length))
    dec_logo_stream = _decoder.decode_messages(rec_codewords)
    dec_logo_stream = np.ravel(dec_logo_stream)
    dec_logo_stream = dec_logo_stream[:-pad_width]
    dec_logo = np.reshape(dec_logo_stream, shape_img)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    axs[1].imshow(dec_logo)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="Mat-file with transmitted data")
    parser.add_argument("-o", '--output', help="Output file name. If not provided, a default will be used")
    parser.add_argument("-n", "--code_length", type=int, default=8192)
    parser.add_argument("-b", "--bf_bob", type=float, default=.42)
    parser.add_argument("-e", "--bf_eve", type=float, default=.49)
    #parser.add_argument("-k", "--info_length", type=int, default=149)
    #parser.add_argument("-r", "--random_length", type=int, default=2)
    args = vars(parser.parse_args())
    main(**args)
    #plt.show()
