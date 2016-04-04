"""
Usage: python util_inference_example.py image annotations
Adapted from the dense_inference.py to demonstate the usage of the util
functions.
"""

import sys
import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
import matplotlib.pylab as plt
from skimage.segmentation import relabel_sequential

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian

def unaryOcclusionModel(img, fgMask, probs):

    """
    Simple classifier that is 50% certain that the annotation is correct
    (same as in the inference example).
    """

    fg_energy = -np.log(probs[0])
    occ_energy = -np.log(probs[1])

    probBackground = 0.99

    U = np.zeros((3, fgMask.size), dtype='float32')
    
    U[1, :] = -np.log((1-probBackground))

    U[2, :] = -np.log(probBackground)

    U[0, fgMask] = fg_energy
    U[1, fgMask] = occ_energy
    U[2, fgMask] = -np.log(0)

    return U


def crfInference(img, fgMask, probs):
    ##################################
    ### Read images and annotation ###
    ##################################
    img = np.uint8(img*255)
    labels = relabel_sequential(cv2.imread(fn_anno, 0))[0].flatten()

    M = 3 # forground, background, occluding object.

    ###########################
    ### Setup the CRF model ###
    ###########################

    # Example using the DenseCRF class and the util functions
    d = dcrf.DenseCRF(img.shape[0] * img.shape[1], M)

    # get unary potentials (neg log probability)
    # U = compute_unary(labels, M)
    U = unaryOcclusionModel(img, fgMask, probs)

    d.setUnaryEnergy(U)

    # This creates the color-independent features and then add them to the CRF
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                      img=img, chdim=2)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)


    ####################################
    ### Do inference and compute map ###
    ####################################
    Q = d.inference(5)
    mapseg = np.argmax(Q, axis=0).reshape(img.shape[:2])

    # res = map.astype('float32') * 255 / map.max()
    # plt.imshow(res)
    # plt.show()


    # # Manually inference
    # Q, tmp1, tmp2 = d.startInference()
    # for i in range(5):
    #     print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
    #     d.stepInference(Q, tmp1, tmp2)

    return mapseg