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
import skimage
from skimage.segmentation import slic
import skimage.segmentation
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import ipdb

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian

def unaryOcclusionModel(img, fgMask, probs):

    """
    Simple classifier that is 50% certain that the annotation is correct
    (same as in the inference example).
    """
    fg_energy = -np.log(probs[0])
    occ_energy = -np.log(probs[1])

    probBackground = 0.6

    U = np.zeros((3, fgMask.size), dtype='float32')

    U[0, :] = -np.log(0.0001)

    U[1, :] = -np.log((1-probBackground))

    U[2, :] = -np.log(probBackground)

    U[0, fgMask] = fg_energy
    U[1, fgMask] = occ_energy
    U[2, fgMask] = -np.log(0.0001)

    return U

def superpixelUnary(img, U, fgMask, iouProb):

    segments = skimage.segmentation.quickshift(img, ratio=1, max_dist=10, convert2lab=True)

    # plt.imshow(segments)

    fgMaskReshaped = fgMask.reshape([img.shape[0], img.shape[1]])

    for seg_i in np.arange(np.max(segments)):
        currentSegment = segments == seg_i
        masksCat = np.concatenate([fgMaskReshaped[:,:,None], currentSegment[:,:,None]], axis=2)
        segmentationIOU = np.sum(np.all(masksCat, axis=2)) / np.sum(currentSegment)

        if segmentationIOU > 0.05 and segmentationIOU < 0.95:
            U[0, currentSegment.ravel()*fgMask] = -np.log(1 - iouProb)
            U[1, currentSegment.ravel()*fgMask] = -np.log(iouProb)

            U[2, currentSegment.ravel()*(~fgMask)] = -np.log(1 - iouProb)
            U[1, currentSegment.ravel()*(~fgMask)] = -np.log(iouProb)

        # # Edge pixels inside the mask (and far from boundary) are occluder pixels. How to choose the right one? (interior pixels)

        # if segmentationIOU >= 0.95 :
        #
        #     U[0, currentSegment.ravel()*fgMask] = -np.log(iouProb)
        #     U[1, currentSegment.ravel()*fgMask] = -np.log(1-iouProb)


        # # Segments with edges that match with boundary of the mask are fg.
        # if segmentationIOU >= 0.99:
        #     U[0, currentSegment*fgMask] = 1 - iouProb
        #     U[1, currentSegment*fgMask] = iouProb

        #     U[2, currentSegment*(~fgMask)] = 1 - iouProb
        #     U[1, currentSegment*(~fgMask)] = iouProb

    return U

def boundaryUnary(img, U, fgMask, boundProb):

    from damascene import damascene

    #img must be uint8 0-255
    borders, textons, orientations = damascene(np.uint8(img[:,:,0:3]*255), device_num=0)

    


    return U

def crfInference(imageGT, fgMask, probs):
    ##################################
    ### Read images and annotation ###
    ##################################
    # img = np.uint8(img*255)
    img = skimage.color.rgb2lab(imageGT)

    M = 3 # forground, background, occluding object.

    ###########################
    ### Setup the CRF model ###
    ###########################

    # Example using the DenseCRF class and the util functions
    crfmodel = dcrf.DenseCRF(img.shape[0] * img.shape[1], M)

    # get unary potentials (neg log probability)
    # U = compute_unary(labels, M)
    U = unaryOcclusionModel(img, fgMask, probs)

    U = superpixelUnary(imageGT, U, fgMask, 0.8)

    crfmodel.setUnaryEnergy(U)

    # This creates the color-independent features and then add them to the CRF
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    crfmodel.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(30, 30), schan=(13, 13, 13),
                                      img=img, chdim=2)
    crfmodel.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    ####################################
    ### Do inference and compute map ###
    ####################################
    Q = crfmodel.inference(5)
    mapseg = np.argmax(Q, axis=0).reshape(img.shape[:2])

    # res = map.astype('float32') * 255 / map.max()
    # plt.imshow(res)
    # plt.show()

    # # Manually inference
    # Q, tmp1, tmp2 = crfmodel.startInference()
    # for i in range(5):
    #     print("KL-divergence at {}: {}".format(i, crfmodel.klDivergence(Q)))
    #     crfmodel.stepInference(Q, tmp1, tmp2)

    return mapseg, np.array(Q)