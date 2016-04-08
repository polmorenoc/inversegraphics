"""
Usage: python util_inference_example.py image annotations
Adapted from the dense_inference.py to demonstate the usage of the util
functions.
"""

# from damascene import damascene
import pydensecrf.densecrf as dcrf

import sys
import numpy as np

import matplotlib.pylab as plt
from skimage.segmentation import relabel_sequential
import skimage
from skimage.segmentation import slic

import cv2
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

    probBackground = 0.9

    U = np.zeros((3, fgMask.size), dtype='float32')

    U[0, :] = -np.log(0.01)

    U[1, :] = -np.log((1-probBackground))

    U[2, :] = -np.log(probBackground)

    U[0, fgMask] = fg_energy
    U[1, fgMask] = occ_energy
    U[2, fgMask] = -np.log(0.01)

    return U

def superpixelUnary(img, U, fgMask, iouProb):

    segments = skimage.segmentation.quickshift(img, ratio=1, max_dist=10, convert2lab=True)
    plt.imsave('superpixelsegments',mark_boundaries(img, segments))

    # plt.imshow(segments)

    fgMaskReshaped = fgMask.reshape([img.shape[0], img.shape[1]])

    for seg_i in np.arange(np.max(segments)):
        currentSegment = segments == seg_i
        masksCat = np.concatenate([fgMaskReshaped[:,:,None], currentSegment[:,:,None]], axis=2)
        segmentationIOU = np.sum(np.all(masksCat, axis=2)) / np.sum(currentSegment)

        if segmentationIOU > 0.1 and segmentationIOU < 0.75:
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

def boundaryUnary(img, U, fgMask, fgBoundary, fgBoundProb):

    #img must be uint8 0-255
    # borders, textons, orientations = damascene(np.uint8(img[:,:,0:3]*255), device_num=0)
    from skimage import feature

    borders= feature.canny(skimage.color.rgb2gray(img), sigma=3)

    gtBorders = borders > 0.5

    plt.imsave('cannyedges', gtBorders)

    maskDistTransform = cv2.distanceTransform(np.uint8(~fgBoundary.reshape([150,150])), cv2.DIST_L2, 3)

    closeEdges = maskDistTransform.ravel() < 5

    imCloseEdges = np.zeros([150,150])
    imCloseEdges.ravel()[closeEdges * gtBorders.ravel() * fgMask] = 1
    plt.imsave('closeEdges', imCloseEdges)

    U[0, closeEdges * gtBorders.ravel() * fgMask] = -np.log(fgBoundProb)
    U[1, closeEdges * gtBorders.ravel() * fgMask] = -np.log(1 - fgBoundProb)

    z = np.argmin(U.swapaxes(0,1).reshape([150,150,3]), axis=2)
    Urgb = np.concatenate([np.array(z==0)[:,:,None], np.array(z==1)[:,:,None], np.array(z==2)[:,:,None]], axis=2)
    plt.imshow(Urgb)

    plt.savefig('boundaryEdges.png')

    farEdges = maskDistTransform.ravel() > 10
    imfarEdges = np.zeros([150,150])
    imfarEdges.ravel()[farEdges* gtBorders.ravel() * fgMask] = 1
    plt.imsave('farEdges', imfarEdges)

    U[0, farEdges * gtBorders.ravel() * fgMask] = -np.log(1-fgBoundProb)
    U[1, farEdges * gtBorders.ravel() * fgMask] = -np.log(fgBoundProb)

    return U

def crfInference(imageGT, fgMask, fgBoundary,  probs):
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

    U = unaryOcclusionModel(img, fgMask.ravel(), probs)

    plt.figure()
    z = np.argmin(U.swapaxes(0,1).reshape([150,150,3]), axis=2)
    Urgb = np.concatenate([np.array(z==0)[:,:,None], np.array(z==1)[:,:,None], np.array(z==2)[:,:,None]], axis=2)
    plt.imshow(Urgb)
    plt.savefig('unary.png')

    U = superpixelUnary(imageGT, U, fgMask.ravel(), 0.65)
    z = np.argmin(U.swapaxes(0,1).reshape([150,150,3]), axis=2)
    Urgb = np.concatenate([np.array(z==0)[:,:,None], np.array(z==1)[:,:,None], np.array(z==2)[:,:,None]], axis=2)
    plt.imshow(Urgb)
    plt.savefig('superpixel.png')

    U = boundaryUnary(imageGT, U, fgMask.ravel(), fgBoundary.ravel(), 0.9)
    z = np.argmin(U.swapaxes(0,1).reshape([150,150,3]), axis=2)
    Urgb = np.concatenate([np.array(z==0)[:,:,None], np.array(z==1)[:,:,None], np.array(z==2)[:,:,None]], axis=2)
    plt.imshow(Urgb)
    plt.savefig('boundarycomplete.png')

    crfmodel.setUnaryEnergy(U)

    # # This creates the color-independent features and then add them to the CRF
    feats = create_pairwise_gaussian(sdims=(10, 10), shape=img.shape[:2])
    crfmodel.addPairwiseEnergy(feats, compat=np.array([[0,3,3], [3,0,3],[3,3,0]], dtype=np.float32),
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(30, 30), schan=(30., 20., 20.),
                                      img=img, chdim=2)

    crfmodel.addPairwiseEnergy(feats, compat=np.array([[0.,10,10], [10,0,10],[10,10,0]], dtype=np.float32),
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(-75, -75), schan=(-60., -150., -150.),
                                      img=img, chdim=2)

    crfmodel.addPairwiseEnergy(feats, compat=np.array([[20,0,0], [0,0,0],[0,0,0]], dtype=np.float32),
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # # This creates the color-dependent features and then add them to the CRF
    # feats = create_pairwise_bilateral(sdims=(-100, -100), schan=(-75., -100., -100.),
    #                                   img=img, chdim=2)
    #
    # crfmodel.addPairwiseEnergy(feats, compat=np.array([[0,0,0], [0,1,0],[0,0,0]], dtype=np.float32),
    #                     kernel=dcrf.DIAG_KERNEL,
    #                     normalization=dcrf.NORMALIZE_SYMMETRIC)

    # # This creates the color-dependent features and then add them to the CRF
    # feats = create_pairwise_bilateral(sdims=(30, 30), schan=(20., 10., 10.),
    #                                   img=img, chdim=2)
    #
    # crfmodel.addPairwiseEnergy(feats, compat=np.array([[0.,0,0], [0,0,0],[0,0,0]], dtype=np.float32),
    #                     kernel=dcrf.DIAG_KERNEL,
    #                     normalization=dcrf.NORMALIZE_SYMMETRIC)

    ####################################
    ### Do inference and compute map ###
    ####################################

    Q = np.array(crfmodel.inference(5))
    mapseg = np.argmax(Q, axis=0).reshape(img.shape[:2])

    # res = map.astype('float32') * 255 / map.max()
    # plt.imshow(res)
    # plt.show()

    # # Manually inference
    # Q, tmp1, tmp2 = crfmodel.startInference()
    # for i in range(5):
    #     print("KL-divergence at {}: {}".format(i, crfmodel.klDivergence(Q)))
    #     crfmodel.stepInference(Q, tmp1, tmp2)

    plt.figure()
    plt.imshow(mapseg)
    plt.clim(0,2)
    plt.savefig('mapseg.png')

    plt.figure()

    plt.imshow(Q.swapaxes(0,1).reshape([150,150,3]))
    plt.savefig('Q.png')

    return mapseg, np.array(Q)