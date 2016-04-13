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

    probBackground = 0.7

    U = np.zeros((3, fgMask.size), dtype='float32')

    U[0, :] = -np.log(0.01)

    U[1, :] = -np.log((1-probBackground))

    U[2, :] = -np.log(probBackground)

    U[0, fgMask] = fg_energy
    U[1, fgMask] = occ_energy
    U[2, fgMask] = -np.log(probs[2])

    return U

def superpixelUnary(img, segments, U, fgMask, iouProb, figsDir):

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

def entropyUnary(img, U, fgMask, figsDir):

    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    entr_img1 = entropy(img[:,:,0], disk(5))
    entr_img2 = entropy(img[:,:,1], disk(5))
    entr_img3 = entropy(img[:,:,2], disk(5))

    avgEntropy = (entr_img1 + entr_img2 + entr_img3)/3

    plt.imsave('avgEntropy', avgEntropy)

    # ipdb.set_trace()

    return U

def boundaryUnary(img, U, fgMask, fgBoundary, fgBoundProb, figsDir):

    #img must be uint8 0-255
    # borders, textons, orientations = damascene(np.uint8(img[:,:,0:3]*255), device_num=0)
    from skimage import feature

    borders= feature.canny(skimage.color.rgb2gray(img), sigma=2)

    gtBorders = borders > 0.5

    # plt.imsave(figsDir + 'cannyedges', gtBorders)

    maskDistTransform = cv2.distanceTransform(np.uint8(~fgBoundary.reshape([150,150])), cv2.DIST_L2, 3)

    closeEdges = maskDistTransform.ravel() < 5

    imCloseEdges = np.zeros([150,150])
    imCloseEdges.ravel()[closeEdges * gtBorders.ravel() * fgMask] = 1
    # plt.imsave(figsDir + 'closeEdges', imCloseEdges)

    U[0, closeEdges * gtBorders.ravel() * fgMask] = -np.log(fgBoundProb)
    U[1, closeEdges * gtBorders.ravel() * fgMask] = -np.log(1 - fgBoundProb)

    # U[0, closeEdges * gtBorders.ravel() * fgMask] = -np.log(fgBoundProb)
    # U[1, closeEdges * gtBorders.ravel() * fgMask] = -np.log(1 - fgBoundProb)

    z = np.argmin(U.swapaxes(0,1).reshape([150,150,3]), axis=2)
    Urgb = np.concatenate([np.array(z==0)[:,:,None], np.array(z==1)[:,:,None], np.array(z==2)[:,:,None]], axis=2)


    # plt.figure()
    # plt.imshow(Urgb)
    # plt.savefig(figsDir + 'boundaryEdges.png')
    # plt.close()

    farEdgesRegion = maskDistTransform.ravel() > 5
    farEdges = farEdgesRegion* gtBorders.ravel() * fgMask
    imfarEdges = np.zeros([150,150])
    imfarEdges.ravel()[farEdges] = 1

    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    entr_img1 = entropy(img[:,:,0], disk(5))
    entr_img2 = entropy(img[:,:,1], disk(5))
    entr_img3 = entropy(img[:,:,2], disk(5))

    farEdgeDistTransform = cv2.distanceTransform(np.uint8(~farEdges.reshape([150,150])), cv2.DIST_L2, 3) < 5

    avgEntropy = (entr_img1 + entr_img2 + entr_img3)/3

    entropyTh = avgEntropy > 3

    occlusionPixels = farEdgeDistTransform.ravel() * entropyTh.ravel()

    U[0, occlusionPixels.ravel()] = -np.log(1-fgBoundProb)
    U[1, occlusionPixels.ravel()] = -np.log(fgBoundProb)


    # plt.imsave('farEdges', imfarEdges)

    U[0, farEdgesRegion * gtBorders.ravel() * fgMask] = -np.log(1-fgBoundProb)
    U[1, farEdgesRegion * gtBorders.ravel() * fgMask] = -np.log(fgBoundProb)

    return U

def crfInference(imageGT, fgMask, fgBoundary,  probs, figsDir):
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

    figsub, (vax1, vax2, vax3, vax4, vax5, vax6, vax7) = plt.subplots(1, 7, subplot_kw={'aspect':'equal'})

    z = np.argmin(U.swapaxes(0,1).reshape([150,150,3]), axis=2)
    Urgb = np.concatenate([np.array(z==0)[:,:,None], np.array(z==1)[:,:,None], np.array(z==2)[:,:,None]], axis=2)

    vax1.imshow(imageGT)
    vax1.axis('off')
    vax2.imshow(fgMask)
    vax2.axis('off')

    segments = skimage.segmentation.quickshift(imageGT, ratio=1, max_dist=10, convert2lab=True)
    U = superpixelUnary(imageGT, segments, U, fgMask.ravel(), 0.8, figsDir)
    z = np.argmin(U.swapaxes(0,1).reshape([150,150,3]), axis=2)
    Urgb = np.concatenate([np.array(z==0)[:,:,None], np.array(z==1)[:,:,None], np.array(z==2)[:,:,None]], axis=2)

    vax3.imshow(mark_boundaries(imageGT, segments))
    vax3.axis('off')

    vax4.imshow(Urgb)
    vax4.axis('off')

    U = entropyUnary(imageGT, U, fgMask, figsDir)

    U = boundaryUnary(imageGT, U, fgMask.ravel(), fgBoundary.ravel(), 0.9, figsDir)
    z = np.argmin(U.swapaxes(0,1).reshape([150,150,3]), axis=2)
    Urgb = np.concatenate([np.array(z==0)[:,:,None], np.array(z==1)[:,:,None], np.array(z==2)[:,:,None]], axis=2)

    vax5.axis('off')
    vax5.imshow(Urgb)

    smoothedU = -np.log(skimage.filters.gaussian(np.exp(-U.swapaxes(0,1).reshape([150,150,3])), 5, output=None, mode='nearest', cval=0, multichannel=True))
    vax6.axis('off')
    vax6.imshow(smoothedU)


    # crfmodel.setUnaryEnergy(np.ascontiguousarray(smoothedU.reshape([-1,3]).swapaxes(0,1).astype(np.float32)))
    crfmodel.setUnaryEnergy(U)

    # # This creates the color-independent features and then add them to the CRF
    feats = create_pairwise_gaussian(sdims=(5,5), shape=img.shape[:2])
    crfmodel.addPairwiseEnergy(feats, compat=5,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(30, 30), schan=(5., 5., 5.),
                                      img=img, chdim=2)

    crfmodel.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # # This creates the color-dependent features and then add them to the CRF
    # feats = create_pairwise_bilateral(sdims=(5, 5), schan=(5., 5., 5.),
    #                                   img=img, chdim=2)
    #
    # crfmodel.addPairwiseEnergy(feats, compat=10,
    #                     kernel=dcrf.DIAG_KERNEL,
    #                     normalization=dcrf.NORMALIZE_SYMMETRIC)

    # # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(10, 10), schan=(100., 250., 250.),
                                      img=img, chdim=2)
    crfmodel.addPairwiseEnergy(feats, compat=np.array([[80,0,0], [0,0,0],[0,0,0]], dtype=np.float32),
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    # #
    # #
    # # # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(20, 20), schan=(5., 5., 5.),
                                      img=img, chdim=2)
    # #
    crfmodel.addPairwiseEnergy(feats, compat=np.array([[-100,0,0], [0,0,0],[0,0,0]], dtype=np.float32),
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    #
    # feats = create_pairwise_bilateral(sdims=(20, 20), schan=(20., 15., 15.),
    #                                   img=img, chdim=2)
    #
    # crfmodel.addPairwiseEnergy(feats, compat=np.array([[0,0,0], [0,-50,0],[0,0,0]], dtype=np.float32),
    #                     kernel=dcrf.DIAG_KERNEL,
    #                     normalization=dcrf.NORMALIZE_SYMMETRIC)

    feats = create_pairwise_bilateral(sdims=(15, 15), schan=(50., 60., 60.),
                                      img=img, chdim=2)

    crfmodel.addPairwiseEnergy(feats, compat=np.array([[0,0,0], [0, -20,0],[0,0,0]], dtype=np.float32),
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    feats = create_pairwise_bilateral(sdims=(5, 5), schan=(50., 60., 60.),
                                      img=img, chdim=2)

    crfmodel.addPairwiseEnergy(feats, compat=np.array([[0,0,0], [0, 0,0],[0,0,-5]], dtype=np.float32),
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    # feats = create_pairwise_bilateral(sdims=(20, 20), schan=(20., 15., 15.),
    #                                   img=img, chdim=2)
    #
    # crfmodel.addPairwiseEnergy(feats, compat=np.array([[0,0,0], [0,0,0],[0,0,-20]], dtype=np.float32),
    #                     kernel=dcrf.DIAG_KERNEL,
    #                     normalization=dcrf.NORMALIZE_SYMMETRIC)

    # # This creates the color-dependent features and then add them to the CRF
    # feats = create_pairwise_bilateral(sdims=(100, 100), schan=(85., 200., 200.),
    #                                   img=img, chdim=2)
    #
    # crfmodel.addPairwiseEnergy(feats, compat=np.array([[0,0,0], [0,10,0],[0,0,0]], dtype=np.float32),
    #                     kernel=dcrf.DIAG_KERNEL,
    #                     normalization=dcrf.NORMALIZE_SYMMETRIC)
    #
    # # This creates the color-dependent features and then add them to the CRF
    # feats = create_pairwise_bilateral(sdims=(100, 100), schan=(30., 20., 20.),
    #                                   img=img, chdim=2)
    #
    # crfmodel.addPairwiseEnergy(feats, compat=np.array([[0,0,0], [0,-25,0],[0,0,0]], dtype=np.float32),
    #                     kernel=dcrf.DIAG_KERNEL,
    #                     normalization=dcrf.NORMALIZE_SYMMETRIC)
    #
    # This creates the color-dependent features and then add them to the CRF
    # feats = create_pairwise_bilateral(sdims=(5, 5), schan=(15., 10., 10.),
    #                                   img=img, chdim=2)
    #
    # crfmodel.addPairwiseEnergy(feats, compat=np.array([[-25,50,0], [50,0,0],[0,0,0]], dtype=np.float32),
    #                     kernel=dcrf.DIAG_KERNEL,
    #                     normalization=dcrf.NORMALIZE_SYMMETRIC)

    # # This creates the color-dependent features and then add them to the CRF
    # feats = create_pairwise_bilateral(sdims=(5, 5), schan=(5., 2., 2.),
    #                                   img=img, chdim=2)
    #
    # crfmodel.addPairwiseEnergy(feats, compat=np.array([[-1,0,0], [0,0,0],[0,0,0]], dtype=np.float32),
    #                     kernel=dcrf.DIAG_KERNEL,
    #                     normalization=dcrf.NORMALIZE_SYMMETRIC)

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


    vax7.axis('off')
    vax7.imshow(Q.swapaxes(0,1).reshape([150,150,3]))

    figsub.savefig(figsDir, bbox_inches='tight')

    plt.close(figsub)

    return mapseg, np.array(Q)