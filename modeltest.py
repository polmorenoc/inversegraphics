__author__ = 'pol'
import matplotlib.pyplot as plt
import numpy
from PIL import Image
from score_image import *

im1 = numpy.zeros((2,2,3))
im1[:,:,0] = 1
im2 = numpy.zeros((2,2,3))
im2[:,:,1] = 1
im2[:,:,0] = 1
im3 = numpy.ones((2,2,3))
im3[:,:,0] = 0

im4 = numpy.zeros((2,2,3))
im1[:,:,0] = 1
im5 = numpy.ones((2,2,3))
im5[:,:,0] = 0

test = numpy.zeros([2,2,3])
test[:,:,0] = 1.0

mask1 = numpy.ones((2, 2), dtype=bool)
mask1[:,1] = False
mask2 = numpy.ones((2, 2), dtype=bool)
mask2[1,:] = False
mask3 = numpy.ones((2, 2), dtype=bool)
mask3[1,1] = False
masks = [mask1, mask2, mask3]

mask4 = numpy.ones((2, 2), dtype=bool)
mask4[:,1] = False

mask5 = numpy.ones((2, 2), dtype=bool)
mask5[:,1] = False

sqDist1 = sqDistImages(im1, test)
sqDist2 = sqDistImages(im2, test)
sqDist3 = sqDistImages(im3, test)
sqDistsSeq = [sqDist1, sqDist2, sqDist3]

sqDist4 = sqDistImages(im4, test)

sqRes = numpy.concatenate([aux[..., numpy.newaxis] for aux in sqDistsSeq], axis=-1)

masks = numpy.concatenate([aux[..., numpy.newaxis] for aux in masks], axis=-1)

vars = computeVariances(sqRes)

layerPriors = layerPriors(masks)

logLikelihood4 = modelLogLikelihood(test, im4, layerPriors, vars)
logLikelihood5 = modelLogLikelihood(test, im5, layerPriors, vars)

fgpost4, bgpost4 = layerPosteriors(test, im4, layerPriors, vars)
fgpost5, bgpost5 = layerPosteriors(test, im5, layerPriors, vars)

plt.imshow(fgpost4)
plt.colorbar()

plt.imshow(bgpost4)
plt.colorbar()

print("lala")