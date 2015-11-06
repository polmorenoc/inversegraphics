__author__ = 'pol'

from skimage.feature import hog
from skimage import data, color, exposure
import numpy as np
import ipdb

# conf.cellSize = cellSize;
# conf.numOrientations = 9;
def computeHoG(image):

    image = color.rgb2gray(image)

    hog_descr, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualise=True)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    #
    # ax1.axis('off')
    # ax1.imshow(image, cmap=plt.cm.gray)
    # ax1.set_title('Input image')

    # # Rescale histogram for better display
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    #
    # ax2.axis('off')
    # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    # plt.show()

    return hog_descr

def computeHoGFeatures(images):
    hogs = []

    for image in images:
        features = computeHoG(image)
        hogs = hogs + [features[None,:] ]

    hogfeats = np.vstack(hogs)
    return hogfeats

def computeIllumFeatures(images, numFreq):
    illum = []
    win = 40
    for image in images:
        features = featuresIlluminationDirection(image, win)[:numFreq,:].ravel()
        illum = illum + [features[None,:]]

    illumfeats = np.vstack(illum)
    return illumfeats


def featuresIlluminationDirection(image,win):
    image = color.rgb2gray(image)
    coeffs = np.fft.fft2(image[image.shape[0]/2-win:image.shape[0]/2+win,image.shape[1]/2-win:image.shape[1]/2+win])
    magnitudes =  np.sqrt(coeffs.real**2 + coeffs.imag**2)

    phases = np.angle(coeffs)

    return np.hstack([magnitudes.ravel()[:,None], phases.ravel()[:,None]])

from chumpy import depends_on, Ch
import chumpy as ch
from math import radians
import cv2
import scipy
import matplotlib.pyplot as plt
class convolve2D(Ch):
    terms = 'filter'
    dterms = 'x'

    def compute_r(self):
        convolved = scipy.signal.convolve2d(self.x, self.filter)
        return convolved[np.int((convolved.shape[0]-self.x.shape[0])/2):np.int((convolved.shape[0]-self.x.shape[0])/2) + self.x.shape[1], np.int((convolved.shape[1]-self.x.shape[1])/2):np.int((convolved.shape[1]-self.x.shape[1])/2) + self.x.shape[1]]

    @depends_on(terms + dterms)
    def dr_wrt_convolution(self):
        widthRolls = self.x.shape[1]
        heightRolls = self.x.shape[0]
        tmpShape = [self.x.shape[0]+self.filter.shape[0], self.x.shape[1]+self.filter.shape[1]]
        template = np.zeros(tmpShape)
        template[0:self.filter.shape[0], 0:self.filter.shape[1]] = self.filter
        jacs = []
        for i in range(heightRolls):
            for j in range(widthRolls):
                templateRolled = np.roll(template, shift=i, axis=0)
                templateRolled = np.roll(templateRolled, shift=j, axis=1)
                templateGrad = templateRolled[tmpShape[0] - self.x.shape[0] - np.int(self.filter.shape[0]/2): tmpShape[0] - np.int(self.filter.shape[0]/2), tmpShape[1] - self.x.shape[1] - np.int(self.filter.shape[1]/2): tmpShape[1] - np.int(self.filter.shape[1]/2)]
                jacs = jacs + [scipy.sparse.coo_matrix(templateGrad.ravel())]
        return scipy.sparse.vstack(jacs).tocsc()

    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            return self.dr_wrt_convolution
        else:
            return None

def diffHog(image, numOrient = 8, cwidth=15, cheight=15):
    image = 0.3*image[:,:,0] +  0.59*image[:,:,1] + 0.11*image[:,:,2]
    gy = image[:-2,1:-1] - image[2:,1:-1] + 0.0001
    gx = image[1:-1,:-2] - image[1:-1, 2:] + 0.0001

    distFilter = np.ones([cheight,cwidth], dtype=np.uint8)
    distFilter[np.int(cheight/2), np.int(cwidth/2)] = 0
    distFilter = (cv2.distanceTransform(distFilter, cv2.DIST_L2, 3)- np.max(cv2.distanceTransform(distFilter, cv2.DIST_L2, 3)))/(-np.max(cv2.distanceTransform(distFilter, cv2.DIST_L2, 3)))

    magn = ch.sqrt(gy**2 + gx**2)

    angles = ch.arctan(gy/gx)

    meanOrient = np.linspace(0, np.pi, numOrient)

    fb = 1./(1. + ch.exp(1 - ch.abs(ch.expand_dims(angles,2) - meanOrient.reshape([1,1,numOrient]))*numOrient/radians(180)))

    Fb = ch.expand_dims(magn,2)*fb

    Fs_list = [convolve2D(x=Fb[:,:,Fbi], filter=distFilter).reshape([Fb.shape[0], Fb.shape[1],1]) for Fbi in range(numOrient)]
    Fs = ch.concatenate(Fs_list, axis=2)
    epsilon = 0.00001

    v = Fs/(ch.sum(Fs**2, axis=2).reshape([Fb.shape[0], Fb.shape[1],1]) + epsilon)

    ipdb.set_trace()

