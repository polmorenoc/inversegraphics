__author__ = 'pol'

from skimage.feature import hog
from skimage import data, color, exposure
import numpy as np
import ipdb

# conf.cellSize = cellSize;
# conf.numOrientations = 9;

def computeHoGFeatures(images):
    hogs = []

    for image in images:
        hogs = hogs + computeHoG(image)

    hogfeats = np.vstack(hogs.reshape[1,:])
    return hogfeats

def computeIllumFeatures(images):
    illum = []
    win = 40
    for image in images:
        illum = illum + featuresIlluminationDirection(image, win)

    illumfeats = np.vstack(illum.reshape[1,:])
    return illumfeats


def featuresIlluminationDirection(image,win):
    # ipdb.set_trace()
    image = color.rgb2gray(image)
    coeffs = np.fft.fft2(image[image.shape[0]/2-win:image.shape[0]/2+win,image.shape[1]/2-win:image.shape[1]/2+win])
    magnitudes =  np.sqrt(coeffs.real**2 + coeffs.imag**2)

    phases = np.angle(coeffs)

    return np.concatenate([magnitudes.ravel(), phases.ravel()])

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