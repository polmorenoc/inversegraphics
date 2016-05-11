__author__ = 'pol'

from skimage.feature import hog
from skimage import data, color, exposure
import numpy as np
import ipdb

import skimage.color

# def xyz2labCh(xyz, illuminant="D65", observer="2"):
#     """XYZ to CIE-LAB color space conversion.
#     Parameters
#     ----------
#     xyz : array_like
#         The image in XYZ format, in a 3- or 4-D array of shape
#         ``(.., ..,[ ..,] 3)``.
#     illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
#         The name of the illuminant (the function is NOT case sensitive).
#     observer : {"2", "10"}, optional
#         The aperture angle of the observer.
#     Returns
#     -------
#     out : ndarray
#         The image in CIE-LAB format, in a 3- or 4-D array of shape
#         ``(.., ..,[ ..,] 3)``.
#     Raises
#     ------
#     ValueError
#         If `xyz` is not a 3-D array of shape ``(.., ..,[ ..,] 3)``.
#     ValueError
#         If either the illuminant or the observer angle is unsupported or
#         unknown.
#     Notes
#     -----
#     By default Observer= 2A, Illuminant= D65. CIE XYZ tristimulus values
#     x_ref=95.047, y_ref=100., z_ref=108.883. See function `get_xyz_coords` for
#     a list of supported illuminants.
#     References
#     ----------
#     .. [1] http://www.easyrgb.com/index.php?X=MATH&H=07#text7
#     .. [2] http://en.wikipedia.org/wiki/Lab_color_space
#     Examples
#     --------
#
#     """
#     arr = xyz
#
#     xyz_ref_white = skimage.color.get_xyz_coords(illuminant, observer)
#
#     # scale by CIE XYZ tristimulus values of the reference white point
#     arr = arr / xyz_ref_white
#
#     # Nonlinear distortion and linear transformation
#     mask = arr > 0.008856
#     notmask = arr <= 0.008856
#     arr[mask] = ch.power(arr[mask], 1. / 3.)
#     arr[notmask] = 7.787 * arr[notmask] + 16. / 116.
#
#     x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]
#
#     # Vector scaling
#     L = (116. * y) - 16.
#     a = 500.0 * (x - y)
#     b = 200.0 * (y - z)
#
#     return ch.concatenate([x[..., np.newaxis] for x in [L, a, b]], axis=-1)
#
# def rgb2xyz(rgb):
#     """RGB to XYZ color space conversion.
#     Parameters
#     ----------
#     rgb : array_like
#         The image in RGB format, in a 3- or 4-D array of shape
#         ``(.., ..,[ ..,] 3)``.
#     Returns
#     -------
#     out : ndarray
#         The image in XYZ format, in a 3- or 4-D array of shape
#         ``(.., ..,[ ..,] 3)``.
#     Raises
#     ------
#     ValueError
#         If `rgb` is not a 3- or 4-D array of shape ``(.., ..,[ ..,] 3)``.
#     Notes
#     -----
#     The CIE XYZ color space is derived from the CIE RGB color space. Note
#     however that this function converts from sRGB.
#     References
#     ----------
#     .. [1] http://en.wikipedia.org/wiki/CIE_1931_color_space
#     Examples
#     --------
#
#     """
#     # Follow the algorithm from http://www.easyrgb.com/index.php
#     # except we don't multiply/divide by 100 in the conversion
#     arr = rgb
#     mask = arr > 0.04045
#     notmask = arr <= 0.04045
#     arr[mask] = ch.power((arr[mask] + 0.055) / 1.055, 2.4)
#     arr[notmask] /= 12.92
#     return _convert(xyz_from_rgb, arr)
def scaleInvariantMSECoeff(x_pred, x_target):
    #Rows: test samples
    #Cols: target variables
    scales = (matrix_multiply(x_pred[:,None,:],x_target[:,:,None])/matrix_multiply(x_pred[:,None,:],x_pred[:,:,None])).ravel()

    return scales

def scaleInvariantColourDifference(rgbGT, rgbPred):
    scaleRGB = scaleInvariantMSECoeff(rgbPred, rgbGT)
    dist = np.sqrt(np.sum((rgbGT - scaleRGB[:,None]*rgbPred)**2, axis=1))
    return dist


def cColourDifference(rgb1, rgb2):
    lab1 = cv2.cvtColor(np.uint8(rgb1.reshape([-1,1,3])*255), cv2.COLOR_RGB2LAB)/255
    lab2 = cv2.cvtColor(np.uint8(rgb2.reshape([-1,1,3])*255), cv2.COLOR_RGB2LAB)/255
    dist = np.sqrt(np.sum((lab1[:,0,1:] - lab2[:,0,1:])**2, axis=1))
    return dist

def eColourDifference(rgb1, rgb2):
    lab1 = cv2.cvtColor(np.uint8(rgb1.reshape([-1,1,3])*255), cv2.COLOR_RGB2LAB)/255
    lab2 = cv2.cvtColor(np.uint8(rgb2.reshape([-1,1,3])*255), cv2.COLOR_RGB2LAB)/255
    dist = np.sqrt(np.sum((lab1[:,0] - lab2[:,0])**2, axis=1))
    return dist


# conf.cellSize = cellSize;
# conf.numOrientations = 9;
def computeHoG(image):

    image = color.rgb2gray(image)

    hog_descr, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualise=True)

    # fig, ( ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
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


def dr_wrt_convolution(x, filter):
    print("Computing convolution gradients")
    widthRolls = x.shape[1]
    heightRolls = x.shape[0]
    tmpShape = [x.shape[0]+filter.shape[0], x.shape[1]+filter.shape[1]]
    template = np.zeros(tmpShape)
    template[0:filter.shape[0], 0:filter.shape[1]] = filter
    jacs = []
    for i in range(heightRolls):
        for j in range(widthRolls):
            templateRolled = np.roll(template, shift=i, axis=0)
            templateRolled = np.roll(templateRolled, shift=j, axis=1)

            templateGrad = templateRolled[tmpShape[0] - x.shape[0] - np.int(filter.shape[0]/2): tmpShape[0] - np.int(filter.shape[0]/2), tmpShape[1] - x.shape[1] - np.int(filter.shape[1]/2): tmpShape[1] - np.int(filter.shape[1]/2)]
            jacs = jacs + [scipy.sparse.coo_matrix(templateGrad.ravel())]

    return scipy.sparse.vstack(jacs).tocsc()



class convolve2D(Ch):
    terms = 'filter'
    dterms = 'x'

    def compute_r(self):
        # convolved = scipy.signal.convolve2d(self.x, self.filter, mode='same')
        convolved = scipy.ndimage.convolve(self.x, self.filter, mode='reflect')
        # return convolved[np.int((convolved.shape[0]-self.x.shape[0])/2):np.int((convolved.shape[0]-self.x.shape[0])/2) + self.x.shape[1], np.int((convolved.shape[1]-self.x.shape[1])/2):np.int((convolved.shape[1]-self.x.shape[1])/2) + self.x.shape[1]]
        return convolved

    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            return self.convolve2DDr
        else:
            return None

class HogImage(Ch):
    terms = 'numOrient', 'cwidth', 'cheight'
    dterms = 'image', 'hog'
    def compute_r(self):
        from skimage import draw
        sy,sx, _ = self.image.shape
        radius = min(self.cwidth, self.cheight) // 2 - 1
        orientations_arr = np.arange(self.numOrient)
        dx_arr = radius * np.cos(orientations_arr / self.numOrient * np.pi)
        dy_arr = radius * np.sin(orientations_arr / self.numOrient * np.pi)
        cr2 = self.cheight + self.cheight
        cc2 = self.cwidth + self.cwidth
        hog_image = np.zeros((sy, sx), dtype=float)
        n_cellsx = int(np.floor(sx // self.cwidth))  # number of cells in x
        n_cellsy = int(np.floor(sy // self.cheight))  # number of cells in y
        for x in range(n_cellsx):
            for y in range(n_cellsy):
                for o, dx, dy in zip(orientations_arr, dx_arr, dy_arr):
                    centre = tuple([y * cr2 // 2, x * cc2 // 2])
                    rr, cc = draw.line(int(centre[0] + dy),
                                       int(centre[1] + dx),
                                       int(centre[0] - dy),
                                       int(centre[1] - dx))
                    hog_image[rr, cc] += self.hog[y, x, o]
        return hog_image

    def compute_dr_wrt(self, wrt):
        return None

import skimage
def diffHog(image, drconv=None, numOrient = 9, cwidth=8, cheight=8):
    imagegray = 0.3*image[:,:,0] +  0.59*image[:,:,1] + 0.11*image[:,:,2]
    sy,sx = imagegray.shape

    # gx = ch.empty(imagegray.shape, dtype=np.double)
    gx = imagegray[:, 2:] - imagegray[:, :-2]
    gx = ch.hstack([np.zeros([sy,1]), gx, np.zeros([sy,1])])

    gy = imagegray[2:, :] - imagegray[:-2, :]
    # gy = imagegray[:, 2:] - imagegray[:, :-2]
    gy = ch.vstack([np.zeros([1,sx]), gy, np.zeros([1,sx])])

    gx += 1e-5
    # gy = imagegray[:-2,1:-1] - imagegray[2:,1:-1] + 0.00001
    # gx = imagegray[1:-1,:-2] - imagegray[1:-1, 2:] + 0.00001

    distFilter = np.ones([2*cheight,2*cwidth], dtype=np.uint8)
    distFilter[np.int(2*cheight/2), np.int(2*cwidth/2)] = 0
    distFilter = (cv2.distanceTransform(distFilter, cv2.DIST_L2, 3)- np.max(cv2.distanceTransform(distFilter, cv2.DIST_L2, 3)))/(-np.max(cv2.distanceTransform(distFilter, cv2.DIST_L2, 3)))

    magn = ch.sqrt(gy**2 + gx**2)*180/np.sqrt(2)

    angles = ch.arctan(gy/gx)*180/np.pi + 90

    # meanOrient = np.linspace(0, 180, numOrient)

    orientations_arr = np.arange(numOrient)

    meanOrient = orientations_arr / numOrient * 180

    fb_resttmp = 1 - ch.abs(ch.expand_dims(angles[:,:],2) - meanOrient[1:].reshape([1,1,numOrient-1]))*numOrient/180
    zeros_rest = np.zeros([sy,sx, numOrient-1, 1])
    fb_rest = ch.max(ch.concatenate([fb_resttmp[:,:,:,None], zeros_rest],axis=3), axis=3)

    chMinOrient0 = ch.min(ch.concatenate([ch.abs(ch.expand_dims(angles[:,:],2) - meanOrient[0].reshape([1,1,1]))[:,:,:,None], ch.abs(180 - ch.expand_dims(angles[:,:],2) - meanOrient[0].reshape([1,1,1]))[:,:,:,None]], axis=3), axis=3)

    zeros_fb0 = np.zeros([sy,sx, 1])
    fb0_tmp =  ch.concatenate([1 - chMinOrient0[:,:]*numOrient/180, zeros_fb0],axis=2)
    fb_0 = ch.max(fb0_tmp,axis=2)

    fb = ch.concatenate([fb_0[:,:,None], fb_rest],axis=2)

    # fb[:,:,0] = ch.max(1 - ch.abs(ch.expand_dims(angles,2) - meanOrient.reshape([1,1,numOrient]))*numOrient/180,0)

    # fb = 1./(1. + ch.exp(1 - ch.abs(ch.expand_dims(angles,2) - meanOrient.reshape([1,1,numOrient]))*numOrient/180))

    Fb = ch.expand_dims(magn,2)*fb

    if drconv is None:
        drconv = dr_wrt_convolution(Fb[:,:,0], distFilter)


    Fs_list = [convolve2D(x=Fb[:,:,Fbi], filter=distFilter, convolve2DDr=drconv).reshape([Fb.shape[0], Fb.shape[1],1]) for Fbi in range(numOrient)]

    # Fs_list = [scipy.signal.convolve2d(Fb[:,:,Fbi], distFilter).reshape([Fb.shape[0], Fb.shape[1],1]) for Fbi in range(numOrient)]
    Fs = ch.concatenate(Fs_list, axis=2)

    # cellCols = np.arange(start=cwidth/2, stop=Fs.shape[1]-cwidth/2 , step=cwidth)
    # cellRows = np.arange(start=cheight/2, stop=Fs.shape[0]-cheight/2 , step=cheight)

    Fcells = Fs[0:Fs.shape[0] :cheight,0:Fs.shape[1] :cwidth,:]

    epsilon = 1e-5

    v = Fcells/ch.sqrt(ch.sum(Fcells**2) + epsilon)
    # v = Fcells

    # hog, hogim = skimage.feature.hog(imagegray,  orientations=numOrient, pixels_per_cell=(cheight, cwidth), visualise=True)
    hog_image = HogImage(image=image, hog=Fcells, numOrient=numOrient, cwidth=cwidth, cheight=cheight)

    # plt.imshow(hog_image)
    # plt.figure()
    # plt.imshow(hogim)
    # ipdb.set_trace()

    return v, hog_image, drconv

import zernike
def zernikeProjection(images, numCoeffs, win=50):

    croppedImages = images[:,images.shape[1]/2-win:images.shape[1]/2+win,images.shape[2]/2-win:images.shape[2]/2+win, :]

    zpolys = zernikePolynomials(imageSize=(croppedImages.shape[1],croppedImages.shape[2]), numCoeffs=numCoeffs)

    coeffs = np.sum(np.sum(croppedImages[:,:,:,:,None]*zpolys.reshape([1,zpolys.shape[0], zpolys.shape[1], 1, -1]), axis=2), axis=1)
    return coeffs

def zernikeProjectionGray(images, numCoeffs, win=50):

    if images.shape[3] == 3:
        images = 0.3*images[:,:,:,0] + 0.59*images[:,:,:,1] + 0.11*images[:,:,:,2]
    croppedImages = images[:,images.shape[1]/2-win:images.shape[1]/2+win,images.shape[2]/2-win:images.shape[2]/2+win]

    zpolys = zernikePolynomials(imageSize=(croppedImages.shape[1],croppedImages.shape[2]), numCoeffs=numCoeffs)

    coeffs = np.sum(np.sum(croppedImages[:,:,:,None]*zpolys.reshape([1,zpolys.shape[0], zpolys.shape[1],  -1]), axis=2), axis=1)
    return coeffs

def chZernikeProjection(image, numCoeffs=20, win=50):
    croppedImage = image[image.shape[0]/2-win:image.shape[0]/2+win,image.shape[1]/2-win:image.shape[1]/2+win, :]
    zpolys = zernikePolynomials(imageSize=(croppedImage.shape[0],croppedImage.shape[1]), numCoeffs=numCoeffs)
    imageProjections = croppedImage[:,:,:,None]*zpolys.reshape([zpolys.shape[0], zpolys.shape[1], 1, -1])
    coeffs = ch.sum(ch.sum(imageProjections, axis=0), axis=0)
    return coeffs, imageProjections

def zernikePolynomials(imageSize=(100,100), numCoeffs=20):

    sy,sx = imageSize

    # distFilter = np.ones([sy,sx], dtype=np.uint8)
    # distFilter[np.int(sy/2), np.int(sx/2)] = 0
    # distFilter = cv2.distanceTransform(distFilter, cv2.DIST_L2, 3)
    # distFilter /= np.max(distFilter)
    # np.arange()

    ones = np.ones([sy,sx], dtype=np.bool)
    imgind = np.where(ones)

    dy = imgind[0] - int(sy/2)
    dx = imgind[1] - int(sx/2)

    pixaz = np.arctan2(dy,dx)
    pixrad = np.sqrt(dy**2 + dx**2)

    imaz = np.zeros([sy,sx])
    imrad = np.zeros([sy,sx])
    imaz[imgind] = pixaz
    imrad[imgind] = pixrad

    outcircle = imrad>=sy/2
    imrad[outcircle] = 0
    imaz[outcircle] = 0

    imrad/=np.max(imrad)

    zpolys = [zernike.zernikel(j, imrad, imaz)[:,:,None] for j in range(numCoeffs)]

    zpolys = np.concatenate(zpolys, axis=2)

    return zpolys