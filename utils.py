import numpy as np
import os
import skimage
import h5py
import ipdb

__author__ = 'pol'

def writeImagesHdf5(imagesDir, imageSet):
    image = skimage.io.imread(imagesDir + 'im' + str(imageSet[0]) + '.jpeg')
    width = image.shape[1]
    height = image.shape[0]
    images = np.zeros([len(imageSet), height, width, 3])
    for imageit, imageid  in enumerate(imageSet):
        image = skimage.io.imread(imagesDir + 'im' + str(imageid) + '.jpeg')
        images[imageit, :, :, :] = image
    gtDataFile = h5py.File(imagesDir + 'images.h5', 'w')
    gtDataFile.create_dataset("images", data=images)
    gtDataFile.close()

def readImages(imagesDir, imageSet, loadFromHdf5):
    if loadFromHdf5:
        if os.path.isfile(imagesDir + 'images.h5'):
            gtDataFile = h5py.File(imagesDir + 'images.h5', 'r')
            return gtDataFile["images"][:][imageSet]
    else:
        image = skimage.io.imread(imagesDir + 'im' + str(imageSet[0]) + '.jpeg')
        width = image.shape[1]
        height = image.shape[0]
        images = np.zeros([len(imageSet), height, width, 3])
        for imageit, imageid  in enumerate(imageSet):
            image = skimage.io.imread(imagesDir + 'im' + str(imageid) + '.jpeg')
            images[imageit, :, :, :] = image

        return images

def generateExperiment(size, prefix, ratio, seed):
    np.random.seed(seed)
    data = np.arange(size)
    np.random.shuffle(data)
    train = data[0:np.int(size*ratio)]
    test = data[np.int(size*ratio)::]

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    np.save(prefix + 'train.npy', train)
    np.save(prefix + 'test.npy', test)