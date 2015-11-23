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
    images = np.zeros([len(imageSet), height, width, 3], dtype=np.uint8)
    for imageit, imageid  in enumerate(imageSet):
        image = skimage.io.imread(imagesDir + 'im' + str(imageid) + '.jpeg').astype(np.uint8)
        images[imageit, :, :, :] = image
    gtDataFile = h5py.File(imagesDir + 'images.h5', 'w')
    gtDataFile.create_dataset("images", data=images)
    gtDataFile.close()

def readImages(imagesDir, imageSet, loadFromHdf5):
    if loadFromHdf5:
        if os.path.isfile(imagesDir + 'images.h5'):
            gtDataFile = h5py.File(imagesDir + 'images.h5', 'r')
            return gtDataFile["images"][:][imageSet].astype(np.float32)/255.0
    else:
        image = skimage.io.imread(imagesDir + 'im' + str(imageSet[0]) + '.jpeg')
        width = image.shape[1]
        height = image.shape[0]
        images = np.zeros([len(imageSet), height, width, 3], dtype=np.float32)
        for imageit, imageid  in enumerate(imageSet):
            image = skimage.io.imread(imagesDir + 'im' + str(imageid) + '.jpeg')
            images[imageit, :, :, :] = image/255.0

        return images

def generateExperiment(size, experimentDir, ratio, seed):
    np.random.seed(seed)
    data = np.arange(size)
    np.random.shuffle(data)
    train = data[0:np.int(size*ratio)]
    test = data[np.int(size*ratio)::]

    if not os.path.exists(experimentDir):
        os.makedirs(experimentDir)

    np.save(experimentDir + 'train.npy', train)
    np.save(experimentDir + 'test.npy', test)

# saveScatter(xaxis*180/np.pi, yaxis[1], 'Azimuth error (ground-truth)', Azimuth (predicted), filename)

import matplotlib.pyplot as plt
def saveScatter(xaxis, yaxis, xlabel, ylabel, filename):
    plt.ioff()
    fig = plt.figure()
    plt.scatter(xaxis, yaxis)
    plt.xlabel('Elevation (degrees)')
    plt.ylabel('Angular error')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,90,-90,90))
    plt.title('Performance scatter plot')
    fig.savefig(filename)
    plt.close(fig)

