import numpy as np
import os
import skimage
import h5py
import ipdb

__author__ = 'pol'


def writeImagesHdf5(imagesDir, writeDir, imageSet, writeGray=False ):
    print("Writing HDF5 file")
    image = skimage.io.imread(imagesDir + 'im' + str(imageSet[0]) + '.jpeg')
    imDtype = image.dtype
    width = image.shape[1]
    height = image.shape[0]
    if not writeGray:
        gtDataFile = h5py.File(writeDir + 'images.h5', 'w')
        images = np.array([], dtype = np.dtype('('+ str(height)+','+ str(width) +',3)uint8'))
        gtDataset = gtDataFile.create_dataset("images", data=images, maxshape=(None,height,width, 3))
        # images = np.zeros([len(imageSet), height, width, 3], dtype=np.uint8)
    else:
        imageGray = 0.3*image[:,:,0] + 0.59*image[:,:,1] + 0.11*image[:,:,2]
        grayDtype = imageGray.dtype

        gtDataFile = h5py.File(writeDir + 'images_gray.h5', 'w')
        # images = np.zeros([], dtype=np.float32)
        images = np.array([], dtype = np.dtype('('+ str(height)+','+ str(width) +')f'))
        gtDataset = gtDataFile.create_dataset("images", data=images, maxshape=(None,height,width))

    for imageit, imageid  in enumerate(imageSet):
        gtDataset.resize(gtDataset.shape[0]+1, axis=0)

        image = skimage.io.imread(imagesDir + 'im' + str(imageid) + '.jpeg').astype(np.uint8)
        if not writeGray:
            gtDataset[-1] = image
        else:
            image = image.astype(np.float32)/255.0
            gtDataset[-1] = 0.3*image[:,:,0] + 0.59*image[:,:,1] + 0.11*image[:,:,2]

        gtDataFile.flush()

    gtDataFile.close()
    print("Ended writing HDF5 file")

def readImages(imagesDir, imageSet, loadGray=False, loadFromHdf5=False):
    if loadFromHdf5:
        if not loadGray:
            if os.path.isfile(imagesDir + 'images.h5'):
                gtDataFile = h5py.File(imagesDir + 'images.h5', 'r')
                boolSet = np.zeros(gtDataFile["images"].shape[0]).astype(np.bool)
                boolSet[imageSet] = True
                return gtDataFile["images"][boolSet,:,:,:].astype(np.float32)/255.0
        else:
            if os.path.isfile(imagesDir + 'images_gray.h5'):
                gtDataFile = h5py.File(imagesDir + 'images_gray.h5', 'r')
                boolSet = np.zeros(gtDataFile["images"].shape[0]).astype(np.bool)
                boolSet[imageSet] = True
                return gtDataFile["images"][boolSet,:,:].astype(np.float32)
    else:
        image = skimage.io.imread(imagesDir + 'im' + str(imageSet[0]) + '.jpeg')
        width = image.shape[1]
        height = image.shape[0]
        if not loadGray:
            images = np.zeros([len(imageSet), height, width, 3], dtype=np.float32)
        else:
            images = np.zeros([len(imageSet), height, width], dtype=np.float32)
        for imageit, imageid  in enumerate(imageSet):
            if os.path.isfile(imagesDir + 'im' + str(imageid) + '.jpeg'):
                image = skimage.io.imread(imagesDir + 'im' + str(imageid) + '.jpeg')
            else:
                print("Image " + str(imageid) + " does not exist!")
                image = np.zeros_like(image)
            image = image/255.0
            if not loadGray:
                images[imageit, :, :, :] =  image
            else:
                images[imageit, :, :] =  0.3*image[:,:,0] + 0.59*image[:,:,1] + 0.11*image[:,:,2]

        return images

def readImagesHdf5(imagesDir, loadGray=False):
    if not loadGray:
        if os.path.isfile(imagesDir + 'images.h5'):
             gtDataFile = h5py.File(imagesDir + 'images.h5', 'r')
             return gtDataFile["images"]
        else:
            if os.path.isfile(imagesDir + 'images_gray.h5'):
                gtDataFile = h5py.File(imagesDir + 'images_gray.h5', 'r')
                return gtDataFile["images"]

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

