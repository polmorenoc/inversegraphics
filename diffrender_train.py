__author__ = 'pol'

import matplotlib
matplotlib.use('Qt4Agg')
from math import radians
import timeit
import time
import image_processing
import numpy as np
import cv2
from utils import *
import generative_models
import matplotlib.pyplot as plt
plt.ion()
import recognition_models
import skimage
import h5py
import ipdb
import pickle

#########################################
# Initialization ends here
#########################################

seed = 1
np.random.seed(seed)

gtPrefix = 'test'
trainPrefix = 'train1'
gtDir = 'groundtruth/' + gtPrefix + '/'
experimentDir = 'experiments/' + trainPrefix + '/'
testPrefix = 'test1'
resultDir = 'results/' + trainPrefix + '_' + testPrefix + '/'

groundTruthFilename = gtDir + 'groundTruth.h5'
gtDataFile = h5py.File(groundTruthFilename, 'r')
groundTruth = gtDataFile[gtPrefix]
print("Reading experiment.")

dataAzsGT = groundTruth['trainAzsGT']
dataObjAzsGT = groundTruth['trainObjAzsGT']
dataElevsGT = groundTruth['trainElevsGT']
dataLightAzsGT = groundTruth['trainLightAzsGT']
dataLightElevsGT = groundTruth['trainLightElevsGT']
dataLightIntensitiesGT = groundTruth['trainLightIntensitiesGT']
dataVColorGT = groundTruth['trainVColorGT']
dataScenes = groundTruth['trainScenes']
dataTeapotIds = groundTruth['trainTeapotIds']
dataEnvMaps = groundTruth['trainEnvMaps']
dataOcclusions = groundTruth['trainOcclusions']
dataTargetIndices = groundTruth['trainTargetIndices']
dataComponentsGT = groundTruth['trainComponentsGT']
dataComponentsGTRel = groundTruth['trainComponentsGTRel']
dataIds = groundTruth['trainIds']

gtDtype = groundTruth.dtype
# gtDtype = [('trainIds', trainIds.dtype.name), ('trainAzsGT', trainAzsGT.dtype.name),('trainObjAzsGT', trainObjAzsGT.dtype.name),('trainElevsGT', trainElevsGT.dtype.name),('trainLightAzsGT', trainLightAzsGT.dtype.name),('trainLightElevsGT', trainLightElevsGT.dtype.name),('trainLightIntensitiesGT', trainLightIntensitiesGT.dtype.name),('trainVColorGT', trainVColorGT.dtype.name, (3,) ),('trainScenes', trainScenes.dtype.name),('trainTeapotIds', trainTeapotIds.dtype.name),('trainEnvMaps', trainEnvMaps.dtype.name),('trainOcclusions', trainOcclusions.dtype.name),('trainTargetIndices', trainTargetIndices.dtype.name), ('trainComponentsGT', trainComponentsGT.dtype, (9,)),('trainComponentsGTRel', trainComponentsGTRel.dtype, (9,))]

# Create experiment simple sepratation.
#

if not os.path.isfile(experimentDir + 'train.npy'):
    generateExperiment(len(dataIds), experimentDir, 0.8, 1)

trainSet = np.load(experimentDir + 'train.npy')

trainAzsRel = np.mod(dataAzsGT - dataObjAzsGT, 2*np.pi)[trainSet]
trainElevsGT = dataElevsGT[trainSet]
trainComponentsGTRel = dataElevsGT[trainSet]

imagesDir = gtDir + 'images/'

loadFromHdf5 = True
writeHdf5 = False

if writeHdf5:
    writeImagesHdf5(imagesDir, dataIds)

if loadFromHdf5:
    images = readImages(imagesDir, dataIds, loadFromHdf5)

loadHogFeatures = False
loadIllumFeatures = False

if loadHogFeatures:
    hogfeatures = np.load(experimentDir + 'hog.npy')
else:
    print("Extracting Hog features .")
    hogfeatures = image_processing.computeHoGFeatures(images)
    np.save(experimentDir + 'hog.npy', hogfeatures)

if loadIllumFeatures:
    illumfeatures =  np.load(experimentDir  + 'illum.npy')
else:
    print("Extracting Illumination features (FFT.")
    illumfeatures = image_processing.computeIllumFeatures(images, images[0].size/12)
    np.save(experimentDir  + 'illum.npy', illumfeatures)

trainHogfeatures = hogfeatures[trainSet]
trainIllumfeatures = illumfeatures[trainSet]

print("Training recognition models.")

print("Training RFs Cos Azs")
randForestModelCosAzs = recognition_models.trainRandomForest(trainHogfeatures, np.cos(trainAzsRel))
print("Training RFs Sin Azs")
randForestModelSinAzs = recognition_models.trainRandomForest(trainHogfeatures, np.sin(trainAzsRel))
print("Training RFs Cos Elevs")
randForestModelCosElevs = recognition_models.trainRandomForest(trainHogfeatures, np.cos(trainElevsGT))
print("Training RFs Sin Elevs")
randForestModelSinElevs = recognition_models.trainRandomForest(trainHogfeatures, np.sin(trainElevsGT))
#
print("Training RFs Components")
randForestModelRelSHComponents = recognition_models.trainRandomForest(trainIllumfeatures, trainComponentsGTRel)
#
# imagesStack = np.vstack([image.reshape([1,-1]) for image in images])
# randForestModelLightIntensity = recognition_models.trainRandomForest(imagesStack, trainLightIntensitiesGT)
#
trainedModels = {'randForestModelCosAzs':randForestModelCosAzs,'randForestModelSinAzs':randForestModelSinAzs,'randForestModelCosElevs':randForestModelCosElevs,'randForestModelSinElevs':randForestModelSinElevs,'randForestModelRelSHComponents':randForestModelRelSHComponents}
with open(experimentDir + 'recognition_models.pickle', 'wb') as pfile:
    pickle.dump(trainedModels, pfile)
#
#
# # # print("Training LR")
# # # linRegModelCosAzs = recognition_models.trainLinearRegression(hogfeatures, np.cos(trainAzsGT))
# # # linRegModelSinAzs = recognition_models.trainLinearRegression(hogfeatures, np.sin(trainAzsGT))
# # # linRegModelCosElevs = recognition_models.trainLinearRegression(hogfeatures, np.cos(trainElevsGT))
# # # linRegModelSinElevs = recognition_models.trainLinearRegression(hogfeatures, np.sin(trainElevsGT))
# #
#
# print("Finished training recognition models.")
