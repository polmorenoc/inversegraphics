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
trainPrefix = gtPrefix + '_' + '1/'
gtDir = 'groundtruth/' + gtPrefix + '/'

groundTruthFilename = gtDir + 'groundTruth.h5'
gtDataFile = h5py.File(groundTruthFilename, 'r')
groundTruth = gtDataFile[gtPrefix]
print("Training recognition models.")

trainAzsGT = groundTruth['trainAzsGT']
trainObjAzsGT = groundTruth['trainObjAzsGT']
trainElevsGT = groundTruth['trainElevsGT']
trainLightAzsGT = groundTruth['trainLightAzsGT']
trainLightElevsGT = groundTruth['trainLightElevsGT']
trainLightIntensitiesGT = groundTruth['trainLightIntensitiesGT']
trainVColorGT = groundTruth['trainVColorGT']
trainScenes = groundTruth['trainScenes']
trainTeapotIds = groundTruth['trainTeapotIds']
trainEnvMaps = groundTruth['trainEnvMaps']
trainOcclusions = groundTruth['trainOcclusions']
trainTargetIndices = groundTruth['trainTargetIndices']
trainComponentsGT = groundTruth['trainComponentsGT']
trainComponentsGTRel = groundTruth['trainComponentsGTRel']
trainIds = groundTruth['trainIds']
gtDtype = [('trainIds', trainIds.dtype.name), ('trainAzsGT', trainAzsGT.dtype.name),('trainObjAzsGT', trainObjAzsGT.dtype.name),('trainElevsGT', trainElevsGT.dtype.name),('trainLightAzsGT', trainLightAzsGT.dtype.name),('trainLightElevsGT', trainLightElevsGT.dtype.name),('trainLightIntensitiesGT', trainLightIntensitiesGT.dtype.name),('trainVColorGT', trainVColorGT.dtype.name, (3,) ),('trainScenes', trainScenes.dtype.name),('trainTeapotIds', trainTeapotIds.dtype.name),('trainEnvMaps', trainEnvMaps.dtype.name),('trainOcclusions', trainOcclusions.dtype.name),('trainTargetIndices', trainTargetIndices.dtype.name), ('trainComponentsGT', trainComponentsGT.dtype, (9,)),('trainComponentsGTRel', trainComponentsGTRel.dtype, (9,))]

trainAzsRel = np.mod(trainAzsGT - trainObjAzsGT, 2*np.pi)
#Create experiment simple sepratation.

generateExperiment(len(trainIds), trainPrefix, 0.8, 1)

trainSet = np.load('experiments/' + trainPrefix + 'train.npy')

imagesDir = gtDir + 'images/'

loadFromHdf5 = False
writeHdf5 = False

if writeHdf5:
    writeImagesHdf5(imagesDir, trainSet)

images = readImages(imagesDir, trainSet, loadFromHdf5)

#Check some images to make sure groundtruth was generated properly.

#Perhaps save these on experiments dir to quickly train new models.
loadHogFeatures = False
loadIllumFeatures = False

experimentPrefix = 'experiments/' + trainPrefix
if loadHogFeatures:
    hogfeatures = np.load(experimentPrefix + 'hog.npy')
else:
    hogfeatures = image_processing.computeHoGFeatures(images)
    np.save(experimentPrefix + 'hog.npy', hogfeatures)

if loadIllumFeatures:
    illumfeatures =  np.save(experimentPrefix  + 'illum.npy')
else:
    illumfeatures = image_processing.computeIllumFeatures(images)
    np.save(experimentPrefix  + 'illum.npy', illumfeatures)

# print("Training RFs")
randForestModelCosAzs = recognition_models.trainRandomForest(hogfeatures, np.cos(trainAzsRel))
randForestModelSinAzs = recognition_models.trainRandomForest(hogfeatures, np.sin(trainAzsRel))
randForestModelCosElevs = recognition_models.trainRandomForest(hogfeatures, np.cos(trainElevsGT))
randForestModelSinElevs = recognition_models.trainRandomForest(hogfeatures, np.sin(trainElevsGT))

randForestModelRelSHComponents = recognition_models.trainRandomForest(illumfeatures, trainComponentsGTRel)

imagesStack = np.vstack([image.reshape([1,-1]) for image in images])
randForestModelLightIntensity = recognition_models.trainRandomForest(imagesStack, trainLightIntensitiesGT)

trainedModels = {'randForestModelCosAzs':randForestModelCosAzs,'randForestModelSinAzs':randForestModelSinAzs,'randForestModelCosElevs':randForestModelCosElevs,'randForestModelSinElevs':randForestModelSinElevs,'randForestModelRelSHComponents':randForestModelRelSHComponents}
with open('experiments/' + trainPrefix + 'models.pickle', 'wb') as pfile:
    pickle.dump(trainedModels, pfile)


# # print("Training LR")
# # linRegModelCosAzs = recognition_models.trainLinearRegression(hogfeatures, np.cos(trainAzsGT))
# # linRegModelSinAzs = recognition_models.trainLinearRegression(hogfeatures, np.sin(trainAzsGT))
# # linRegModelCosElevs = recognition_models.trainLinearRegression(hogfeatures, np.cos(trainElevsGT))
# # linRegModelSinElevs = recognition_models.trainLinearRegression(hogfeatures, np.sin(trainElevsGT))
#

print("Finished training recognition models.")
