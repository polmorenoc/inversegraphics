__author__ = 'pol'

import matplotlib
matplotlib.use('Qt4Agg')
from math import radians
import timeit
import time
import geometry
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

#########################################
# Initialization ends here
#########################################

seed = 1
np.random.seed(seed)

gtPrefix = 'first'
trainPrefix = gtPrefix + '_' + 'train1/'
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


#Create experiment simple sepratation.

generateExperiment(len(trainIds), trainPrefix, 0.8, 1)

trainSet = np.load('experiment/' + trainPrefix + 'train.npy')

imagesDir = gtDir + 'images/'

#Read images
image = skimage.io.imread(imagesDir + 'im' + str(trainSet[0]) + '.png')
width = image.shape[1]
height = image.shape[0]
images = np.zeros([len(trainSet), height, width, 3])
for imidx, imid in enumerate(trainSet):
    imageName = imagesDir + 'im' + str(imid) + '.png'
    images[imidx, :, :, :] =  skimage.io.imread(imageName)

#See some images to make sure groundtruth was generated properly.

ipdb.set_trace()

# hogs = []
# # vcolorsfeats = []
# illumfeats = []
#
# hogfeats = np.vstack(hogs)
# illumfeats = np.vstack(illumfeats)
#
# print("Training RFs")
# randForestModelCosAzs = recognition_models.trainRandomForest(hogfeats, np.cos(trainAzsGT))
# randForestModelSinAzs = recognition_models.trainRandomForest(hogfeats, np.sin(trainAzsGT))
# randForestModelCosElevs = recognition_models.trainRandomForest(hogfeats, np.cos(trainElevsGT))
# randForestModelSinElevs = recognition_models.trainRandomForest(hogfeats, np.sin(trainElevsGT))
#
# randForestModelLightCosAzs = recognition_models.trainRandomForest(illumfeats, np.cos(trainLightAzsGT))
# randForestModelLightSinAzs = recognition_models.trainRandomForest(illumfeats, np.sin(trainLightAzsGT))
# randForestModelLightCosElevs = recognition_models.trainRandomForest(illumfeats, np.cos(trainLightElevsGT))
# randForestModelLightSinElevs = recognition_models.trainRandomForest(illumfeats, np.sin(trainLightElevsGT))
#
# imagesStack = np.vstack([image.reshape([1,-1]) for image in images])
# randForestModelLightIntensity = recognition_models.trainRandomForest(imagesStack, trainLightIntensitiesGT)
#
# trainedModels = {'randForestModelCosAzs':randForestModelCosAzs,'randForestModelSinAzs':randForestModelSinAzs,'randForestModelCosElevs':randForestModelCosElevs,'randForestModelSinElevs':randForestModelSinElevs,'randForestModelLightCosAzs':randForestModelLightCosAzs,'randForestModelLightSinAzs':randForestModelLightSinAzs,'randForestModelLightCosElevs':randForestModelLightCosElevs,'randForestModelLightSinElevs':randForestModelLightSinElevs,'randForestModelLightIntensity':randForestModelLightIntensity}
# with open('experiments/' + trainPrefix + 'models.pickle', 'wb') as pfile:
#     pickle.dump(trainedModels, pfile)
#
#
# # print("Training LR")
# # linRegModelCosAzs = recognition_models.trainLinearRegression(hogfeats, np.cos(trainAzsGT))
# # linRegModelSinAzs = recognition_models.trainLinearRegression(hogfeats, np.sin(trainAzsGT))
# # linRegModelCosElevs = recognition_models.trainLinearRegression(hogfeats, np.cos(trainElevsGT))
# # linRegModelSinElevs = recognition_models.trainLinearRegression(hogfeats, np.sin(trainElevsGT))
#
#
# print("Finished training recognition models.")
# beginTraining = False