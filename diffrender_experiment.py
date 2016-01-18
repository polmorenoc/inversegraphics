__author__ = 'pol'

import matplotlib
matplotlib.use('Qt4Agg')
from math import radians
import timeit
import time
import numpy as np
from utils import *
import matplotlib.pyplot as plt
plt.ion()
import h5py
import ipdb
import pickle

#########################################
# Initialization ends here
#########################################

seed = 1
np.random.seed(seed)

gtPrefix = 'train4_occlusion'
experimentPrefix = 'train4_occlusion_10k'
experimentDescr = 'Synthetic test set with occlusions'
gtDir = 'groundtruth/' + gtPrefix + '/'
experimentDir = 'experiments/' + experimentPrefix + '/'

groundTruthFilename = gtDir + 'groundTruth.h5'
gtDataFile = h5py.File(groundTruthFilename, 'r')

onlySynthetic = True

print("Reading images.")

print("Reading experiment data.")

shapeGT = gtDataFile[gtPrefix].shape

groundTruth = gtDataFile[gtPrefix]

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
dataLightCoefficientsGT = groundTruth['trainLightCoefficientsGT']
dataLightCoefficientsGTRel = groundTruth['trainLightCoefficientsGTRel']
dataAmbientIntensityGT = groundTruth['trainAmbientIntensityGT']
dataIds = groundTruth['trainIds']

gtDtype = groundTruth.dtype

allDataIds = gtDataFile[gtPrefix]['trainIds']
if not os.path.isfile(experimentDir + 'train.npy'):
    generateExperiment(len(allDataIds), experimentDir, 0.9, 1)

# testSamplesIds= [2,4]
# trainSamplesIds = [0,14,20,25,26,1]
#
# dataIdx = np.arange(shapeGT[0])
# train = np.array([],dtype=np.uint16)
# test = np.array([],dtype=np.uint16)
# for testId in testSamplesIds:
#     test = np.append(test, np.where(dataTeapotIds == testId))
#
# for trainId in trainSamplesIds:
#     train = np.append(train, np.where(dataTeapotIds == trainId))
#
# # boolTrainSet = np.ones(shapeGT[0]).astype(np.bool)
# # boolTrainSet[test] = False
# # train = dataIdx[boolTrainSet]
#
# np.random.shuffle(train)
# np.random.shuffle(test)
#
# if not os.path.exists(experimentDir):
#     os.makedirs(experimentDir)
#
# np.save(experimentDir + 'train.npy', train)
# np.save(experimentDir + 'test.npy', test)

with open(experimentDir + 'description.txt', 'w') as expfile:
    expfile.write(experimentDescr)


