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
import lasagne_nn
import theano

#########################################
# Initialization ends here
#########################################

seed = 1
np.random.seed(seed)

gtPrefix = 'train5'
experimentPrefix = 'train5_out_normal'
experimentDescr = 'Out ot sample test set: Normal (2,4), Tall (5,6), Misc (8, 21,22)'
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
# if not os.path.isfile(experimentDir + 'train.npy'):
#     generateExperiment(len(allDataIds), experimentDir, 0.9, 1)

testSamplesIds= [2,4]
trainSamplesIds = [0,14,20,25,26,1]

dataIdx = np.arange(shapeGT[0])
train = np.array([],dtype=np.uint16)
test = np.array([],dtype=np.uint16)
for testId in testSamplesIds:
    test = np.append(test, np.where(dataTeapotIds == testId))

for trainId in trainSamplesIds:
    train = np.append(train, np.where(dataTeapotIds == trainId))

# boolTrainSet = np.ones(shapeGT[0]).astype(np.bool)
# boolTrainSet[test] = False
# train = dataIdx[boolTrainSet]

np.random.shuffle(train)
np.random.shuffle(test)

if not os.path.exists(experimentDir):
    os.makedirs(experimentDir)

with open(experimentDir + 'description.txt', 'w') as expfile:
    expfile.write(experimentDescr)

np.save(experimentDir + 'train.npy', train)
np.save(experimentDir + 'test.npy', test)

ipdb.set_trace()