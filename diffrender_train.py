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

gtPrefix = 'train2'
trainPrefix = 'train2'
gtDir = 'groundtruth/' + gtPrefix + '/'
experimentDir = 'experiments/' + trainPrefix + '/'

groundTruthFilename = gtDir + 'groundTruth.h5'
gtDataFile = h5py.File(groundTruthFilename, 'r')

allDataIds = gtDataFile[gtPrefix]['trainIds']
if not os.path.isfile(experimentDir + 'train.npy'):
    generateExperiment(len(allDataIds), experimentDir, 0.9, 1)

print("Reading images.")

writeHdf5 = False
imagesDir = gtDir + 'images/'
if writeHdf5:
    writeImagesHdf5(imagesDir, allDataIds)

trainSet = np.load(experimentDir + 'train.npy')

#Delete as soon as finished with prototyping:
trainSet = trainSet

print("Reading experiment data.")

shapeGT = gtDataFile[gtPrefix].shape
boolTestSet = np.zeros(shapeGT).astype(np.bool)
boolTestSet[trainSet] = True
trainGroundTruth = gtDataFile[gtPrefix][boolTestSet]
groundTruth = np.zeros(shapeGT, dtype=trainGroundTruth.dtype)
groundTruth[boolTestSet] = trainGroundTruth
groundTruth = groundTruth[trainSet]

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
# gtDtype = [('trainIds', trainIds.dtype.name), ('trainAzsGT', trainAzsGT.dtype.name),('trainObjAzsGT', trainObjAzsGT.dtype.name),('trainElevsGT', trainElevsGT.dtype.name),('trainLightAzsGT', trainLightAzsGT.dtype.name),('trainLightElevsGT', trainLightElevsGT.dtype.name),('trainLightIntensitiesGT', trainLightIntensitiesGT.dtype.name),('trainVColorGT', trainVColorGT.dtype.name, (3,) ),('trainScenes', trainScenes.dtype.name),('trainTeapotIds', trainTeapotIds.dtype.name),('trainEnvMaps', trainEnvMaps.dtype.name),('trainOcclusions', trainOcclusions.dtype.name),('trainTargetIndices', trainTargetIndices.dtype.name), ('trainComponentsGT', trainComponentsGT.dtype, (9,)),('trainComponentsGTRel', trainComponentsGTRel.dtype, (9,))]

# Create experiment simple sepratation.
#
trainAzsRel = np.mod(dataAzsGT - dataObjAzsGT, 2*np.pi)
trainElevsGT = dataElevsGT
trainComponentsGTRel = dataComponentsGTRel
trainVColorGT = dataVColorGT

loadFromHdf5 = True

if loadFromHdf5:
    images = readImages(imagesDir, allDataIds, loadFromHdf5)

loadHogFeatures = False
loadIllumFeatures = False

# if loadHogFeatures:
#     hogfeatures = np.load(experimentDir + 'hog.npy')
# else:
#     print("Extracting Hog features .")
#     hogfeatures = image_processing.computeHoGFeatures(images)
#     np.save(experimentDir + 'hog.npy', hogfeatures)
#
# if loadIllumFeatures:
#     illumfeatures =  np.load(experimentDir  + 'illum.npy')
# else:
#     print("Extracting Illumination features (FFT.")
#     illumfeatures = image_processing.computeIllumFeatures(images, images[0].size/12)
#     np.save(experimentDir  + 'illum.npy', illumfeatures)
#
# trainHogfeatures = hogfeatures[trainSet]
# trainIllumfeatures = illumfeatures[trainSet]

parameterTrainSet = set(['azimuthsRF', 'elevationsRF', 'vcolorsRF', 'spherical_harmonicsNN'])
# parameterTrainSet = set(['vcolorsRF', 'spherical_harmonicsRF'])
parameterTrainSet = set(['spherical_harmonicsNN'])

print("Training recognition models.")

if 'azimuthsRF' in parameterTrainSet:
    print("Training RFs Cos Azs")
    randForestModelCosAzs = recognition_models.trainRandomForest(trainHogfeatures, np.cos(trainAzsRel))
    trainedModel = {'randForestModelCosAzs':randForestModelCosAzs}
    with open(experimentDir + 'randForestModelCosAzs.pickle', 'wb') as pfile:
        pickle.dump(trainedModel, pfile)

    print("Training RFs Sin Azs")
    randForestModelSinAzs = recognition_models.trainRandomForest(trainHogfeatures, np.sin(trainAzsRel))
    trainedModel = {'randForestModelSinAzs':randForestModelSinAzs}
    with open(experimentDir + 'randForestModelSinAzs.pickle', 'wb') as pfile:
        pickle.dump(trainedModel, pfile)

if 'elevationsRF' in parameterTrainSet:
    print("Training RFs Cos Elevs")
    randForestModelCosElevs = recognition_models.trainRandomForest(trainHogfeatures, np.cos(trainElevsGT))
    trainedModel = {'randForestModelCosElevs':randForestModelCosElevs}
    with open(experimentDir + 'randForestModelCosElevs.pickle', 'wb') as pfile:
        pickle.dump(trainedModel, pfile)

    print("Training RFs Sin Elevs")
    randForestModelSinElevs = recognition_models.trainRandomForest(trainHogfeatures, np.sin(trainElevsGT))
    trainedModel = {'randForestModelSinElevs':randForestModelSinElevs}
    with open(experimentDir + 'randForestModelSinElevs.pickle', 'wb') as pfile:
        pickle.dump(trainedModel, pfile)

if 'spherical_harmonicsRF' in parameterTrainSet:
    print("Training RF SH Components")
    randForestModelRelSHComponents = recognition_models.trainRandomForest(trainIllumfeatures, trainComponentsGTRel)
    with open(experimentDir + 'randForestModelRelSHComponents.pickle', 'wb') as pfile:
        pickle.dump(randForestModelRelSHComponents, pfile)

elif 'spherical_harmonicsNN' in parameterTrainSet:
    print("Training NN SH Components")
    validRatio = 0.9
    trainValSet = np.arange(len(trainSet))[:np.uint(len(trainSet)*validRatio)]
    validSet = np.arange(len(trainSet))[np.uint(len(trainSet)*validRatio)::]
    # modelPath = experimentDir + 'neuralNetModelRelSHComponents.npz'

    grayTrainImages =  0.3*images[trainSet[trainValSet]][:,:,:,0] +  0.59*images[trainSet[trainValSet]][:,:,:,1] + 0.11*images[trainSet[trainValSet]][:,:,:,2]
    grayValidImages =  0.3*images[trainSet[validSet]][:,:,:,0] +  0.59*images[trainSet[validSet]][:,:,:,1] + 0.11*images[trainSet[validSet]][:,:,:,2]
    grayTrainImages = grayTrainImages[:,None, :,:]
    grayValidImages = grayValidImages[:,None, :,:]
    # import sys
    # sys.exit("NN")
    modelPath=experimentDir + 'neuralNetModelRelSHLight.pickle'

    SHNNmodel = lasagne_nn.train_nn(grayTrainImages, dataLightCoefficientsGTRel[trainValSet] * dataAmbientIntensityGT[trainValSet].astype(np.float32), grayValidImages, dataLightCoefficientsGTRel[validSet] * dataAmbientIntensityGT[validSet].astype(np.float32), modelType='cnn', num_epochs=500, saveModelAtEpoch=True, modelPath=modelPath)
    # np.savez(modelPath, *SHNNparams)
    with open(modelPath, 'wb') as pfile:
        pickle.dump(SHNNmodel, pfile)

if 'vcolorsRF' in parameterTrainSet:
    print("Training RF on Vertex Colors")
    numTrainSet = images[trainSet].shape[0]
    colorWindow = 30
    image = images[0]
    croppedImages = images[:,image.shape[0]/2-colorWindow:image.shape[0]/2+colorWindow,image.shape[1]/2-colorWindow:image.shape[1]/2+colorWindow,:][:,3]
    randForestModelVColor = recognition_models.trainRandomForest(croppedImages[trainSet].reshape([numTrainSet,-1]), trainVColorGT)
    with open(experimentDir + 'randForestModelVColor.pickle', 'wb') as pfile:
        pickle.dump(randForestModelVColor, pfile)

#
# imagesStack = np.vstack([image.reshape([1,-1]) for image in images])
# randForestModelLightIntensity = recognition_models.trainRandomForest(imagesStack, trainLightIntensitiesGT)
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
