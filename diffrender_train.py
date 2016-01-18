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
import lasagne
import theano

#########################################
# Initialization ends here
#########################################

seed = 1
np.random.seed(seed)

gtPrefix = 'train4_occlusion'
experimentPrefix = 'train4_occlusion_10k'
gtDir = 'groundtruth/' + gtPrefix + '/'
experimentDir = 'experiments/' + experimentPrefix + '/'

groundTruthFilename = gtDir + 'groundTruth.h5'
gtDataFile = h5py.File(groundTruthFilename, 'r')

allDataIds = gtDataFile[gtPrefix]['trainIds']

onlySynthetic = True

print("Reading images.")

writeHdf5 = False
writeGray = False
if onlySynthetic:
    imagesDir = gtDir + 'images_opendr/'
else:
    imagesDir = gtDir + 'images/'

if writeHdf5:
    writeImagesHdf5(imagesDir, imagesDir, allDataIds, writeGray)

if onlySynthetic:
    imagesExpDir = experimentDir + 'opendr_'
else:
    imagesExpDir = experimentDir + ''


# trainSet = np.load(experimentDir + 'train.npy')[:12800]
trainSet = np.load(experimentDir + 'train.npy')[:15000]

writeExpHdf5 = False
if writeExpHdf5:
    writeImagesHdf5(imagesDir, imagesExpDir, trainSet, writeGray)

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
# dataLightAzsGT = groundTruth['trainLightAzsGT']
# dataLightElevsGT = groundTruth['trainLightElevsGT']
# dataLightIntensitiesGT = groundTruth['trainLightIntensitiesGT']
# dataVColorGT = groundTruth['trainVColorGT']
# dataScenes = groundTruth['trainScenes']
dataTeapotIds = groundTruth['trainTeapotIds']
# dataEnvMaps = groundTruth['trainEnvMaps']
# dataOcclusions = groundTruth['trainOcclusions']
# dataTargetIndices = groundTruth['trainTargetIndices']
# dataComponentsGT = groundTruth['trainComponentsGT']
# dataComponentsGTRel = groundTruth['trainComponentsGTRel']
# dataLightCoefficientsGT = groundTruth['trainLightCoefficientsGT']
# dataLightCoefficientsGTRel = groundTruth['trainLightCoefficientsGTRel']
# dataAmbientIntensityGT = groundTruth['trainAmbientIntensityGT']
dataIds = groundTruth['trainIds']

gtDtype = groundTruth.dtype
# gtDtype = [('trainIds', trainIds.dtype.name), ('trainAzsGT', trainAzsGT.dtype.name),('trainObjAzsGT', trainObjAzsGT.dtype.name),('trainElevsGT', trainElevsGT.dtype.name),('trainLightAzsGT', trainLightAzsGT.dtype.name),('trainLightElevsGT', trainLightElevsGT.dtype.name),('trainLightIntensitiesGT', trainLightIntensitiesGT.dtype.name),('trainVColorGT', trainVColorGT.dtype.name, (3,) ),('trainScenes', trainScenes.dtype.name),('trainTeapotIds', trainTeapotIds.dtype.name),('trainEnvMaps', trainEnvMaps.dtype.name),('trainOcclusions', trainOcclusions.dtype.name),('trainTargetIndices', trainTargetIndices.dtype.name), ('trainComponentsGT', trainComponentsGT.dtype, (9,)),('trainComponentsGTRel', trainComponentsGTRel.dtype, (9,))]

# Create experiment simple sepratation.
#
trainAzsRel = np.mod(dataAzsGT - dataObjAzsGT, 2*np.pi)
trainElevsGT = dataElevsGT
# trainComponentsGTRel = dataComponentsGTRel
# trainVColorGT = dataVColorGT

loadFromHdf5 = False

print("Reading images.")

# if loadFromHdf5:

images = readImages(imagesDir, trainSet, False, loadFromHdf5)

loadGray = True
imagesAreH5 = False
loadGrayFromHdf5 = False

# if not imagesAreH5:
#     grayImages = readImages(imagesDir, trainSet, loadGray, loadGrayFromHdf5)
# else:
#     grayImages = h5py.File(imagesExpDir + 'images_gray.h5', 'r')["images"]

grayImages = 0.3*images[:,:,:,0] + 0.59*images[:,:,:,1] + 0.11*images[:,:,:,2]

loadHogFeatures = False
loadFourierFeatures = False
loadZernikeFeatures = False

synthPrefix = '_cycles'
if onlySynthetic:
    synthPrefix = ''

# if loadHogFeatures:
#     hogfeatures = np.load(gtDir + 'hog' + synthPrefix + '.npy')
# else:
#     print("Extracting Hog features .")
#     hogfeatures = image_processing.computeHoGFeatures(images)
#     np.save(gtDir + 'hog' + synthPrefix + '.npy', hogfeatures)

# if loadIllumFeatures:
#     illumfeatures =  np.load(experimentDir  + 'illum.npy')
# else:
#     print("Extracting Illumination features (FFT.")
#     illumfeatures = image_processing.computeIllumFeatures(images, images[0].size/12)
#     np.save(experimentDir  + 'illum.npy', illumfeatures)

# print("Extracting Zernike features.")
#
# numCoeffs=200
# # numTrees = 400
# win=40
# if loadZernikeFeatures:
#     trainZernikeCoeffs =  np.load(gtDir  + 'zernike_numCoeffs' + str(numCoeffs) + '_win' + str(win) + '.npy')[trainSet]
# else:
#     print("Extracting Zernike features.")
#     batchSize = 1000
#
#     trainZernikeCoeffs = np.empty([images.shape[0], numCoeffs])
#     for batch in range(int(images.shape[0]/batchSize)):
#         trainZernikeCoeffs[batchSize*batch:batchSize*batch + batchSize] = image_processing.zernikeProjectionGray(images[batchSize*batch:batchSize*batch + batchSize], numCoeffs=numCoeffs, win=win)
#     np.save(gtDir  + 'zernike_numCoeffs' + str(numCoeffs) + '_win' + str(win) + synthPrefix + '.npy', trainZernikeCoeffs)
#     trainZernikeCoeffs = trainZernikeCoeffs[trainSet]

#
# trainHogfeatures = hogfeatures[trainSet]
# trainIllumfeatures = illumfeatures[trainSet]

parameterTrainSet = set(['azimuthsRF', 'elevationsRF', 'vcolorsRF'])
# parameterTrainSet = set(['vcolorsRF', 'spherical_harmonicsRF'])
# parameterTrainSet = set(['spherical_harmonicsZernike'])
# parameterTrainSet = set(['spherical_harmonicsZernike'])
# parameterTrainSet = set(['spherical_harmonicsNN'])
parameterTrainSet = set(['poseNN'])

print("Training recognition models.")

if 'poseNN' in parameterTrainSet:
    modelType = 'cnn_pose'
    network = lasagne_nn.load_network(modelType=modelType, param_values=[])

    print("Training NN Pose Components")
    validRatio = 0.9

    trainValSet = np.arange(len(trainSet))[:np.uint(len(trainSet)*validRatio)]
    validSet = np.arange(len(trainSet))[np.uint(len(trainSet)*validRatio)::]
    # modelPath = experimentDir + 'neuralNetModelRelSHComponents.npz'

    # grayTrainImages =  grayImages[trainValSet][:,:,:]
    # grayValidImages =  grayImages[validSet][:,:,:]
    # grayTrainImages = grayTrainImages[:,None, :,:]
    # grayValidImages = grayValidImages[:,None, :,:]
    # import sys
    # sys.exit("NN")
    param_values = []

    fineTune = True

    pretrainedExperimentDir =  'experiments/train3_test/'
    if fineTune:
        pretrainedModelFile = pretrainedExperimentDir + 'neuralNetModelPoseCuda.pickle'
        with open(pretrainedModelFile, 'rb') as pfile:
            neuralNetModelPose = pickle.load(pfile)

        meanImage = neuralNetModelPose['mean']
        # ipdb.set_trace()
        modelType = neuralNetModelPose['type']
        param_values = neuralNetModelPose['params']
    else:
        meanImage = np.zeros([150, 150])

    modelPath=experimentDir + 'neuralNetModelPose.pickle'

    poseGT = np.hstack([np.cos(trainAzsRel)[:,None] , np.sin(trainAzsRel)[:,None], np.cos(trainElevsGT)[:,None], np.sin(trainElevsGT)[:,None]])

    poseNNmodel = lasagne_nn.train_nn_h5(grayImages, len(trainValSet), poseGT[trainValSet].astype(np.float32), poseGT[validSet].astype(np.float32), meanImage=meanImage, network=network, modelType=modelType, num_epochs=100, saveModelAtEpoch=True, modelPath=modelPath, param_values=param_values)
    # poseNNmodel = lasagne_nn.train_nn(grayImages, trainSet, validSet, len(trainValSet), poseGT[trainValSet].astype(np.float32), poseGT[validSet].astype(np.float32), meanImage=meanImage, network=network, modelType=modelType, num_epochs=10, saveModelAtEpoch=True, modelPath=modelPath, param_values=param_values)

    # np.savez(modelPath, *SHNNparams)
    with open(modelPath, 'wb') as pfile:
        pickle.dump(poseNNmodel, pfile)

if 'azimuthsRF' in parameterTrainSet:
    print("Training RFs Cos Azs")
    randForestModelCosAzs = recognition_models.trainRandomForest(trainHogfeatures, np.cos(trainAzsRel))
    trainedModel = {'randForestModelCosAzs':randForestModelCosAzs}
    with open(experimentDir + 'randForestModelCosAzs05.pickle', 'wb') as pfile:
        pickle.dump(trainedModel, pfile)

    print("Training RFs Sin Azs")
    randForestModelSinAzs = recognition_models.trainRandomForest(trainHogfeatures, np.sin(trainAzsRel))
    trainedModel = {'randForestModelSinAzs':randForestModelSinAzs}
    with open(experimentDir + 'randForestModelSinAzs05.pickle', 'wb') as pfile:
        pickle.dump(trainedModel, pfile)

if 'elevationsRF' in parameterTrainSet:
    print("Training RFs Cos Elevs")
    randForestModelCosElevs = recognition_models.trainRandomForest(trainHogfeatures, np.cos(trainElevsGT))
    trainedModel = {'randForestModelCosElevs':randForestModelCosElevs}
    with open(experimentDir + 'randForestModelCosElevs05.pickle', 'wb') as pfile:
        pickle.dump(trainedModel, pfile)

    print("Training RFs Sin Elevs")
    randForestModelSinElevs = recognition_models.trainRandomForest(trainHogfeatures, np.sin(trainElevsGT))
    trainedModel = {'randForestModelSinElevs':randForestModelSinElevs}
    with open(experimentDir + 'randForestModelSinElevs05.pickle', 'wb') as pfile:
        pickle.dump(trainedModel, pfile)

if 'spherical_harmonicsNN' in parameterTrainSet or 'spherical_harmonicsNN2' in parameterTrainSet:
    print("Training NN SH Components")
    validRatio = 0.9
    trainValSet = np.arange(len(trainSet))[:np.uint(len(trainSet)*validRatio)]
    validSet = np.arange(len(trainSet))[np.uint(len(trainSet)*validRatio)::]
    # modelPath = experimentDir + 'neuralNetModelRelSHComponents.npz'

    grayTrainImages =  0.3*images[trainValSet][:,:,:,0] + 0.59*images[trainValSet][:,:,:,1] + 0.11*images[trainValSet][:,:,:,2]
    grayValidImages =  0.3*images[validSet][:,:,:,0] + 0.59*images[validSet][:,:,:,1] + 0.11*images[validSet][:,:,:,2]
    grayTrainImages = grayTrainImages[:,None, :,:]
    grayValidImages = grayValidImages[:,None, :,:]
    # import sys
    # sys.exit("NN")

    if 'spherical_harmonicsNN' in parameterTrainSet:
        modelPath=experimentDir + 'neuralNetModelRelSHLight.pickle'
        SHNNmodel = lasagne_nn.train_nn(grayTrainImages, dataLightCoefficientsGTRel[trainValSet].astype(np.float32) * dataAmbientIntensityGT[trainValSet][:,None].astype(np.float32), grayValidImages, dataLightCoefficientsGTRel[validSet].astype(np.float32) * dataAmbientIntensityGT[validSet][:,None].astype(np.float32), modelType='cnn', num_epochs=200, saveModelAtEpoch=True, modelPath=modelPath)
        # np.savez(modelPath, *SHNNparams)
        with open(modelPath, 'wb') as pfile:
            pickle.dump(SHNNmodel, pfile)

    if 'spherical_harmonicsNN2' in parameterTrainSet:
        modelPath=experimentDir + 'neuralNetModelRelSHLight2.pickle'
        SHNNmodel = lasagne_nn.train_nn(grayTrainImages, dataLightCoefficientsGTRel[trainValSet].astype(np.float32) * dataAmbientIntensityGT[trainValSet][:,None].astype(np.float32),grayValidImages, dataLightCoefficientsGTRel[validSet].astype(np.float32) * dataAmbientIntensityGT[validSet][:,None].astype(np.float32), modelType='cnn2', num_epochs=200, saveModelAtEpoch=True, modelPath=modelPath)
        with open(modelPath, 'wb') as pfile:
            pickle.dump(SHNNmodel, pfile)


if 'spherical_harmonicsZernike' in parameterTrainSet:
    print("Training on Zernike features")

    # linRegModelZernikeSH = recognition_models.trainLinearRegression(trainZernikeCoeffs,dataLightCoefficientsGTRel * dataAmbientIntensityGT[:,None])
    # with open(experimentDir + 'linRegModelZernike' + str(numCoeffs) +'_win' + str(win) + '.pickle', 'wb') as pfile:
    #     pickle.dump(linRegModelZernikeSH, pfile)
    trainZernikeCoeffs[trainZernikeCoeffs >= 1000] = 35
    randForestModelRelZernikeSH = recognition_models.trainRandomForest(trainZernikeCoeffs, dataLightCoefficientsGTRel * dataAmbientIntensityGT[:,None])
    with open(experimentDir + 'randomForestModelZernike400' + str(numCoeffs) + '_win' + str(win) + '.pickle', 'wb') as pfile:
        pickle.dump(randForestModelRelZernikeSH, pfile)


if 'vcolorsRF' in parameterTrainSet:
    print("Training RF on Vertex Colors")
    numTrainSet = images.shape[0]
    colorWindow = 30
    image = images[0]
    croppedImages = images[:,image.shape[0]/2-colorWindow:image.shape[0]/2+colorWindow,image.shape[1]/2-colorWindow:image.shape[1]/2+colorWindow,:]
    randForestModelVColor = recognition_models.trainRandomForest(croppedImages.reshape([numTrainSet,-1]), trainVColorGT)
    with open(experimentDir + 'randForestModelVColor05.pickle', 'wb') as pfile:
        pickle.dump(randForestModelVColor, pfile)


if  'vcolorsLR' in parameterTrainSet:
    print("Training LR on Vertex Colors")
    numTrainSet = images.shape[0]
    colorWindow = 30
    image = images[0]
    croppedImages = images[:,image.shape[0]/2-colorWindow:image.shape[0]/2+colorWindow,image.shape[1]/2-colorWindow:image.shape[1]/2+colorWindow,:]
    linRegModelVColor = recognition_models.trainLinearRegression(croppedImages.reshape([numTrainSet,-1]), trainVColorGT)
    with open(experimentDir + 'linearRegressionModelVColor.pickle', 'wb') as pfile:
        pickle.dump(linRegModelVColor, pfile)
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
