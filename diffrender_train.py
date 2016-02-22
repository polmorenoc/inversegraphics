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

gtPrefix = 'train4_occlusion_shapemodel'
experimentPrefix = 'train4_occlusion_shapemodel_10k'
gtDir = 'groundtruth/' + gtPrefix + '/'
experimentDir = 'experiments/' + experimentPrefix + '/'

groundTruthFilename = gtDir + 'groundTruth.h5'
gtDataFile = h5py.File(groundTruthFilename, 'r')

allDataIds = gtDataFile[gtPrefix]['trainIds']

onlySynthetic = True

# trainSet = np.load(experimentDir + 'train.npy')[:12800]
trainSet = np.load(experimentDir + 'train.npy')[:]


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
dataVColorGT = groundTruth['trainVColorGT']
# dataScenes = groundTruth['trainScenes']
dataTeapotIds = groundTruth['trainTeapotIds']
# dataEnvMaps = groundTruth['trainEnvMaps']
dataOcclusions = groundTruth['trainOcclusions']
# dataTargetIndices = groundTruth['trainTargetIndices']
# dataLightCoefficientsGT = groundTruth['trainLightCoefficientsGT']
dataLightCoefficientsGTRel = groundTruth['trainLightCoefficientsGTRel']
dataAmbientIntensityGT = groundTruth['trainAmbientIntensityGT']
dataShapeModelCoeffsGT = groundTruth['trainShapeModelCoeffsGT']
dataIds = groundTruth['trainIds']

gtDtype = groundTruth.dtype
# gtDtype = [('trainIds', trainIds.dtype.name), ('trainAzsGT', trainAzsGT.dtype.name),('trainObjAzsGT', trainObjAzsGT.dtype.name),('trainElevsGT', trainElevsGT.dtype.name),('trainLightAzsGT', trainLightAzsGT.dtype.name),('trainLightElevsGT', trainLightElevsGT.dtype.name),('trainLightIntensitiesGT', trainLightIntensitiesGT.dtype.name),('trainVColorGT', trainVColorGT.dtype.name, (3,) ),('trainScenes', trainScenes.dtype.name),('trainTeapotIds', trainTeapotIds.dtype.name),('trainEnvMaps', trainEnvMaps.dtype.name),('trainOcclusions', trainOcclusions.dtype.name),('trainTargetIndices', trainTargetIndices.dtype.name), ('trainComponentsGT', trainComponentsGT.dtype, (9,)),('trainComponentsGTRel', trainComponentsGTRel.dtype, (9,))]

# Create experiment simple sepratation.
#
trainAzsRel = np.mod(dataAzsGT - dataObjAzsGT, 2*np.pi)
trainElevsGT = dataElevsGT
# trainComponentsGTRel = dataComponentsGTRel
trainVColorGT = dataVColorGT

loadFromHdf5 = False

print("Reading images.")

if onlySynthetic:
    imagesDir = gtDir + 'images_opendr/'
else:
    imagesDir = gtDir + 'images/'

loadGray = False
imagesAreH5 = False
loadGrayFromHdf5 = False

filter = np.ones(len(trainSet)).astype(np.bool)
filter = np.array(tuple(dataOcclusions > 0) and tuple(dataOcclusions < 0.9))
# filter = dataOcclusions < 0.3
trainSet = trainSet[filter]

trainLightCoefficientsGTRel = dataLightCoefficientsGTRel[filter]
trainShapeModelCoeffsGT = dataShapeModelCoeffsGT[filter]
trainAzsRel=trainAzsRel[filter]
trainElevsGT =trainElevsGT[filter]
trainVColorGT=trainVColorGT[filter]
trainAmbientIntensityGT = dataAmbientIntensityGT[filter]

grayImages = readImages(imagesDir, trainSet, True, loadGrayFromHdf5)

# images = readImages(imagesDir, trainSet, False, False)

# grayImages = 0.3*images[:,:,:,0] + 0.59*images[:,:,:,1] + 0.11*images[:,:,:,2]

loadMask = False
gtDirMask = 'groundtruth/train4_occlusion_mask/'
masksDir =  gtDirMask + 'masks_occlusion/'
if loadMask:
    masksGT = loadMasks(masksDir, trainSet)

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
parameterTrainSet = set(['appearanceAndLightNN'])
parameterTrainSet = set(['appearanceNN'])
parameterTrainSet = set(['neuralNetModelLight'])
parameterTrainSet = set(['neuralNetModelShape'])
parameterTrainSet = set(['poseNN'])

# parameterTrainSet = set(['appearanceNN'])
# parameterTrainSet = set(['poseNN'])

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

    fineTune = False

    pretrainedExperimentDir =  'experiments/train3_test/'
    if fineTune:
        pretrainedModelFile = pretrainedExperimentDir + 'neuralNetModelPose.pickle'
        with open(pretrainedModelFile, 'rb') as pfile:
            neuralNetModelPose = pickle.load(pfile)

        meanImage = neuralNetModelPose['mean']
        # ipdb.set_trace()
        modelType = neuralNetModelPose['type']
        param_values = neuralNetModelPose['params']
    else:
        meanImage = np.mean(grayImages, axis=0)


    modelPath=experimentDir + 'neuralNetModelPose2.pickle'

    poseGT = np.hstack([np.cos(trainAzsRel)[:,None] , np.sin(trainAzsRel)[:,None], np.cos(trainElevsGT)[:,None], np.sin(trainElevsGT)[:,None]])

    poseNNmodel = lasagne_nn.train_nn_h5(grayImages.reshape([grayImages.shape[0],1,grayImages.shape[1],grayImages.shape[2]]), len(trainValSet), poseGT[trainValSet].astype(np.float32), poseGT[validSet].astype(np.float32), meanImage=meanImage, network=network, modelType=modelType, num_epochs=150, saveModelAtEpoch=True, modelPath=modelPath, param_values=param_values)
    # poseNNmodel = lasagne_nn.train_nn(grayImages, trainSet, validSet, len(trainValSet), poseGT[trainValSet].astype(np.float32), poseGT[validSet].astype(np.float32), meanImage=meanImage, network=network, modelType=modelType, num_epochs=10, saveModelAtEpoch=True, modelPath=modelPath, param_values=param_values)

    # np.savez(modelPath, *SHNNparams)
    with open(modelPath, 'wb') as pfile:
        pickle.dump(poseNNmodel, pfile)

if 'poseNNColor' in parameterTrainSet:
    modelType = 'cnn_pose_color'
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

    fineTune = False

    pretrainedExperimentDir =  'experiments/train3_test/'
    if fineTune:
        pretrainedModelFile = pretrainedExperimentDir + 'neuralNetModelPoseColor.pickle'
        with open(pretrainedModelFile, 'rb') as pfile:
            neuralNetModelPose = pickle.load(pfile)

        meanImage = neuralNetModelPose['mean']
        # ipdb.set_trace()
        modelType = neuralNetModelPose['type']
        param_values = neuralNetModelPose['params']
    else:
        meanImage = np.mean(images, axis=0)

    modelPath=experimentDir + 'neuralNetModelPoseColor.pickle'

    poseGT = np.hstack([np.cos(trainAzsRel)[:,None] , np.sin(trainAzsRel)[:,None], np.cos(trainElevsGT)[:,None], np.sin(trainElevsGT)[:,None]])

    poseNNmodel = lasagne_nn.train_nn_h5(images.reshape([images.shape[0],3,images.shape[1],images.shape[2]]), len(trainValSet), poseGT[trainValSet].astype(np.float32), poseGT[validSet].astype(np.float32), meanImage=meanImage, network=network, modelType=modelType, num_epochs=150, saveModelAtEpoch=True, modelPath=modelPath, param_values=param_values)
    # poseNNmodel = lasagne_nn.train_nn(grayImages, trainSet, validSet, len(trainValSet), poseGT[trainValSet].astype(np.float32), poseGT[validSet].astype(np.float32), meanImage=meanImage, network=network, modelType=modelType, num_epochs=10, saveModelAtEpoch=True, modelPath=modelPath, param_values=param_values)

    # np.savez(modelPath, *SHNNparams)
    with open(modelPath, 'wb') as pfile:
        pickle.dump(poseNNmodel, pfile)

if 'appearanceAndLightNN' in parameterTrainSet:
    modelType = 'cnn_appLight'
    network = lasagne_nn.load_network(modelType=modelType, param_values=[])

    print("Training NN Appereance and Light Components")
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

    pretrainedExperimentDir =  'experiments/train4_occlusion_10k/'
    if fineTune:
        pretrainedModelFile = pretrainedExperimentDir + 'neuralNetModelAppLight.pickle'
        with open(pretrainedModelFile, 'rb') as pfile:
            neuralNetModelAppLight = pickle.load(pfile)

        meanImage = neuralNetModelAppLight['mean']
        # ipdb.set_trace()
        modelType = neuralNetModelAppLight['type']
        param_values = neuralNetModelAppLight['params']
    else:
        meanImage = np.mean(images, axis=0)

    modelPath=experimentDir + 'neuralNetModelAppLight.pickle'

    appLightGT = np.hstack([trainLightCoefficientsGTRel*trainAmbientIntensityGT[:,None] , trainVColorGT])

    appLightNNmodel = lasagne_nn.train_nn_h5(images.reshape([images.shape[0],3,images.shape[1],images.shape[2]]), len(trainValSet), appLightGT[trainValSet].astype(np.float32), appLightGT[validSet].astype(np.float32), meanImage=meanImage, network=network, modelType=modelType, num_epochs=150, saveModelAtEpoch=True, modelPath=modelPath, param_values=param_values)
    # poseNNmodel = lasagne_nn.train_nn(grayImages, trainSet, validSet, len(trainValSet), poseGT[trainValSet].astype(np.float32), poseGT[validSet].astype(np.float32), meanImage=meanImage, network=network, modelType=modelType, num_epochs=10, saveModelAtEpoch=True, modelPath=modelPath, param_values=param_values)

    # np.savez(modelPath, *SHNNparams)
    with open(modelPath, 'wb') as pfile:
        pickle.dump(appLightNNmodel, pfile)

if 'appearanceNN' in parameterTrainSet:
    modelType = 'cnn_app'
    network = lasagne_nn.load_network(modelType=modelType, param_values=[])

    print("Training NN Appeareance Components")
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

    fineTune = False

    pretrainedExperimentDir =  'experiments/train3_test/'

    if fineTune:
        pretrainedModelFile = pretrainedExperimentDir + 'neuralNetModelAppearance.pickle'
        with open(pretrainedModelFile, 'rb') as pfile:
            neuralNetModelAppearance = pickle.load(pfile)

        meanImage = neuralNetModelAppearance['mean']
        # ipdb.set_trace()
        modelType = neuralNetModelAppearance['type']
        param_values = neuralNetModelAppearance['params']
    else:
        meanImage = np.mean(images, axis=0)


    modelPath=experimentDir + 'neuralNetModelAppearance.pickle'

    appNNmodel = lasagne_nn.train_nn_h5(images.reshape([images.shape[0],3,images.shape[1],images.shape[2]]), len(trainValSet), trainVColorGT[trainValSet].astype(np.float32), trainVColorGT[validSet].astype(np.float32), meanImage=meanImage, network=network, modelType=modelType, num_epochs=150, saveModelAtEpoch=True, modelPath=modelPath, param_values=param_values)
    # poseNNmodel = lasagne_nn.train_nn(grayImages, trainSet, validSet, len(trainValSet), poseGT[trainValSet].astype(np.float32), poseGT[validSet].astype(np.float32), meanImage=meanImage, network=network, modelType=modelType, num_epochs=10, saveModelAtEpoch=True, modelPath=modelPath, param_values=param_values)

    # np.savez(modelPath, *SHNNparams)
    with open(modelPath, 'wb') as pfile:
        pickle.dump(appNNmodel, pfile)

if 'maskNN' in parameterTrainSet:
    modelType = 'cnn_mask_large'
    network = lasagne_nn.load_network(modelType=modelType, param_values=[])

    print("Training NN Appeareance Components")
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

    pretrainedExperimentDir =  experimentDir
    if fineTune:
        pretrainedModelFile = pretrainedExperimentDir + 'neuralNetModelMaskLarge.pickle'
        with open(pretrainedModelFile, 'rb') as pfile:
            neuralNetModelMask = pickle.load(pfile)

        meanImage = neuralNetModelMask['mean']
        # ipdb.set_trace()
        modelType = neuralNetModelMask['type']
        param_values = neuralNetModelMask['params']
    else:
        meanImage = np.mean(images, axis=0)

    modelPath=experimentDir + 'neuralNetModelMaskLarge.pickle'

    # masksGT = masksGT.reshape([-1, 150,150])

    rsMasksGt = np.zeros([len(masksGT), 50,50])
    for mask_i, mask in enumerate(masksGT):
        rsMasksGt[mask_i] = skimage.transform.resize(mask, [50,50])

    meanImage = np.mean(images, axis=0)
    rsMasksGt = rsMasksGt.reshape([rsMasksGt.shape[0], 50*50])

    maskNNmodel = lasagne_nn.train_nn_h5(images.reshape([images.shape[0],3,images.shape[1],images.shape[2]]), len(trainValSet), rsMasksGt[trainValSet].astype(np.float32), rsMasksGt[validSet].astype(np.float32), meanImage=meanImage, network=network, modelType=modelType, num_epochs=150, saveModelAtEpoch=True, modelPath=modelPath, param_values=param_values)
    # poseNNmodel = lasagne_nn.train_nn(grayImages, trainSet, validSet, len(trainValSet), poseGT[trainValSet].astype(np.float32), poseGT[validSet].astype(np.float32), meanImage=meanImage, network=network, modelType=modelType, num_epochs=10, saveModelAtEpoch=True, modelPath=modelPath, param_values=param_values)

    # np.savez(modelPath, *SHNNparams)
    with open(modelPath, 'wb') as pfile:
        pickle.dump(maskNNmodel, pfile)

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

if 'neuralNetModelLight' in parameterTrainSet:

    modelType = 'cnn_light'
    network = lasagne_nn.load_network(modelType=modelType, param_values=[])

    print("Training NN SH Components")
    validRatio = 0.9
    trainValSet = np.arange(len(trainSet))[:np.uint(len(trainSet)*validRatio)]
    validSet = np.arange(len(trainSet))[np.uint(len(trainSet)*validRatio)::]
    # modelPath = experimentDir + 'neuralNetModelRelSHComponents.npz'

    param_values = []

    fineTune = False
    pretrainedExperimentDir =  'experiments/train3_test/'
    if fineTune:
        pretrainedModelFile = pretrainedExperimentDir + 'neuralNetModelLight.pickle'
        with open(pretrainedModelFile, 'rb') as pfile:
            neuralNetModelLight = pickle.load(pfile)

        meanImage = neuralNetModelLight['mean']
        # ipdb.set_trace()
        modelType = neuralNetModelLight['type']
        param_values = neuralNetModelLight['params']
    else:
        meanImage = np.mean(images, axis=0)

    lightGT = trainLightCoefficientsGTRel*trainAmbientIntensityGT[:,None]

    modelPath=experimentDir + 'neuralNetModelLight.pickle'
    lightNNmodel = lasagne_nn.train_nn_h5(images.reshape([images.shape[0],3,images.shape[1],images.shape[2]]), len(trainValSet), lightGT[trainValSet].astype(np.float32), lightGT[validSet].astype(np.float32), meanImage=meanImage, network=network, modelType=modelType, num_epochs=150, saveModelAtEpoch=True, modelPath=modelPath, param_values=param_values)
    # np.savez(modelPath, *SHNNparams)
    with open(modelPath, 'wb') as pfile:
        pickle.dump(lightNNmodel, pfile)

if 'neuralNetModelShape' in parameterTrainSet:

    modelType = 'cnn_shape'
    network = lasagne_nn.load_network(modelType=modelType, param_values=[])

    print("Training NN Shape Components")
    validRatio = 0.9
    trainValSet = np.arange(len(trainSet))[:np.uint(len(trainSet)*validRatio)]
    validSet = np.arange(len(trainSet))[np.uint(len(trainSet)*validRatio)::]
    # modelPath = experimentDir + 'neuralNetModelRelSHComponents.npz'

    param_values = []

    fineTune = True
    pretrainedExperimentDir =  'experiments/train4_occlusion_shapemodel_10k/'
    if fineTune:
        pretrainedModelFile = pretrainedExperimentDir + 'neuralNetModelShape.pickle'
        with open(pretrainedModelFile, 'rb') as pfile:
            neuralNetModelShape = pickle.load(pfile)

        meanImage = neuralNetModelShape['mean']
        # ipdb.set_trace()
        modelType = neuralNetModelShape['type']
        param_values = neuralNetModelShape['params']
    else:
        meanImage = np.mean(grayImages, axis=0)

    modelPath=experimentDir + 'neuralNetModelShape.pickle'
    shapeNNmodel = lasagne_nn.train_nn_h5(grayImages.reshape([grayImages.shape[0],1,grayImages.shape[1],grayImages.shape[2]]), len(trainValSet), trainShapeModelCoeffsGT[trainValSet].astype(np.float32), trainShapeModelCoeffsGT[validSet].astype(np.float32), meanImage=meanImage, network=network, modelType=modelType, num_epochs=150, saveModelAtEpoch=True, modelPath=modelPath, param_values=param_values)
    # np.savez(modelPath, *SHNNparams)
    with open(modelPath, 'wb') as pfile:
        pickle.dump(shapeNNmodel, pfile)


if 'spherical_harmonicsZernike' in parameterTrainSet:
    print("Training on Zernike features")

    # linRegModelZernikeSH = recognition_models.trainLinearRegression(trainZernikeCoeffs,dataLightCoefficientsGTRel * dataAmbientIntensityGT[:,None])
    # with open(experimentDir + 'linRegModelZernike' + str(numCoeffs) +'_win' + str(win) + '.pickle', 'wb') as pfile:
    #     pickle.dump(linRegModelZernikeSH, pfile)
    trainZernikeCoeffs[trainZernikeCoeffs >= 1000] = 35
    randForestModelRelZernikeSH = recognition_models.trainRandomForest(trainZernikeCoeffs, trainLightCoefficientsGTRel * trainAmbientIntensityGT[:,None])
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