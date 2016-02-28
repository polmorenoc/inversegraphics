__author__ = 'pol'

import matplotlib
matplotlib.use('Qt4Agg')
import scene_io_utils
from math import radians
import timeit
import time
import opendr
import chumpy as ch
import geometry
import image_processing
import numpy as np
import cv2
import generative_models
import recognition_models
import matplotlib.pyplot as plt
from opendr_utils import *
from utils import *

seed = 1
np.random.seed(seed)

# testPrefix = 'train4_occlusion_opt_train4occlusion10k_100s_dropoutsamples_std01_nnsampling_minSH'
testPrefix = 'train4_occlusion_shapemodel_10k_background_type7_vcolorFirstNoLight5_noPose_all_100samples__method1errorFun1_std0.05_shapePen0'


gtPrefix = 'train4_occlusion_shapemodel'
experimentPrefix = 'train4_occlusion_shapemodel_10k'

gtDir = 'groundtruth/' + gtPrefix + '/'
featuresDir = gtDir

experimentDir = 'experiments/' + experimentPrefix + '/'

resultDir = 'results/' + testPrefix + '/'

ignore = []
if os.path.isfile(gtDir + 'ignore.npy'):
    ignore = np.load(gtDir + 'ignore.npy')

groundTruthFilename = gtDir + 'groundTruth.h5'
gtDataFile = h5py.File(groundTruthFilename, 'r')

numTests = 100
testSet = np.load(experimentDir + 'test.npy')[:numTests]
# testSet = np.load(experimentDir + 'test.npy')[[ 3,  5, 14, 21, 35, 36, 54, 56, 59, 60, 68, 70, 72, 79, 83, 85, 89,94]]
# [13:14]

#Bad samples for data set train5 out normal.
# testSet = np.array([162371, 410278, 132297, 350815, 104618, 330181,  85295,  95047,
#        410233, 393785, 228626, 452094, 117242,  69433,  35352,  31030,
#        268444, 147111, 117287, 268145, 478618, 334784])

shapeGT = gtDataFile[gtPrefix].shape
boolTestSet = np.zeros(shapeGT).astype(np.bool)
boolTestSet[testSet] = True
testGroundTruth = gtDataFile[gtPrefix][boolTestSet]
groundTruthTest = np.zeros(shapeGT, dtype=testGroundTruth.dtype)
groundTruthTest[boolTestSet] = testGroundTruth
groundTruth = groundTruthTest[testSet]
dataTeapotIdsTest = groundTruth['trainTeapotIds']
test = np.arange(len(testSet))

# testSamplesIds= [2]
# test = np.array([],dtype=np.uint16)
# for testId in testSamplesIds:
#     test = np.append(test, np.where(dataTeapotIdsTest == testId))
#
# groundTruth = groundTruth[test]
#

testSet = testSet[test]

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
dataLightCoefficientsGT = groundTruth['trainLightCoefficientsGT']
dataLightCoefficientsGTRel = groundTruth['trainLightCoefficientsGTRel']
dataAmbientIntensityGT = groundTruth['trainAmbientIntensityGT']
dataIds = groundTruth['trainIds']

gtDtype = groundTruth.dtype

# testSet = np.array([ 11230, 3235, 10711,  9775, 11230, 10255,  5060, 12784,  5410,  1341,14448, 12935, 13196,  6728,  9002,  7946,  1119,  5827,  4842,12435,  8152,  4745,  9512,  9641,  7165, 13950,  3567,   860,4105, 10330,  7218, 10176,  2310,  5325])

testSetFixed = testSet
whereBad = []
for test_it, test_id in enumerate(testSet):
    if test_id in ignore:
        bad = np.where(testSetFixed==test_id)
        testSetFixed = np.delete(testSetFixed, bad)
        whereBad = whereBad + [bad]

# testSet = testSetFixed

loadFromHdf5 = False
useShapeModel = True
syntheticGroundtruth = True

synthPrefix = '_cycles'
if syntheticGroundtruth:
    synthPrefix = ''

if syntheticGroundtruth:
    imagesDir = gtDir + 'images_opendr/'
else:
    imagesDir = gtDir + 'images/'
images = readImages(imagesDir, dataIds, loadFromHdf5)

print("Backprojecting and fitting estimates.")
# testSet = np.arange(len(images))[0:10]

testAzsGT = dataAzsGT
testObjAzsGT = dataObjAzsGT
testElevsGT = dataElevsGT
testLightAzsGT = dataLightAzsGT
testLightElevsGT = dataLightElevsGT
testLightIntensitiesGT = dataLightIntensitiesGT
testVColorGT = dataVColorGT
testOcclusions = dataOcclusions

if useShapeModel:
    dataShapeModelCoeffsGT = groundTruth['trainShapeModelCoeffsGT']

    testShapeParamsGT = dataShapeModelCoeffsGT

testLightCoefficientsGTRel = dataLightCoefficientsGTRel * dataAmbientIntensityGT[:,None]

testAzsRel = np.mod(testAzsGT - testObjAzsGT, 2*np.pi)

if os.path.isfile(resultDir + 'samples.npz'):
    samplesfile = np.load(resultDir + 'samples.npz')
    azsPred=samplesfile['azsPred']
    elevsPred=samplesfile['elevsPred']
    fittedAzs=samplesfile['fittedAzs']
    fittedElevs=samplesfile['fittedElevs']
    vColorsPred=samplesfile['vColorsPred']
    fittedVColors=samplesfile['fittedVColors']
    relLightCoefficientsGTPred=samplesfile['relLightCoefficientsGTPred']
    fittedRelLightCoeffs=samplesfile['fittedRelLightCoeffs']
    if useShapeModel:
        fittedShapeParams =samplesfile['fittedShapeParams']
        shapeParamsPred=samplesfile['shapeParamsPred']

if os.path.isfile(resultDir + 'samples.npz'):
    performancefile = np.load(resultDir + 'performance_samples.npz')
    predictedErrorFuns = performancefile['predictedErrorFuns']
    fittedErrorFuns = performancefile['fittedErrorFuns']
    if useShapeModel:
        errorsFittedShapeParams=performancefile['errorsFittedShapeParams']
        errorsFittedShapeVertices=performancefile['errorsFittedShapeVertices']

        errorsShapeParams=performancefile['errorsShapeParams']
        errorsShapeVertices=performancefile['errorsShapeVertices']
