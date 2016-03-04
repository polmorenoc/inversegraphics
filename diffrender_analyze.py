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
testPrefix = 'train4_occlusion_shapemodel_10k_background_POSESHAPESH-ALL_1000samples__method1errorFun1_std0.03_shapePen0'

gtPrefix = 'train4_occlusion_shapemodel'
experimentPrefix = 'train4_occlusion_shapemodel_10k'

gtDir = 'groundtruth/' + gtPrefix + '/'
featuresDir = gtDir

experimentDir = 'experiments/' + experimentPrefix + '/'

resultDir = 'results/' + testPrefix + '/'

useShapeModel = True

if useShapeModel:
    import shape_model
    #%% Load data
    filePath = 'data/teapotModel.pkl'
    teapotModel = shape_model.loadObject(filePath)
    faces = teapotModel['faces']

    #%% Sample random shape Params
    latentDim = np.shape(teapotModel['ppcaW'])[1]
    shapeParams = np.random.randn(latentDim)
    chShapeParams = ch.Ch(shapeParams)

    meshLinearTransform=teapotModel['meshLinearTransform']
    W=teapotModel['ppcaW']
    b=teapotModel['ppcaB']

    chVertices = shape_model.VerticesModel(chShapeParams=chShapeParams,meshLinearTransform=meshLinearTransform,W = W,b=b)
    chVertices.init()

    chVertices = ch.dot(geometry.RotateZ(-np.pi/2)[0:3,0:3],chVertices.T).T

    chVertices = chVertices - ch.mean(chVertices, axis=0)
    minZ = ch.min(chVertices[:,2])

    chMinZ = ch.min(chVertices[:,2])

    zeroZVerts = chVertices[:,2]- chMinZ
    chVertices = ch.hstack([chVertices[:,0:2] , zeroZVerts.reshape([-1,1])])

    chVertices = chVertices*0.09
    smCenter = ch.array([0,0,0.1])

    smVertices = [chVertices]


ignore = []
if os.path.isfile(gtDir + 'ignore.npy'):
    ignore = np.load(gtDir + 'ignore.npy')

groundTruthFilename = gtDir + 'groundTruth.h5'
gtDataFile = h5py.File(groundTruthFilename, 'r')

rangeTests = np.arange(100,1100)
testSet = np.load(experimentDir + 'test.npy')[rangeTests]


shapeGT = gtDataFile[gtPrefix].shape
boolTestSet = np.zeros(shapeGT).astype(np.bool)
boolTestSet[testSet] = True
testGroundTruth = gtDataFile[gtPrefix][boolTestSet]
groundTruthTest = np.zeros(shapeGT, dtype=testGroundTruth.dtype)
groundTruthTest[boolTestSet] = testGroundTruth
groundTruth = groundTruthTest[testSet]
dataTeapotIdsTest = groundTruth['trainTeapotIds']
test = np.arange(len(testSet))


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


######## Read different datasets
# if os.path.isfile(resultDir + 'samples.npz'):
#     samplesfile = np.load(resultDir + 'samples.npz')
#
#     fittedAzs=samplesfile['fittedAzs']
#     fittedElevs=samplesfile['fittedElevs']
#     fittedVColors=samplesfile['fittedVColors']
#     fittedRelLightCoeffs=samplesfile['fittedRelLightCoeffs']
#     if useShapeModel:
#         fittedShapeParams =samplesfile['fittedShapeParams']
#
# envMapTexture = np.zeros([180,360,3])
# approxProjectionsFittedList = []
# for test_i in range(len(testSet)):
#     pEnvMap = SHProjection(envMapTexture, np.concatenate([fittedRelLightCoeffs[test_i][:,None], fittedRelLightCoeffs[test_i][:,None], fittedRelLightCoeffs[test_i][:,None]], axis=1))
#     approxProjection = np.sum(pEnvMap, axis=(2,3))
#     approxProjectionsFittedList = approxProjectionsFittedList + [approxProjection[None,:]]
# approxProjectionsFitted = np.vstack(approxProjectionsFittedList)
#
#
# testPrefix = 'train4_occlusion_shapemodel_10k_background_gaussian_ALL_1000samples__method1errorFun0_std0.1_shapePen0'
# resultDir = 'results/' + testPrefix + '/'
#
# if os.path.isfile(resultDir + 'samples.npz'):
#     samplesfile = np.load(resultDir + 'samples.npz')
#     fittedAzsGaussian=samplesfile['fittedAzs']
#     fittedElevsGaussian=samplesfile['fittedElevs']
#     fittedVColorsGaussian=samplesfile['fittedVColors']
#     fittedRelLightCoeffsGaussian=samplesfile['fittedRelLightCoeffs']
#     if useShapeModel:
#         fittedShapeParamsGaussian =samplesfile['fittedShapeParams']
#
# envMapTexture = np.zeros([180,360,3])
# approxProjectionsFittedList = []
# for test_i in range(len(testSet)):
#     pEnvMap = SHProjection(envMapTexture, np.concatenate([fittedRelLightCoeffsGaussian[test_i][:,None], fittedRelLightCoeffsGaussian[test_i][:,None], fittedRelLightCoeffsGaussian[test_i][:,None]], axis=1))
#     approxProjection = np.sum(pEnvMap, axis=(2,3))
#     approxProjectionsFittedList = approxProjectionsFittedList + [approxProjection[None,:]]
# approxProjectionsFittedGaussian = np.vstack(approxProjectionsFittedList)
#
#
# testPrefix = 'train4_occlusion_shapemodel_10k_nearestNeighbour_predict_1000samples__method1errorFun0_std0.1_shapePen0'
# resultDir = 'results/' + testPrefix + '/'
#
# expFilename = resultDir + 'experiment.pickle'
# with open(expFilename, 'rb') as pfile:
#     experimentDic = pickle.load(pfile)
#
# methodsPred = experimentDic['methodsPred']
# testOcclusions = experimentDic[ 'testOcclusions']
# testPrefixBase = experimentDic[ 'testPrefixBase']
# parameterRecognitionModels = experimentDic[ 'parameterRecognitionModels']
# azimuths = experimentDic[ 'azimuths']
# azimuths[1] = azimuths[1].ravel()
# elevations = experimentDic[ 'elevations']
# elevations[1] = elevations[1].ravel()
# vColors = experimentDic[ 'vColors']
# lightCoeffs = experimentDic[ 'lightCoeffs']
# approxProjections = experimentDic[ 'approxProjections']
# approxProjectionsGT = experimentDic[ 'approxProjectionsGT']
# shapeParams = experimentDic['shapeParams']
#
# nearestNeighbours = False
# if 'Nearest Neighbour' in set(methodsPred):
#     nearestNeighbours = True
#
# plotColors = ['k']
# if nearestNeighbours:
#     # methodsPred = methodsPred + ["Nearest Neighbours"]
#     plotColors = plotColors + ['g']
#
# methodsPred = methodsPred + ["Gaussian Fit"]
# plotColors = plotColors + ['b']
#
# methodsPred = methodsPred + ["Robust Fit"]
# plotColors = plotColors + ['r']
#
# azimuths = azimuths + [fittedAzsGaussian]
# elevations = elevations + [fittedElevsGaussian]
# vColors = vColors + [fittedVColorsGaussian]
# lightCoeffs = lightCoeffs + [fittedRelLightCoeffsGaussian]
# approxProjections = approxProjections + [approxProjectionsFittedGaussian]
# shapeParams = shapeParams + [fittedShapeParamsGaussian]
#
# azimuths = azimuths + [fittedAzs]
# elevations = elevations + [fittedElevs]
# vColors = vColors + [fittedVColors]
# lightCoeffs = lightCoeffs + [fittedRelLightCoeffs]
# approxProjections = approxProjections + [approxProjectionsFitted]
# shapeParams = shapeParams + [fittedShapeParams]
#
# errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList, \
#     = computeErrors(np.arange(len(rangeTests)), azimuths, testAzsRel, elevations, testElevsGT, vColors, testVColorGT, lightCoeffs, testLightCoefficientsGTRel, approxProjections,  approxProjectionsGT, shapeParams, testShapeParamsGT, useShapeModel, chShapeParams, chVertices)
#
# meanAbsErrAzsList, meanAbsErrElevsList, medianAbsErrAzsList, medianAbsErrElevsList, meanErrorsLightCoeffsList, meanErrorsShapeParamsList, meanErrorsShapeVerticesList, meanErrorsLightCoeffsCList, meanErrorsEnvMapList, meanErrorsVColorsEList, meanErrorsVColorsCList \
# = computeErrorMeans(np.arange(len(rangeTests)), useShapeModel, errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList)
#
# testOcclusionsFull = testOcclusions.copy()
#
# testPrefix = 'train4_occlusion_shapemodel_10k_ECCV'
#
# resultDir = 'results/' + testPrefix + '/'
#
#
# if not os.path.exists(resultDir):
#     os.makedirs(resultDir)
# if not os.path.exists(resultDir + 'imgs/'):
#     os.makedirs(resultDir + 'imgs/')
# if not os.path.exists(resultDir +  'imgs/samples/'):
#     os.makedirs(resultDir + 'imgs/samples/')
#
# experimentDic = {'testSet':testSet, 'methodsPred':methodsPred, 'testOcclusions':testOcclusions, 'testPrefixBase':testPrefixBase, 'parameterRecognitionModels':parameterRecognitionModels, 'azimuths':azimuths, 'elevations':elevations, 'vColors':vColors, 'lightCoeffs':lightCoeffs, 'approxProjections':approxProjections, 'shapeParams':shapeParams, 'approxProjectionsGT':approxProjectionsGT, 'errorsPosePredList':errorsPosePredList, 'errorsLightCoeffsList':errorsLightCoeffsList, 'errorsShapeParamsLis':errorsShapeParamsList, 'errorsShapeVerticesList':errorsShapeVerticesList, 'errorsEnvMapList':errorsEnvMapList, 'errorsLightCoeffsCList':errorsLightCoeffsCList, 'errorsVColorsEList':errorsVColorsEList, 'errorsVColorsCList':errorsVColorsCList}
#
# with open(resultDir + 'experiment.pickle', 'wb') as pfile:
#     pickle.dump(experimentDic, pfile)

# ipdb.set_trace()

testOcclusionsFull = testOcclusions.copy()

testPrefix = 'train4_occlusion_shapemodel_10k_ECCV'
testPrefixBase = testPrefix

resultDir = 'results/' + testPrefix + '/'

expFilename = resultDir + 'experiment.pickle'
with open(expFilename, 'rb') as pfile:
    experimentDic = pickle.load(pfile)

plt.ioff()

methodsPred = experimentDic['methodsPred']
testOcclusions = experimentDic[ 'testOcclusions']
testPrefixBase = experimentDic[ 'testPrefixBase']
parameterRecognitionModels = experimentDic[ 'parameterRecognitionModels']
azimuths = experimentDic[ 'azimuths']
elevations = experimentDic[ 'elevations']
vColors = experimentDic[ 'vColors']
lightCoeffs = experimentDic[ 'lightCoeffs']
approxProjections = experimentDic[ 'approxProjections']
approxProjectionsGT = experimentDic[ 'approxProjectionsGT']
shapeParams = experimentDic['shapeParams']
errorsPosePredList = experimentDic['errorsPosePredList']
errorsLightCoeffsList = experimentDic['errorsLightCoeffsList']
errorsShapeParamsList = experimentDic['errorsShapeParamsLis']
errorsShapeVerticesList = experimentDic['errorsShapeVerticesList']
errorsEnvMapList = experimentDic['errorsEnvMapList']
errorsLightCoeffsCList = experimentDic['errorsLightCoeffsCList']
errorsVColorsEList = experimentDic['errorsVColorsEList']
errorsVColorsCList = experimentDic['errorsVColorsCList']

nearestNeighbours = False
if 'Nearest Neighbours' in set(methodsPred):
    nearestNeighbours = True

plotColors = ['k']
if nearestNeighbours:
    # methodsPred = methodsPred + ["Nearest Neighbours"]
    plotColors = plotColors + ['g']

plotColors = plotColors + ['b']

plotColors = plotColors + ['y']

plotColors = plotColors + ['r']

plotMethodsIndices = [0,2,3,4]

meanAbsErrAzsList, meanAbsErrElevsList, medianAbsErrAzsList, medianAbsErrElevsList, meanErrorsLightCoeffsList, meanErrorsShapeParamsList, meanErrorsShapeVerticesList, meanErrorsLightCoeffsCList, meanErrorsEnvMapList, meanErrorsVColorsEList, meanErrorsVColorsCList \
= computeErrorMeans(np.arange(len(rangeTests)), useShapeModel, errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList)

meanAbsErrAzsArr = []
meanAbsErrElevsArr = []
medianAbsErrAzsArr = []
medianAbsErrElevsArr = []
meanErrorsLightCoeffsArr = []
meanErrorsEnvMapArr = []
meanErrorsShapeParamsArr = []
meanErrorsShapeVerticesArr = []
meanErrorsLightCoeffsCArr = []
meanErrorsVColorsEArr = []
meanErrorsVColorsCArr = []

for method_i in range(len(methodsPred)):
    meanAbsErrAzsArr = meanAbsErrAzsArr + [np.array([])]
    meanAbsErrElevsArr = meanAbsErrElevsArr + [np.array([])]
    medianAbsErrAzsArr = medianAbsErrAzsArr + [np.array([])]
    medianAbsErrElevsArr = medianAbsErrElevsArr + [np.array([])]
    meanErrorsLightCoeffsArr = meanErrorsLightCoeffsArr + [np.array([])]
    meanErrorsShapeParamsArr = meanErrorsShapeParamsArr + [np.array([])]
    meanErrorsShapeVerticesArr = meanErrorsShapeVerticesArr + [np.array([])]
    meanErrorsLightCoeffsCArr = meanErrorsLightCoeffsCArr + [np.array([])]
    meanErrorsVColorsEArr = meanErrorsVColorsEArr + [np.array([])]
    meanErrorsVColorsCArr = meanErrorsVColorsCArr + [np.array([])]
    meanErrorsEnvMapArr = meanErrorsEnvMapArr + [np.array([])]
occlusions = []

for occlusionLevel in range(100):

    setUnderOcclusionLevel = testOcclusionsFull * 100 < occlusionLevel

    if np.any(setUnderOcclusionLevel):
        occlusions = occlusions + [occlusionLevel]
        testOcclusions = testOcclusionsFull[setUnderOcclusionLevel]

        colors = matplotlib.cm.plasma(testOcclusions)

        for method_i in range(len(methodsPred)):

            meanAbsErrAzsArr[method_i] = np.append(meanAbsErrAzsArr[method_i], np.mean(np.abs(errorsPosePredList[method_i][0][setUnderOcclusionLevel])))
            meanAbsErrElevsArr[method_i] = np.append(meanAbsErrElevsArr[method_i], np.mean(np.abs(errorsPosePredList[method_i][1][setUnderOcclusionLevel])))

            medianAbsErrAzsArr[method_i] = np.append(medianAbsErrAzsArr[method_i], np.median(np.abs(errorsPosePredList[method_i][0][setUnderOcclusionLevel])))
            medianAbsErrElevsArr[method_i] = np.append(medianAbsErrElevsArr[method_i], np.median(np.abs(errorsPosePredList[method_i][1][setUnderOcclusionLevel])))

            meanErrorsLightCoeffsArr[method_i] = np.append(meanErrorsLightCoeffsArr[method_i],np.mean(np.mean(errorsLightCoeffsList[method_i][setUnderOcclusionLevel], axis=1), axis=0))
            meanErrorsLightCoeffsCArr[method_i] = np.append(meanErrorsLightCoeffsCArr[method_i],np.mean(np.mean(errorsLightCoeffsCList[method_i][setUnderOcclusionLevel], axis=1), axis=0))

            if useShapeModel:
                meanErrorsShapeParamsArr[method_i] = np.append(meanErrorsShapeParamsArr[method_i],np.mean(np.mean(errorsShapeParamsList[method_i][setUnderOcclusionLevel], axis=1), axis=0))
                meanErrorsShapeVerticesArr[method_i] = np.append(meanErrorsShapeVerticesArr[method_i], np.mean(errorsShapeVerticesList[method_i][setUnderOcclusionLevel], axis=0))

            meanErrorsEnvMapArr[method_i] = np.append(meanErrorsEnvMapArr[method_i], np.mean(errorsEnvMapList[method_i][setUnderOcclusionLevel]))
            meanErrorsVColorsEArr[method_i] = np.append(meanErrorsVColorsEArr[method_i], np.mean(errorsVColorsEList[method_i][setUnderOcclusionLevel], axis=0))
            meanErrorsVColorsCArr[method_i] = np.append(meanErrorsVColorsCArr[method_i], np.mean(errorsVColorsCList[method_i][setUnderOcclusionLevel], axis=0))


# methodsPred = [methodsPred[ind] for ind in plotMethodsIndices]
# plotColors = [plotColors[ind] for ind in plotMethodsIndices]

saveOcclusionPlots(resultDir, occlusions, methodsPred, plotColors, plotMethodsIndices, useShapeModel, meanAbsErrAzsArr, meanAbsErrElevsArr, meanErrorsVColorsCArr, meanErrorsVColorsEArr, meanErrorsLightCoeffsArr, meanErrorsShapeParamsArr, meanErrorsShapeVerticesArr, meanErrorsLightCoeffsCArr, meanErrorsEnvMapArr)

SHModel = ""

for occlusionLevel in [25, 50, 75, 100]:

    resultDirOcclusion = 'results/' + testPrefix + '/occlusion' + str(occlusionLevel) + '/'
    if not os.path.exists(resultDirOcclusion):
        os.makedirs(resultDirOcclusion)

    setUnderOcclusionLevel = testOcclusionsFull * 100 < occlusionLevel
    testOcclusions = testOcclusionsFull[setUnderOcclusionLevel]

    # if len(stdevsFull) > 0:
    #     stdevs = stdevsFull[setUnderOcclusionLevel]

    colors = matplotlib.cm.plasma(testOcclusions)

    meanAbsErrAzsList, meanAbsErrElevsList, medianAbsErrAzsList, medianAbsErrElevsList, meanErrorsLightCoeffsList, meanErrorsShapeParamsList, meanErrorsShapeVerticesList, meanErrorsLightCoeffsCList, meanErrorsEnvMapList, meanErrorsVColorsEList, meanErrorsVColorsCList = computeErrorMeans(setUnderOcclusionLevel, useShapeModel, errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList)

    # Write statistics to file.

    import tabulate

    headers = ["Errors"] + methodsPred

    table = [["Azimuth"] +  meanAbsErrAzsList,
             ["Elevation"] + meanAbsErrElevsList,
             ["VColor C"] + meanErrorsVColorsCList,
             ["SH Light"] + meanErrorsLightCoeffsList,
             ["SH Light C"] + meanErrorsLightCoeffsCList,
             ["SH Env Map"] + meanErrorsEnvMapList,
             ["Shape Params"] + meanErrorsShapeParamsList,
             ["Shape Vertices"] + meanErrorsShapeVerticesList
             ]
    performanceTable = tabulate.tabulate(table, headers=headers, tablefmt="latex", floatfmt=".3f")
    with open(resultDirOcclusion + 'performance.tex', 'w') as expfile:
        expfile.write(performanceTable)

    headers = ["", "l=0", "SH $l=0,m=-1$", "SH $l=1,m=0$", "SH $l=1,m=1$", "SH $l=1,m=-2$", "SH $l=2,m=-1$",
               "SH $l=2,m=0$", "SH $l=2,m=1$", "SH $l=2,m=2$"]


    for method_i in plotMethodsIndices:

        SMSE_SH = np.mean(errorsLightCoeffsList[method_i][setUnderOcclusionLevel], axis=0)
        table = [
            [SHModel, SMSE_SH[0], SMSE_SH[1], SMSE_SH[2], SMSE_SH[3], SMSE_SH[4], SMSE_SH[5], SMSE_SH[6], SMSE_SH[7],
             SMSE_SH[8]],
        ]
        performanceTable = tabulate.tabulate(table, headers=headers, tablefmt="latex", floatfmt=".3f")
        with open(resultDir + 'performance_SH_' + methodsPred[method_i] + '.tex', 'w') as expfile:
            expfile.write(performanceTable)

        #     expfile.write(performanceTable)

        SMSE_SH = np.mean(errorsLightCoeffsCList[method_i][setUnderOcclusionLevel], axis=0)
        table = [
            [SHModel, SMSE_SH[0], SMSE_SH[1], SMSE_SH[2], SMSE_SH[3], SMSE_SH[4], SMSE_SH[5], SMSE_SH[6], SMSE_SH[7],
             SMSE_SH[8]],
        ]
        performanceTable = tabulate.tabulate(table, headers=headers, tablefmt="latex", floatfmt=".3f")
        with open(resultDir + 'performance_SH_C_' + methodsPred[method_i] + '.tex', 'w') as expfile:
            expfile.write(performanceTable)


        if useShapeModel:
            SMSE_SHAPE_PARAMS = np.mean(errorsShapeParamsList[method_i][setUnderOcclusionLevel], axis=0)
            table = [[SHModel, SMSE_SHAPE_PARAMS[0], SMSE_SHAPE_PARAMS[1], SMSE_SHAPE_PARAMS[2], SMSE_SHAPE_PARAMS[3],
                      SMSE_SHAPE_PARAMS[4], SMSE_SHAPE_PARAMS[5], SMSE_SHAPE_PARAMS[6], SMSE_SHAPE_PARAMS[7],
                      SMSE_SHAPE_PARAMS[8], SMSE_SHAPE_PARAMS[9]]]
            performanceTable = tabulate.tabulate(table, headers=headers, tablefmt="latex", floatfmt=".3f")
            with open(resultDir + 'performance_ShapeParams_' + methodsPred[method_i] + '.tex', 'w') as expfile:
                expfile.write(performanceTable)

plt.ion()
print("Finished.")



