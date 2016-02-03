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
testPrefix = 'train4_occlusion_1000s_allplots_robust_std001'

# testPrefixWrite= 'train4_occlusion_1000s_allplots_robust_std001'

optimizationType = 1
optimizationTypeDescr = ["predict", "optimize", "joint"]
computePredErrorFuns = True
# testPrefix = 'train4_occlusion_opt_train4occlusion10k_10s_std01_bad'

parameterRecognitionModels = set(['randForestAzs', 'randForestElevs', 'randForestVColors', 'linearRegressionVColors', 'neuralNetModelSHLight', ])
parameterRecognitionModels = set(['randForestAzs', 'randForestElevs', 'randForestVColors', 'linearRegressionVColors', 'linRegModelSHZernike' ])
parameterRecognitionModels = set(['randForestAzs', 'randForestElevs','linearRegressionVColors','neuralNetModelSHLight' ])
parameterRecognitionModels = set(['neuralNetPose', 'linearRegressionVColors','constantSHLight' ])
parameterRecognitionModels = set(['neuralNetPose', 'neuralNetApperanceAndLight', 'neuralNetVColors' ])
parameterRecognitionModels = set(['neuralNetPose', 'neuralNetModelSHLight', 'neuralNetVColors' ])
# parameterRecognitionModels = set(['neuralNetPose', 'neuralNetApperanceAndLight'])

# parameterRecognitionModels = set(['randForestAzs', 'randForestElevs','randForestVColors','randomForestSHZernike' ])

gtPrefix = 'train4_occlusion'
experimentPrefix = 'train4_occlusion'
trainPrefixPose = 'train4_occlusion_10k'
trainPrefixVColor = 'train4_occlusion_10k'
trainPrefixLightCoeffs = 'train4_occlusion_10k'
trainModelsDirAppLight = 'train4_occlusion_10k'
gtDir = 'groundtruth/' + gtPrefix + '/'
featuresDir = gtDir

experimentDir = 'experiments/' + experimentPrefix + '/'
trainModelsDirPose = 'experiments/' + trainPrefixPose + '/'
trainModelsDirVColor = 'experiments/' + trainPrefixVColor + '/'
trainModelsDirLightCoeffs = 'experiments/' + trainPrefixLightCoeffs + '/'
trainModelsDirAppLight = 'experiments/' + trainModelsDirAppLight + '/'
resultDir = 'results/' + testPrefix + '/'

ignoreGT = True
ignore = []
if os.path.isfile(gtDir + 'ignore.npy'):
    ignore = np.load(gtDir + 'ignore.npy')

groundTruthFilename = gtDir + 'groundTruth.h5'
gtDataFile = h5py.File(groundTruthFilename, 'r')

numTests = 1000
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
dataComponentsGT = groundTruth['trainComponentsGT']
dataComponentsGTRel = groundTruth['trainComponentsGTRel']
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
testComponentsGT = dataComponentsGT
testComponentsGTRel = dataComponentsGTRel
testOcclusions = dataOcclusions

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

if os.path.isfile(resultDir + 'samples.npz'):
    performancefile = np.load(resultDir + 'performance_samples.npz')
    predictedErrorFuns = performancefile['predictedErrorFuns']
    fittedErrorFuns = performancefile['fittedErrorFuns']

SHModel = ""
azsPredictions = []
relLightCoefficientsPred = relLightCoefficientsGTPred

errorsPosePred = recognition_models.evaluatePrediction(testAzsRel, testElevsGT, azsPred, elevsPred)

testVColorGTGray = 0.3*testVColorGT[:,0] + 0.59*testVColorGT[:,1] + 0.11*testVColorGT[:,2]
vColorsPredGray = 0.3*vColorsPred[:,0] + 0.59*vColorsPred[:,1] + 0.11*vColorsPred[:,2]

errorsLightCoeffsC = (testVColorGTGray[:,None] * testLightCoefficientsGTRel - vColorsPredGray[:,None] * relLightCoefficientsPred) ** 2
errorsLightCoeffs = (testLightCoefficientsGTRel - relLightCoefficientsPred) ** 2
errorsVColorsE = image_processing.eColourDifference(testVColorGT, vColorsPred)
errorsVColorsC = image_processing.cColourDifference(testVColorGT, vColorsPred)

meanAbsErrAzs = np.mean(np.abs(errorsPosePred[0]))
meanAbsErrElevs = np.mean(np.abs(errorsPosePred[1]))

medianAbsErrAzs = np.median(np.abs(errorsPosePred[0]))
medianAbsErrElevs = np.median(np.abs(errorsPosePred[1]))

meanErrorsLightCoeffs = np.sqrt(np.mean(np.mean(errorsLightCoeffs,axis=1), axis=0))
meanErrorsLightCoeffsC = np.sqrt(np.mean(np.mean(errorsLightCoeffsC,axis=1), axis=0))
meanErrorsVColorsE = np.mean(errorsVColorsE, axis=0)
meanErrorsVColorsC = np.mean(errorsVColorsC, axis=0)

errorsPoseFitted = (np.array([]), np.array([]))
errorsPoseFitted = recognition_models.evaluatePrediction(testAzsRel, testElevsGT, fittedAzs, fittedElevs)
stdevs = np.array([])
if len(azsPredictions) > 0:
    stdevs = np.array([])
    for test_i, test_id in enumerate(testSet):
        stdevs = np.append(stdevs, np.sqrt(-np.log(np.min([np.mean(sinAzsPredSamples[test_i])**2 + np.mean(cosAzsPredSamples[test_i])**2,1]))))

if optimizationTypeDescr[optimizationType] != 'predict':
    fittedRelLightCoeffs = np.vstack(fittedRelLightCoeffs)
    fittedVColors = np.vstack(fittedVColors)

    fittedVColorsGray = 0.3*fittedVColors[:,0] + 0.59*fittedVColors[:,1] + 0.11*fittedVColors[:,2]

    errorsFittedLightCoeffsC = (testVColorGTGray[:,None]*testLightCoefficientsGTRel - fittedVColorsGray[:,None]*fittedRelLightCoeffs)**2

    errorsFittedLightCoeffs = (testLightCoefficientsGTRel - fittedRelLightCoeffs)**2
    errorsFittedVColorsE = image_processing.eColourDifference(testVColorGT, fittedVColors)
    errorsFittedVColorsC = image_processing.cColourDifference(testVColorGT, fittedVColors)


##Print and plotting results now:

plt.ioff()
import seaborn

colors = matplotlib.cm.plasma(testOcclusions)

if len(stdevs) > 0:
    #Show scatter correlations with occlusions.
    directory = resultDir + 'occlusion_nnazsamplesstdev'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scat = ax.scatter(testOcclusions * 100.0, stdevs*180/np.pi, s=20, vmin=0, vmax=100, c='b')
    ax.set_xlabel('Occlusion (%)')
    ax.set_ylabel('NN Az samples stdev')
    x1,x2 = ax.get_xlim()
    y1,y2 = ax.get_ylim()
    ax.set_xlim((0,100))
    ax.set_ylim((0,180))
    ax.set_title('Occlusion vs az predictions correlations')
    fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)


directory = resultDir + 'occlusion_vcolorserrorsE'
#Show scatter correlations with occlusions.
fig = plt.figure()
ax = fig.add_subplot(111)
scat = ax.scatter(testOcclusions * 100.0, errorsVColorsE, s=20, vmin=0, vmax=100, c='b')
ax.set_xlabel('Occlusion (%)')
ax.set_ylabel('Vertex Colors errors')
x1,x2 = ax.get_xlim()
y1,y2 = ax.get_ylim()
ax.set_xlim((0,100))
ax.set_ylim((-0.0,1))
ax.set_title('Occlusion - vertex color prediction errors')

fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
plt.close(fig)

directory = resultDir + 'occlusion_vcolorserrorsC'
#Show scatter correlations with occlusions.
fig = plt.figure()
ax = fig.add_subplot(111)
scat = ax.scatter(testOcclusions * 100.0, errorsVColorsC, s=20, vmin=0, vmax=100, c='b')
ax.set_xlabel('Occlusion (%)')
ax.set_ylabel('Vertex Colors errors')
x1,x2 = ax.get_xlim()
y1,y2 = ax.get_ylim()
ax.set_xlim((0,100))
ax.set_ylim((-0.0,1))
ax.set_title('Occlusion - vertex color prediction errors')
fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
plt.close(fig)
#Show scatter correlations with occlusions.


directory = resultDir + 'occlusion_shcoeffsserrors'
#Show scatter correlations with occlusions.
fig = plt.figure()
ax = fig.add_subplot(111)
scat = ax.scatter(testOcclusions * 100.0, np.sqrt(np.mean(errorsLightCoeffs,axis=1)), s=20, vmin=0, vmax=100, c='b', )
ax.set_xlabel('Occlusion (%)')
ax.set_ylabel('SH coefficient errors')
x1,x2 = ax.get_xlim()
y1,y2 = ax.get_ylim()
ax.set_xlim((0,100))
ax.set_ylim((-0.0,1))
ax.set_title('Occlusion - SH coefficients fitted errors')
fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
plt.close(fig)


directory = resultDir + 'occlusion_shcoeffsserrorsC'
#Show scatter correlations with occlusions.
fig = plt.figure()
ax = fig.add_subplot(111)
scat = ax.scatter(testOcclusions * 100.0, np.sqrt(np.mean(errorsLightCoeffsC,axis=1)), s=20, vmin=0, vmax=100, c='b')
ax.set_xlabel('Occlusion (%)')
ax.set_ylabel('SH coefficient errors')
x1,x2 = ax.get_xlim()
y1,y2 = ax.get_ylim()
ax.set_xlim((0,100))
ax.set_ylim((-0.0,1))
ax.set_title('Occlusion - SH coefficients fitted errors')
fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
plt.close(fig)
#Show scatter correlations with occlusions.



# Show scatter correlations of occlusion with predicted and fitted color.
if not optimizationTypeDescr[optimizationType] == 'predict':
    directory = resultDir + 'fitted-occlusion_vcolorserrorsE'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scat = ax.scatter(testOcclusions * 100.0, errorsFittedVColorsE, s=20, vmin=0, vmax=100)
    ax.set_xlabel('Occlusion (%)')
    ax.set_ylabel('Vertex Colors errors')
    x1,x2 = ax.get_xlim()
    y1,y2 = ax.get_ylim()
    ax.set_xlim((0,100))
    ax.set_ylim((-0.0,1))
    ax.set_title('Occlusion - vertex color fitted errors')
    fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)


    directory = resultDir + 'fitted-occlusion_vcolorserrorsC'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scat = ax.scatter(testOcclusions * 100.0, errorsFittedVColorsC, s=20, vmin=0, vmax=100)
    ax.set_xlabel('Occlusion (%)')
    ax.set_ylabel('Vertex Colors errors')
    x1,x2 = ax.get_xlim()
    y1,y2 = ax.get_ylim()
    ax.set_xlim((0,100))
    ax.set_ylim((-0.0,1))
    ax.set_title('Occlusion - vertex color fitted errors')
    fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)
    #Show scatter correlations with occlusions.

    directory = resultDir + 'fitted-occlusion_shcoeffsserrors'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scat = ax.scatter(testOcclusions * 100.0, np.sqrt(np.mean(errorsFittedLightCoeffs,axis=1)), s=20, vmin=0, vmax=100)
    ax.set_xlabel('Occlusion (%)')
    ax.set_ylabel('Fitted SH coefficients errors')
    x1,x2 = ax.get_xlim()
    y1,y2 = ax.get_ylim()
    ax.set_xlim((0,100))
    ax.set_ylim((-0.0,1))
    ax.set_title('Occlusion - SH coefficients fitted errors')
    fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)
    #Show scatter correlations with occlusions.

    directory = resultDir + 'fitted-occlusion_shcoeffsserrorsC'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scat = ax.scatter(testOcclusions * 100.0, np.sqrt(np.mean(errorsFittedLightCoeffsC,axis=1)), s=20, vmin=0, vmax=100)
    ax.set_xlabel('Occlusion (%)')
    ax.set_ylabel('Fitted SH coefficients errors')
    x1,x2 = ax.get_xlim()
    y1,y2 = ax.get_ylim()
    ax.set_xlim((0,100))
    ax.set_ylim((-0.0,1))
    ax.set_title('Occlusion - SH coefficients fitted errors')
    fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)

directory = resultDir + 'azimuth-pose-prediction'

fig = plt.figure()
ax = fig.add_subplot(111)
scat = ax.scatter(testOcclusions * 100.0, errorsPosePred[0], s=20, vmin=0, vmax=100)
ax.set_xlabel('Occlusion (%)')
ax.set_ylabel('Angular error')
x1,x2 = ax.get_xlim()
y1,y2 = ax.get_ylim()
ax.set_xlim((0,100))
ax.set_ylim((-180,180))
ax.set_title('Performance scatter plot')
fig.savefig(directory + '_occlusion-performance-scatter.png', bbox_inches='tight')
plt.close(fig)

directory = resultDir + 'elevation-pose-prediction'
fig = plt.figure()
ax = fig.add_subplot(111)
scat = ax.scatter(testOcclusions * 100.0, errorsPosePred[1], s=20, vmin=0, vmax=100)
ax.set_xlabel('Occlusion (%)')
ax.set_ylabel('Angular error')
x1,x2 = ax.get_xlim()
y1,y2 = ax.get_ylim()
ax.set_xlim((0,100))
ax.set_ylim((-90.0,90))
ax.set_title('Performance scatter plot')
fig.savefig(directory + '_occlusion-performance-scatter.png', bbox_inches='tight')
plt.close(fig)

if not optimizationTypeDescr[optimizationType] == 'predict':
    directory = resultDir + 'fitted-azimuth-prediction'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scat = ax.scatter(testOcclusions * 100.0, errorsPoseFitted[0], s=20, vmin=0, vmax=100)

    ax.set_xlabel('Occlusion (%)')
    ax.set_ylabel('Angular error')
    x1,x2 = ax.get_xlim()
    y1,y2 = ax.get_ylim()
    ax.set_xlim((0,100))
    ax.set_ylim((-180.0,180))
    ax.set_title('Performance scatter plot')
    fig.savefig(directory + '_occlusion-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)


    directory = resultDir + 'fitted-elevation-pose-prediction'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scat = ax.scatter(testOcclusions * 100.0, errorsPoseFitted[1], s=20, vmin=0, vmax=100)
    ax.set_xlabel('Occlusion (%)')
    ax.set_ylabel('Angular error')
    x1,x2 = ax.get_xlim()
    y1,y2 = ax.get_ylim()
    ax.set_xlim((0,100))
    ax.set_ylim((-90.0,90))
    ax.set_title('Performance scatter plot')
    fig.savefig(directory + '_occlusion-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)


errorsAzPredFull = errorsPosePred[0].copy()
errorsElPredFull = errorsPosePred[1].copy()
errorsLightCoeffsFull = errorsLightCoeffs.copy()
errorsLightCoeffsCFull = errorsLightCoeffsC.copy()
errorsVColorsEFull = errorsVColorsE.copy()
errorsVColorsCFull = errorsVColorsC.copy()
errorsAzFittedFull = errorsPoseFitted[0].copy()
errorsElFittedFull = errorsPoseFitted[1].copy()
errorsFittedLightCoeffsFull = errorsFittedLightCoeffs.copy()
errorsFittedLightCoeffsCFull = errorsFittedLightCoeffsC.copy()
errorsFittedVColorsCFull = errorsFittedVColorsC.copy()
errorsFittedVColorsEFull = errorsFittedVColorsE.copy()
testOcclusionsFull = testOcclusions.copy()
stdevsFull = stdevs.copy()


meanAbsErrAzsArr = np.array([])
meanAbsErrElevsArr = np.array([])
medianAbsErrAzsArr = np.array([])
medianAbsErrElevsArr = np.array([])
meanErrorsLightCoeffsArr = np.array([])
meanErrorsLightCoeffsCArr = np.array([])
meanErrorsVColorsEArr = np.array([])
meanErrorsVColorsCArr = np.array([])
meanAbsErrAzsFittedArr = np.array([])
meanAbsErrElevsFittedArr = np.array([])
medianAbsErrAzsFittedArr = np.array([])
medianAbsErrElevsFittedArr = np.array([])
meanErrorsFittedLightCoeffsArr = np.array([])
meanErrorsFittedLightCoeffsCArr = np.array([])
meanErrorsFittedVColorsCArr = np.array([])
meanErrorsFittedVColorsEArr = np.array([])

occlusions = []
for occlusionLevel in range(100):

    setUnderOcclusionLevel = testOcclusionsFull*100 < occlusionLevel

    if np.any(setUnderOcclusionLevel):
        occlusions = occlusions + [occlusionLevel]
        testOcclusions = testOcclusionsFull[setUnderOcclusionLevel]

        if len(stdevsFull) >0:
            stdevs = stdevsFull[setUnderOcclusionLevel]

        colors = matplotlib.cm.plasma(testOcclusions)

        errorsPosePred = np.vstack([errorsAzPredFull[setUnderOcclusionLevel], errorsElPredFull[setUnderOcclusionLevel]])
        errorsLightCoeffs = errorsLightCoeffsFull[setUnderOcclusionLevel]
        errorsLightCoeffsC = errorsLightCoeffsCFull[setUnderOcclusionLevel]
        errorsVColorsE = errorsVColorsEFull[setUnderOcclusionLevel]
        errorsVColorsC = errorsVColorsCFull[setUnderOcclusionLevel]

        if optimizationTypeDescr[optimizationType] != 'predict':
            errorsPoseFitted =  np.vstack([errorsAzFittedFull[setUnderOcclusionLevel], errorsElFittedFull[setUnderOcclusionLevel]])
            errorsFittedLightCoeffs = errorsFittedLightCoeffsFull[setUnderOcclusionLevel]
            errorsFittedLightCoeffsC = errorsFittedLightCoeffsCFull[setUnderOcclusionLevel]
            errorsFittedVColorsC = errorsFittedVColorsCFull[setUnderOcclusionLevel]
            errorsFittedVColorsE = errorsFittedVColorsEFull[setUnderOcclusionLevel]

        meanAbsErrAzsArr = np.append(meanAbsErrAzsArr,np.mean(np.abs(errorsPosePred[0])))
        meanAbsErrElevsArr = np.append(meanAbsErrElevsArr,np.mean(np.abs(errorsPosePred[1])))

        medianAbsErrAzsArr = np.append(medianAbsErrAzsArr,np.median(np.abs(errorsPosePred[0])))
        medianAbsErrElevsArr = np.append(medianAbsErrElevsArr,np.median(np.abs(errorsPosePred[1])))

        meanErrorsLightCoeffsArr = np.append(meanErrorsLightCoeffsArr,np.sqrt(np.mean(np.mean(errorsLightCoeffs,axis=1), axis=0)))
        meanErrorsLightCoeffsCArr = np.append(meanErrorsLightCoeffsCArr,np.sqrt(np.mean(np.mean(errorsLightCoeffsC,axis=1), axis=0)))
        meanErrorsVColorsEArr = np.append(meanErrorsVColorsEArr,np.mean(errorsVColorsE, axis=0))
        meanErrorsVColorsCArr = np.append(meanErrorsVColorsCArr,np.mean(errorsVColorsC, axis=0))

        if optimizationTypeDescr[optimizationType] != 'predict':

            meanAbsErrAzsFittedArr = np.append(meanAbsErrAzsFittedArr,np.mean(np.abs(errorsPoseFitted[0])))
            meanAbsErrElevsFittedArr = np.append(meanAbsErrElevsFittedArr,np.mean(np.abs(errorsPoseFitted[1])))
            medianAbsErrAzsFittedArr = np.append(medianAbsErrAzsFittedArr,np.median(np.abs(errorsPoseFitted[0])))
            medianAbsErrElevsFittedArr = np.append(medianAbsErrElevsFittedArr,np.median(np.abs(errorsPoseFitted[1])))

        if optimizationTypeDescr[optimizationType] != 'predict':
            meanErrorsFittedLightCoeffsArr = np.append(meanErrorsFittedLightCoeffsArr,np.sqrt(np.mean(np.mean(errorsFittedLightCoeffs,axis=1), axis=0)))
            meanErrorsFittedLightCoeffsCArr = np.append(meanErrorsFittedLightCoeffsCArr,np.sqrt(np.mean(np.mean(errorsFittedLightCoeffsC,axis=1), axis=0)))
            meanErrorsFittedVColorsCArr = np.append(meanErrorsFittedVColorsCArr,np.mean(errorsFittedVColorsC, axis=0))
            meanErrorsFittedVColorsEArr = np.append(meanErrorsFittedVColorsEArr,np.mean(errorsFittedVColorsE, axis=0))

directory = resultDir + 'predictionMeanError-Azimuth'
#Show scatter correlations with predicted and azimuth error.

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(occlusions, meanAbsErrAzsArr, c='b', label="Recognition")
if optimizationTypeDescr[optimizationType] != 'predict':
    ax.plot(occlusions, meanAbsErrAzsFittedArr, c='r', label="Fit")
legend = ax.legend()
ax.set_xlabel('Occlusion (%)')
ax.set_ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
x1,x2 = ax.get_xlim()
y1,y2 = ax.get_ylim()
ax.set_xlim((0,100))
ax.set_ylim((-0.0,y2))
ax.set_title('Cumulative prediction per occlusion level')
fig.savefig(directory + '-performance-plot.png', bbox_inches='tight')
plt.close(fig)

directory = resultDir + 'predictionMeanError-Elev'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(occlusions, meanAbsErrElevsArr, c='b',label="Recognition")
if optimizationTypeDescr[optimizationType] != 'predict':
    ax.plot(occlusions, meanAbsErrElevsFittedArr, c='r',label="Fit")
legend = ax.legend()

ax.set_xlabel('Occlusion (%)')
ax.set_ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
x1,x2 = ax.get_xlim()
y1,y2 = ax.get_ylim()
ax.set_xlim((0,100))
ax.set_ylim((-0.0,y2))
ax.set_title('Cumulative prediction per occlusion level')
fig.savefig(directory + '-performance-plot.png', bbox_inches='tight')
plt.close(fig)

directory = resultDir + 'predictionMeanError-VColors-C'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(occlusions, meanErrorsVColorsCArr, c='b',label="Recognition")
if optimizationTypeDescr[optimizationType] != 'predict':
    ax.plot(occlusions, meanErrorsFittedVColorsCArr, c='r',label="Fit")
legend = ax.legend()
ax.set_xlabel('Occlusion (%)')
ax.set_ylabel('Mean VColor C Error change')
x1,x2,y1,y2 = plt.axis()
x1,x2 = ax.get_xlim()
y1,y2 = ax.get_ylim()
ax.set_xlim((0,100))
ax.set_ylim((-0.0,y2))
ax.set_title('Cumulative prediction per occlusion level')
fig.savefig(directory + '-performance-plot.png', bbox_inches='tight')
plt.close(fig)

directory = resultDir + 'predictionMeanError-VColors-E'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(occlusions, meanErrorsVColorsEArr, c='b',label="Recognition")
if optimizationTypeDescr[optimizationType] != 'predict':
    ax.plot(occlusions, meanErrorsFittedVColorsEArr, c='r',label="Fit")
legend = ax.legend()
ax.set_xlabel('Occlusion (%)')
ax.set_ylabel('Mean VColor C Error change')
x1,x2,y1,y2 = plt.axis()
x1,x2 = ax.get_xlim()
y1,y2 = ax.get_ylim()
ax.set_xlim((0,100))
ax.set_ylim((-0.0,y2))
ax.set_title('Cumulative prediction  per occlusion level')
fig.savefig(directory + '-performance-plot.png', bbox_inches='tight')
plt.close(fig)

directory = resultDir + 'predictionMeanError-SH'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(occlusions, meanErrorsLightCoeffsArr, c='b',label="Recognition")
if optimizationTypeDescr[optimizationType] != 'predict':
    ax.plot(occlusions, meanErrorsFittedLightCoeffsArr, c='r',label="Fit")
legend = ax.legend()
ax.set_xlabel('Occlusion (%)')
ax.set_ylabel('Mean SH coefficients Error change')
x1,x2,y1,y2 = plt.axis()
x1,x2 = ax.get_xlim()
y1,y2 = ax.get_ylim()
ax.set_xlim((0,100))
ax.set_ylim((-0.0,y2))
ax.set_title('Cumulative prediction per occlusion level')
fig.savefig(directory + '-performance-plot.png', bbox_inches='tight')
plt.close(fig)

directory = resultDir + 'predictionMeanError-SH-C'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(occlusions, meanErrorsLightCoeffsCArr, c='b',label="Recognition")
if optimizationTypeDescr[optimizationType] != 'predict':
    ax.plot(occlusions, meanErrorsFittedLightCoeffsCArr, c='r',label="Fit")
legend = ax.legend()
ax.set_xlabel('Occlusion (%)')
ax.set_ylabel('Mean SH coefficients Error change')
x1,x2,y1,y2 = plt.axis()
x1,x2 = ax.get_xlim()
y1,y2 = ax.get_ylim()
ax.set_xlim((0,100))
ax.set_ylim((-0.0,y2))
ax.set_title('Cumulative prediction per occlusion level')
fig.savefig(directory + '-performance-plot.png', bbox_inches='tight')
plt.close(fig)
if optimizationTypeDescr[optimizationType] != 'predict':
    directory = resultDir + 'relativeAzImprovement'


    #Show scatter correlations with predicted and azimuth error.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(occlusions, meanAbsErrAzsFittedArr- meanAbsErrAzsArr)
    legend = ax.legend()
    ax.set_xlabel('Occlusion (%)')
    ax.set_ylabel('Mean Azimuth Error change')
    x1,x2,y1,y2 = plt.axis()
    x1,x2 = ax.get_xlim()
    y1,y2 = ax.get_ylim()
    ax.set_xlim((0,100))
    ax.set_ylim((min(y1,0),max(0,y2)))
    ax.plot([0,100], [0, 0], ls="--", c=".3")
    ax.set_title('Cumulative relative change per occlusion level')
    fig.savefig(directory + '-performance-plot.png', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'relativeElevsImprovement'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(occlusions, meanAbsErrElevsFittedArr - meanAbsErrElevsArr)
    legend = ax.legend()
    ax.set_xlabel('Occlusion (%)')
    ax.set_ylabel('Mean elevation error change')
    x1,x2,y1,y2 = plt.axis()
    x1,x2 = ax.get_xlim()
    y1,y2 = ax.get_ylim()
    ax.set_xlim((0,100))
    ax.set_ylim((min(y1,0),max(0,y2)))
    ax.plot([0,100], [0, 0], ls="--", c=".3")
    ax.set_title('Cumulative relative change per occlusion level')
    fig.savefig(directory + '-performance-plot.png', bbox_inches='tight')


    directory = resultDir + 'relativeVColors-C-Improvement'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(occlusions, meanErrorsFittedVColorsCArr - meanErrorsVColorsCArr)
    legend = ax.legend()
    ax.set_xlabel('Occlusion (%)')
    ax.set_ylabel('Mean vertex color error change')
    x1,x2,y1,y2 = plt.axis()
    x1,x2 = ax.get_xlim()
    y1,y2 = ax.get_ylim()
    ax.set_xlim((0,100))
    ax.set_ylim((min(y1,0),max(0,y2)))
    ax.plot([0,100], [0, 0], ls="--", c=".3")
    ax.set_title('Cumulative relative change per occlusion level')
    fig.savefig(directory + '-performance-plot.png', bbox_inches='tight')

    directory = resultDir + 'relativeVColors-E-Improvement'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(occlusions, meanErrorsFittedVColorsEArr - meanErrorsVColorsEArr)
    legend = ax.legend()
    ax.set_xlabel('Occlusion (%)')
    ax.set_ylabel('Mean vertex color error change')
    x1,x2,y1,y2 = plt.axis()
    x1,x2 = ax.get_xlim()
    y1,y2 = ax.get_ylim()
    ax.set_xlim((0,100))
    ax.set_ylim((min(y1,0),max(0,y2)))
    ax.plot([0,100], [0, 0], ls="--", c=".3")
    ax.set_title('Cumulative relative change per occlusion level')
    fig.savefig(directory + '-performance-plot.png', bbox_inches='tight')

    directory = resultDir + 'relativeSH-Improvement'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(occlusions, meanErrorsFittedLightCoeffsArr - meanErrorsLightCoeffsArr)
    legend = ax.legend()
    ax.set_xlabel('Occlusion (%)')
    ax.set_ylabel('Mean SH coefficients Error change')
    x1,x2,y1,y2 = plt.axis()
    x1,x2 = ax.get_xlim()
    y1,y2 = ax.get_ylim()
    ax.set_xlim((0,100))
    ax.set_ylim((min(y1,0),max(0,y2)))
    ax.plot([0,100], [0, 0], ls="--", c=".3")
    ax.set_title('Cumulative relative change per occlusion level')
    fig.savefig(directory + '-performance-plot.png', bbox_inches='tight')

    directory = resultDir + 'relativeSH-C-Improvement'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(occlusions, meanErrorsFittedLightCoeffsCArr - meanErrorsLightCoeffsCArr)
    legend = ax.legend()
    ax.set_xlabel('Occlusion (%)')
    ax.set_ylabel('Mean SH coefficients error change')
    x1,x2,y1,y2 = plt.axis()
    x1,x2 = ax.get_xlim()
    y1,y2 = ax.get_ylim()
    ax.set_xlim((0,100))
    ax.set_ylim((min(y1,0),max(0,y2)))
    ax.plot([0,100], [0, 0], ls="--", c=".3")
    ax.set_title('Cumulative relative change per occlusion level')
    fig.savefig(directory + '-performance-plot.png', bbox_inches='tight')

for occlusionLevel in [25,50,75,100]:

    resultDir = 'results/' + testPrefix + '/occlusion' + str(occlusionLevel)  + '/'
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)

    setUnderOcclusionLevel = testOcclusionsFull*100 < occlusionLevel
    testOcclusions = testOcclusionsFull[setUnderOcclusionLevel]

    if len(stdevsFull) > 0:
        stdevs = stdevsFull[setUnderOcclusionLevel]

    colors = matplotlib.cm.plasma(testOcclusions)

    errorsPosePred = np.vstack([errorsAzPredFull[setUnderOcclusionLevel], errorsElPredFull[setUnderOcclusionLevel]])
    errorsLightCoeffs = errorsLightCoeffsFull[setUnderOcclusionLevel]
    errorsLightCoeffsC = errorsLightCoeffsCFull[setUnderOcclusionLevel]
    errorsVColorsE = errorsVColorsEFull[setUnderOcclusionLevel]
    errorsVColorsC = errorsVColorsCFull[setUnderOcclusionLevel]

    if optimizationTypeDescr[optimizationType] != 'predict':
        errorsPoseFitted =  np.vstack([errorsAzFittedFull[setUnderOcclusionLevel], errorsElFittedFull[setUnderOcclusionLevel]])
        errorsFittedLightCoeffs = errorsFittedLightCoeffsFull[setUnderOcclusionLevel]
        errorsFittedLightCoeffsC = errorsFittedLightCoeffsCFull[setUnderOcclusionLevel]
        errorsFittedVColorsC = errorsFittedVColorsCFull[setUnderOcclusionLevel]
        errorsFittedVColorsE = errorsFittedVColorsEFull[setUnderOcclusionLevel]

    meanAbsErrAzs = np.mean(np.abs(errorsPosePred[0]))
    meanAbsErrElevs = np.mean(np.abs(errorsPosePred[1]))

    medianAbsErrAzs = np.median(np.abs(errorsPosePred[0]))
    medianAbsErrElevs = np.median(np.abs(errorsPosePred[1]))

    meanErrorsLightCoeffs = np.sqrt(np.mean(np.mean(errorsLightCoeffs,axis=1), axis=0))
    meanErrorsLightCoeffsC = np.sqrt(np.mean(np.mean(errorsLightCoeffsC,axis=1), axis=0))
    meanErrorsVColorsE = np.mean(errorsVColorsE, axis=0)
    meanErrorsVColorsC = np.mean(errorsVColorsC, axis=0)

    if optimizationTypeDescr[optimizationType] != 'predict':

        meanAbsErrAzsFitted = np.mean(np.abs(errorsPoseFitted[0]))
        meanAbsErrElevsFitted = np.mean(np.abs(errorsPoseFitted[1]))
        medianAbsErrAzsFitted = np.median(np.abs(errorsPoseFitted[0]))
        medianAbsErrElevsFitted = np.median(np.abs(errorsPoseFitted[1]))

    if optimizationTypeDescr[optimizationType] != 'predict':
        meanErrorsFittedLightCoeffs = np.sqrt(np.mean(np.mean(errorsFittedLightCoeffs,axis=1), axis=0))
        meanErrorsFittedLightCoeffsC = np.sqrt(np.mean(np.mean(errorsFittedLightCoeffsC,axis=1), axis=0))
        meanErrorsFittedVColorsC = np.mean(errorsFittedVColorsC, axis=0)
        meanErrorsFittedVColorsE = np.mean(errorsFittedVColorsE, axis=0)

    if len(stdevs) > 0:

        directory = resultDir + 'nnazsamples_azerror'

        fig = plt.figure()
        ax = fig.add_subplot(111)
        scat = ax.scatter(testOcclusions * 100.0, errorsPosePred[0], s=20, vmin=0, vmax=100, c=testOcclusions*100, cmap=matplotlib.cm.plasma)
        cbar = fig.colorbar(scat, ticks=[0, 50, 100])
        cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
        ax.set_xlabel('Occlusion (%)')
        ax.set_ylabel('Vertex Colors errors')
        x1,x2 = ax.get_xlim()
        y1,y2 = ax.get_ylim()
        ax.set_xlim((0,180))
        ax.set_ylim((-180,180))
        ax.set_title('Neural net samples Performance scatter plot')

        fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
        plt.close(fig)


        directory = resultDir + 'nnazsamples_vcolorserrorsC'

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        scat = ax.scatter(testOcclusions * 100.0, errorsVColorsC, s=20, vmin=0, vmax=100, c=testOcclusions*100, cmap=matplotlib.cm.plasma)
        cbar = fig.colorbar(scat, ticks=[0, 50, 100])
        cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
        ax.set_xlabel('Occlusion (%)')
        ax.set_ylabel('Vertex Colors errors')
        x1,x2 = ax.get_xlim()
        y1,y2 = ax.get_ylim()
        ax.set_xlim((0,180))
        ax.set_ylim((0,1))
        ax.set_title('Neural net samples performance scatter plot')

        fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
        plt.close(fig)

    if not optimizationTypeDescr[optimizationType] == 'predict':

        directory = resultDir + 'pred-azimuth-errors_fitted-azimutherror'
        #Show scatter correlations with occlusions.

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        scat = ax.scatter(abs(errorsPosePred[0]), abs(errorsPoseFitted[0]), s=20, vmin=0, vmax=100, c=testOcclusions*100, cmap=matplotlib.cm.plasma)
        cbar = fig.colorbar(scat, ticks=[0, 50, 100])
        cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
        ax.set_xlabel('Predicted azimuth errors')
        ax.set_ylabel('Fitted azimuth errors')
        x1,x2 = ax.get_xlim()
        y1,y2 = ax.get_ylim()
        ax.set_xlim((0,180))
        ax.set_ylim((0,180))
        ax.plot([0, 180], [0, 180], ls="--", c=".3")
        ax.set_title('Predicted vs fitted azimuth errors')
        fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
        plt.close(fig)


        directory = resultDir + 'pred-elevation-errors_fitted-elevation-error'
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        scat = ax.scatter(abs(errorsPosePred[1]), abs(errorsPoseFitted[1]), s=20, vmin=0, vmax=100, c=testOcclusions*100, cmap=matplotlib.cm.plasma)
        cbar = fig.colorbar(scat, ticks=[0, 50, 100])
        cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
        ax.set_xlabel('Predicted elevation errors')
        ax.set_ylabel('Fitted elevation errors')
        x1,x2 = ax.get_xlim()
        y1,y2 = ax.get_ylim()
        ax.set_xlim((0,90))
        ax.set_ylim((0,90))
        ax.plot([0,90], [0,90], ls="--", c=".3")
        ax.set_title('Predicted vs fitted azimuth errors')
        fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
        plt.close(fig)


        directory = resultDir + 'shcoeffsserrorsC_fitted-shcoeffsserrorsC'

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        scat = ax.scatter(np.sqrt(np.mean(errorsLightCoeffsC,axis=1)), np.sqrt(np.mean(errorsFittedLightCoeffsC,axis=1)), s=20, vmin=0, vmax=100, c=testOcclusions*100, cmap=matplotlib.cm.plasma)
        cbar = fig.colorbar(scat, ticks=[0, 50, 100])
        cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
        ax.set_xlabel('Predicted SH coefficients errors')
        ax.set_ylabel('Fitted SH coefficients errors')
        x1,x2 = ax.get_xlim()
        y1,y2 = ax.get_ylim()
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.plot([0, 1], [0, 1], ls="--", c=".3")
        ax.set_title('Predicted vs fitted SH coefficients errors')
        fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
        plt.close(fig)

        directory = resultDir + 'shcoeffsserrors_fitted-shcoeffsserrors'
        #Show scatter correlations with occlusions.
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        scat = ax.scatter(np.sqrt(np.mean(errorsLightCoeffs,axis=1)), np.sqrt(np.mean(errorsFittedLightCoeffs,axis=1)), s=20, vmin=0, vmax=100, c=testOcclusions*100, cmap=matplotlib.cm.plasma)
        cbar = fig.colorbar(scat, ticks=[0, 50, 100])
        cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
        ax.set_xlabel('Predicted SH coefficients errors')
        ax.set_ylabel('Fitted SH coefficients errors')
        x1,x2 = ax.get_xlim()
        y1,y2 = ax.get_ylim()
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.plot([0, 1], [0, 1], ls="--", c=".3")
        ax.set_title('Predicted vs fitted SH coefficients errors')
        fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
        plt.close(fig)


        directory = resultDir + 'vColorsE_fitted-vColorsE'

        #Show scatter correlations with occlusions.
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        scat = ax.scatter(errorsVColorsE, errorsFittedVColorsE, s=20, vmin=0, vmax=100, c=testOcclusions*100, cmap=matplotlib.cm.plasma)
        cbar = fig.colorbar(scat, ticks=[0, 50, 100])
        cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
        ax.set_xlabel('Predicted VColor E coefficients errors')
        ax.set_ylabel('Fitted VColor E coefficients errors')
        x1,x2 = ax.get_xlim()
        y1,y2 = ax.get_ylim()
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.plot([0, 1], [0, 1], ls="--", c=".3")
        ax.set_title('Predicted vs fitted vertex color errors')
        fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
        plt.close(fig)


        directory = resultDir + 'vColorsC_fitted-vColorsC'

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        scat = ax.scatter(errorsVColorsC, errorsFittedVColorsC, s=20, vmin=0, vmax=100, c=testOcclusions*100, cmap=matplotlib.cm.plasma)
        cbar = fig.colorbar(scat, ticks=[0, 50, 100])
        cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar
        ax.set_xlabel('Predicted VColor C coefficients errors')
        ax.set_ylabel('Fitted VColor C coefficients errors')
        x1,x2 = ax.get_xlim()
        y1,y2 = ax.get_ylim()
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.plot([0, 1], [0, 1], ls="--", c=".3")
        ax.set_title('Predicted vs fitted vertex color errors')
        fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
        plt.close(fig)

    directory = resultDir + 'predicted-azimuth-error'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(errorsPosePred[0], bins=18)
    ax.set_xlabel('Angular error')
    ax.set_ylabel('Counts')
    x1,x2 = ax.get_xlim()
    y1,y2 = ax.get_ylim()
    ax.set_xlim((-180,180))
    ax.set_ylim((y1, y2))
    ax.set_title('Error histogram')
    fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'predicted-elevation-error'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(errorsPosePred[1], bins=18)
    ax.set_xlabel('Angular error')
    ax.set_ylabel('Counts')
    x1,x2 = ax.get_xlim()
    y1,y2 = ax.get_ylim()
    ax.set_xlim((-90,90))
    ax.set_ylim((y1, y2))
    ax.set_title('Error histogram')
    fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)


    #Fitted predictions plots:

    directory = resultDir + 'fitted-azimuth-error'
    if not optimizationTypeDescr[optimizationType] == 'predict':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(errorsPoseFitted[0], bins=18)
        ax.set_xlabel('Angular error')
        ax.set_ylabel('Counts')
        x1,x2 = ax.get_xlim()
        y1,y2 = ax.get_ylim()
        ax.set_xlim((-180,180))
        ax.set_ylim((y1, y2))
        ax.set_title('Error histogram')
        fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
        plt.close(fig)


        directory = resultDir + 'fitted-elevation-error'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(errorsPoseFitted[1], bins=18)
        ax.set_xlabel('Angular error')
        ax.set_ylabel('Counts')
        x1,x2 = ax.get_xlim()
        y1,y2 = ax.get_ylim()
        ax.set_xlim((-90,90))
        ax.set_ylim((y1, y2))
        ax.set_title('Error histogram')
        fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
        plt.close(fig)


    #Write statistics to file.
    with open(resultDir + 'performance.txt', 'w') as expfile:
        # expfile.write(str(z))
        expfile.write("Average occlusions: " + str(np.mean(testOcclusions*100)))
        expfile.write("Avg Pred NLL    :" +  str(np.mean(predictedErrorFuns))+ '\n')
        expfile.write("Avg Fitt NLL    :" +  str(np.mean(fittedErrorFuns))+ '\n')
        expfile.write("Mean Azimuth Error (predicted) " +  str(meanAbsErrAzs) + '\n')
        expfile.write("Mean Elevation Error (predicted) " +  str(meanAbsErrElevs)+ '\n')
        expfile.write("Median Azimuth Error (predicted) " +  str(medianAbsErrAzs) + '\n')
        expfile.write("Median Elevation Error (predicted) " +  str(medianAbsErrElevs)+ '\n')
        if not optimizationTypeDescr[optimizationType] == 'predict':
            expfile.write("Mean Azimuth Error (fitted) " + str(meanAbsErrAzsFitted) + '\n')
            expfile.write("Mean Elevation Error (fitted) " + str(meanAbsErrElevsFitted) + '\n')
            expfile.write("Median Azimuth Error (fitted) " + str(medianAbsErrAzsFitted) + '\n')
            expfile.write("Median Elevation Error (fitted) " + str(medianAbsErrElevsFitted) + '\n')
        expfile.write("Mean SH Components Error (predicted) " +  str(meanErrorsLightCoeffs)+ '\n')
        expfile.write("Mean SH Components Error C (predicted) " +  str(meanErrorsLightCoeffsC)+ '\n')
        expfile.write("Mean Vertex Colors Error E (predicted) " +  str(meanErrorsVColorsE)+ '\n')
        expfile.write("Mean Vertex Colors Error C (predicted) " +  str(meanErrorsVColorsC)+ '\n')
        if not optimizationTypeDescr[optimizationType] == 'predict':
            expfile.write("Mean SH Components Error (fitted) " +  str(meanErrorsFittedLightCoeffs)+ '\n')
            expfile.write("Mean SH Components Error C (fitted) " +  str(meanErrorsFittedLightCoeffsC)+ '\n')
            expfile.write("Mean Vertex Colors Error E (fitted) " +  str(meanErrorsFittedVColorsE)+ '\n')
            expfile.write("Mean Vertex Colors Error C (fitted) " +  str(meanErrorsFittedVColorsC)+ '\n')

    # if not optimizationTypeDescr[optimizationType] == 'predict':
    #     headerDesc = "Pred NLL    :" + "Fitt NLL    :" + "Err Pred Az :" + "Err Pred El :" + "Err Fitted Az :" + "Err Fitted El :" + "Occlusions  :"
    #     perfSamplesData = np.hstack([predictedErrorFuns.reshape([-1,1]), fittedErrorFuns.reshape([-1,1]), errorsPosePred[0].reshape([-1, 1]), errorsPosePred[1].reshape([-1, 1]), errorsPoseFitted[0].reshape([-1, 1]), errorsPoseFitted[1].reshape([-1, 1]), testOcclusions.reshape([-1, 1])])
    # elif optimizationTypeDescr[optimizationType] == 'predict' and computePredErrorFuns:
    #     headerDesc = "Pred NLL    :" + "Err Pred Az :" + "Err Pred El :"  +  "Occlusions  :"
    #     perfSamplesData = np.hstack([predictedErrorFuns.reshape([-1,1]), errorsPosePred[0].reshape([-1, 1]), errorsPosePred[1].reshape([-1, 1]), testOcclusions.reshape([-1, 1])])
    # else:
    #     headerDesc = "Err Pred Az :" + "Err Pred El :"  +  "Occlusions  :"
    #     perfSamplesData = np.hstack([errorsPosePred[0].reshape([-1, 1]), errorsPosePred[1].reshape([-1, 1]), testOcclusions.reshape([-1, 1])])
    #
    # np.savetxt(resultDir + 'performance_samples.txt', perfSamplesData, delimiter=',', fmt="%g", header=headerDesc)

    import tabulate
    headers=["Errors", "Pred (mean)", "Fitted (mean)"]

    table = [["NLL", np.mean(predictedErrorFuns), 0, np.mean(fittedErrorFuns), 0],
             ["Azimuth", np.mean(np.abs(errorsPosePred[0])), np.mean(np.abs(errorsPoseFitted[0]))],
             ["Elevation", np.mean(np.abs(errorsPosePred[1])), np.mean(np.abs(errorsPoseFitted[1]))],
             ["VColor C", meanErrorsVColorsC, meanErrorsFittedVColorsC],
             ["SH Light", meanErrorsLightCoeffs, meanErrorsFittedLightCoeffs],
             ["SH Light C", meanErrorsLightCoeffsC, meanErrorsFittedLightCoeffsC]
             ]
    performanceTable = tabulate.tabulate(table, headers=headers,tablefmt="latex", floatfmt=".2f")
    with open(resultDir + 'performance.tex', 'w') as expfile:
        expfile.write(performanceTable)


    headers=["Method", "l=0", "SH $l=0,m=-1$", "SH $l=1,m=0$", "SH $l=1,m=1$", "SH $l=1,m=-2$" , "SH $l=2,m=-1$", "SH $l=2,m=0$", "SH $l=2,m=1$", "SH $l=2,m=2$"]


    # SMSE_SH_std = np.mean((testLightCoefficientsGTRel - relLightCoefficientsGTPred)**2 + 1e-5, axis=0)/(np.var(testLightCoefficientsGTRel, axis=0) + 1e-5)
    # table = [[SHModel, SMSE_SH_std[0], SMSE_SH_std[1], SMSE_SH_std[2],SMSE_SH_std[3], SMSE_SH_std[4], SMSE_SH_std[5],SMSE_SH_std[6], SMSE_SH_std[7], SMSE_SH_std[8] ],
    #         ]
    # performanceTable = tabulate.tabulate(table, headers=headers,tablefmt="latex", floatfmt=".2f")
    # with open(resultDir + 'performance_SH_standardised.tex', 'w') as expfile:
    #     expfile.write(performanceTable)

    SMSE_SH = np.sqrt(np.mean(errorsLightCoeffs, axis=0))
    table = [[SHModel, SMSE_SH[0], SMSE_SH[1], SMSE_SH[2],SMSE_SH[3], SMSE_SH[4], SMSE_SH[5],SMSE_SH[6], SMSE_SH[7], SMSE_SH[8] ],
            ]
    performanceTable = tabulate.tabulate(table, headers=headers,tablefmt="latex", floatfmt=".2f")
    with open(resultDir + 'performance_SH.tex', 'w') as expfile:
        expfile.write(performanceTable)

    # SMSE_SH_std = np.mean((testLightCoefficientsGTRel - fittedRelLightCoeffs)**2 + 1e-5, axis=0)/(np.var(testLightCoefficientsGTRel, axis=0) + 1e-5)
    # table = [[SHModel, SMSE_SH_std[0], SMSE_SH_std[1], SMSE_SH_std[2],SMSE_SH_std[3], SMSE_SH_std[4], SMSE_SH_std[5],SMSE_SH_std[6], SMSE_SH_std[7], SMSE_SH_std[8] ],
    #         ]
    # performanceTable = tabulate.tabulate(table, headers=headers,tablefmt="latex", floatfmt=".2f")
    # with open(resultDir + 'performance_SH_fit_standardised.tex', 'w') as expfile:
    #     expfile.write(performanceTable)

    if not optimizationTypeDescr[optimizationType] == 'predict':
        SMSE_SH = np.sqrt(np.mean(errorsFittedLightCoeffs, axis=0))
        table = [[SHModel, SMSE_SH[0], SMSE_SH[1], SMSE_SH[2],SMSE_SH[3], SMSE_SH[4], SMSE_SH[5],SMSE_SH[6], SMSE_SH[7], SMSE_SH[8] ],
                ]
        performanceTable = tabulate.tabulate(table, headers=headers,tablefmt="latex", floatfmt=".2f")
        with open(resultDir + 'performance_SH_fit.tex', 'w') as expfile:
            expfile.write(performanceTable)


    SMSE_SH = np.sqrt(np.mean(errorsLightCoeffsC, axis=0))
    table = [[SHModel, SMSE_SH[0], SMSE_SH[1], SMSE_SH[2],SMSE_SH[3], SMSE_SH[4], SMSE_SH[5],SMSE_SH[6], SMSE_SH[7], SMSE_SH[8] ],
            ]
    performanceTable = tabulate.tabulate(table, headers=headers,tablefmt="latex", floatfmt=".2f")
    with open(resultDir + 'performance_SH_C.tex', 'w') as expfile:
        expfile.write(performanceTable)

    # SMSE_SH_std = np.mean((testLightCoefficientsGTRel - fittedRelLightCoeffs)**2 + 1e-5, axis=0)/(np.var(testLightCoefficientsGTRel, axis=0) + 1e-5)
    # table = [[SHModel, SMSE_SH_std[0], SMSE_SH_std[1], SMSE_SH_std[2],SMSE_SH_std[3], SMSE_SH_std[4], SMSE_SH_std[5],SMSE_SH_std[6], SMSE_SH_std[7], SMSE_SH_std[8] ],
    #         ]
    # performanceTable = tabulate.tabulate(table, headers=headers,tablefmt="latex", floatfmt=".2f")
    # with open(resultDir + 'performance_SH_fit_standardised.tex', 'w') as expfile:
    #     expfile.write(performanceTable)

    if not optimizationTypeDescr[optimizationType] == 'predict':
        SMSE_SH = np.sqrt(np.mean(errorsFittedLightCoeffsC, axis=0))
        table = [[SHModel, SMSE_SH[0], SMSE_SH[1], SMSE_SH[2],SMSE_SH[3], SMSE_SH[4], SMSE_SH[5],SMSE_SH[6], SMSE_SH[7], SMSE_SH[8] ],
                ]
        performanceTable = tabulate.tabulate(table, headers=headers,tablefmt="latex", floatfmt=".2f")
        with open(resultDir + 'performance_SH_fit_C.tex', 'w') as expfile:
            expfile.write(performanceTable)

plt.ion()
print("Finished backprojecting and fitting estimates.")

