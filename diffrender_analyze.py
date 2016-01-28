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
testPrefix = 'train4_occlusion_opt_train4occlusion10k_1000s_std01_optAll_nnLight9_nnApp'

testPrefixWrite= 'train4_occlusion_opt_train4occlusion10k_1000s_std01_optAll_nnLight9_nnApp_ignore'

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
experimentPrefix = 'train4_occlusion_10k'
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

testLightCoefficientsGTRel = dataLightCoefficientsGTRel * dataAmbientIntensityGT[:,None]

testAzsRel = np.mod(testAzsGT - testObjAzsGT, 2*np.pi)

if os.path.isfile(resultDir + 'samples.npy'):
    samplesfile = np.load(resultDir + 'performance_samples.npz')
    azsPred=samplesfile['azsPred']
    elevsPred=samplesfile['elevsPred']
    fittedAzs=samplesfile['fittedAzs']
    fittedElevs=samplesfile['fittedElevs']
    vColorsPred=samplesfile['vColorsPred']
    fittedVColors=samplesfile['fittedVColors']
    relLightCoefficientsGTPred=samplesfile['relLightCoefficientsGTPred']
    fittedRelLightCoeffs=samplesfile['fittedRelLightCoeffs']

performancefile = np.load(resultDir + 'performance_samples.npz')
predictedErrorFuns = performancefile['predictedErrorFuns']
fittedErrorFuns = performancefile['fittedErrorFuns']
predErrorAzs = performancefile['predErrorAzs']
predErrorElevs = performancefile['predErrorElevs']
errorsLightCoeffs = performancefile['errorsLightCoeffs']
errorsVColors = performancefile['errorsVColors']
errorsFittedAzs = performancefile['errorsFittedAzs']
errorsFittedElevs = performancefile['errorsFittedElevs']
errorsFittedLightCoeffs = performancefile['errorsFittedLightCoeffs']
fittedRelLightCoeffs = errorsFittedLightCoeffs
errorsFittedVColors = performancefile['errorsFittedVColors']
testOcclusions = performancefile['testOcclusions']

errorsPosePred = np.vstack([predErrorAzs[None,:],predErrorElevs[None,:]])
errorsPoseFitted = np.vstack([errorsFittedAzs[None,:],errorsFittedElevs[None,:]])


meanAbsErrAzs = np.mean(np.abs(errorsPosePred[0]))
meanAbsErrElevs = np.mean(np.abs(errorsPosePred[1]))

medianAbsErrAzs = np.median(np.abs(errorsPosePred[0]))
medianAbsErrElevs = np.median(np.abs(errorsPosePred[1]))

meanErrorsLightCoeffs = np.mean(errorsLightCoeffs)
meanErrorsVColors = np.mean(errorsVColors)

stdevs = np.load(resultDir + 'nn_azpred_samples.npz')['arr_0']

plt.ioff()
import seaborn


if len(stdevs) > 0:

    directory = resultDir + 'nnazsamples_azerror'

    #Show scatter correlations with predicted and azimuth error.
    fig = plt.figure()
    plt.scatter(stdevs*180/np.pi, errorsPosePred[0])
    plt.xlabel('NN Az samples stdev')
    plt.ylabel('Angular error')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,180,-180,180))
    plt.title('Neural net samples Performance scatter plot')
    fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)

    #Show scatter correlations with occlusions.
    directory = resultDir + 'occlusion_nnazsamplesstdev'
    fig = plt.figure()
    plt.scatter(testOcclusions * 100.0, stdevs*180/np.pi)
    plt.xlabel('Occlusion (%)')
    plt.ylabel('NN Az samples stdev')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,100,0,180))
    plt.title('Occlusion vs az predictions correlations')
    fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'nnazsamples_vcolorserrors'

    #Show scatter correlations with predicted and azimuth error.
    fig = plt.figure()
    plt.scatter(stdevs*180/np.pi, errorsVColors)
    plt.xlabel('NN Az samples stdev')
    plt.ylabel('VColor error')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,180,-0.0,1))
    plt.title('Neural net samples performance scatter plot')
    fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)


directory = resultDir + 'occlusion_vcolorserrors'
#Show scatter correlations with occlusions.
fig = plt.figure()
plt.scatter(testOcclusions * 100.0, errorsVColors)
plt.xlabel('Occlusion (%)')
plt.ylabel('Vertex Colors errors')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,100,-0.0,1))
plt.title('Occlusion - vertex color predictions')
fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
plt.close(fig)

directory = resultDir + 'occlusion_vcolorserrors'
#Show scatter correlations with occlusions.
fig = plt.figure()
plt.scatter(testOcclusions * 100.0, errorsVColors)
plt.xlabel('Occlusion (%)')
plt.ylabel('Vertex Colors errors')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,100,-0.0,1))
plt.title('Occlusion - vertex color fitted predictions')
fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
plt.close(fig)

directory = resultDir + 'occlusion_shcoeffsserrors'
#Show scatter correlations with occlusions.
fig = plt.figure()
plt.scatter(testOcclusions * 100.0, errorsLightCoeffs)
plt.xlabel('Occlusion (%)')
plt.ylabel('Vertex Colors errors')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,100,-0.0,1))
plt.title('Occlusion - SH coefficients fitted errors')
fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
plt.close(fig)


# Show scatter correlations of occlusion with predicted and fitted color.
if not optimizationTypeDescr[optimizationType] == 'predict':
    directory = resultDir + 'fitted-occlusion_vcolorserrors'
    #Show scatter correlations with occlusions.
    fig = plt.figure()
    plt.scatter(testOcclusions * 100.0, errorsVColors)
    plt.xlabel('Occlusion (%)')
    plt.ylabel('Vertex Colors errors')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,100,-0.0,1))
    plt.title('Occlusion - vertex color fitted errors')
    fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'fitted-occlusion_shcoeffsserrors'
    #Show scatter correlations with occlusions.
    fig = plt.figure()
    plt.scatter(testOcclusions * 100.0, errorsFittedLightCoeffs)
    plt.xlabel('Occlusion (%)')
    plt.ylabel('Fitted SH coefficients errors')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,100,-0.0,1))
    plt.title('Occlusion - SH coefficients fitted errors')
    fig.savefig(directory + '-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)

#Show performance averages for different occlusion cut-offs. E.g.


if optimizationTypeDescr[optimizationType] != 'predict':

    meanAbsErrAzsFitted = np.mean(np.abs(errorsPoseFitted[0]))
    meanAbsErrElevsFitted = np.mean(np.abs(errorsPoseFitted[1]))
    medianAbsErrAzsFitted = np.median(np.abs(errorsPoseFitted[0]))
    medianAbsErrElevsFitted = np.median(np.abs(errorsPoseFitted[1]))

if optimizationTypeDescr[optimizationType] != 'predict':
    meanErrorsFittedLightCoeffs = np.mean(errorsFittedLightCoeffs)
    meanErrorsFittedVColors = np.mean(errorsFittedVColors)


directory = resultDir + 'predicted-azimuth-error'

fig = plt.figure()
plt.scatter(testElevsGT * 180 / np.pi, errorsPosePred[0])
plt.xlabel('Elevation (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,90,-90,90))
plt.title('Performance scatter plot')
fig.savefig(directory + '_elev-performance-scatter.png', bbox_inches='tight')
plt.close(fig)

fig = plt.figure()
plt.scatter(testOcclusions * 100.0, errorsPosePred[0])
plt.xlabel('Occlusion (%)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,100,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory + '_occlusion-performance-scatter.png', bbox_inches='tight')
plt.close(fig)

fig = plt.figure()
plt.scatter(testAzsRel * 180 / np.pi, errorsPosePred[0])
plt.xlabel('Azimuth (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,360,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory  + '_azimuth-performance-scatter.png', bbox_inches='tight')
plt.close(fig)

fig = plt.figure()
plt.hist(errorsPosePred[0], bins=18)
plt.xlabel('Angular error')
plt.ylabel('Counts')
x1,x2,y1,y2 = plt.axis()
plt.axis((-180,180,y1, y2))
plt.title('Performance histogram')
fig.savefig(directory  + '_performance-histogram.png', bbox_inches='tight')
plt.close(fig)

directory = resultDir + 'predicted-elevation-error'

fig = plt.figure()
plt.scatter(testElevsGT * 180 / np.pi, errorsPosePred[1])
plt.xlabel('Elevation (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,90,-90,90))
plt.title('Performance scatter plot')
fig.savefig(directory + '_elev-performance-scatter.png', bbox_inches='tight')
plt.close(fig)

fig = plt.figure()
plt.scatter(testOcclusions * 100.0, errorsPosePred[1])
plt.xlabel('Occlusion (%)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,100,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory + '_occlusion-performance-scatter.png', bbox_inches='tight')
plt.close(fig)

fig = plt.figure()
plt.scatter(testAzsRel * 180 / np.pi, errorsPosePred[1])
plt.xlabel('Azimuth (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,360,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory  + '_azimuth-performance-scatter.png', bbox_inches='tight')
plt.close(fig)

fig = plt.figure()
plt.hist(errorsPosePred[1], bins=18)
plt.xlabel('Angular error')
plt.ylabel('Counts')
x1,x2,y1,y2 = plt.axis()
plt.axis((-180,180,y1, y2))
plt.title('Performance histogram')
fig.savefig(directory  + '_performance-histogram.png', bbox_inches='tight')
plt.close(fig)

#Fitted predictions plots:

directory = resultDir + 'fitted-azimuth-error'
if not optimizationTypeDescr[optimizationType] == 'predict':
    fig = plt.figure()
    plt.scatter(testElevsGT * 180 / np.pi, errorsPoseFitted[0])
    plt.xlabel('Elevation (degrees)')
    plt.ylabel('Angular error')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,90,-90,90))
    plt.title('Performance scatter plot')
    fig.savefig(directory + '_elev-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(testOcclusions * 100.0, errorsPoseFitted[0])
    plt.xlabel('Occlusion (%)')
    plt.ylabel('Angular error')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,100,-180,180))
    plt.title('Performance scatter plot')
    fig.savefig(directory + '_occlusion-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(testAzsRel * 180 / np.pi, errorsPoseFitted[0])
    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('Angular error')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,360,-180,180))
    plt.title('Performance scatter plot')
    fig.savefig(directory  + '_azimuth-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.hist(errorsPoseFitted[0], bins=18)
    plt.xlabel('Angular error')
    plt.ylabel('Counts')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((-180,180,y1, y2))
    plt.title('Performance histogram')
    fig.savefig(directory  + '_performance-histogram.png', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'fitted-elevation-error'

    fig = plt.figure()
    plt.scatter(testElevsGT * 180 / np.pi, errorsPoseFitted[1])
    plt.xlabel('Elevation (degrees)')
    plt.ylabel('Angular error')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,90,-90,90))
    plt.title('Performance scatter plot')
    fig.savefig(directory + '_elev-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(testOcclusions * 100.0, errorsPoseFitted[1])
    plt.xlabel('Occlusion (%)')
    plt.ylabel('Angular error')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,100,-180,180))
    plt.title('Performance scatter plot')
    fig.savefig(directory + '_occlusion-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(testAzsRel * 180 / np.pi, errorsPoseFitted[1])
    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('Angular error')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,360,-180,180))
    plt.title('Performance scatter plot')
    fig.savefig(directory  + '_azimuth-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.hist(errorsPoseFitted[1], bins=18)
    plt.xlabel('Angular error')
    plt.ylabel('Counts')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((-180,180,y1, y2))
    plt.title('Performance histogram')
    fig.savefig(directory  + '_performance-histogram.png', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'fitted-robust-azimuth-error'

plt.ion()

#Write statistics to file.
with open(resultDir + 'performance.txt', 'w') as expfile:
    # expfile.write(str(z))
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
    expfile.write("Mean Vertex Colors Error (predicted) " +  str(meanErrorsVColors)+ '\n')
    if not optimizationTypeDescr[optimizationType] == 'predict':
        expfile.write("Mean SH Components Error (fitted) " +  str(meanErrorsFittedLightCoeffs)+ '\n')
        expfile.write("Mean Vertex Colors Error (fitted) " +  str(meanErrorsFittedVColors)+ '\n')

if not optimizationTypeDescr[optimizationType] == 'predict':
    headerDesc = "Pred NLL    :" + "Fitt NLL    :" + "Err Pred Az :" + "Err Pred El :" + "Err Fitted Az :" + "Err Fitted El :" + "Occlusions  :"
    perfSamplesData = np.hstack([predictedErrorFuns.reshape([-1,1]), fittedErrorFuns.reshape([-1,1]), errorsPosePred[0].reshape([-1, 1]), errorsPosePred[1].reshape([-1, 1]), errorsPoseFitted[0].reshape([-1, 1]), errorsPoseFitted[1].reshape([-1, 1]), testOcclusions.reshape([-1, 1])])
elif optimizationTypeDescr[optimizationType] == 'predict' and computePredErrorFuns:
    headerDesc = "Pred NLL    :" + "Err Pred Az :" + "Err Pred El :"  +  "Occlusions  :"
    perfSamplesData = np.hstack([predictedErrorFuns.reshape([-1,1]), errorsPosePred[0].reshape([-1, 1]), errorsPosePred[1].reshape([-1, 1]), testOcclusions.reshape([-1, 1])])
else:
    headerDesc = "Err Pred Az :" + "Err Pred El :"  +  "Occlusions  :"
    perfSamplesData = np.hstack([errorsPosePred[0].reshape([-1, 1]), errorsPosePred[1].reshape([-1, 1]), testOcclusions.reshape([-1, 1])])

np.savetxt(resultDir + 'performance_samples.txt', perfSamplesData, delimiter=',', fmt="%g", header=headerDesc)

import tabulate
headers=["Errors", "Pred (mean)", "Stdv", "Fitted (mean)", "Stdv"]

table = [["NLL", np.mean(predictedErrorFuns), 0, np.mean(fittedErrorFuns), 0],
         ["Azimuth", np.mean(np.abs(errorsPosePred[0])), np.std(np.abs(errorsPosePred[0])), np.mean(np.abs(errorsPoseFitted[0])), np.std(np.abs(errorsPoseFitted[0]))],
         ["Elevation", np.mean(np.abs(errorsPosePred[1])), np.std(np.abs(errorsPosePred[1])), np.mean(np.abs(errorsPoseFitted[1])), np.std(np.abs(errorsPoseFitted[1]))],
         ["VColor", np.mean(errorsVColors), np.std(errorsVColors), np.mean(errorsFittedVColors), np.std(errorsFittedVColors)],
         ["SH Light", np.mean(errorsLightCoeffs), np.std(errorsLightCoeffs), np.mean(errorsFittedLightCoeffs), np.std(np.abs(errorsFittedLightCoeffs))],
         ]
performanceTable = tabulate.tabulate(table, headers=headers,tablefmt="latex", floatfmt=".2f")
with open(resultDir + 'performance.tex', 'w') as expfile:
    expfile.write(performanceTable)


headers=["Method", "l=0", "SH $l=0,m=-1$", "SH $l=1,m=0$", "SH $l=1,m=1$", "SH $l=1,m=-2$" , "SH $l=2,m=-1$", "SH $l=2,m=0$", "SH $l=2,m=1$", "SH $l=2,m=2$"]

SHModel = ""

# SMSE_SH_std = np.mean((testLightCoefficientsGTRel - relLightCoefficientsGTPred)**2 + 1e-5, axis=0)/(np.var(testLightCoefficientsGTRel, axis=0) + + 1e-5)
# table = [[SHModel, SMSE_SH_std[0], SMSE_SH_std[1], SMSE_SH_std[2],SMSE_SH_std[3], SMSE_SH_std[4], SMSE_SH_std[5],SMSE_SH_std[6], SMSE_SH_std[7], SMSE_SH_std[8] ],
#         ]
# performanceTable = tabulate.tabulate(table, headers=headers,tablefmt="latex", floatfmt=".2f")
# with open(resultDir + 'performance_SH_standardised.tex', 'w') as expfile:
#     expfile.write(performanceTable)
#
# SMSE_SH = np.mean((testLightCoefficientsGTRel - relLightCoefficientsGTPred)**2, axis=0)
# table = [[SHModel, SMSE_SH[0], SMSE_SH[1], SMSE_SH[2],SMSE_SH[3], SMSE_SH[4], SMSE_SH[5],SMSE_SH[6], SMSE_SH[7], SMSE_SH[8] ],
#         ]
# performanceTable = tabulate.tabulate(table, headers=headers,tablefmt="latex", floatfmt=".2f")
# with open(resultDir + 'performance_SH.tex', 'w') as expfile:
#     expfile.write(performanceTable)
#
# SMSE_SH_std = np.mean((testLightCoefficientsGTRel - fittedRelLightCoeffs)**2 + 1e-5, axis=0)/(np.var(testLightCoefficientsGTRel, axis=0) + 1e-5)
# table = [[SHModel, SMSE_SH_std[0], SMSE_SH_std[1], SMSE_SH_std[2],SMSE_SH_std[3], SMSE_SH_std[4], SMSE_SH_std[5],SMSE_SH_std[6], SMSE_SH_std[7], SMSE_SH_std[8] ],
#         ]
# performanceTable = tabulate.tabulate(table, headers=headers,tablefmt="latex", floatfmt=".2f")
# with open(resultDir + 'performance_SH_fit_standardised.tex', 'w') as expfile:
#     expfile.write(performanceTable)
#
# SMSE_SH = np.mean((testLightCoefficientsGTRel - fittedRelLightCoeffs)**2, axis=0)
# table = [[SHModel, SMSE_SH[0], SMSE_SH[1], SMSE_SH[2],SMSE_SH[3], SMSE_SH[4], SMSE_SH[5],SMSE_SH[6], SMSE_SH[7], SMSE_SH[8] ],
#         ]
# performanceTable = tabulate.tabulate(table, headers=headers,tablefmt="latex", floatfmt=".2f")
# with open(resultDir + 'performance_SH_fit.tex', 'w') as expfile:
#     expfile.write(performanceTable)

print("Finished backprojecting and fitting estimates.")
