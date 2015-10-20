__author__ = 'pol'

import matplotlib
matplotlib.use('Qt4Agg')
import scene_io_utils
import mathutils
from math import radians
import timeit
import time
import opendr
import chumpy as ch
import geometry
import image_processing
import numpy as np
import cv2
import glfw
import generative_models
import recognition_models
import matplotlib.pyplot as plt
from opendr_utils import *
from utils import *
import OpenGL.GL as GL
import light_probes
from OpenGL import contextdata


plt.ion()

#########################################
# Initialization starts here
#########################################

#Main script options:

glModes = ['glfw','mesa']
glMode = glModes[0]

width, height = (100, 100)
win = -1

if glMode == 'glfw':
    #Initialize base GLFW context for the Demo and to share context among all renderers.
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.DEPTH_BITS,32)
    if demoMode:
        glfw.window_hint(glfw.VISIBLE, GL.GL_TRUE)
    else:
        glfw.window_hint(glfw.VISIBLE, GL.GL_FALSE)
    win = glfw.create_window(width, height, "Demo",  None, None)
    glfw.make_context_current(win)

angle = 60 * 180 / numpy.pi
clip_start = 0.05
clip_end = 10
frustum = {'near': clip_start, 'far': clip_end, 'width': width, 'height': height}
camDistance = 0.4

teapots = [line.strip() for line in open('teapots.txt')]

renderTeapotsList = np.arange(len(teapots))

targetModels = []

v_teapots, f_list_teapots, vc_teapots, vn_teapots, uv_teapots, haveTextures_list_teapots, textures_list_teapots, vflat, varray, center_teapots, blender_teapots = scene_io_utils.loadTeapotsOpenDRData(renderTeapotsList, useBlender, unpackModelsFromBlender, targetModels)

azimuth = np.pi
chCosAz = ch.Ch([np.cos(azimuth)])
chSinAz = ch.Ch([np.sin(azimuth)])

chAz = 2*ch.arctan(chSinAz/(ch.sqrt(chCosAz**2 + chSinAz**2) + chCosAz))
chAz = ch.Ch([np.pi/4])
chObjAz = ch.Ch([np.pi/4])
chAzRel = chAz - chObjAz

elevation = 0
chLogCosEl = ch.Ch(np.log(np.cos(elevation)))
chLogSinEl = ch.Ch(np.log(np.sin(elevation)))
chEl = 2*ch.arctan(ch.exp(chLogSinEl)/(ch.sqrt(ch.exp(chLogCosEl)**2 + ch.exp(chLogSinEl)**2) + ch.exp(chLogCosEl)))
chEl =  ch.Ch([0.95993109])
chDist = ch.Ch([camDistance])

chComponent = ch.Ch(np.array([2, 0.25, 0.25, 0.12,-0.17,0.36,0.1,0.,0.]))

chPointLightIntensity = ch.Ch([1])

chLightAz = ch.Ch([0.0])
chLightEl = ch.Ch([np.pi/2])
chLightDist = ch.Ch([0.5])

light_color = ch.ones(3)*chPointLightIntensity

chVColors = ch.Ch([0.4,0.4,0.4])

chDisplacement = ch.Ch([0.0, 0.0,0.0])
chScale = ch.Ch([1.0,1.0,1.0])

currentTeapotModel = 0

center = center_teapots[currentTeapotModel]

#########################################
# Initialization ends here
#########################################


seed = 1
np.random.seed(seed)

gtPrefix = 'test'
trainPrefix = gtPrefix + '_' + '1/'
gtDir = 'groundtruth/' + gtPrefix + '/'

rendererGT = ch.Ch(renderer.r)
numPixels = width*height

E_raw = renderer - rendererGT
SE_raw = ch.sum(E_raw*E_raw, axis=2)

SSqE_raw = ch.SumOfSquares(E_raw)/numPixels

initialPixelStdev = 0.075
reduceVariance = False
# finalPixelStdev = 0.05
stds = ch.Ch([initialPixelStdev])
variances = stds ** 2
globalPrior = ch.Ch([0.8])

negLikModel = -generative_models.modelLogLikelihoodCh(rendererGT, renderer, vis_im, 'FULL', variances)/numPixels

negLikModelRobust = -generative_models.modelLogLikelihoodRobustCh(rendererGT, renderer, vis_im, 'FULL', globalPrior, variances)/numPixels

pixelLikelihoodCh = generative_models.logPixelLikelihoodCh(rendererGT, renderer, vis_im, 'FULL', variances)

pixelLikelihoodRobustCh = ch.log(generative_models.pixelLikelihoodRobustCh(rendererGT, renderer, vis_im, 'FULL', globalPrior, variances))

post = generative_models.layerPosteriorsRobustCh(rendererGT, renderer, vis_im, 'FULL', globalPrior, variances)[0]

models = [negLikModel, negLikModelRobust, negLikModelRobust]
pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh, pixelLikelihoodRobustCh]
modelsDescr = ["Gaussian Model", "Outlier model", "Outler model (variance reduction)"]
# , negLikModelPyr, negLikModelRobustPyr, SSqE_raw
model = 0
pixelErrorFun = pixelModels[model]
errorFun = models[model]
iterat = 0

t = time.time()

def cb(_):
    global t
    elapsed_time = time.time() - t
    print("Ended interation in  " + str(elapsed_time))

    global pixelErrorFun
    global errorFun
    global iterat
    iterat = iterat + 1
    print("Callback! " + str(iterat))
    print("Sq Error: " + str(errorFun.r))
    global imagegt
    global renderer
    global gradAz
    global gradEl
    global performance
    global azimuths
    global elevations

    t = time.time()

free_variables = [ chAz, chEl]

mintime = time.time()
boundEl = (0, np.pi/2.0)
boundAz = (0, None)
boundscomponents = (0,None)
bounds = [boundAz,boundEl]
bounds = [(None , None ) for sublist in free_variables for item in sublist]

methods=['dogleg', 'minimize', 'BFGS', 'L-BFGS-B', 'Nelder-Mead']
method = 1
exit = False
minimize = False
plotMinimization = False
createGroundTruth = False

chAzOld = chAz.r[0]
chElOld = chEl.r[0]
print("Backprojecting and fitting estimates.")

with open(testDataName, 'rb') as pfile:
    testData = pickle.load(pfile)

testAzsGT = testData['testAzsGT']
testElevsGT = testData['testElevsGT']
testLightAzsGT = testData['testLightAzsGT']
testLightElevsGT = testData['testLightElevsGT']
testLightIntensitiesGT = testData['testLightIntensitiesGT']
testVColorGT = testData['testVColorGT']

if trainedModels == {}:
    with open('experiments/' + trainprefix +  'models.pickle', 'rb') as pfile:
        trainedModels = pickle.load(pfile)

    randForestModelCosAzs = trainedModels['randForestModelCosAzs']
    randForestModelSinAzs = trainedModels['randForestModelSinAzs']
    randForestModelCosElevs = trainedModels['randForestModelCosElevs']
    randForestModelSinElevs = trainedModels['randForestModelSinElevs']
    randForestModelLightCosAzs = trainedModels['randForestModelLightCosAzs']
    randForestModelLightSinAzs = trainedModels['randForestModelLightSinAzs']
    randForestModelLightCosElevs = trainedModels['randForestModelLightCosElevs']
    randForestModelLightSinElevs = trainedModels['randForestModelLightSinElevs']
    randForestModelLightIntensity = trainedModels['randForestModelLightIntensity']



print("Predicting with RFs")
testHogfeats = np.vstack(testHogs)
testIllumfeats = np.vstack(testIllumfeats)
testPredVColors = np.vstack(testPredVColors)
cosAzsPredRF = recognition_models.testRandomForest(randForestModelCosAzs, testHogfeats)
sinAzsPredRF = recognition_models.testRandomForest(randForestModelSinAzs, testHogfeats)
cosElevsPredRF = recognition_models.testRandomForest(randForestModelCosElevs, testHogfeats)
sinElevsPredRF = recognition_models.testRandomForest(randForestModelSinElevs, testHogfeats)

cosAzsLightPredRF = recognition_models.testRandomForest(randForestModelLightCosAzs, testIllumfeats)
sinAzsLightPredRF = recognition_models.testRandomForest(randForestModelLightSinAzs, testIllumfeats)
cosElevsLightPredRF = recognition_models.testRandomForest(randForestModelLightCosElevs, testIllumfeats)
sinElevsLightPredRF = recognition_models.testRandomForest(randForestModelLightSinElevs, testIllumfeats)

# print("Predicting with LR")
# cosAzsPredLR = recognition_models.testLinearRegression(linRegModelCosAzs, testHogfeats)
# sinAzsPredLR = recognition_models.testLinearRegression(linRegModelSinAzs, testHogfeats)
# cosElevsPredLR = recognition_models.testLinearRegression(linRegModelCosElevs, testHogfeats)
# sinElevsPredLR = recognition_models.testLinearRegression(linRegModelSinElevs, testHogfeats)

elevsPredRF = np.arctan2(sinElevsPredRF, cosElevsPredRF)
azsPredRF = np.arctan2(sinAzsPredRF, cosAzsPredRF)


for test_i in range(len(testAzsGT)):
    # testPredPoseGMMs = testPredPoseGMMs + [recognition_models.poseGMM(testAzsGT[test_i], testElevsGT[test_i])]
    testPredPoseGMMs = testPredPoseGMMs + [recognition_models.poseGMM(azsPredRF[test_i], elevsPredRF[test_i])]

lightElevsPredRF = np.arctan2(sinElevsLightPredRF, cosElevsLightPredRF)
lightAzsPredRF = np.arctan2(sinAzsLightPredRF, cosAzsLightPredRF)

testImagesStack = np.vstack([testImage.reshape([1,-1]) for testImage in testImages])
lightIntensityPredRF = recognition_models.testRandomForest(randForestModelLightIntensity, testImagesStack)

componentPreds=[]
for test_i in range(len(testAzsGT)):

    shDirLightGTTest = chZonalToSphericalHarmonics(zGT, np.pi/2 - lightElevsPredRF[test_i], lightAzsPredRF[test_i] - np.pi/2) * clampedCosCoeffs

    # componentPreds = componentPreds + [chAmbientSHGT + shDirLightGTTest*lightIntensityPredRF[test_i]]
    componentPreds = componentPreds + [chAmbientSHGT]

componentPreds = np.vstack(componentPreds)

# elevsPredLR = np.arctan2(sinElevsPredLR, cosElevsPredLR)
# azsPredLR = np.arctan2(sinAzsPredLR, cosAzsPredLR)

errorsRF = recognition_models.evaluatePrediction(testAzsGT, testElevsGT, azsPredRF, elevsPredRF)

errorsLightRF = recognition_models.evaluatePrediction(testLightAzsGT, testLightElevsGT, lightAzsPredRF, lightElevsPredRF)
# errorsLR = recognition_models.evaluatePrediction(testAzsGT, testElevsGT, azsPredLR, elevsPredLR)

meanAbsErrAzsRF = np.mean(np.abs(errorsRF[0]))
meanAbsErrElevsRF = np.mean(np.abs(errorsRF[1]))

meanAbsErrLightAzsRF = np.mean(np.abs(errorsLightRF[0]))
meanAbsErrLightElevsRF = np.mean(np.abs(errorsLightRF[1]))

# meanAbsErrAzsLR = np.mean(np.abs(errorsLR[0]))
# meanAbsErrElevsLR = np.mean(np.abs(errorsLR[1]))

#Fit:
print("Fitting predictions")

model = 0
print("Using " + modelsDescr[model])
errorFun = models[model]
pixelErrorFun = pixelModels[model]
fittedAzsGaussian = np.array([])
fittedElevsGaussian = np.array([])
fittedLightAzsGaussian = np.array([])
fittedLightElevsGaussian = np.array([])
testOcclusions = np.array([])
free_variables = [chVColors, chComponent, chAz, chEl]


if not os.path.exists('results/' + testprefix + 'imgs/'):
    os.makedirs('results/' + testprefix + 'imgs/')
if not os.path.exists('results/' + testprefix + 'imgs/samples/'):
    os.makedirs('results/' + testprefix + 'imgs/samples/')

if not os.path.exists('results/' + testprefix ):
    os.makedirs('results/' + testprefix )

model = 1
print("Using " + modelsDescr[model])
errorFun = models[model]
pixelErrorFun = pixelModels[model]

maxiter = 5
for test_i in range(len(testAzsGT)):

    bestPredAz = chAz.r
    bestPredEl = chEl.r
    bestModelLik = np.finfo('f').max
    bestVColors = chVColors.r
    bestComponent = chComponent.r

    colorGMM = testPredVColorGMMs[test_i]
    poseComps, vmAzParams, vmElParams = testPredPoseGMMs[test_i]
    print("Minimizing loss of prediction " + str(test_i) + "of " + str(len(testAzsGT)))
    chAzGT[:] = testAzsGT[test_i]
    chElGT[:] = testElevsGT[test_i]
    chLightAzGT[:] = testLightAzsGT[test_i]
    chLightElGT[:] = testLightElevsGT[test_i]
    chLightIntensityGT[:] = testLightIntensitiesGT[test_i]
    chVColorsGT[:] = testVColorGT[test_i]

    image = cv2.cvtColor(numpy.uint8(rendererGT.r*255), cv2.COLOR_RGB2BGR)
    cv2.imwrite('results/' + testprefix + 'imgs/test'+ str(test_i) + 'groundtruth' + '.png', image)
    for sample in range(10):
        from numpy.random import choice

        sampleComp = choice(len(poseComps), size=1, p=poseComps)
        az = np.random.vonmises(vmAzParams[sampleComp][0],vmAzParams[sampleComp][1],1)
        el = np.random.vonmises(vmElParams[sampleComp][0],vmElParams[sampleComp][1],1)
        color = colorGMM.sample(n_samples=1)[0]
        chAz[:] = az
        chEl[:] = el

        chVColors[:] = color.copy()
        # chVColors[:] = testPredVColors[test_i]
        chComponent[:] = componentPreds[test_i].copy()
        image = cv2.cvtColor(numpy.uint8(renderer.r*255), cv2.COLOR_RGB2BGR)
        cv2.imwrite('results/' + testprefix + 'imgs/samples/test'+ str(test_i) + '_sample' + str(sample) +  '_predicted'+ '.png',image)
        ch.minimize({'raw': errorFun}, bounds=None, method=methods[method], x0=free_variables, callback=cb, options={'disp':False, 'maxiter':maxiter})

        if errorFun.r < bestModelLik:
            bestModelLik = errorFun.r.copy()
            bestPredAz = chAz.r.copy()
            bestPredEl = chEl.r.copy()
            bestVColors = chVColors.r.copy()
            bestComponent = chComponent.r.copy()
            image = cv2.cvtColor(numpy.uint8(renderer.r*255), cv2.COLOR_RGB2BGR)
            cv2.imwrite('results/' + testprefix + 'imgs/test'+ str(test_i) + '_best'+ '.png',image)
        image = cv2.cvtColor(numpy.uint8(renderer.r*255), cv2.COLOR_RGB2BGR)
        cv2.imwrite('results/' + testprefix + 'imgs/samples/test'+ str(test_i) + '_sample' + str(sample) +  '_fitted'+ '.png',image)

    chDisplacement[:] = np.array([0.0, 0.0,0.0])
    chScale[:] = np.array([1.0,1.0,1.0])
    testOcclusions = np.append(testOcclusions, getOcclusionFraction(rendererGT))

    fittedAzsGaussian = np.append(fittedAzsGaussian, bestPredAz)
    fittedElevsGaussian = np.append(fittedElevsGaussian, bestPredEl)
    # fittedLightAzsGaussian = np.append(fittedLightAzsGaussian, chLightAz.r[0])
    # fittedLightElevsGaussian = np.append(fittedLightElevsGaussian, chLightEl.r[0])
errorsFittedRFGaussian = recognition_models.evaluatePrediction(testAzsGT, testElevsGT, fittedAzsGaussian, fittedElevsGaussian)
# errorsLightFittedRFGaussian = recognition_models.evaluatePrediction(testLightAzsGT, testLightElevsGT, fittedLightAzsGaussian, fittedLightElevsGaussian)
meanAbsErrAzsFittedRFGaussian = np.mean(np.abs(errorsFittedRFGaussian[0]))
meanAbsErrElevsFittedRFGaussian = np.mean(np.abs(errorsFittedRFGaussian[1]))
# meanAbsErrLightAzsFittedRFGaussian = np.mean(np.abs(errorsLightFittedRFGaussian[0]))
# meanAbsErrLightElevsFittedRFGaussian = np.mean(np.abs(errorsLightFittedRFGaussian[1]))

# model = 1
# print("Using " + modelsDescr[model])
# errorFun = models[model]
# pixelErrorFun = pixelModels[model]
# fittedAzsRobust = np.array([])
# fittedElevsRobust = np.array([])
# fittedLightAzsRobust = np.array([])
# fittedLightElevsRobust = np.array([])
# for test_i in range(len(testAzsGT)):
#     print("Minimizing loss of prediction " + str(test_i) + "of " + str(len(testAzsGT)))
#     chAzGT[:] = testAzsGT[test_i]
#     chElGT[:] = testElevsGT[test_i]
#     chLightAzGT[:] = testLightAzsGT[test_i]
#     chLightElGT[:] = testLightElevsGT[test_i]
#     chLightIntensityGT[:] = testLightIntensitiesGT[test_i]
#     chVColorsGT[:] = testVColorGT[test_i]
#     chAz[:] = azsPredRF[test_i]
#     chEl[:] = elevsPredRF[test_i]
#     chVColors[:] = testPredVColors[test_i]
#     chComponent[:] = componentPreds[test_i]
#
#     chDisplacement[:] = np.array([0.0, 0.0,0.0])
#     chScale[:] = np.array([1.0,1.0,1.0])
#
#     ch.minimize({'raw': errorFun}, bounds=bounds, method=methods[method], x0=free_variables, callback=cb, options={'disp':False, 'maxiter':maxiter})
#     image = cv2.cvtColor(numpy.uint8(renderer.r*255), cv2.COLOR_RGB2BGR)
#     cv2.imwrite('results/' + testprefix + 'imgs/test'+ str(test_i) + 'fitted-robust' + '.png', image)
#     fittedAzsRobust = np.append(fittedAzsRobust, chAz.r[0])
#     fittedElevsRobust = np.append(fittedElevsRobust, chEl.r[0])

    # fittedLightAzsRobust = np.append(fittedLightAzsRobust, chLightAz.r[0])
    # fittedLightElevsRobust = np.append(fittedLightElevsRobust, chLightEl.r[0])

# errorsFittedRFRobust = recognition_models.evaluatePrediction(testAzsGT, testElevsGT, fittedAzsRobust, fittedElevsRobust)
# meanAbsErrAzsFittedRFRobust = np.mean(np.abs(errorsFittedRFRobust[0]))
# meanAbsErrElevsFittedRFRobust = np.mean(np.abs(errorsFittedRFRobust[1]))
# errorsLightFittedRFRobust = recognition_models.evaluatePrediction(testLightAzsGT, testLightElevsGT, fittedLightAzsRobust, fittedLightElevsRobust)
# meanAbsErrLightAzsFittedRFRobust = np.mean(np.abs(errorsLightFittedRFRobust[0]))
# meanAbsErrLightElevsFittedRFRobust = np.mean(np.abs(errorsLightFittedRFRobust[1]))

# model = 1
# print("Using Both")
# errorFun = models[model]
# pixelErrorFun = pixelModels[model]
# fittedAzsBoth = np.array([])
# fittedElevsBoth = np.array([])
# for test_i in range(len(testAzsGT)):
#     print("Minimizing loss of prediction " + str(test_i) + "of " + str(len(testAzsGT)))
#     chAzGT[:] = testAzsGT[test_i]
#     chElGT[:] = testElevsGT[test_i]
#     chLightAzGT[:] = testLightAzsGT[test_i]
#     chLightElGT[:] = testLightElevsGT[test_i]
#     chLightIntensityGT[:] = testLightIntensitiesGT[test_i]
#     chVColorsGT[:] = testLightIntensitiesGT[test_i]
#     chAz[:] = azsPredRF[test_i]
#     chEl[:] = elevsPredRF[test_i]
#     chVColors[:] = testPredVColors[test_i]
#     chComponent[:] = componentPreds[test_i]
#     chDisplacement[:] = np.array([0.0, 0.0,0.0])
#     chScale[:] = np.array([1.0,1.0,1.0])
#
#     model = 0
#     errorFun = models[model]
#     pixelErrorFun = pixelModels[model]
#     ch.minimize({'raw': errorFun}, bounds=bounds, method=methods[method], x0=free_variables, callback=cb, options={'disp':False})
#     model = 1
#     errorFun = models[model]
#     pixelErrorFun = pixelModels[model]
#     ch.minimize({'raw': errorFun}, bounds=bounds, method=methods[method], x0=free_variables, callback=cb, options={'disp':False})
#     image = cv2.cvtColor(numpy.uint8(renderer.r*255), cv2.COLOR_RGB2BGR)
#     cv2.imwrite('results/imgs/fitted-robust' + str(test_i) + '.png', image)
#     fittedAzsBoth = np.append(fittedAzsBoth, chAz.r[0])
#     fittedElevsBoth = np.append(fittedElevsBoth, chEl.r[0])
#
# errorsFittedRFBoth = recognition_models.evaluatePrediction(testAzsGT, testElevsGT, fittedAzsBoth, fittedElevsBoth)
# meanAbsErrAzsFittedRFBoth = np.mean(np.abs(errorsFittedRFBoth[0]))
# meanAbsErrElevsFittedRFBoth = np.mean(np.abs(errorsFittedRFBoth[1]))

plt.ioff()

directory = 'results/' + testprefix + 'predicted-azimuth-error'

fig = plt.figure()
plt.scatter(testElevsGT*180/np.pi, errorsRF[0])
plt.xlabel('Elevation (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,90,-90,90))
plt.title('Performance scatter plot')
fig.savefig(directory + '_elev-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.scatter(testOcclusions*100.0,errorsRF[0])
plt.xlabel('Occlusion (%)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,100,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory + '_occlusion-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.scatter(testAzsGT*180/np.pi, errorsRF[0])
plt.xlabel('Azimuth (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,360,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory  + '_azimuth-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.hist(np.abs(errorsRF[0]), bins=18)
plt.xlabel('Angular error')
plt.ylabel('Counts')
x1,x2,y1,y2 = plt.axis()
plt.axis((-180,180,y1, y2))
plt.title('Performance histogram')
fig.savefig(directory  + '_performance-histogram.png')
plt.close(fig)

directory = 'results/' + testprefix + 'predicted-elevation-error'

fig = plt.figure()
plt.scatter(testElevsGT*180/np.pi, errorsRF[1])
plt.xlabel('Elevation (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,90,-90,90))
plt.title('Performance scatter plot')
fig.savefig(directory + '_elev-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.scatter(testOcclusions*100.0,errorsRF[1])
plt.xlabel('Occlusion (%)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,100,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory + '_occlusion-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.scatter(testAzsGT*180/np.pi, errorsRF[1])
plt.xlabel('Azimuth (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,360,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory  + '_azimuth-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.hist(np.abs(errorsRF[1]), bins=18)
plt.xlabel('Angular error')
plt.ylabel('Counts')
x1,x2,y1,y2 = plt.axis()
plt.axis((-180,180,y1, y2))
plt.title('Performance histogram')
fig.savefig(directory  + '_performance-histogram.png')
plt.close(fig)

#Fitted predictions plots:

directory = 'results/' + testprefix + 'fitted-azimuth-error'

fig = plt.figure()
plt.scatter(testElevsGT*180/np.pi, errorsFittedRFGaussian[0])
plt.xlabel('Elevation (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,90,-90,90))
plt.title('Performance scatter plot')
fig.savefig(directory + '_elev-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.scatter(testOcclusions*100.0,errorsFittedRFGaussian[0])
plt.xlabel('Occlusion (%)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,100,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory + '_occlusion-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.scatter(testAzsGT*180/np.pi, errorsFittedRFGaussian[0])
plt.xlabel('Azimuth (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,360,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory  + '_azimuth-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.hist(np.abs(errorsFittedRFGaussian[0]), bins=18)
plt.xlabel('Angular error')
plt.ylabel('Counts')
x1,x2,y1,y2 = plt.axis()
plt.axis((-180,180,y1, y2))
plt.title('Performance histogram')
fig.savefig(directory  + '_performance-histogram.png')
plt.close(fig)

directory = 'results/' + testprefix + 'fitted-elevation-error'

fig = plt.figure()
plt.scatter(testElevsGT*180/np.pi, errorsFittedRFGaussian[1])
plt.xlabel('Elevation (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,90,-90,90))
plt.title('Performance scatter plot')
fig.savefig(directory + '_elev-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.scatter(testOcclusions*100.0,errorsFittedRFGaussian[1])
plt.xlabel('Occlusion (%)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,100,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory + '_occlusion-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.scatter(testAzsGT*180/np.pi, errorsFittedRFGaussian[1])
plt.xlabel('Azimuth (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,360,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory  + '_azimuth-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.hist(np.abs(errorsFittedRFGaussian[1]), bins=18)
plt.xlabel('Angular error')
plt.ylabel('Counts')
x1,x2,y1,y2 = plt.axis()
plt.axis((-180,180,y1, y2))
plt.title('Performance histogram')
fig.savefig(directory  + '_performance-histogram.png')
plt.close(fig)

directory = 'results/' + testprefix + 'fitted-robust-azimuth-error'

# fig = plt.figure()
# plt.scatter(testElevsGT*180/np.pi, errorsFittedRFRobust[0])
# plt.xlabel('Elevation (degrees)')
# plt.ylabel('Angular error')
# x1,x2,y1,y2 = plt.axis()
# plt.axis((0,90,-90,90))
# plt.title('Performance scatter plot')
# fig.savefig(directory + '_elev-performance-scatter.png')
# plt.close(fig)
#
# fig = plt.figure()
# plt.scatter(testOcclusions*100.0,errorsFittedRFRobust[0])
# plt.xlabel('Occlusion (%)')
# plt.ylabel('Angular error')
# x1,x2,y1,y2 = plt.axis()
# plt.axis((0,100,-180,180))
# plt.title('Performance scatter plot')
# fig.savefig(directory + '_occlusion-performance-scatter.png')
# plt.close(fig)
#
# fig = plt.figure()
# plt.scatter(testAzsGT*180/np.pi, errorsFittedRFRobust[0])
# plt.xlabel('Azimuth (degrees)')
# plt.ylabel('Angular error')
# x1,x2,y1,y2 = plt.axis()
# plt.axis((0,360,-180,180))
# plt.title('Performance scatter plot')
# fig.savefig(directory  + '_azimuth-performance-scatter.png')
# plt.close(fig)
#
# fig = plt.figure()
# plt.hist(np.abs(errorsFittedRFRobust[0]), bins=18)
# plt.xlabel('Angular error')
# plt.ylabel('Counts')
# x1,x2,y1,y2 = plt.axis()
# plt.axis((-180,180,y1, y2))
# plt.title('Performance histogram')
# fig.savefig(directory  + '_performance-histogram.png')
# plt.close(fig)
#
# directory = 'results/' + testprefix + 'fitted-robust-elevation-error'
#
# fig = plt.figure()
# plt.scatter(testElevsGT*180/np.pi, errorsFittedRFRobust[1])
# plt.xlabel('Elevation (degrees)')
# plt.ylabel('Angular error')
# x1,x2,y1,y2 = plt.axis()
# plt.axis((0,90,-90,90))
# plt.title('Performance scatter plot')
# fig.savefig(directory + '_elev-performance-scatter.png')
# plt.close(fig)
#
# fig = plt.figure()
# plt.scatter(testOcclusions*100.0,errorsFittedRFRobust[1])
# plt.xlabel('Occlusion (%)')
# plt.ylabel('Angular error')
# x1,x2,y1,y2 = plt.axis()
# plt.axis((0,100,-180,180))
# plt.title('Performance scatter plot')
# fig.savefig(directory + '_occlusion-performance-scatter.png')
# plt.close(fig)
#
# fig = plt.figure()
# plt.scatter(testAzsGT*180/np.pi, errorsFittedRFRobust[1])
# plt.xlabel('Azimuth (degrees)')
# plt.ylabel('Angular error')
# x1,x2,y1,y2 = plt.axis()
# plt.axis((0,360,-180,180))
# plt.title('Performance scatter plot')
# fig.savefig(directory  + '_azimuth-performance-scatter.png')
# plt.close(fig)
#
# fig = plt.figure()
# plt.hist(np.abs(errorsFittedRFRobust[1]), bins=18)
# plt.xlabel('Angular error')
# plt.ylabel('Counts')
# x1,x2,y1,y2 = plt.axis()
# plt.axis((-180,180,y1, y2))
# plt.title('Performance histogram')
# fig.savefig(directory  + '_performance-histogram.png')
# plt.close(fig)

plt.ion()

#Write statistics to file.
with open('results/' + testprefix + 'performance.txt', 'w') as expfile:
    # expfile.write(str(z))
    expfile.write("Mean Azimuth Error (predicted) " +  str(meanAbsErrAzsRF) + '\n')
    expfile.write("Mean Elevation Error (predicted) " +  str(meanAbsErrElevsRF)+ '\n')
    expfile.write("Mean Azimuth Error (gaussian) " +  str(meanAbsErrAzsFittedRFGaussian)+ '\n')
    expfile.write("Mean Elevation Error (gaussian) " +  str(meanAbsErrElevsFittedRFGaussian)+ '\n')
    # expfile.write("Mean Azimuth Error (robust) " +  str(meanAbsErrAzsFittedRFRobust)+ '\n')
    # expfile.write("Mean Elevation Error (robust) " +  str(meanAbsErrElevsFittedRFRobust)+ '\n\n')
    # expfile.write("Mean Light Azimuth Error (predicted) " +  str(meanAbsErrLightAzsRF)+ '\n')
    # expfile.write("Mean Light Elevation Error (predicted) " +  str(meanAbsErrLightElevsRF)+ '\n')
    # expfile.write("Mean Light Azimuth Error (gaussian) " +  str(meanAbsErrLightAzsFittedRFGaussian)+ '\n')
    # expfile.write("Mean Light Elevation Error (gaussian)" +  str(meanAbsErrLightElevsFittedRFGaussian)+ '\n')
    # expfile.write("Mean Light Azimuth Error (robust)" +  str(meanAbsErrLightAzsFittedRFRobust)+ '\n')
    # expfile.write("Mean Light Elevation Error (robust) " +  str(meanAbsErrLightElevsFittedRFRobust)+ '\n')
    # expfile.write("meanAbsErrAzsFittedRFBoth " +  str(meanAbsErrAzsFittedRFBoth)+ '\n')
    # expfile.write("meanAbsErrElevsFittedRFBoth " +  str(meanAbsErrElevsFittedRFBoth)+ '\n')

    expfile.write("Occlusions " +  str(testOcclusions)+ '\n')

chAz[:] = chAzOld
chEl[:] = chElOld

print("Finished backprojecting and fitting estimates.")
