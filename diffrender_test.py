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
import lasagne_nn

plt.ion()

#########################################
# Initialization starts here
#########################################

#Main script options:

glModes = ['glfw','mesa']
glMode = glModes[0]

width, height = (150, 150)
win = -1

if glMode == 'glfw':
    #Initialize base GLFW context for the Demo and to share context among all renderers.
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.DEPTH_BITS,32)
    glfw.window_hint(glfw.VISIBLE, GL.GL_FALSE)
    win = glfw.create_window(width, height, "Demo",  None, None)
    glfw.make_context_current(win)

angle = 60 * 180 / np.pi
clip_start = 0.05
clip_end = 10
frustum = {'near': clip_start, 'far': clip_end, 'width': width, 'height': height}
camDistance = 0.4

teapots = [line.strip() for line in open('teapots.txt')]

# renderTeapotsList = np.arange(len(teapots))
renderTeapotsList = [0]

targetModels = []

v_teapots, f_list_teapots, vc_teapots, vn_teapots, uv_teapots, haveTextures_list_teapots, textures_list_teapots, vflat, varray, center_teapots, blender_teapots = scene_io_utils.loadTeapotsOpenDRData(renderTeapotsList, False, False, targetModels)

azimuth = np.pi
chCosAz = ch.Ch([np.cos(azimuth)])
chSinAz = ch.Ch([np.sin(azimuth)])

chAz = 2*ch.arctan(chSinAz/(ch.sqrt(chCosAz**2 + chSinAz**2) + chCosAz))
chAz = ch.Ch([0])
chObjAz = ch.Ch([0])
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

# vcch[0] = np.ones_like(vcflat[0])*chVColorsGT.reshape([1,3])
renderer_teapots = []

for teapot_i in range(len(renderTeapotsList)):
    vmod = v_teapots[teapot_i]
    fmod_list = f_list_teapots[teapot_i]
    vcmod = vc_teapots[teapot_i]
    vnmod = vn_teapots[teapot_i]
    uvmod = uv_teapots[teapot_i]
    haveTexturesmod_list = haveTextures_list_teapots[teapot_i]
    texturesmod_list = textures_list_teapots[teapot_i]
    centermod = center_teapots[teapot_i]
    renderer = createRendererTarget(glMode, chAz, chObjAz, chEl, chDist, centermod, vmod, vcmod, fmod_list, vnmod, light_color, chComponent, chVColors, 0, chDisplacement, chScale, width,height, uvmod, haveTexturesmod_list, texturesmod_list, frustum, win )
    renderer.r
    renderer_teapots = renderer_teapots + [renderer]

currentTeapotModel = 0

center = center_teapots[currentTeapotModel]

#########################################
# Initialization ends here
#########################################

#########################################
# Generative model set up
#########################################

rendererGT = ch.Ch(renderer.r.copy())
numPixels = width*height

E_raw = renderer - rendererGT
SE_raw = ch.sum(E_raw*E_raw, axis=2)

SSqE_raw = ch.SumOfSquares(E_raw)/numPixels

initialPixelStdev = 0.01
reduceVariance = False
# finalPixelStdev = 0.05
stds = ch.Ch([initialPixelStdev])
variances = stds ** 2
globalPrior = ch.Ch([0.9])

negLikModel = -generative_models.modelLogLikelihoodCh(rendererGT, renderer, np.array([]), 'FULL', variances)/numPixels

negLikModelRobust = -generative_models.modelLogLikelihoodRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances)/numPixels

pixelLikelihoodCh = generative_models.logPixelLikelihoodCh(rendererGT, renderer, np.array([]), 'FULL', variances)

pixelLikelihoodRobustCh = ch.log(generative_models.pixelLikelihoodRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances))

post = generative_models.layerPosteriorsRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances)[0]

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


#########################################
# Generative model setup ends here.
#########################################

seed = 1
np.random.seed(seed)

testPrefix = 'test1_pred_100_simplex_ov_std001_msaa'

gtPrefix = 'train1'
trainPrefix = 'train1'
gtDir = 'groundtruth/' + gtPrefix + '/'
imagesDir = gtDir + 'images/'
experimentDir = 'experiments/' + trainPrefix + '/'
resultDir = 'results/' + testPrefix + '/'

groundTruthFilename = gtDir + 'groundTruth.h5'
gtDataFile = h5py.File(groundTruthFilename, 'r')

testSet = np.load(experimentDir + 'test.npy')[:100]

shapeGT = gtDataFile[gtPrefix].shape
boolTestSet = np.zeros(shapeGT).astype(np.bool)
boolTestSet[testSet] = True
testGroundTruth = gtDataFile[gtPrefix][boolTestSet]
groundTruth = np.zeros(shapeGT, dtype=testGroundTruth.dtype)
groundTruth[boolTestSet] = testGroundTruth
groundTruth = groundTruth[testSet]

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
dataIds = groundTruth['trainIds']

gtDtype = groundTruth.dtype

loadFromHdf5 = False

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

testAzsRel = np.mod(testAzsGT - testObjAzsGT, 2*np.pi)

loadHogFeatures = True
loadIllumFeatures = True

hogfeatures = np.load(experimentDir + 'hog.npy')
illumfeatures =  np.load(experimentDir  + 'illum.npy')

testHogfeatures = hogfeatures[testSet]
testIllumfeatures = illumfeatures[testSet]

recognitionTypeDescr = ["near", "mean", "sampling"]
recognitionType = 1
method = 4
model = 1
maxiter = 500
numSamples = 10

free_variables = [ chAz, chEl, chComponent, chVColors]
free_variables = [ chAz, chEl]

mintime = time.time()
boundEl = (0, np.pi/2.0)
boundAz = (0, None)
boundscomponents = (0,None)
bounds = [boundAz,boundEl]
bounds = [(None , None ) for sublist in free_variables for item in sublist]

methods=['dogleg', 'minimize', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'SGDMom']

options={'disp':False, 'maxiter':maxiter, 'lr':0.001, 'momentum':0.7, 'decay':0.99}
# testRenderer = np.int(dataTeapotIds[testSet][0])
testRenderer = np.int(dataTeapotIds[0])
renderer = renderer_teapots[testRenderer]
nearGTOffsetRelAz = 0
nearGTOffsetEl = 0
nearGTOffsetSHComponent = np.zeros(9)
nearGTOffsetVColor = np.zeros(3)

#Load trained recognition models

parameterRecognitionModels = set(['randForestAzs', 'randForestElevs', 'randForestVColors', 'randForestRelSHComponents'])

if 'randForestAzs' in parameterRecognitionModels:
    with open(experimentDir + 'randForestModelCosAzs.pickle', 'rb') as pfile:
        randForestModelCosAzs = pickle.load(pfile)
    cosAzsPred = recognition_models.testRandomForest(randForestModelCosAzs, testHogfeatures)

    with open(experimentDir + 'randForestModelSinAzs.pickle', 'rb') as pfile:
        randForestModelSinAzs = pickle.load(pfile)
    sinAzsPred = recognition_models.testRandomForest(randForestModelSinAzs, testHogfeatures)

if 'randForestElevs' in parameterRecognitionModels:
    with open(experimentDir + 'randForestModelCosElevs.pickle', 'rb') as pfile:
        randForestModelCosElevs = pickle.load(pfile)
    cosElevsPred = recognition_models.testRandomForest(randForestModelCosElevs, testHogfeatures)

    with open(experimentDir + 'randForestModelSinElevs.pickle', 'rb') as pfile:
        randForestModelSinElevs = pickle.load(pfile)
    sinElevsPred = recognition_models.testRandomForest(randForestModelSinElevs, testHogfeatures)
if 'randForestVColors' in parameterRecognitionModels:
    with open(experimentDir + 'randForestModelVColor.pickle', 'rb') as pfile:
        randForestModelVColor = pickle.load(pfile)
    vColorsPred = recognition_models.testRandomForest(randForestModelVColor, images)

if 'randForestRelSHComponents' in parameterRecognitionModels:
    with open(experimentDir + 'randForestModelRelSHComponents.pickle', 'rb') as pfile:
        randForestModelRelSHComponents = pickle.load(pfile)
    relSHComponentsPred = recognition_models.testRandomForest(randForestModelRelSHComponents, images)

if 'neuralNetRelSHComponents' in parameterRecognitionModels:
    modelPath = experimentDir + 'neuralNetModelRelSHComponents.npz'
    with np.load(modelPath) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    relSHComponentsPred = lasagne_nn.get_predictions(images, model='cnn', param_values=param_values)

elevsPred = np.arctan2(sinElevsPred, cosElevsPred)
azsPred = np.arctan2(sinAzsPred, cosAzsPred)

testPredPoseGMMs = []
colorGMMs = []
if recognitionType == 2:
    for test_i in range(len(testAzsRel)):
        testPredPoseGMMs = testPredPoseGMMs + [recognition_models.poseGMM(azsPred[test_i], elevsPred[test_i])]
        colorGMMs = colorGMMs + [recognition_models.colorGMM(images[test_i], 40)]

# errors = recognition_models.evaluatePrediction(testAzsRel, testElevsGT, testAzsRel, testElevsGT)
errors = recognition_models.evaluatePrediction(testAzsRel, testElevsGT, azsPred, elevsPred)

meanAbsErrAzs = np.mean(np.abs(errors[0]))
meanAbsErrElevs = np.mean(np.abs(errors[1]))

#Fit:
print("Fitting predictions")

fittedAzsGaussian = np.array([])
fittedElevsGaussian = np.array([])
fittedRelSHComponentsGaussian = np.array([])

predictedErrorFuns = np.array([])
fittedErrorFuns = np.array([])

if not os.path.exists(resultDir + 'imgs/'):
    os.makedirs(resultDir + 'imgs/')
if not os.path.exists(resultDir +  'imgs/samples/'):
    os.makedirs(resultDir + 'imgs/samples/')

print("Using " + modelsDescr[model])
errorFun = models[model]
pixelErrorFun = pixelModels[model]

testSamples = 1
if recognitionType == 2:
    testSamples  = numSamples

chDisplacement[:] = np.array([0.0, 0.0,0.0])
chScale[:] = np.array([1.0,1.0,1.0])
chObjAz[:] = 0
shapeIm = [height, width]

#Update all error functions with the right renderers.
print("Using " + modelsDescr[model])
negLikModel = -generative_models.modelLogLikelihoodCh(rendererGT, renderer, np.array([]), 'FULL', variances)/numPixels
negLikModelRobust = -generative_models.modelLogLikelihoodRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances)/numPixels
pixelLikelihoodCh = generative_models.logPixelLikelihoodCh(rendererGT, renderer, np.array([]), 'FULL', variances)
pixelLikelihoodRobustCh = ch.log(generative_models.pixelLikelihoodRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances))
post = generative_models.layerPosteriorsRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances)[0]
models = [negLikModel, negLikModelRobust, negLikModelRobust]
pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh, pixelLikelihoodRobustCh]
pixelErrorFun = pixelModels[model]
errorFun = models[model]

for test_i in range(len(testAzsRel)):

    bestPredAz = chAz.r
    bestPredEl = chEl.r
    bestModelLik = np.finfo('f').max
    bestVColors = chVColors.r
    bestComponent = chComponent.r

    testId = dataIds[test_i]
    print("Minimizing loss of prediction " + str(test_i) + "of " + str(len(testAzsRel)))

    rendererGT[:] = images[test_i]

    cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '_id' + str(testId) +'_groundtruth' + '.png', cv2.cvtColor(np.uint8(rendererGT.r*255), cv2.COLOR_RGB2BGR))

    for sample in range(testSamples):
        from numpy.random import choice
        if recognitionType == 0:
            #Prediction from (near) ground truth.
            color = testVColorGT[test_i] + nearGTOffsetVColor
            az = testAzsRel[test_i] + nearGTOffsetRelAz
            el = testElevsGT[test_i] + nearGTOffsetEl
            SHcomponents = testComponentsGTRel[test_i]
        elif recognitionType == 1:
            #Point (mean) estimate:
            az = azsPred[test_i]
            el = elevsPred[test_i]
            # color = recognition_models.meanColor(rendererGT.r, 40)
            # color = recognition_models.filteredMean(rendererGT.r, 40)
            color = recognition_models.midColor(rendererGT.r)
            color = testVColorGT[test_i]
            SHcomponents = relSHComponentsPred[test_i].copy()
            SHcomponents = testComponentsGTRel[test_i]
        else:
            #Sampling
            poseComps, vmAzParams, vmElParams = testPredPoseGMMs[test_i]
            sampleComp = choice(len(poseComps), size=1, p=poseComps)
            az = np.random.vonmises(vmAzParams[sampleComp][0],vmAzParams[sampleComp][1],1)
            el = np.random.vonmises(vmElParams[sampleComp][0],vmElParams[sampleComp][1],1)
            SHcomponents = relSHComponentsPred[test_i].copy()
            colorGMM = colorGMMs[test_i]
            color = colorGMM.sample(n_samples=1)[0]

        chAz[:] = az
        chEl[:] = el
        chVColors[:] = color.copy()
        # chVColors[:] = testPredVColors[test_i]
        chComponent[:] = SHcomponents

        #Update all error functions with the right renderers.
        negLikModel = -generative_models.modelLogLikelihoodCh(rendererGT, renderer, np.array([]), 'FULL', variances)/numPixels
        negLikModelRobust = -generative_models.modelLogLikelihoodRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances)/numPixels
        pixelLikelihoodCh = generative_models.logPixelLikelihoodCh(rendererGT, renderer, np.array([]), 'FULL', variances)
        pixelLikelihoodRobustCh = ch.log(generative_models.pixelLikelihoodRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances))
        post = generative_models.layerPosteriorsRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances)[0]
        models = [negLikModel, negLikModelRobust, negLikModelRobust]
        pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh, pixelLikelihoodRobustCh]
        pixelErrorFun = pixelModels[model]
        errorFun = models[model]

        cv2.imwrite(resultDir + 'imgs/samples/test'+ str(test_i) + '_sample' + str(sample) +  '_predicted'+ '.png', cv2.cvtColor(np.uint8(renderer.r*255), cv2.COLOR_RGB2BGR))
        z = -pixelErrorFun.dr_wrt(chAz).reshape(shapeIm[0],shapeIm[1],1).reshape(shapeIm[0],shapeIm[1],1)
        plt.imsave(resultDir + 'imgs/test'+ str(test_i) + '_id' + str(testId) +'_groundtruth_drAz' + '.png', z.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)

        predictedErrorFuns = np.append(predictedErrorFuns, errorFun.r)

        global iterat
        iterat = 0

        ch.minimize({'raw': errorFun}, bounds=None, method=methods[method], x0=free_variables, callback=cb, options=options)

        if errorFun.r < bestModelLik:
            bestModelLik = errorFun.r.copy()
            bestPredAz = chAz.r.copy()
            bestPredEl = chEl.r.copy()
            bestVColors = chVColors.r.copy()
            bestComponent = chComponent.r.copy()
            cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '_best'+ '.png', cv2.cvtColor(np.uint8(renderer.r*255), cv2.COLOR_RGB2BGR))

        cv2.imwrite(resultDir + 'imgs/samples/test'+ str(test_i) + '_sample' + str(sample) +  '_fitted'+ '.png',cv2.cvtColor(np.uint8(renderer.r*255), cv2.COLOR_RGB2BGR))

    fittedErrorFuns = np.append(fittedErrorFuns, bestModelLik)
    fittedAzsGaussian = np.append(fittedAzsGaussian, bestPredAz)
    fittedElevsGaussian = np.append(fittedElevsGaussian, bestPredEl)

testOcclusions = dataOcclusions

errorsFittedRFGaussian = recognition_models.evaluatePrediction(testAzsRel, testElevsGT, fittedAzsGaussian, fittedElevsGaussian)
meanAbsErrAzsFittedRFGaussian = np.mean(np.abs(errorsFittedRFGaussian[0]))
meanAbsErrElevsFittedRFGaussian = np.mean(np.abs(errorsFittedRFGaussian[1]))

plt.ioff()

directory = resultDir + 'predicted-azimuth-error'

fig = plt.figure()
plt.scatter(testElevsGT*180/np.pi, errors[0])
plt.xlabel('Elevation (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,90,-90,90))
plt.title('Performance scatter plot')
fig.savefig(directory + '_elev-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.scatter(testOcclusions*100.0,errors[0])
plt.xlabel('Occlusion (%)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,100,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory + '_occlusion-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.scatter(testAzsRel*180/np.pi, errors[0])
plt.xlabel('Azimuth (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,360,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory  + '_azimuth-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.hist(np.abs(errors[0]), bins=18)
plt.xlabel('Angular error')
plt.ylabel('Counts')
x1,x2,y1,y2 = plt.axis()
plt.axis((-180,180,y1, y2))
plt.title('Performance histogram')
fig.savefig(directory  + '_performance-histogram.png')
plt.close(fig)

directory = resultDir + 'predicted-elevation-error'

fig = plt.figure()
plt.scatter(testElevsGT*180/np.pi, errors[1])
plt.xlabel('Elevation (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,90,-90,90))
plt.title('Performance scatter plot')
fig.savefig(directory + '_elev-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.scatter(testOcclusions*100.0,errors[1])
plt.xlabel('Occlusion (%)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,100,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory + '_occlusion-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.scatter(testAzsRel*180/np.pi, errors[1])
plt.xlabel('Azimuth (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,360,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory  + '_azimuth-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.hist(np.abs(errors[1]), bins=18)
plt.xlabel('Angular error')
plt.ylabel('Counts')
x1,x2,y1,y2 = plt.axis()
plt.axis((-180,180,y1, y2))
plt.title('Performance histogram')
fig.savefig(directory  + '_performance-histogram.png')
plt.close(fig)

#Fitted predictions plots:

directory = resultDir + 'fitted-azimuth-error'

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
plt.scatter(testAzsRel*180/np.pi, errorsFittedRFGaussian[0])
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

directory = resultDir + 'fitted-elevation-error'

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
plt.scatter(testAzsRel*180/np.pi, errorsFittedRFGaussian[1])
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

directory = resultDir + 'fitted-robust-azimuth-error'

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
# plt.scatter(testAzsRel*180/np.pi, errorsFittedRFRobust[0])
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
# directory = resultDir + 'fitted-robust-elevation-error'
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
# plt.scatter(testAzsRel*180/np.pi, errorsFittedRFRobust[1])
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
with open(resultDir + 'performance.txt', 'w') as expfile:
    # expfile.write(str(z))
    expfile.write("Avg Pred NLL    :" +  str(np.mean(predictedErrorFuns))+ '\n')
    expfile.write("Avg Fitt NLL    :" +  str(np.mean(fittedErrorFuns))+ '\n')
    expfile.write("Mean Azimuth Error (predicted) " +  str(meanAbsErrAzs) + '\n')
    expfile.write("Mean Elevation Error (predicted) " +  str(meanAbsErrElevs)+ '\n')
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
#Write statistics to file.

headerDesc = "Pred NLL    :" + "Fitt NLL    :" + "Err Pred Az :" + "Err Pred El :" + "Err Fitted Az :" + "Err Fitted El :" + "Occlusions  :"
perfSamplesData = np.hstack([predictedErrorFuns.reshape([-1,1]),fittedErrorFuns.reshape([-1,1]),errors[0].reshape([-1,1]),errors[1].reshape([-1,1]),errorsFittedRFGaussian[0].reshape([-1,1]),errorsFittedRFGaussian[1].reshape([-1,1]),testOcclusions.reshape([-1,1])])

np.savetxt(resultDir + 'performance_samples.txt', perfSamplesData, delimiter=',', fmt="%g", header=headerDesc)
# with open(resultDir + 'performance_samples.txt', 'w') as expfile:
#     # expfile.write(str(z))
#     expfile.write("Pred NLL    :" +  str(predictedErrorFuns)+ '\n')
#     expfile.write("Fitt NLL    :" +  str(fittedErrorFuns)+ '\n')
#     expfile.write("Err Pred Az :" +  str(errors[0])+ '\n')
#     expfile.write("Err Pred El :" +  str(errors[1])+ '\n')
#     expfile.write("Err Fitted Az :" +  str(errorsFittedRFGaussian[0])+ '\n')
#     expfile.write("Err Fitted El :" +  str(errorsFittedRFGaussian[1])+ '\n')
#     expfile.write("Occlusions  :" +  str(testOcclusions)+ '\n')

print("Finished backprojecting and fitting estimates.")
