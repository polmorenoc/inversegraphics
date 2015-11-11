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
modelsDescr = [" Model", "Outlier model", "Outler model (variance reduction)"]
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

testPrefix = 'test1_pred_minimize_ov_msaa_ITERATIVE'

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

optimizationTypeDescr = ["normal", "joint"]
optimizationType = 0

method = 1
model = 2
maxiter = 500
numSamples = 10

free_variables = [ chAz, chEl]
# free_variables = [ chAz, chEl]

mintime = time.time()
boundEl = (0, np.pi/2.0)
boundAz = (0, None)
boundscomponents = (0,None)
bounds = [boundAz,boundEl]
bounds = [(None , None ) for sublist in free_variables for item in sublist]

methods=['dogleg', 'minimize', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'SGDMom']

options={'disp':False, 'maxiter':maxiter, 'lr':0.0001, 'momentum':0.1, 'decay':0.99}
# options={'disp':False, 'maxiter':maxiter}
# testRenderer = np.int(dataTeapotIds[testSet][0])
testRenderer = np.int(dataTeapotIds[0])
renderer = renderer_teapots[testRenderer]
nearGTOffsetRelAz = 0
nearGTOffsetEl = 0
nearGTOffsetSHComponent = np.zeros(9)
nearGTOffsetVColor = np.zeros(3)

#Load trained recognition models

parameterRecognitionModels = set(['randForestAzs', 'randForestElevs', 'randForestVColors', 'neuralNetRelSHComponents'])

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

    colorWindow = 30
    image = images[0]
    croppedImages = images[:,image.shape[0]/2-colorWindow:image.shape[0]/2+colorWindow,image.shape[1]/2-colorWindow:image.shape[1]/2+colorWindow,:][:,3]
    vColorsPred = recognition_models.testRandomForest(randForestModelVColor, croppedImages.reshape([len(testSet),-1]))

if 'randForestRelSHComponents' in parameterRecognitionModels:
    with open(experimentDir + 'randForestModelRelSHComponents.pickle', 'rb') as pfile:
        randForestModelRelSHComponents = pickle.load(pfile)
    relSHComponentsPred = recognition_models.testRandomForest(randForestModelRelSHComponents, testIllumfeatures)

elif 'neuralNetRelSHComponents' in parameterRecognitionModels:
    # modelPath = experimentDir + 'neuralNetModelRelSHComponents.npz'
    # with np.load(modelPath) as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    with open(experimentDir + 'neuralNetModelRelSHComponents.pickle', 'rb') as pfile:
        neuralNetModelRelSHComponents = pickle.load(pfile)

    meanImage = neuralNetModelRelSHComponents['mean']
    modelType = neuralNetModelRelSHComponents['type']
    param_values = neuralNetModelRelSHComponents['params']
    grayTestImages =  0.3*images[:,:,:,0] +  0.59*images[:,:,:,1] + 0.11*images[:,:,:,2]
    grayTestImages = grayTestImages[:,None, :,:]
    grayTestImages = grayTestImages - meanImage
    relSHComponentsPred = lasagne_nn.get_predictions(grayTestImages, model=modelType, param_values=param_values)

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

errorsSHComponents = np.linalg.norm(testComponentsGTRel - relSHComponentsPred, axis=1)
errorsVColors = np.linalg.norm(testVColorGT - vColorsPred, axis=1)

meanAbsErrAzs = np.mean(np.abs(errors[0]))
meanAbsErrElevs = np.mean(np.abs(errors[1]))

meanErrorsSHComponents = np.mean(errorsSHComponents)
meanErrorsVColors = np.mean(errorsVColors)

#Fit:
print("Fitting predictions")

fittedAzs = np.array([])
fittedElevs = np.array([])
fittedRelSHComponents = []
fittedVColors = []

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

hogGT, hogImGT, drconv = image_processing.diffHog(rendererGT)
hogRenderer, hogImRenderer, _ = image_processing.diffHog(renderer, drconv)

hogE_raw = hogGT - hogRenderer
hogCellErrors = ch.sum(hogE_raw*hogE_raw, axis=2)
hogError = ch.SumOfSquares(hogE_raw)

models = [negLikModel, negLikModelRobust, hogError]
pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh, hogCellErrors]
modelsDescr = ["Gaussian Model", "Outlier model", "HOG"]

# models = [negLikModel, negLikModelRobust, negLikModelRobust]
# pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh, pixelLikelihoodRobustCh]
# pixelErrorFun = pixelModels[model]
errorFun = models[model]

for test_i in range(len(testAzsRel)):

    bestPredAz = chAz.r
    bestPredEl = chEl.r
    bestModelLik = np.finfo('f').max
    bestVColors = chVColors.r
    bestComponent = chComponent.r

    testId = dataIds[test_i]
    print("************** Minimizing loss of prediction " + str(test_i) + "of " + str(len(testAzsRel)))

    rendererGT[:] = images[test_i]

    if not os.path.exists(resultDir + 'imgs/test'+ str(test_i) + '/'):
        os.makedirs(resultDir + 'imgs/test'+ str(test_i) + '/')

    cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/id' + str(testId) +'_groundtruth' + '.png', cv2.cvtColor(np.uint8(rendererGT.r*255), cv2.COLOR_RGB2BGR))

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
            # color = testVColorGT[test_i]
            color = vColorsPred[test_i]
            SHcomponents = relSHComponentsPred[test_i].copy()
            # SHcomponents = testComponentsGTRel[test_i]
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

        hogGT, hogImGT, drconv = image_processing.diffHog(rendererGT,drconv)
        hogRenderer, hogImRenderer, _ = image_processing.diffHog(renderer, drconv)

        hogE_raw = hogGT - hogRenderer
        hogCellErrors = ch.sum(hogE_raw*hogE_raw, axis=2)
        hogError = ch.SumOfSquares(hogE_raw)

        models = [negLikModel, negLikModelRobust, hogError]
        pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh, hogCellErrors]
        modelsDescr = ["Gaussian Model", "Outlier model", "HOG"]

        pixelErrorFun = pixelModels[model]
        errorFun = models[model]

        cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/sample' + str(sample) +  '_predicted'+ '.png', cv2.cvtColor(np.uint8(renderer.r*255), cv2.COLOR_RGB2BGR))
        # plt.imsave(resultDir + 'imgs/test'+ str(test_i) + '/id' + str(testId) +'_groundtruth_drAz' + '.png', z.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)

        predictedErrorFuns = np.append(predictedErrorFuns, errorFun.r)

        global iterat
        iterat = 0

        sys.stdout.flush()
        if optimizationTypeDescr[optimizationType] == 'normal':
            ch.minimize({'raw': errorFun}, bounds=None, method=methods[method], x0=free_variables, callback=cb, options=options)

        elif optimizationTypeDescr[optimizationType] == 'joint':
            currPoseError = recognition_models.evaluatePrediction(testAzsRel[test_i], testElevsGT[test_i], chAz.r, chEl.r)
            currSHError = np.linalg.norm(testComponentsGTRel[test_i] - chComponent.r)
            currVColorError = np.linalg.norm(testVColorGT[test_i] - chVColors.r)
            currErrorGaussian = models[0].r
            currErrorRobust = models[1].r
            currErrorHoG = models[2].r
            print("Predicted errors:")
            print("Az error: " + str(currPoseError[0]))
            print("El error: " + str(currPoseError[1]))
            print("VColor error: " + str(currVColorError))
            print("SH error: " + str(currSHError))
            print("Gaussian likelihood: " + str(currErrorGaussian))
            print("Robust likelihood: " + str(currErrorRobust))
            print("HoG error: " + str(currErrorHoG))

            free_variables = [ chAz, chEl]
            ch.minimize({'raw': models[2]}, bounds=None, method=methods[4], x0=free_variables, callback=cb, options=options)

            sys.stdout.flush()
            currPoseError = recognition_models.evaluatePrediction(testAzsRel[test_i], testElevsGT[test_i], chAz.r, chEl.r)
            currErrorGaussian = models[0].r
            currErrorRobust = models[1].r
            currErrorHoG = models[2].r
            print("HoG fitted errors:")
            print("Az error: " + str(currPoseError[0]))
            print("El error: " + str(currPoseError[1]))
            print("Gaussian likelihood: " + str(currErrorGaussian))
            print("Robust likelihood: " + str(currErrorRobust))
            print("HoG error: " + str(currErrorHoG))

            free_variables = [ chAz, chEl, chVColors, chComponent]

            ch.minimize({'raw': models[1]}, bounds=None, method=methods[1], x0=free_variables, callback=cb, options=options)

            sys.stdout.flush()
            currPoseError = recognition_models.evaluatePrediction(testAzsRel[test_i], testElevsGT[test_i], chAz.r, chEl.r)
            currSHError = np.linalg.norm(testComponentsGTRel[test_i] - chComponent.r)
            currVColorError = np.linalg.norm(testVColorGT[test_i] - chVColors.r)
            currErrorGaussian = models[0].r
            currErrorRobust = models[1].r
            currErrorHoG = models[2].r
            print("Joint fitted errors:")
            print("Az error: " + str(currPoseError[0]))
            print("El error: " + str(currPoseError[1]))
            print("VColor error: " + str(currVColorError))
            print("SH error: " + str(currSHError))
            print("Gaussian likelihood: " + str(currErrorGaussian))
            print("Robust likelihood: " + str(currErrorRobust))
            print("HoG error: " + str(currErrorHoG))

            sys.stdout.flush()

        if errorFun.r < bestModelLik:
            bestModelLik = errorFun.r.copy()
            bestPredAz = chAz.r.copy()
            bestPredEl = chEl.r.copy()
            bestVColors = chVColors.r.copy()
            bestComponent = chComponent.r.copy()
            cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/best'+ '.png', cv2.cvtColor(np.uint8(renderer.r*255), cv2.COLOR_RGB2BGR))

        cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/sample' + str(sample) +  '_fitted'+ '.png',cv2.cvtColor(np.uint8(renderer.r*255), cv2.COLOR_RGB2BGR))

    fittedErrorFuns = np.append(fittedErrorFuns, bestModelLik)
    fittedAzs = np.append(fittedAzs, bestPredAz)
    fittedElevs = np.append(fittedElevs, bestPredEl)
    fittedVColors = fittedVColors + [bestVColors]
    fittedRelSHComponents = fittedRelSHComponents + [bestComponent]

fittedVColors = np.vstack(fittedVColors)
fittedRelSHComponents = np.vstack(fittedRelSHComponents)
testOcclusions = dataOcclusions

errorsFittedRF = recognition_models.evaluatePrediction(testAzsRel, testElevsGT, fittedAzs, fittedElevs)
meanAbsErrAzsFittedRF = np.mean(np.abs(errorsFittedRF[0]))
meanAbsErrElevsFittedRF = np.mean(np.abs(errorsFittedRF[1]))

errorsFittedSHComponents = np.linalg.norm(testComponentsGTRel - fittedRelSHComponents, axis=1)
errorsFittedVColors = np.linalg.norm(testVColorGT - fittedVColors, axis=1)
meanErrorsFittedSHComponents = np.mean(errorsFittedSHComponents)
meanErrorsFittedVColors = np.mean(errorsFittedVColors)

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
plt.scatter(testElevsGT*180/np.pi, errorsFittedRF[0])
plt.xlabel('Elevation (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,90,-90,90))
plt.title('Performance scatter plot')
fig.savefig(directory + '_elev-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.scatter(testOcclusions*100.0,errorsFittedRF[0])
plt.xlabel('Occlusion (%)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,100,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory + '_occlusion-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.scatter(testAzsRel*180/np.pi, errorsFittedRF[0])
plt.xlabel('Azimuth (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,360,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory  + '_azimuth-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.hist(np.abs(errorsFittedRF[0]), bins=18)
plt.xlabel('Angular error')
plt.ylabel('Counts')
x1,x2,y1,y2 = plt.axis()
plt.axis((-180,180,y1, y2))
plt.title('Performance histogram')
fig.savefig(directory  + '_performance-histogram.png')
plt.close(fig)

directory = resultDir + 'fitted-elevation-error'

fig = plt.figure()
plt.scatter(testElevsGT*180/np.pi, errorsFittedRF[1])
plt.xlabel('Elevation (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,90,-90,90))
plt.title('Performance scatter plot')
fig.savefig(directory + '_elev-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.scatter(testOcclusions*100.0,errorsFittedRF[1])
plt.xlabel('Occlusion (%)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,100,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory + '_occlusion-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.scatter(testAzsRel*180/np.pi, errorsFittedRF[1])
plt.xlabel('Azimuth (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,360,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory  + '_azimuth-performance-scatter.png')
plt.close(fig)

fig = plt.figure()
plt.hist(np.abs(errorsFittedRF[1]), bins=18)
plt.xlabel('Angular error')
plt.ylabel('Counts')
x1,x2,y1,y2 = plt.axis()
plt.axis((-180,180,y1, y2))
plt.title('Performance histogram')
fig.savefig(directory  + '_performance-histogram.png')
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
    expfile.write("Mean Azimuth Error (fitted) " +  str(meanAbsErrAzsFittedRF)+ '\n')
    expfile.write("Mean Elevation Error (fitted) " +  str(meanAbsErrElevsFittedRF)+ '\n')
    expfile.write("Mean SH Components Error (predicted) " +  str(meanErrorsSHComponents)+ '\n')
    expfile.write("Mean Vertex Colors Error (predicted) " +  str(meanErrorsVColors)+ '\n')
    expfile.write("Mean SH Components Error (fitted) " +  str(meanErrorsFittedSHComponents)+ '\n')
    expfile.write("Mean Vertex Colors Error (fitted) " +  str(meanErrorsFittedVColors)+ '\n')




headerDesc = "Pred NLL    :" + "Fitt NLL    :" + "Err Pred Az :" + "Err Pred El :" + "Err Fitted Az :" + "Err Fitted El :" + "Occlusions  :"
perfSamplesData = np.hstack([predictedErrorFuns.reshape([-1,1]),fittedErrorFuns.reshape([-1,1]),errors[0].reshape([-1,1]),errors[1].reshape([-1,1]),errorsFittedRF[0].reshape([-1,1]),errorsFittedRF[1].reshape([-1,1]),testOcclusions.reshape([-1,1])])

np.savetxt(resultDir + 'performance_samples.txt', perfSamplesData, delimiter=',', fmt="%g", header=headerDesc)


print("Finished backprojecting and fitting estimates.")
