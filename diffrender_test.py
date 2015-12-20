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
import theano
# theano.sandbox.cuda.use('cpu')
import lasagne
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
renderTeapotsList = np.arange(len(teapots))

targetModels = []

v_teapots, f_list_teapots, vc_teapots, vn_teapots, uv_teapots, haveTextures_list_teapots, textures_list_teapots, vflat, varray, center_teapots = scene_io_utils.loadTeapotsOpenDRData(renderTeapotsList, False, False, targetModels)

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

chLightSHCoeffs = ch.Ch(np.array([2, 0.25, 0.25, 0.12,-0.17,0.36,0.1,0.,0.]))

clampedCosCoeffs = clampedCosineCoefficients()
chComponent = chLightSHCoeffs * clampedCosCoeffs

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

negLikModel = -ch.sum(generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances))/numPixels
negLikModelRobust = -ch.sum(generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances))/numPixels
pixelLikelihoodCh = generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances)
pixelLikelihoodRobustCh = generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances)
# negLikModel = -generative_models.modelLogLikelihoodCh(rendererGT, renderer, np.array([]), 'FULL', variances)/numPixels
# negLikModelRobust = -generative_models.modelLogLikelihoodRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances)/numPixels
# pixelLikelihoodCh = generative_models.logPixelLikelihoodCh(rendererGT, renderer, np.array([]), 'FULL', variances)
# pixelLikelihoodRobustCh = ch.log(generative_models.pixelLikelihoodRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances))

post = generative_models.layerPosteriorsRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances)[0]

# hogGT, hogImGT, drconv = image_processing.diffHog(rendererGT)
# hogRenderer, hogImRenderer, _ = image_processing.diffHog(renderer, drconv)
#
# hogE_raw = hogGT - hogRenderer
# hogCellErrors = ch.sum(hogE_raw*hogE_raw, axis=2)
# hogError = ch.SumOfSquares(hogE_raw)

# models = [negLikModel, negLikModelRobust, hogError]
models = [negLikModel, negLikModelRobust, negLikModelRobust]
# pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh, hogCellErrors]
pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh, pixelLikelihoodRobustCh]
modelsDescr = ["Gaussian Model", "Outlier model", "HOG"]

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


#########################################
# Test code starts here:
#########################################

seed = 1
np.random.seed(seed)

testPrefix = 'train5_normalteapots_nnposepred_robust_render0'

parameterRecognitionModels = set(['randForestAzs', 'randForestElevs', 'randForestVColors', 'linearRegressionVColors', 'neuralNetModelSHLight', ])
parameterRecognitionModels = set(['randForestAzs', 'randForestElevs', 'randForestVColors', 'linearRegressionVColors', 'linRegModelSHZernike' ])
parameterRecognitionModels = set(['randForestAzs', 'randForestElevs','linearRegressionVColors','neuralNetModelSHLight' ])
parameterRecognitionModels = set(['neuralNetPose', 'linearRegressionVColors','neuralNetModelSHLight' ])

# parameterRecognitionModels = set(['randForestAzs', 'randForestElevs','randForestVColors','randomForestSHZernike' ])

gtPrefix = 'train5'
experimentPrefix = 'train5_out_normal'
trainPrefix = 'train4'
trainPrefixPose = 'train4'
trainPrefixVColor = 'train4'
trainPrefixLightCoeffs = 'train4'
gtDir = 'groundtruth/' + gtPrefix + '/'
featuresDir = gtDir

experimentDir = 'experiments/' + experimentPrefix + '/'
trainModelsDirPose = 'experiments/' + trainPrefixPose + '/'
trainModelsDirVColor = 'experiments/' + trainPrefixVColor + '/'
trainModelsDirLightCoeffs = 'experiments/' + trainPrefixLightCoeffs + '/'
resultDir = 'results/' + testPrefix + '/'

ignoreGT = True
ignore = []
if os.path.isfile(gtDir + 'ignore.npy'):
    ignore = np.load(gtDir + 'ignore.npy')

groundTruthFilename = gtDir + 'groundTruth.h5'
gtDataFile = h5py.File(groundTruthFilename, 'r')

testSet = np.load(experimentDir + 'test.npy')[:200]

# testSet = np.array([ 11230, 3235, 10711,  9775, 11230, 10255,  5060, 12784,  5410,  1341,14448, 12935, 13196,  6728,  9002,  7946,  1119,  5827,  4842,12435,  8152,  4745,  9512,  9641,  7165, 13950,  3567,   860,4105, 10330,  7218, 10176,  2310,  5325])
testSetFixed = testSet
whereBad = []
for test_it, test_id in enumerate(testSet):
    if test_id in ignore:
        bad = np.where(testSetFixed==test_id)
        testSetFixed = np.delete(testSetFixed, bad)
        whereBad = whereBad + [bad]

testSet = testSetFixed

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
dataLightCoefficientsGT = groundTruth['trainLightCoefficientsGT']
dataLightCoefficientsGTRel = groundTruth['trainLightCoefficientsGTRel']
dataAmbientIntensityGT = groundTruth['trainAmbientIntensityGT']
dataIds = groundTruth['trainIds']

gtDtype = groundTruth.dtype

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

# loadHogFeatures = True
# loadZernikeFeatures = False
#
# if loadHogFeatures:
#     hogfeatures = np.load(featuresDir  +  'hog' + synthPrefix + '.npy')
#
# if loadZernikeFeatures:
#     numCoeffs=100
#     win=40
#     zernikefeatures = np.load(featuresDir  + 'zernike_numCoeffs' + str(numCoeffs) + '_win' + str(win) + synthPrefix + '.npy')
#
# testHogfeatures = hogfeatures[testSet]
# testZernikefeatures = zernikefeatures[testSet]
# testIllumfeatures = illumfeatures[testSet]

recognitionTypeDescr = ["near", "mean", "sampling"]
recognitionType = 1

optimizationTypeDescr = ["predict", "optimize", "joint"]
optimizationType = 1
computePredErrorFuns = False

method = 1
model = 1
maxiter = 500
numSamples = 1

# free_variables = [ chAz, chEl]
free_variables = [ chAz, chEl, chVColors, chLightSHCoeffs]
# free_variables = [ chAz, chEl]

mintime = time.time()
boundEl = (-np.pi, np.pi)
boundAz = (-3*np.pi, 3*np.pi)
boundscomponents = (0,None)
bounds = [boundAz,boundEl]
bounds = [(None , None ) for sublist in free_variables for item in sublist]

methods=['dogleg', 'minimize', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'SGDMom']

options={'disp':False, 'maxiter':maxiter, 'lr':0.0001, 'momentum':0.1, 'decay':0.99}
options={'disp':False, 'maxiter':maxiter}
# options={'disp':False, 'maxiter':maxiter}
# testRenderer = np.int(dataTeapotIds[testSet][0])
# testRenderer = np.int(dataTeapotIds[0])
testRenderer = 0

# testRenderer = 2
renderer = renderer_teapots[testRenderer]
nearGTOffsetRelAz = 0
nearGTOffsetEl = 0
nearGTOffsetLighCoeffs = np.zeros(9)
nearGTOffsetVColor = np.zeros(3)

#Load trained recognition models

if 'neuralNetPose' in parameterRecognitionModels:
    poseModel = ""

    with open(trainModelsDirPose + 'neuralNetModelPose.pickle', 'rb') as pfile:
        neuralNetModelPose = pickle.load(pfile)

    meanImage = neuralNetModelPose['mean']
    # ipdb.set_trace()
    modelType = neuralNetModelPose['type']
    param_values = neuralNetModelPose['params']
    grayTestImages =  0.3*images[:,:,:,0] +  0.59*images[:,:,:,1] + 0.11*images[:,:,:,2]
    grayTestImages = grayTestImages[:,None, :,:]
    grayTestImages = grayTestImages - meanImage

    network = lasagne_nn.load_network(modelType=modelType, param_values=param_values)
    posePredictionFun = lasagne_nn.get_prediction_fun(network)
    posePredictions = posePredictionFun(grayTestImages)
    cosAzsPred = posePredictions[:,0]
    sinAzsPred = posePredictions[:,1]
    cosElevsPred = posePredictions[:,2]
    sinElevsPred = posePredictions[:,3]

if 'randForestAzs' in parameterRecognitionModels:
    with open(trainModelsDirPose + 'randForestModelCosAzs.pickle', 'rb') as pfile:
        randForestModelCosAzs = pickle.load(pfile)['randForestModelCosAzs']
    cosAzsPred = recognition_models.testRandomForest(randForestModelCosAzs, testHogfeatures)
    cosAzsPredictions = np.vstack([estimator.predict(testHogfeatures) for estimator in randForestModelCosAzs.estimators_])

    with open(trainModelsDirPose + 'randForestModelSinAzs.pickle', 'rb') as pfile:
        randForestModelSinAzs = pickle.load(pfile)['randForestModelSinAzs']
    sinAzsPred = recognition_models.testRandomForest(randForestModelSinAzs, testHogfeatures)
    sinAzsPredictions = np.vstack([estimator.predict(testHogfeatures) for estimator in randForestModelSinAzs.estimators_])

if 'randForestElevs' in parameterRecognitionModels:
    with open(trainModelsDirPose + 'randForestModelCosElevs.pickle', 'rb') as pfile:
        randForestModelCosElevs = pickle.load(pfile)['randForestModelCosElevs']
    cosElevsPred = recognition_models.testRandomForest(randForestModelCosElevs, testHogfeatures)

    cosElevsPredictions = np.vstack([estimator.predict(testHogfeatures) for estimator in randForestModelCosElevs.estimators_])

    with open(trainModelsDirPose + 'randForestModelSinElevs.pickle', 'rb') as pfile:
        randForestModelSinElevs = pickle.load(pfile)['randForestModelSinElevs']
    sinElevsPred = recognition_models.testRandomForest(randForestModelSinElevs, testHogfeatures)
    sinElevsPredictions = np.vstack([estimator.predict(testHogfeatures) for estimator in randForestModelSinElevs.estimators_])

if 'randForestVColors' in parameterRecognitionModels:
    with open(trainModelsDirVColor + 'randForestModelVColor.pickle', 'rb') as pfile:
        randForestModelVColor = pickle.load(pfile)

    colorWindow = 30
    image = images[0]
    croppedImages = images[:,image.shape[0]/2-colorWindow:image.shape[0]/2+colorWindow,image.shape[1]/2-colorWindow:image.shape[1]/2+colorWindow,:]
    vColorsPred = recognition_models.testRandomForest(randForestModelVColor, croppedImages.reshape([len(testSet),-1]))

if 'linearRegressionVColors' in parameterRecognitionModels:
    with open(trainModelsDirVColor + 'linearRegressionModelVColor.pickle', 'rb') as pfile:
        linearRegressionModelVColor = pickle.load(pfile)

    colorWindow = 30
    image = images[0]
    croppedImages = images[:,image.shape[0]/2-colorWindow:image.shape[0]/2+colorWindow,image.shape[1]/2-colorWindow:image.shape[1]/2+colorWindow,:]
    vColorsPred = recognition_models.testLinearRegression(linearRegressionModelVColor, croppedImages.reshape([len(testSet),-1]))

if 'medianVColors' in parameterRecognitionModels:
    # recognition_models.medianColor(image, win)
    colorWindow = 30
    imagesWin = images[:,images.shape[1]/2-colorWindow:images.shape[1]/2+colorWindow,images.shape[2]/2-colorWindow:images.shape[2]/2+colorWindow,:]
    vColorsPred = np.median(imagesWin.reshape([images.shape[0],-1,3]), axis=1)/1.4
    # return color

import theano
import theano.tensor as T
SHModel = ""

with open(trainModelsDirLightCoeffs + 'neuralNetModelPose.pickle', 'rb') as pfile:
    neuralNetModelPose = pickle.load(pfile)

meanImage = neuralNetModelPose['mean'].reshape([150,150])
# ipdb.set_trace()
modelType = neuralNetModelPose['type']
param_values = neuralNetModelPose['params']
network = lasagne_nn.load_network(modelType=modelType, param_values=param_values)
layer = lasagne.layers.get_all_layers(network)[-2]
inputLayer = lasagne.layers.get_all_layers(network)[0]
layer_output = lasagne.layers.get_output(layer, deterministic=True)
dim_output= layer.output_shape[1]

networkGT = lasagne_nn.load_network(modelType=modelType, param_values=param_values)
layerGT = lasagne.layers.get_all_layers(networkGT)[-2]
inputLayerGT = lasagne.layers.get_all_layers(networkGT)[0]
layer_outputGT = lasagne.layers.get_output(layerGT, deterministic=True)

rendererGray =  0.3*renderer[:,:,0] +  0.59*renderer[:,:,1] + 0.11*renderer[:,:,2]
rendererGrayGT =  0.3*rendererGT[:,:,0] +  0.59*rendererGT[:,:,1] + 0.11*rendererGT[:,:,2]

chThError = TheanoFunOnOpenDR(theano_input=inputLayer.input_var, theano_output=layer_output, opendr_input=rendererGray - meanImage, dim_output = dim_output,
                              theano_input_gt=inputLayerGT.input_var, theano_output_gt=layer_outputGT, opendr_input_gt=rendererGrayGT - meanImage)

chThError.compileFunctions(layer_output, theano_input=inputLayer.input_var, dim_output=dim_output, theano_input_gt=inputLayerGT.input_var, theano_output_gt=layer_outputGT)

chThError.r


# chThFunGT = TheanoFunOnOpenDR(theano_input=inputLayer.input_var, theano_output=layer_output, opendr_input=rendererGrayGT - meanImage, dim_output = dim_output)

# chThFun.dr_wrt(chThFun.opendr_input)
# chThFun.old_grads()

# chNNAz = 2*ch.arctan(chThSHFun[1]/(ch.sqrt(chThSHFun[0]**2 + chThSHFun[1]**2) + chThSHFun[0]))
# chNNEl = 2*ch.arctan(chThSHFun[3]/(ch.sqrt(chThSHFun[2]**2 + chThSHFun[3]**2) + chThSHFun[2]))
#
# chNNAzGT = 2*ch.arctan(chThSHFunGT[1]/(ch.sqrt(chThSHFunGT[0]**2 + chThSHFunGT[1]**2) + chThSHFunGT[0]))
# chNNElGT = 2*ch.arctan(chThSHFunGT[3]/(ch.sqrt(chThSHFunGT[2]**2 + chThSHFunGT[3]**2) + chThSHFunGT[2]))

# nnPoseError = ch.sum((chThFun - chThFunGT)**2)

if 'neuralNetModelSHLight' in parameterRecognitionModels:
    SHModel = 'neuralNetModelSHLight'
    # modelPath = experimentDir + 'neuralNetModelRelSHComponents.npz'
    # with np.load(modelPath) as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    with open(trainModelsDirLightCoeffs + 'neuralNetModelRelSHLight.pickle', 'rb') as pfile:
        neuralNetModelSHLight = pickle.load(pfile)

    meanImage = neuralNetModelSHLight['mean']
    # ipdb.set_trace()
    modelType = neuralNetModelSHLight['type']
    param_values = neuralNetModelSHLight['params']
    grayTestImages =  0.3*images[:,:,:,0] +  0.59*images[:,:,:,1] + 0.11*images[:,:,:,2]
    grayTestImages = grayTestImages[:,None, :,:]
    grayTestImages = grayTestImages - meanImage

    network = lasagne_nn.load_network(modelType=modelType, param_values=param_values)
    shPredictionFun = lasagne_nn.get_prediction_fun(network)
    relLightCoefficientsGTPred = shPredictionFun(grayTestImages)


if 'randomForestSHZernike' in parameterRecognitionModels:

    SHModel = 'randomForestSHZernike'
    with open(trainModelsDirLightCoeffs  + 'randomForestModelZernike' + str(numCoeffs) + '_win' + str(win) + '.pickle', 'rb') as pfile:
        randForestModelLightCoeffs = pickle.load(pfile)

    relLightCoefficientsGTPred = recognition_models.testRandomForest(randForestModelLightCoeffs, testZernikefeatures)

if 'meanSHLight' in parameterRecognitionModels:
    SHModel = 'meanSHLight'
    relLightCoefficientsGTPred = np.tile(np.array([  7.85612876e-01,  -8.33688916e-03,   2.42002031e-01,
         6.63450677e-03,  -7.45523901e-04,   2.82965294e-03,
         3.09943249e-01,  -1.15105810e-03,  -3.26439948e-03])[None,:], [testLightCoefficientsGTRel.shape[0],1])

if 'constantSHLight' in parameterRecognitionModels:
    SHModel = 'Constant'
    relLightCoefficientsGTPred = np.tile(np.array([  0.75,  0,   0.15, 0,  0,   0, 0,  0,  0])[None,:], [testLightCoefficientsGTRel.shape[0],1])

if 'linRegModelSHZernike' in parameterRecognitionModels:
    with open(trainModelsDirLightCoeffs  + 'linRegModelZernike' + str(numCoeffs) + '_win' + str(win) + '.pickle', 'rb') as pfile:
        linearRegressionModelLightCoeffs = pickle.load(pfile)

    relLightCoefficientsGTPred = recognition_models.testLinearRegression(linearRegressionModelLightCoeffs, testZernikefeatures)

if not os.path.exists(resultDir + 'imgs/'):
    os.makedirs(resultDir + 'imgs/')
if not os.path.exists(resultDir +  'imgs/samples/'):
    os.makedirs(resultDir + 'imgs/samples/')


elevsPred = np.arctan2(sinElevsPred, cosElevsPred)
azsPred = np.arctan2(sinAzsPred, cosAzsPred)

azsGmms = []
elevsGmms = []
if recognitionType == 2:
    azsPredictions = np.arctan2(sinAzsPredictions, cosAzsPredictions)
    directory = resultDir
    plt.ioff()
    fig = plt.figure()
    for sample_i, sampleAzsPredictions in enumerate(azsPredictions.T):
        sample = (sample_i + 1)*2
        plt.scatter(np.ones_like(sampleAzsPredictions)*sample, np.mod(sampleAzsPredictions*180/np.pi, 360), c='b')

        from sklearn import mixture
        gmm = mixture.GMM(n_components=5, covariance_type='spherical', min_covar=radians(5)**2, init_params='wmc', params='wmc')

        gmm.fit(np.hstack([sinAzsPredictions[:,sample_i][:,None], cosAzsPredictions[:,sample_i][:,None]]))
        azsGmms = azsGmms + [gmm]
        for comp_i, weight in enumerate(gmm.weights_):
            meanSin = gmm.means_[comp_i][0]
            meanCos = gmm.means_[comp_i][1]
            azCompMean = np.arctan2(meanSin, meanCos)
            plt.plot(sample-0.4, np.mod(azCompMean*180/np.pi, 360), marker='o', ms=weight*50, c='y')

        plt.plot(sample,testAzsRel[sample_i]*180/np.pi, marker='o', ms=20, c='g')
        plt.plot(sample, np.mod(azsPred[sample_i]*180/np.pi, 360), marker='o', ms=20, c='r')

    plt.xlabel('Sample')
    plt.ylabel('Angular error')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,len(testSet+1)*2,0,360))
    plt.title('Random forest multiple predictions')
    fig.savefig(directory + 'randomForest_Az_scatter.png', bbox_inches='tight')
    plt.close(fig)
    # elsPredictions = [np.arctan2(sinElsPredictions[i], cosAzsPredictions[i]) for i in range(len(cosAzsPredictions))]

if recognitionType == 2:
    elevsPredictions = np.arctan2(sinElevsPredictions, cosElevsPredictions)
    directory = resultDir
    plt.ioff()
    fig = plt.figure()
    for sample_i, sampleElevsPredictions in enumerate(elevsPredictions.T):
        sample = (sample_i + 1)*2
        plt.scatter(np.ones_like(sampleElevsPredictions)*sample, np.mod(sampleElevsPredictions*180/np.pi, 90), c='b')

        from sklearn import mixture
        gmm = mixture.GMM(n_components=5, covariance_type='spherical', min_covar=radians(5)**2, init_params='wmc', params='wmc')

        gmm.fit(np.hstack([sinElevsPredictions[:,sample_i][:,None], cosElevsPredictions[:,sample_i][:,None]]))
        elevsGmms = elevsGmms + [gmm]
        for comp_i, weight in enumerate(gmm.weights_):
            meanSin = gmm.means_[comp_i][0]
            meanCos = gmm.means_[comp_i][1]
            elCompMean = np.arctan2(meanSin, meanCos)
            plt.plot(sample-0.4, np.mod(elCompMean*180/np.pi, 90), marker='o', ms=weight*50, c='y')

        plt.plot(sample, testElevsGT[sample_i]*180/np.pi, marker='o', ms=20, c='g')
        plt.plot(sample, np.mod(elevsPred[sample_i]*180/np.pi, 90), marker='o', ms=20, c='r')

    plt.xlabel('Sample')
    plt.ylabel('Angular error')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,len(testSet+1)*2,0,90))
    plt.title('Random forest multiple predictions')
    fig.savefig(directory + 'randomForest_Elev_scatter.png', bbox_inches='tight')
    plt.close(fig)
    # elsPredictions = [np.arctan2(sinElsPredictions[i], cosAzsPredictions[i]) for i in range(len(cosAzsPredictions))]

# testPredPoseGMMs = []
# colorGMMs = []
# if recognitionType == 2:
#     for test_i in range(len(testAzsRel)):
#         testPredPoseGMMs = testPredPoseGMMs + [recognition_models.poseGMM(azsPred[test_i], elevsPred[test_i])]
#         colorGMMs = colorGMMs + [recognition_models.colorGMM(images[test_i], 40)]

# errors = recognition_models.evaluatePrediction(testAzsRel, testElevsGT, testAzsRel, testElevsGT)
errors = recognition_models.evaluatePrediction(testAzsRel, testElevsGT, azsPred, elevsPred)

errorsLightCoeffs = np.linalg.norm(testLightCoefficientsGTRel - relLightCoefficientsGTPred, axis=1)
errorsVColors = np.linalg.norm(testVColorGT - vColorsPred, axis=1)

meanAbsErrAzs = np.mean(np.abs(errors[0]))
meanAbsErrElevs = np.mean(np.abs(errors[1]))

meanErrorsLightCoeffs = np.mean(errorsLightCoeffs)
meanErrorsVColors = np.mean(errorsVColors)

#Fit:
print("Fitting predictions")

fittedAzs = np.array([])
fittedElevs = np.array([])
fittedRelLightCoeffs = []
fittedVColors = []

predictedErrorFuns = np.array([])
fittedErrorFuns = np.array([])

print("Using " + modelsDescr[model])
errorFun = models[model]
pixelErrorFun = pixelModels[model]

testSamples = 1
if recognitionType == 2:
    testSamples  = numSamples
predSamples = 50

chDisplacement[:] = np.array([0.0, 0.0,0.0])
chScale[:] = np.array([1.0,1.0,1.0])
chObjAz[:] = 0
shapeIm = [height, width]

#Update all error functions with the right renderers.
print("Using " + modelsDescr[model])

negLikModel = -ch.sum(generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances))/numPixels
negLikModelRobust = -ch.sum(generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances))/numPixels
pixelLikelihoodCh = generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances)
pixelLikelihoodRobustCh = generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances)
# negLikModel = -generative_models.modelLogLikelihoodCh(rendererGT, renderer, np.array([]), 'FULL', variances)/numPixels
# negLikModelRobust = -generative_models.modelLogLikelihoodRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances)/numPixels
# pixelLikelihoodCh = generative_models.logPixelLikelihoodCh(rendererGT, renderer, np.array([]), 'FULL', variances)
# pixelLikelihoodRobustCh = ch.log(generative_models.pixelLikelihoodRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances))

post = generative_models.layerPosteriorsRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances)[0]

# hogGT, hogImGT, drconv = image_processing.diffHog(rendererGT)
# hogRenderer, hogImRenderer, _ = image_processing.diffHog(renderer, drconv)
#
# hogE_raw = hogGT - hogRenderer
# hogCellErrors = ch.sum(hogE_raw*hogE_raw, axis=2)
# hogError = ch.SumOfSquares(hogE_raw)

# models = [negLikModel, negLikModelRobust, hogError]
models = [negLikModel, negLikModelRobust, negLikModelRobust]
# pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh, hogCellErrors]
pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh, pixelLikelihoodRobustCh]
modelsDescr = ["Gaussian Model", "Outlier model", "HOG"]

# models = [negLikModel, negLikModelRobust, negLikModelRobust]
# pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh, pixelLikelihoodRobustCh]
# pixelErrorFun = pixelModels[model]
errorFun = models[model]

# if optimizationTypeDescr[optimizationType] != 'predict':
startTime = time.time()
if (computePredErrorFuns and optimizationType == 0) or optimizationType != 0:
    for test_i in range(len(testAzsRel)):

        bestFittedAz = chAz.r
        bestFittedEl = chEl.r
        bestModelLik = np.finfo('f').max
        bestVColors = chVColors.r
        bestLightSHCoeffs = chLightSHCoeffs.r

        testId = dataIds[test_i]
        print("************** Minimizing loss of prediction " + str(test_i) + "of " + str(len(testAzsRel)))

        rendererGT[:] = srgb2lin(images[test_i])

        if not os.path.exists(resultDir + 'imgs/test'+ str(test_i) + '/'):
            os.makedirs(resultDir + 'imgs/test'+ str(test_i) + '/')

        cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/id' + str(testId) +'_groundtruth' + '.png', cv2.cvtColor(np.uint8(lin2srgb(rendererGT.r.copy())*255), cv2.COLOR_RGB2BGR))

        for sample in range(testSamples):
            from numpy.random import choice
            if recognitionType == 0:
                #Prediction from (near) ground truth.
                color = testVColorGT[test_i] + nearGTOffsetVColor
                az = testAzsRel[test_i] + nearGTOffsetRelAz
                el = testElevsGT[test_i] + nearGTOffsetEl
                lightCoefficientsRel = testLightCoefficientsGTRel[test_i]
            elif recognitionType == 1 or recognitionType == 2:
                #Point (mean) estimate:
                az = azsPred[test_i]
                el = min(max(elevsPred[test_i],radians(1)), np.pi/2-radians(1))

                color = vColorsPred[test_i]

                # color = testVColorGT[test_i]

                lightCoefficientsRel = relLightCoefficientsGTPred[test_i]
                # color = testVColorGT[test_i]
                # lightCoefficientsGTRel = testLightCoefficientsGTRel[test_i]
            # else:
            #     #Sampling
            #     # poseComps, vmAzParams, vmElParams = testPredPoseGMMs[test_i]
            #     sampleComp = choice(len(poseComps), size=1, p=poseComps)
            #     # az = np.random.vonmises(vmAzParams[sampleComp][0],vmAzParams[sampleComp][1],1)
            #     # el = np.random.vonmises(vmElParams[sampleComp][0],vmElParams[sampleComp][1],1)
            #     az = azsPred[test_i]
            #     el = elevsPred[test_i]
            #
            #     color = vColorsPred[test_i]
            #     lightCoefficientsGTRel = relLightCoefficientsGTPred[test_i]

            # chAz[:] = testAzsRel[test_i]
            # chEl[:] = testElevsGT[test_i]
            # chVColors[:] =  testVColorGT[test_i]
            # # chVColors[:] = testPredVColors[test_i]
            #
            # chLightSHCoeffs[:] =testLightCoefficientsGTRel[test_i]
            #
            # cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/sample' + str(sample) +  '_reconstructed'+ '.png', cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))

            chAz[:] = az
            chEl[:] = el
            chVColors[:] = color
            chLightSHCoeffs[:] = lightCoefficientsRel

            #Update all error functions with the right renderers.
            # negLikModel = -ch.sum(generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances))/numPixels
            # negLikModelRobust = -ch.sum(generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances))/numPixels
            # negLikModel = -generative_models.modelLogLikelihoodCh(rendererGT, renderer, np.array([]), 'FULL', variances)/numPixels
            # negLikModelRobust = -generative_models.modelLogLikelihoodRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances)/numPixels
            # pixelLikelihoodCh = generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances)
            # pixelLikelihoodRobustCh = generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances)

            # negLikModel = -generative_models.modelLogLikelihoodCh(rendererGT, renderer, np.array([]), 'FULL', variances)/numPixels
            # negLikModelRobust = -generative_models.modelLogLikelihoodRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances)/numPixels
            # pixelLikelihoodCh = generative_models.logPixelLikelihoodCh(rendererGT, renderer, np.array([]), 'FULL', variances)
            # pixelLikelihoodRobustCh = ch.log(generative_models.pixelLikelihoodRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances))

            # post = generative_models.layerPosteriorsRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances)[0]
            #
            # hogGT, hogImGT, drconv = image_processing.diffHog(rendererGT,drconv)
            # hogRenderer, hogImRenderer, _ = image_processing.diffHog(renderer, drconv)
            #
            # hogE_raw = hogGT - hogRenderer
            # hogCellErrors = ch.sum(hogE_raw*hogE_raw, axis=2)
            # hogError = ch.SumOfSquares(hogE_raw)


            stds[:] = 0.1

            model = 1
            errorFun = models[model]

            bestPredAz = chAz.r
            bestPredEl = chEl.r
            bestPredModelLik = errorFun.r

            cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/sample' + str(sample) +  '_predicted'+ '.png', cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))

            if recognitionType == 2:
                continueSampling = True
                numPredSamples = 0
                previousAngles = np.array([[az, el]])
                numGoodPreviousAngles = 0
                numberSkipped = 0
                while continueSampling:
                    if numPredSamples >= predSamples:
                        continueSampling = False
                        print("Last sample!")
                    print("Sample " + str(numPredSamples))
                    azGmm = azsGmms[test_i]
                    elGmm = elevsGmms[test_i]
                    (sinAzSample, cosAzSample) = azGmm.sample()[0]
                    azsample = np.arctan2(sinAzSample, cosAzSample)
                    (sinElevSample, cosElevSample) = elGmm.sample()[0]
                    elsample = min(max(np.arctan2(sinElevSample, cosElevSample),radians(1)), np.pi/2-radians(1))
                    skipSample = False

                    for previousAngle in previousAngles:
                        diffAngles = recognition_models.evaluatePrediction(previousAngle[0], previousAngle[1], azsample, elsample)
                        if abs(diffAngles[0] < 5) and abs(diffAngles[1] < 5):
                            skipSample = True
                            print("Seen this angle before!")
                            break

                    if skipSample:
                        numberSkipped += 1
                        if numberSkipped > 100:
                            continueSampling = False
                        continue

                    numberSkipped = 0
                    numPredSamples += 1

                    chAz[:] = azsample
                    chEl[:] = elsample
                    print("Minimizing first step")
                    model = 1
                    errorFun = models[model]
                    method=1
                    stds[:] = 0.01
                    free_variables = [chVColors, chLightSHCoeffs[1:4]]
                    options={'disp':False, 'maxiter':5}
                    ch.minimize({'raw': errorFun}, bounds=None, method=methods[method], x0=free_variables, callback=cb, options=options)
                    # print("Minimizing second step")
                    # method=2
                    # free_variables = [ chAz, chEl, chVColors, chLightSHCoeffs]
                    # options={'disp':False, 'maxiter':5}
                    # ch.minimize({'raw': errorFun}, bounds=None, method=methods[method], x0=free_variables, callback=cb, options=options)

                    if errorFun.r < bestPredModelLik:
                        diffAngles = recognition_models.evaluatePrediction(chAz.r, chEl.r, bestPredAz, bestPredEl)
                        if abs(diffAngles[0] < 5) and abs(diffAngles[1] < 5):
                            numGoodPreviousAngles += 1
                            if numGoodPreviousAngles >= 3:
                                continueSampling = False
                                print("Enough good samples!")
                        print("Found best angle!")
                        bestPredModelLik = errorFun.r.copy()
                        bestPredAz = chAz.r.copy()
                        bestPredEl = min(max(chEl.r.copy(),radians(1)), np.pi/2-radians(1))
                        bestPredVColors = chVColors.r.copy()
                        bestPredLightSHCoeffs = chLightSHCoeffs.r.copy()
                        bestModelLik = errorFun.r.copy()
                        cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/best_predSample' + str(numPredSamples) + '.png', cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))


                    previousAngles = np.vstack([previousAngles, np.array([[azsample, elsample],[chAz.r.copy(), chEl.r.copy()]])])

            chAz[:] = az
            chEl[:] = min(max(el,radians(1)), np.pi/2-radians(1))
            chVColors[:] = color
            chLightSHCoeffs[:] = lightCoefficientsRel.copy()

            # cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/best_sample' + str(sample) +  '_predicted'+ '.png', cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))
            # plt.imsave(resultDir + 'imgs/test'+ str(test_i) + '/id' + str(testId) +'_groundtruth_drAz' + '.png', z.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)

            predictedErrorFuns = np.append(predictedErrorFuns, errorFun.r)

            global iterat
            iterat = 0

            sys.stdout.flush()
            if optimizationTypeDescr[optimizationType] == 'optimize':
                print("** Minimizing intial predicted parameters. **")
                model=1
                errorFun = models[model]
                # errorFun = chThError
                method=1
                stds[:] = 0.1

                options={'disp':False, 'maxiter':10}
                # options={'disp':False, 'maxiter':maxiter, 'lr':0.0001, 'momentum':0.1, 'decay':0.99}
                free_variables = [ chAz, chEl]
                ch.minimize({'raw': errorFun}, bounds=None, method=methods[method], x0=free_variables, callback=cb, options=options)

            if errorFun.r < bestModelLik:
                bestModelLik = errorFun.r.copy()
                bestFittedAz = chAz.r.copy()
                bestFittedEl = min(max(chEl.r.copy(),radians(1)), np.pi/2-radians(1))
                bestVColors = chVColors.r.copy()
                bestLightSHCoeffs = chLightSHCoeffs.r.copy()
                cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/best'+ '.png', cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))
            else:
                bestFittedAz = bestPredAz.copy()
                bestFittedEl = min(max(bestPredEl.copy(),radians(1)), np.pi/2-radians(1))
                bestVColors = bestPredVColors.copy()
                bestLightSHCoeffs = bestPredLightSHCoeffs.copy()


            chAz[:] = bestFittedAz
            chEl[:] = min(max(bestFittedEl,radians(1)), np.pi/2-radians(1))
            chVColors[:] = bestVColors
            chLightSHCoeffs[:] = bestLightSHCoeffs

            # model=1
            # errorFun = models[model]
            # method=1
            # stds[:] = 0.1
            # options={'disp':False, 'maxiter':5}
            # free_variables = [ chAz, chEl, chVColors, chLightSHCoeffs]
            # ch.minimize({'raw': errorFun}, bounds=None, method=methods[method], x0=free_variables, callback=cb, options=options)

            cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/fitted'+ '.png',cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))

        if not optimizationTypeDescr[optimizationType] == 'predict':
            fittedErrorFuns = np.append(fittedErrorFuns, bestModelLik)
            fittedAzs = np.append(fittedAzs, bestFittedAz)
            fittedElevs = np.append(fittedElevs, bestFittedEl)
            fittedVColors = fittedVColors + [bestVColors]
            fittedRelLightCoeffs = fittedRelLightCoeffs + [bestLightSHCoeffs]

totalTime = time.time() - startTime
print("Took " + str(totalTime/len(testSet)) + " time per instance.")

if optimizationTypeDescr[optimizationType] != 'predict':
    fittedVColors = np.vstack(fittedVColors)
    fittedRelLightCoeffs = np.vstack(fittedRelLightCoeffs)

testOcclusions = dataOcclusions

errorsFittedRF = (np.array([]),np.array([]))
if optimizationTypeDescr[optimizationType] != 'predict':
    errorsFittedRF = recognition_models.evaluatePrediction(testAzsRel, testElevsGT, fittedAzs, fittedElevs)
    meanAbsErrAzsFittedRF = np.mean(np.abs(errorsFittedRF[0]))
    meanAbsErrElevsFittedRF = np.mean(np.abs(errorsFittedRF[1]))

errorsFittedLightCoeffs = np.array([])
errorsFittedVColors = np.array([])
if optimizationTypeDescr[optimizationType] != 'predict':
    errorsFittedLightCoeffs = np.linalg.norm(testLightCoefficientsGTRel - fittedRelLightCoeffs, axis=1)
    errorsFittedVColors = np.linalg.norm(testVColorGT - fittedVColors, axis=1)
    meanErrorsFittedLightCoeffs = np.mean(errorsFittedLightCoeffs)
    meanErrorsFittedVColors = np.mean(errorsFittedVColors)

plt.ioff()
import seaborn


directory = resultDir + 'predicted-azimuth-error'

fig = plt.figure()
plt.scatter(testElevsGT*180/np.pi, errors[0])
plt.xlabel('Elevation (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,90,-90,90))
plt.title('Performance scatter plot')
fig.savefig(directory + '_elev-performance-scatter.png', bbox_inches='tight')
plt.close(fig)

fig = plt.figure()
plt.scatter(testOcclusions*100.0,errors[0])
plt.xlabel('Occlusion (%)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,100,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory + '_occlusion-performance-scatter.png', bbox_inches='tight')
plt.close(fig)

fig = plt.figure()
plt.scatter(testAzsRel*180/np.pi, errors[0])
plt.xlabel('Azimuth (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,360,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory  + '_azimuth-performance-scatter.png', bbox_inches='tight')
plt.close(fig)

fig = plt.figure()
plt.hist(errors[0], bins=18)
plt.xlabel('Angular error')
plt.ylabel('Counts')
x1,x2,y1,y2 = plt.axis()
plt.axis((-180,180,y1, y2))
plt.title('Performance histogram')
fig.savefig(directory  + '_performance-histogram.png', bbox_inches='tight')
plt.close(fig)

# fig = plt.figure()
# plt.hist(predErrorAzs, bins=30)
# plt.xlabel('Angular error')
# plt.ylabel('Counts')
# x1,x2,y1,y2 = plt.axis()
# plt.axis((-180,180,y1, 700))
# plt.title('Predicted Azimuth')
# fig.savefig('histaz1.eps', bbox_inches='tight')
#
# fig = plt.figure()
# plt.hist(predErrorElevs, bins=30)
# plt.xlabel('Angular error')
# plt.ylabel('Counts')
# x1,x2,y1,y2 = plt.axis()
# plt.axis((-90,90,y1, 700))
# plt.title('Predicted Elevation')
# fig.savefig('histel1.eps', bbox_inches='tight')
#
# fig = plt.figure()
# plt.hist(errorsFittedAzs, bins=30)
# plt.xlabel('Angular error')
# plt.ylabel('Counts')
# x1,x2,y1,y2 = plt.axis()
# plt.axis((-180,180,y1, 700))
# plt.title('Fitted azimuth')
# fig.savefig('histaz2.eps', bbox_inches='tight')
#
# fig = plt.figure()
# plt.hist(errorsFittedElevs, bins=30)
# plt.xlabel('Angular error')
# plt.ylabel('Counts')
# x1,x2,y1,y2 = plt.axis()
# plt.axis((-90,90,y1, 700))
# plt.title('Fitted Elevation')
# fig.savefig('histel2.eps', bbox_inches='tight')

directory = resultDir + 'predicted-elevation-error'

fig = plt.figure()
plt.scatter(testElevsGT*180/np.pi, errors[1])
plt.xlabel('Elevation (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,90,-90,90))
plt.title('Performance scatter plot')
fig.savefig(directory + '_elev-performance-scatter.png', bbox_inches='tight')
plt.close(fig)

fig = plt.figure()
plt.scatter(testOcclusions*100.0,errors[1])
plt.xlabel('Occlusion (%)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,100,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory + '_occlusion-performance-scatter.png', bbox_inches='tight')
plt.close(fig)

fig = plt.figure()
plt.scatter(testAzsRel*180/np.pi, errors[1])
plt.xlabel('Azimuth (degrees)')
plt.ylabel('Angular error')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,360,-180,180))
plt.title('Performance scatter plot')
fig.savefig(directory  + '_azimuth-performance-scatter.png', bbox_inches='tight')
plt.close(fig)

fig = plt.figure()
plt.hist(errors[1], bins=18)
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
    plt.scatter(testElevsGT*180/np.pi, errorsFittedRF[0])
    plt.xlabel('Elevation (degrees)')
    plt.ylabel('Angular error')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,90,-90,90))
    plt.title('Performance scatter plot')
    fig.savefig(directory + '_elev-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(testOcclusions*100.0,errorsFittedRF[0])
    plt.xlabel('Occlusion (%)')
    plt.ylabel('Angular error')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,100,-180,180))
    plt.title('Performance scatter plot')
    fig.savefig(directory + '_occlusion-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(testAzsRel*180/np.pi, errorsFittedRF[0])
    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('Angular error')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,360,-180,180))
    plt.title('Performance scatter plot')
    fig.savefig(directory  + '_azimuth-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.hist(errorsFittedRF[0], bins=18)
    plt.xlabel('Angular error')
    plt.ylabel('Counts')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((-180,180,y1, y2))
    plt.title('Performance histogram')
    fig.savefig(directory  + '_performance-histogram.png', bbox_inches='tight')
    plt.close(fig)

    directory = resultDir + 'fitted-elevation-error'

    fig = plt.figure()
    plt.scatter(testElevsGT*180/np.pi, errorsFittedRF[1])
    plt.xlabel('Elevation (degrees)')
    plt.ylabel('Angular error')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,90,-90,90))
    plt.title('Performance scatter plot')
    fig.savefig(directory + '_elev-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(testOcclusions*100.0,errorsFittedRF[1])
    plt.xlabel('Occlusion (%)')
    plt.ylabel('Angular error')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,100,-180,180))
    plt.title('Performance scatter plot')
    fig.savefig(directory + '_occlusion-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(testAzsRel*180/np.pi, errorsFittedRF[1])
    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('Angular error')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,360,-180,180))
    plt.title('Performance scatter plot')
    fig.savefig(directory  + '_azimuth-performance-scatter.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.hist(errorsFittedRF[1], bins=18)
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
    if not optimizationTypeDescr[optimizationType] == 'predict':
        expfile.write("Mean Azimuth Error (fitted) " +  str(meanAbsErrAzsFittedRF)+ '\n')
        expfile.write("Mean Elevation Error (fitted) " +  str(meanAbsErrElevsFittedRF)+ '\n')
    expfile.write("Mean SH Components Error (predicted) " +  str(meanErrorsLightCoeffs)+ '\n')
    expfile.write("Mean Vertex Colors Error (predicted) " +  str(meanErrorsVColors)+ '\n')
    if not optimizationTypeDescr[optimizationType] == 'predict':
        expfile.write("Mean SH Components Error (fitted) " +  str(meanErrorsFittedLightCoeffs)+ '\n')
        expfile.write("Mean Vertex Colors Error (fitted) " +  str(meanErrorsFittedVColors)+ '\n')

if not optimizationTypeDescr[optimizationType] == 'predict':
    headerDesc = "Pred NLL    :" + "Fitt NLL    :" + "Err Pred Az :" + "Err Pred El :" + "Err Fitted Az :" + "Err Fitted El :" + "Occlusions  :"
    perfSamplesData = np.hstack([predictedErrorFuns.reshape([-1,1]),fittedErrorFuns.reshape([-1,1]),errors[0].reshape([-1,1]),errors[1].reshape([-1,1]),errorsFittedRF[0].reshape([-1,1]),errorsFittedRF[1].reshape([-1,1]),testOcclusions.reshape([-1,1])])
elif optimizationTypeDescr[optimizationType] == 'predict' and computePredErrorFuns:
    headerDesc = "Pred NLL    :" + "Err Pred Az :" + "Err Pred El :"  +  "Occlusions  :"
    perfSamplesData = np.hstack([predictedErrorFuns.reshape([-1,1]),errors[0].reshape([-1,1]),errors[1].reshape([-1,1]),testOcclusions.reshape([-1,1])])
else:
    headerDesc = "Err Pred Az :" + "Err Pred El :"  +  "Occlusions  :"
    perfSamplesData = np.hstack([errors[0].reshape([-1,1]),errors[1].reshape([-1,1]),testOcclusions.reshape([-1,1])])

np.savetxt(resultDir + 'performance_samples.txt', perfSamplesData, delimiter='\t', fmt="%g", header=headerDesc)

np.savez(resultDir + 'performance_samples.npz',  predictedErrorFuns=predictedErrorFuns, fittedErrorFuns= fittedErrorFuns, predErrorAzs=errors[0], predErrorElevs=errors[1], errorsLightCoeffs=errorsLightCoeffs, errorsVColors=errorsVColors, errorsFittedAzs=errorsFittedRF[0], errorsFittedElevs=errorsFittedRF[1], errorsFittedLightCoeffs=errorsFittedLightCoeffs, errorsFittedVColors=errorsFittedVColors,testOcclusions=testOcclusions )

import tabulate
headers=["Errors", "Pred (mean)", "Stdv", "Fitted (mean)", "Stdv"]

table = [["NLL", np.mean(predictedErrorFuns), 0, np.mean(fittedErrorFuns), 0],
         ["Azimuth", np.mean(np.abs(errors[0])), np.std(np.abs(errors[0])), np.mean(np.abs(errorsFittedRF[0])), np.std(np.abs(errorsFittedRF[0]))],
         ["Elevation", np.mean(np.abs(errors[1])), np.std(np.abs(errors[1])), np.mean(np.abs(errorsFittedRF[1])), np.std(np.abs(errorsFittedRF[1]))],
         ["VColor", np.mean(errorsVColors), np.std(errorsVColors), np.mean(errorsFittedVColors), np.std(errorsFittedVColors)],
         ["SH Light", np.mean(errorsLightCoeffs), np.std(errorsLightCoeffs), np.mean(errorsFittedLightCoeffs), np.std(np.abs(errorsFittedLightCoeffs))],
         ]
performanceTable = tabulate.tabulate(table, headers=headers,tablefmt="latex", floatfmt=".2f")
with open(resultDir + 'performance.tex', 'w') as expfile:
    expfile.write(performanceTable)


headers=["Method", "l=0", "SH $l=0,m=-1$", "SH $l=1,m=0$", "SH $l=1,m=1$", "SH $l=1,m=-2$" , "SH $l=2,m=-1$", "SH $l=2,m=0$", "SH $l=2,m=1$", "SH $l=2,m=2$"]

# ipdb.set_trace()
SMSE_SH = np.mean((testLightCoefficientsGTRel - relLightCoefficientsGTPred)**2 + 1e-5, axis=0)/np.var(testLightCoefficientsGTRel + 1e-5, axis=0)
table = [[SHModel, SMSE_SH[0], SMSE_SH[1], SMSE_SH[2],SMSE_SH[3], SMSE_SH[4], SMSE_SH[5],SMSE_SH[6], SMSE_SH[7], SMSE_SH[8] ],
        ]
performanceTable = tabulate.tabulate(table, headers=headers,tablefmt="latex", floatfmt=".2f")
with open(resultDir + 'performance_SH.tex', 'w') as expfile:
    expfile.write(performanceTable)

print("Finished backprojecting and fitting estimates.")


np.mean(errorsVColors)