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
import OpenGL.GL as GL
import glfw

useShapeModel = True

#########################################
# OpenDR Initialization starts here
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
renderTeapotsList = np.arange(len(teapots))[0:1]

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
    renderer = createRendererTarget(glMode, True, chAz, chObjAz, chEl, chDist, centermod, vmod, vcmod, fmod_list, vnmod, light_color, chComponent, chVColors, 0, chDisplacement, chScale, width,height, uvmod, haveTexturesmod_list, texturesmod_list, frustum, win )
    renderer.msaa = True
    renderer.r
    renderer_teapots = renderer_teapots + [renderer]

currentTeapotModel = 0

center = center_teapots[currentTeapotModel]

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

    chNormals = shape_model.chGetNormals(chVertices, faces)

    smNormals = [chNormals]
    smFaces = [[faces]]
    smVColors = [chVColors*np.ones(chVertices.shape)]
    smUVs = [ch.Ch(np.zeros([chVertices.shape[0],2]))]
    smHaveTextures = [[False]]
    smTexturesList = [[None]]

    chVertices = chVertices - ch.mean(chVertices, axis=0)
    minZ = ch.min(chVertices[:,2])

    chMinZ = ch.min(chVertices[:,2])

    zeroZVerts = chVertices[:,2]- chMinZ
    chVertices = ch.hstack([chVertices[:,0:2] , zeroZVerts.reshape([-1,1])])

    chVertices = chVertices*0.09
    smCenter = ch.array([0,0,0.1])

    smVertices = [chVertices]
    chNormals = shape_model.chGetNormals(chVertices, faces)
    smNormals = [chNormals]

    renderer = createRendererTarget(glMode, True, chAz, chObjAz, chEl, chDist, smCenter, [smVertices], [smVColors], [smFaces], [smNormals], light_color, chComponent, chVColors, 0, chDisplacement, chScale, width,height, [smUVs], [smHaveTextures], [smTexturesList], frustum, win )
    renderer.msaa = True
    renderer.overdraw = True

    chShapeParams[:] = np.zeros([latentDim])
    chVerticesMean = chVertices.r.copy()
else:
    renderer = renderer_teapots[0]

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

negLikModel = -ch.sum(generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances))
negLikModelRobust = -ch.sum(generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances))
pixelLikelihoodCh = generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances)
pixelLikelihoodRobustCh = generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances)
# negLikModel = -generative_models.modelLogLikelihoodCh(rendererGT, renderer, np.array([]), 'FULL', variances)/numPixels
# negLikModelRobust = -generative_models.modelLogLikelihoodRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances)/numPixels
# pixelLikelihoodCh = generative_models.logPixelLikelihoodCh(rendererGT, renderer, np.array([]), 'FULL', variances)
# pixelLikelihoodRobustCh = ch.log(generative_models.pixelLikelihoodRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances))

post = generative_models.layerPosteriorsRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances)[0]

# modelLogLikelihoodRobustRegionCh = -ch.sum(generative_models.LogRobustModelRegion(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances))/numPixels
#
# pixelLikelihoodRobustRegionCh = generative_models.LogRobustModelRegion(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances)

import opendr.filters
robPyr = opendr.filters.gaussian_pyramid(renderer - rendererGT, n_levels=6, normalization='size')
robPyrSum = -ch.sum(ch.log(ch.exp(-0.5*robPyr**2/variances) + 1))

# models = [negLikModel, negLikModelRobust, hogError]
models = [negLikModel, negLikModelRobust, robPyrSum]
pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh, robPyr]
modelsDescr = ["Gaussian Model", "Outlier model", "Region Robust" ]

model = 1
pixelErrorFun = pixelModels[model]
errorFun = models[model]



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


# ######## Read different datasets
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
# ipdb.set_trace()
#
# errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList, errorsSegmentation \
#         = computeErrors(np.arange(len(rangeTests)), azimuths, testAzsRel, elevations, testElevsGT, vColors, testVColorGT, lightCoeffs, testLightCoefficientsGTRel, approxProjections,  approxProjectionsGT, shapeParams, testShapeParamsGT, useShapeModel, chShapeParams, chVertices, posteriors, masksGT)
#
# meanAbsErrAzsList, meanAbsErrElevsList, medianAbsErrAzsList, medianAbsErrElevsList, meanErrorsLightCoeffsList, meanErrorsShapeParamsList, meanErrorsShapeVerticesList, meanErrorsLightCoeffsCList, meanErrorsEnvMapList, meanErrorsVColorsEList, meanErrorsVColorsCList, meanErrorsSegmentation \
#     = computeErrorMeans(np.arange(len(rangeTests)), useShapeModel, errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList, errorsSegmentation)

testOcclusionsFull = testOcclusions.copy()


testPrefix = 'train4_occlusion_shapemodel_10k_ECCV'
testPrefixBase = testPrefix

resultDir = 'results/' + testPrefix + '/'

if not os.path.exists(resultDir):
    os.makedirs(resultDir)
if not os.path.exists(resultDir + 'imgs/'):
    os.makedirs(resultDir + 'imgs/')
if not os.path.exists(resultDir +  'imgs/samples/'):
    os.makedirs(resultDir + 'imgs/samples/')

expFilename = resultDir + 'experiment.pickle'
with open(expFilename, 'rb') as pfile:
    experimentDic = pickle.load(pfile)

methodsPred = experimentDic['methodsPred']
testOcclusions = experimentDic[ 'testOcclusions']
testPrefixBase = experimentDic[ 'testPrefixBase']
parameterRecognitionModels = experimentDic[ 'parameterRecognitionModels']
azimuths = experimentDic[ 'azimuths']
elevations = experimentDic[ 'elevations']
vColors = experimentDic[ 'vColors']
lightCoeffs = experimentDic[ 'lightCoeffs']
likelihoods = experimentDic['likelihoods']
# approxProjections = experimentDic[ 'approxProjections']
# approxProjectionsGT = experimentDic[ 'approxProjectionsGT']

shapeParams = experimentDic['shapeParams']

expErrorsFilename = resultDir + 'experiment_errors.pickle'
with open(expErrorsFilename, 'rb') as pfile:
    experimentErrorsDic = pickle.load(pfile)

errorsPosePredList = experimentErrorsDic['errorsPosePredList']
errorsLightCoeffsList = experimentErrorsDic['errorsLightCoeffsList']
errorsShapeParamsList = experimentErrorsDic['errorsShapeParamsLis']
errorsShapeVerticesList = experimentErrorsDic['errorsShapeVerticesList']
errorsEnvMapList = experimentErrorsDic['errorsEnvMapList']
errorsLightCoeffsCList = experimentErrorsDic['errorsLightCoeffsCList']
errorsVColorsEList = experimentErrorsDic['errorsVColorsEList']
errorsVColorsCList = experimentErrorsDic['errorsVColorsCList']
errorsSegmentationList = experimentErrorsDic['errorsSegmentationList']

loadMask = True
if loadMask:
    masksGT = loadMasks(gtDir + '/masks_occlusion/', testSet)

## Get likelihood function values:

# #
# #
# fittedPosteriorsList= []
# recognitionPosteriorList = []
#
# posteriorsComp = []
# fittedPosteriorsList1= []
# fittedPosteriorsList2= []
# fittedPosteriorsList3= []
# fittedPosteriorsList4= []
#
# likelihoods=[None,None,None,None,None]
# posteriors = [None,None,None,None,None]
#
# for test_i in range(len(testAzsRel)):
#
#     testId = dataIds[test_i]
#     print("************** Minimizing loss of prediction " + str(test_i) + "of " + str(len(testAzsRel)))
#     image = skimage.transform.resize(images[test_i], [height,width])
#     imageSrgb = image.copy()
#     rendererGT[:] = srgb2lin(image)
#
#     negLikModel = -ch.sum(generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances))/numPixels
#     negLikModelRobust = -ch.sum(generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances))/numPixels
#     models = [negLikModel, negLikModelRobust]
#
#     errorFun = models[1]
#
#     modelsDescr = ["Gaussian Model", "Outlier model"]
#
#     # color = testVColorGT[test_i]
#     # az = testAzsRel[test_i]
#     # el = testElevsGT[test_i]
#     # lightCoefficientsRel = testLightCoefficientsGTRel[test_i]
#     #
#     #
#     # chAz[:] = az
#     # chEl[:] = el
#     # chVColors[:] = color
#     # chLightSHCoeffs[:] = lightCoefficientsRel
#     # if useShapeModel:
#     #     chShapeParams[:] = testShapeParamsGT[test_i]
#
#     #Recognition
#
#     # chAz[:] = azimuths[2][test_i]
#     # chEl[:] = elevations[2][test_i]
#     # chVColors[:] = vColors[2][test_i]
#     # chLightSHCoeffs[:] = lightCoeffs[2][test_i]
#     # if useShapeModel:
#     #     chShapeParams[:] = shapeParams[2][test_i]
#     # recognitionPosteriorList = recognitionPosteriorList + [np.array(renderer.indices_image==1).copy().astype(np.bool)[None,:]]
#     #
#
#
#     #Recognition
#
#     chAz[:] = azimuths[2][test_i]
#     chEl[:] = elevations[2][test_i]
#     chVColors[:] = vColors[2][test_i]
#     chLightSHCoeffs[:] = lightCoeffs[2][test_i]
#     if useShapeModel:
#         chShapeParams[:] = shapeParams[2][test_i]
#     recognitionPosteriorList = recognitionPosteriorList + [np.array(renderer.indices_image==1).copy().astype(np.bool)[None,:]]
#
#
#     #Robust
#
#     chAz[:] = azimuths[4][test_i]
#     chEl[:] = elevations[4][test_i]
#     chVColors[:] = vColors[4][test_i]
#     chLightSHCoeffs[:] = lightCoeffs[4][test_i]
#     if useShapeModel:
#         chShapeParams[:] = shapeParams[4][test_i]
#
#     #Robust
#     stds[:] = 0.05
#     vis_im = np.array(renderer.indices_image==1).copy().astype(np.bool)
#     post = generative_models.layerPosteriorsRobustCh(rendererGT, renderer, vis_im, 'MASK', globalPrior, variances)[0].r>0.5
#     fittedPosteriorsList = fittedPosteriorsList2 + [post[None,:]]
#
# recognitionSegmentation = np.vstack(recognitionPosteriorList)
# fittedPosteriors = np.vstack(fittedPosteriorsList)
#
# segmentations = [None, None, recognitionSegmentation, None, fittedPosteriors]
# #
# segmentationsDic = {'segmentations':segmentations}
# with open(resultDir + 'segmentations.pickle', 'wb') as pfile:
#     pickle.dump(segmentationsDic, pfile)

with open(resultDir + 'segmentations.pickle', 'rb') as pfile:
    segmentationsDic = pickle.load(pfile)

segmentations = segmentationsDic['segmentations']

testPrefix = 'train4_occlusion_shapemodel_10k_ECCV_detection'
testPrefixBase = testPrefix

resultDir = 'results/' + testPrefix + '/'

if not os.path.exists(resultDir):
    os.makedirs(resultDir)
if not os.path.exists(resultDir + 'imgs/'):
    os.makedirs(resultDir + 'imgs/')
if not os.path.exists(resultDir +  'imgs/samples/'):
    os.makedirs(resultDir + 'imgs/samples/')

SHModel = ""

testi = 0
badShapes = np.array([],dtype=np.bool)
for testCase in range(len(testSet)):
    badShape = False
    shapeVals = shapeParams[4][testCase]
    chShapeParams[:] = shapeVals
    shapeNorm = np.linalg.norm(chVertices.r - chVerticesMean)
    if np.linalg.norm(chVertices.r - chVerticesMean) >= 3:
        badShape = True
    badShapes = np.append(badShapes, badShape)
    testi = testi + 1

detected = np.logical_or.reduce([np.any(shapeParams[4] >= 4, axis=1), elevations[4] >= np.pi/2 - 0.05, elevations[4] <= 0.05, badShapes])
robustIdx = 4

azimuths[robustIdx][detected] = azimuths[2][detected]
elevations[robustIdx][detected] = elevations[2][detected]
vColors[robustIdx][detected] = vColors[2][detected]
lightCoeffs[robustIdx][detected] = lightCoeffs[2][detected]
shapeParams[robustIdx][detected] = shapeParams[2][detected]

errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, _, _, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList, errorsSegmentationList \
        = computeErrors(np.arange(len(rangeTests)), azimuths, testAzsRel, elevations, testElevsGT, vColors, testVColorGT, lightCoeffs, testLightCoefficientsGTRel, None,  None, shapeParams, testShapeParamsGT, useShapeModel, chShapeParams, chVertices, segmentations, masksGT)

experimentDic = {'testSet':testSet, 'methodsPred':methodsPred, 'testOcclusions':testOcclusions, 'likelihoods':likelihoods, 'testPrefixBase':testPrefixBase, 'parameterRecognitionModels':parameterRecognitionModels, 'azimuths':azimuths, 'elevations':elevations, 'vColors':vColors, 'lightCoeffs':lightCoeffs, 'shapeParams':shapeParams}

with open(resultDir + 'experiment.pickle', 'wb') as pfile:
    pickle.dump(experimentDic, pfile)

experimentErrorsDic = {'errorsPosePredList':errorsPosePredList, 'errorsLightCoeffsList':errorsLightCoeffsList, 'errorsShapeParamsLis':errorsShapeParamsList, 'errorsShapeVerticesList':errorsShapeVerticesList, 'errorsEnvMapList':errorsEnvMapList, 'errorsLightCoeffsCList':errorsLightCoeffsCList, 'errorsVColorsEList':errorsVColorsEList, 'errorsVColorsCList':errorsVColorsCList, 'errorsSegmentationList':errorsSegmentationList}
#
with open(resultDir + 'experiment_errors.pickle', 'wb') as pfile:
    pickle.dump(experimentErrorsDic, pfile)

meanAbsErrAzsList, meanAbsErrElevsList, meanErrorsLightCoeffsList, meanErrorsShapeParamsList, meanErrorsShapeVerticesList, meanErrorsLightCoeffsCList, meanErrorsEnvMapList, meanErrorsVColorsEList, meanErrorsVColorsCList, meanErrorsSegmentation \
    = computeErrorAverages(np.mean, np.arange(len(rangeTests)), useShapeModel, errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList, errorsSegmentationList)

nearestNeighbours = False
if 'Nearest Neighbours' in set(methodsPred):
    nearestNeighbours = True

plotColors = ['k']
if nearestNeighbours:
    # methodsPred = methodsPred + ["Nearest Neighbours"]
    plotColors = plotColors + ['m']

plotColors = plotColors + ['b']

plotColors = plotColors + ['g']

plotColors = plotColors + ['r']

plotMethodsIndices = [0,2,3,4]
recognitionIdx = 2
robustIdx = 4
#
print("Printing occlusin-likelihood plots!")
# meanLikelihoodArr = [np.array([]), np.array([]), np.array([]), np.array([])]
# occlusions = []
# for occlusionLevel in range(100):
#
#     setUnderOcclusionLevel = testOcclusionsFull * 100 < occlusionLevel
#
#     if np.any(setUnderOcclusionLevel):
#         occlusions = occlusions + [occlusionLevel]
#         testOcclusions = testOcclusionsFull[setUnderOcclusionLevel]
#
#         if likelihoods[0] is not None:
#             meanLikelihoodArr[0] = np.append(meanLikelihoodArr[0], np.mean(likelihoods[0][setUnderOcclusionLevel]))
#         if likelihoods[1] is not None:
#             meanLikelihoodArr[1] = np.append(meanLikelihoodArr[1], np.mean(likelihoods[1][setUnderOcclusionLevel]))
#         if likelihoods[2] is not None:
#             meanLikelihoodArr[2] = np.append(meanLikelihoodArr[2], np.mean(likelihoods[2][setUnderOcclusionLevel]))
#         if likelihoods[3] is not None:
#             meanLikelihoodArr[3] = np.append(meanLikelihoodArr[3], np.mean(likelihoods[3][setUnderOcclusionLevel]))

# saveLikelihoodPlots(resultDir, occlusions, methodsPred, plotColors, plotMethodsIndices, meanLikelihoodArr)
#
# print("Computing means!")
#
meanAbsErrAzsArr = []
meanAbsErrElevsArr = []
meanErrorsLightCoeffsArr = []
meanErrorsEnvMapArr = []
meanErrorsShapeParamsArr = []
meanErrorsShapeVerticesArr = []
meanErrorsLightCoeffsCArr = []
meanErrorsVColorsEArr = []
meanErrorsVColorsCArr = []
meanErrorsSegmentationArr = []
for method_i in range(len(methodsPred)):
    meanAbsErrAzsArr = meanAbsErrAzsArr + [np.array([])]
    meanAbsErrElevsArr = meanAbsErrElevsArr + [np.array([])]
    meanErrorsLightCoeffsArr = meanErrorsLightCoeffsArr + [np.array([])]
    meanErrorsShapeParamsArr = meanErrorsShapeParamsArr + [np.array([])]
    meanErrorsShapeVerticesArr = meanErrorsShapeVerticesArr + [np.array([])]
    meanErrorsLightCoeffsCArr = meanErrorsLightCoeffsCArr + [np.array([])]
    meanErrorsVColorsEArr = meanErrorsVColorsEArr + [np.array([])]
    meanErrorsVColorsCArr = meanErrorsVColorsCArr + [np.array([])]
    meanErrorsEnvMapArr = meanErrorsEnvMapArr + [np.array([])]
    meanErrorsSegmentationArr = meanErrorsSegmentationArr + [np.array([])]

occlusions = []

print("Printing occlusin-error plots!")

for occlusionLevel in range(100):

    setUnderOcclusionLevel = testOcclusionsFull * 100 < occlusionLevel

    if np.any(setUnderOcclusionLevel):
        occlusions = occlusions + [occlusionLevel]
        testOcclusions = testOcclusionsFull[setUnderOcclusionLevel]

        colors = matplotlib.cm.plasma(testOcclusions)

        for method_i in range(len(methodsPred)):

            meanAbsErrAzsArr[method_i] = np.append(meanAbsErrAzsArr[method_i], np.mean(np.abs(errorsPosePredList[method_i][0][setUnderOcclusionLevel])))
            meanAbsErrElevsArr[method_i] = np.append(meanAbsErrElevsArr[method_i], np.mean(np.abs(errorsPosePredList[method_i][1][setUnderOcclusionLevel])))

            meanErrorsLightCoeffsArr[method_i] = np.append(meanErrorsLightCoeffsArr[method_i],np.mean(np.mean(errorsLightCoeffsList[method_i][setUnderOcclusionLevel], axis=1), axis=0))
            meanErrorsLightCoeffsCArr[method_i] = np.append(meanErrorsLightCoeffsCArr[method_i],np.mean(np.mean(errorsLightCoeffsCList[method_i][setUnderOcclusionLevel], axis=1), axis=0))

            if useShapeModel:
                meanErrorsShapeParamsArr[method_i] = np.append(meanErrorsShapeParamsArr[method_i],np.mean(np.mean(errorsShapeParamsList[method_i][setUnderOcclusionLevel], axis=1), axis=0))
                meanErrorsShapeVerticesArr[method_i] = np.append(meanErrorsShapeVerticesArr[method_i], np.mean(errorsShapeVerticesList[method_i][setUnderOcclusionLevel], axis=0))

            meanErrorsEnvMapArr[method_i] = np.append(meanErrorsEnvMapArr[method_i], np.mean(errorsEnvMapList[method_i][setUnderOcclusionLevel]))
            meanErrorsVColorsEArr[method_i] = np.append(meanErrorsVColorsEArr[method_i], np.mean(errorsVColorsEList[method_i][setUnderOcclusionLevel], axis=0))
            meanErrorsVColorsCArr[method_i] = np.append(meanErrorsVColorsCArr[method_i], np.mean(errorsVColorsCList[method_i][setUnderOcclusionLevel], axis=0))

            if errorsSegmentationList[method_i] is not None:
                meanErrorsSegmentationArr[method_i] = np.append(meanErrorsSegmentationArr[method_i], np.mean(errorsSegmentationList[method_i][setUnderOcclusionLevel], axis=0))
            else:
                meanErrorsSegmentationArr[method_i] = None

print("Printing occlusin-error plots - median!")
saveOcclusionPlots(resultDir, 'mean',occlusions, methodsPred, plotColors, plotMethodsIndices, useShapeModel, meanAbsErrAzsArr, meanAbsErrElevsArr, meanErrorsVColorsCArr, meanErrorsVColorsEArr, meanErrorsLightCoeffsArr, meanErrorsShapeParamsArr, meanErrorsShapeVerticesArr, meanErrorsLightCoeffsCArr, meanErrorsEnvMapArr, meanErrorsSegmentationArr)

medianAbsErrAzsArr = []
medianAbsErrElevsArr = []
medianErrorsLightCoeffsArr = []
medianErrorsEnvMapArr = []
medianErrorsShapeParamsArr = []
medianErrorsShapeVerticesArr = []
medianErrorsLightCoeffsCArr = []
medianErrorsVColorsEArr = []
medianErrorsVColorsCArr = []
medianErrorsSegmentationArr = []

for method_i in range(len(methodsPred)):
    medianAbsErrAzsArr = medianAbsErrAzsArr + [np.array([])]
    medianAbsErrElevsArr = medianAbsErrElevsArr + [np.array([])]
    medianErrorsLightCoeffsArr = medianErrorsLightCoeffsArr + [np.array([])]
    medianErrorsShapeParamsArr = medianErrorsShapeParamsArr + [np.array([])]
    medianErrorsShapeVerticesArr = medianErrorsShapeVerticesArr + [np.array([])]
    medianErrorsLightCoeffsCArr = medianErrorsLightCoeffsCArr + [np.array([])]
    medianErrorsVColorsEArr = medianErrorsVColorsEArr + [np.array([])]
    medianErrorsVColorsCArr = medianErrorsVColorsCArr + [np.array([])]
    medianErrorsEnvMapArr = medianErrorsEnvMapArr + [np.array([])]
    medianErrorsSegmentationArr = medianErrorsSegmentationArr + [np.array([])]

occlusions = []

print("Printing occlusin-error plots!")

for occlusionLevel in range(100):

    setUnderOcclusionLevel = testOcclusionsFull * 100 < occlusionLevel

    if np.any(setUnderOcclusionLevel):
        occlusions = occlusions + [occlusionLevel]
        testOcclusions = testOcclusionsFull[setUnderOcclusionLevel]

        colors = matplotlib.cm.plasma(testOcclusions)

        for method_i in range(len(methodsPred)):

            medianAbsErrAzsArr[method_i] = np.append(medianAbsErrAzsArr[method_i], np.median(np.abs(errorsPosePredList[method_i][0][setUnderOcclusionLevel])))
            medianAbsErrElevsArr[method_i] = np.append(medianAbsErrElevsArr[method_i], np.median(np.abs(errorsPosePredList[method_i][1][setUnderOcclusionLevel])))

            medianErrorsLightCoeffsArr[method_i] = np.append(medianErrorsLightCoeffsArr[method_i],np.median(np.median(errorsLightCoeffsList[method_i][setUnderOcclusionLevel], axis=1), axis=0))
            medianErrorsLightCoeffsCArr[method_i] = np.append(medianErrorsLightCoeffsCArr[method_i],np.median(np.median(errorsLightCoeffsCList[method_i][setUnderOcclusionLevel], axis=1), axis=0))

            if useShapeModel:
                medianErrorsShapeParamsArr[method_i] = np.append(medianErrorsShapeParamsArr[method_i],np.median(np.median(errorsShapeParamsList[method_i][setUnderOcclusionLevel], axis=1), axis=0))
                medianErrorsShapeVerticesArr[method_i] = np.append(medianErrorsShapeVerticesArr[method_i], np.median(errorsShapeVerticesList[method_i][setUnderOcclusionLevel], axis=0))

            medianErrorsEnvMapArr[method_i] = np.append(medianErrorsEnvMapArr[method_i], np.median(errorsEnvMapList[method_i][setUnderOcclusionLevel]))
            medianErrorsVColorsEArr[method_i] = np.append(medianErrorsVColorsEArr[method_i], np.median(errorsVColorsEList[method_i][setUnderOcclusionLevel], axis=0))
            medianErrorsVColorsCArr[method_i] = np.append(medianErrorsVColorsCArr[method_i], np.median(errorsVColorsCList[method_i][setUnderOcclusionLevel], axis=0))

            if errorsSegmentationList[method_i] is not None:
                medianErrorsSegmentationArr[method_i] = np.append(medianErrorsSegmentationArr[method_i], np.median(errorsSegmentationList[method_i][setUnderOcclusionLevel], axis=0))
            else:
                medianErrorsSegmentationArr[method_i] = None

saveOcclusionPlots(resultDir, 'median', occlusions,methodsPred, plotColors, plotMethodsIndices, useShapeModel, medianAbsErrAzsArr, medianAbsErrElevsArr, medianErrorsVColorsCArr, medianErrorsVColorsEArr, medianErrorsLightCoeffsArr, medianErrorsShapeParamsArr, medianErrorsShapeVerticesArr, medianErrorsLightCoeffsCArr, medianErrorsEnvMapArr, medianErrorsSegmentationArr)


for occlusionLevel in [25,75,100]:

    resultDirOcclusion = 'results/' + testPrefix + '/occlusion' + str(occlusionLevel) + '/'
    if not os.path.exists(resultDirOcclusion):
        os.makedirs(resultDirOcclusion)

    setUnderOcclusionLevel = testOcclusionsFull * 100 < occlusionLevel
    testOcclusions = testOcclusionsFull[setUnderOcclusionLevel]

    errorsPosePred = [errorsPosePredList[recognitionIdx][0][setUnderOcclusionLevel], errorsPosePredList[recognitionIdx][1][setUnderOcclusionLevel]]
    errorsLightCoeffs = errorsLightCoeffsList[recognitionIdx][setUnderOcclusionLevel]
    errorsShapeParams = errorsShapeParamsList[recognitionIdx][setUnderOcclusionLevel]
    errorsShapeVertices= errorsShapeVerticesList[recognitionIdx][setUnderOcclusionLevel]
    errorsEnvMap= errorsEnvMapList[recognitionIdx][setUnderOcclusionLevel]
    errorsLightCoeffsC= errorsLightCoeffsCList[recognitionIdx][setUnderOcclusionLevel]
    errorsVColorsE= errorsVColorsEList[recognitionIdx][setUnderOcclusionLevel]
    errorsVColorsC= errorsVColorsCList[recognitionIdx][setUnderOcclusionLevel]
    errorsSegmentation = errorsSegmentationList[recognitionIdx][setUnderOcclusionLevel]

    errorsPoseFitted =  [errorsPosePredList[robustIdx][0][setUnderOcclusionLevel], errorsPosePredList[robustIdx][1][setUnderOcclusionLevel]]
    errorsFittedLightCoeffs = errorsLightCoeffsList[robustIdx][setUnderOcclusionLevel]
    errorsFittedShapeParams = errorsShapeParamsList[robustIdx][setUnderOcclusionLevel]
    errorsFittedShapeVertices= errorsShapeVerticesList[robustIdx][setUnderOcclusionLevel]
    errorsFittedEnvMap= errorsEnvMapList[robustIdx][setUnderOcclusionLevel]
    errorsFittedLightCoeffsC= errorsLightCoeffsCList[robustIdx][setUnderOcclusionLevel]
    errorsFittedVColorsE= errorsVColorsEList[robustIdx][setUnderOcclusionLevel]
    errorsFittedVColorsC= errorsVColorsCList[robustIdx][setUnderOcclusionLevel]
    errorsFittedSegmentation = errorsSegmentationList[recognitionIdx][setUnderOcclusionLevel]

    # azTop10 = np.argsort(errorsPoseFitted[0] - errorsPosePred[0])[:10]
    # elTop10 = np.argsort(errorsPoseFitted[1] - errorsPosePred[1])[:10]
    # vColorTop10 = np.argsort(errorsFittedVColorsC - errorsVColorsC)[:10]
    # shTop10 = np.argsort(errorsFittedEnvMap - errorsEnvMap)[:10]
    # shapeTop10 = np.argsort(errorsFittedShapeVertices - errorsShapeVertices)[:10]
    # segmentationTop10 = np.argsort(errorsFittedSegmentation - errorsSegmentation)[:10]
    #
    # shapeWorst = np.argsort(errorsFittedShapeVertices)[::-1]
    # chVColors[:] = np.array([0.5,0.5,0.5])
    #
    # testi = 0
    # for testCase in shapeWorst:
    #     shapeVals = shapeParams[4][testCase]
    #     chShapeParams[:] = shapeVals
    #     shapeNorm = np.linalg.norm(chVertices.r - chVerticesMean)
    #     indiv = np.any(np.abs(chShapeParams.r) >= 2.48)
    #     plt.imsave('tmp/shape/render' + str(testi) + '_' + str(indiv) +  '_' + str(shapeNorm) + '.png', renderer.r)
    #     testi = testi + 1
    # chShapeParams[:] = np.zeros([10])
    #
    # vColorsWorst = np.argsort(errorsFittedVColorsC)[::-1]
    # testi = 0
    # for testCase in vColorsWorst:
    #     chVColors[:] = vColors[4][testCase]
    #     plt.imsave('tmp/vcolors/render' + str(testi) +  '.png', renderer.r)
    #     testi = testi + 1
    # chVColors[:] = np.array([0.5,0.5,0.5])
    #
    # errorsFittedEnvMapWorst = np.argsort(errorsFittedEnvMap)[::-1]
    # testi = 0
    # for testCase in errorsFittedEnvMapWorst:
    #     chLightSHCoeffs[:] = lightCoeffs[4][testCase]
    #     plt.imsave('tmp/illumination/render' + str(testi) +  '.png', renderer.r)
    #     testi = testi + 1

    # with open(resultDirOcclusion + 'badSamples.txt', "w") as samples:
    #     samples.write('Top 10 worse change from Recognition to Fit.\n')
    #     samples.write('Azimuth\n')
    #     samples.write(str(azTop10) + '\n')
    #     samples.write('Elevation\n')
    #     samples.write(str(elTop10) + '\n')
    #     samples.write('VColor\n')
    #     samples.write(str(vColorTop10) + '\n')
    #     samples.write('SH env map\n')
    #     samples.write(str(shTop10) + '\n')
    #     samples.write('Shape\n')
    #     samples.write(str(shapeTop10) + '\n')
    #     samples.write('Segmentation\n')
    #     samples.write(str(segmentationTop10) + '\n')

    saveScatterPlots(resultDirOcclusion, testOcclusions, useShapeModel, errorsPosePred, errorsPoseFitted,errorsLightCoeffsC,errorsFittedLightCoeffsC,errorsEnvMap,errorsFittedEnvMap,errorsLightCoeffs,errorsFittedLightCoeffs,errorsShapeParams,errorsFittedShapeParams,errorsShapeVertices,errorsFittedShapeVertices,errorsVColorsE,errorsFittedVColorsE,errorsVColorsC,errorsFittedVColorsC)

    # saveLikelihoodScatter(resultDirOcclusion, setUnderOcclusionLevel, testOcclusions,  likelihoods)

    # if len(stdevsFull) > 0:
    #     stdevs = stdevsFull[setUnderOcclusionLevel]

    colors = matplotlib.cm.plasma(testOcclusions)

    meanAbsErrAzsList, meanAbsErrElevsList, meanErrorsLightCoeffsList, meanErrorsShapeParamsList, meanErrorsShapeVerticesList, meanErrorsLightCoeffsCList, meanErrorsEnvMapList, meanErrorsVColorsEList, meanErrorsVColorsCList, meanErrorsSegmentationList \
        = computeErrorAverages(np.mean, setUnderOcclusionLevel, useShapeModel, errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList, errorsSegmentationList)

    medianAbsErrAzsList, medianAbsErrElevsList, medianErrorsLightCoeffsList, medianErrorsShapeParamsList, medianErrorsShapeVerticesList, medianErrorsLightCoeffsCList, medianErrorsEnvMapList, medianErrorsVColorsEList, medianErrorsVColorsCList, medianErrorsSegmentationList  \
        = computeErrorAverages(np.median, setUnderOcclusionLevel, useShapeModel, errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList, errorsSegmentationList)
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
             ["Shape Vertices"] + meanErrorsShapeVerticesList,
             ["Segmentation"] + meanErrorsSegmentationList
             ]
    performanceTable = tabulate.tabulate(table, headers=headers, tablefmt="latex", floatfmt=".3f")
    with open(resultDirOcclusion + 'performance.tex', 'w') as expfile:
        expfile.write(performanceTable)


    table = [["Azimuth"] +  medianAbsErrAzsList,
             ["Elevation"] + medianAbsErrElevsList,
             ["VColor C"] + medianErrorsVColorsCList,
             ["SH Light"] + medianErrorsLightCoeffsList,
             ["SH Light C"] + medianErrorsLightCoeffsCList,
             ["SH Env Map"] + medianErrorsEnvMapList,
             ["Shape Params"] + medianErrorsShapeParamsList,
             ["Shape Vertices"] + medianErrorsShapeVerticesList,
             ["Segmentation"] + medianErrorsSegmentationList
             ]
    performanceTable = tabulate.tabulate(table, headers=headers, tablefmt="latex", floatfmt=".3f")
    with open(resultDirOcclusion + 'median-performance.tex', 'w') as expfile:
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



