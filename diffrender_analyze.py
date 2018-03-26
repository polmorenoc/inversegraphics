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
    vmod, vnmod, _ = transformObject(vmod, vnmod, chScale, chObjAz, ch.Ch([0]), ch.Ch([0]), np.array([0, 0, 0]))
    renderer = createRendererTarget(glMode, chAz, chEl, chDist, centermod, vmod, vcmod, fmod_list, vnmod, light_color, chComponent, chVColors, 0, chDisplacement, width,height, uvmod, haveTexturesmod_list, texturesmod_list, frustum, win )
    renderer.overdraw = True
    renderer.nsamples = 8
    renderer.msaa = False
    renderer.initGL()
    renderer.initGLTexture()
    # renderer.initGL_AnalyticRenderer()
    renderer.imageGT = None
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
    smVertices, smNormals, _ = transformObject(smVertices, smNormals, chScale, chObjAz, ch.Ch([0]), ch.Ch([0]), np.array([0, 0, 0]))
    renderer = createRendererTarget(glMode, chAz, chEl, chDist, smCenter, [smVertices], [smVColors], [smFaces], [smNormals], light_color, chComponent, chVColors, 0, chDisplacement, width,height, [smUVs], [smHaveTextures], [smTexturesList], frustum, win )
    renderer.overdraw = True
    renderer.nsamples = 8
    renderer.msaa = False
    renderer.initGL()
    renderer.initGLTexture()
    # renderer.initGL_AnalyticRenderer()
    renderer.imageGT = None
    renderer.r
    chShapeParams[:] = np.zeros([latentDim])
    chVerticesMean = chVertices.r.copy()
else:
    renderer = renderer_teapots[0]


##GT Renderer
chObjAzGT = ch.Ch([0])
chAzGT = ch.Ch([0])
chAzRelGT = chAzGT - chObjAzGT
chElGT = ch.Ch(chEl.r[0])
chDistGT = ch.Ch([camDistance])
phiOffset = ch.Ch([0])
totalOffset = phiOffset + chObjAzGT
chVColorsGT = ch.Ch([0.8,0.8,0.8])
chAmbientIntensityGT = ch.Ch([0.025])
clampedCosCoeffs = clampedCosineCoefficients()

SHFilename = 'data/LightSHCoefficients.pickle'
with open(SHFilename, 'rb') as pfile:
    envMapDic = pickle.load(pfile)
hdritems = list(envMapDic.items())

envMapCoeffs = ch.Ch(list(envMapDic.items())[0][1][1])

envMapCoeffsRotated = ch.Ch(np.dot(light_probes.chSphericalHarmonicsZRotation(totalOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]])
envMapCoeffsRotatedRel = ch.Ch(np.dot(light_probes.chSphericalHarmonicsZRotation(phiOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]])

shCoeffsRGB = envMapCoeffsRotated
shCoeffsRGBRel = envMapCoeffsRotatedRel
chShCoeffs = 0.3*shCoeffsRGB[:,0] + 0.59*shCoeffsRGB[:,1] + 0.11*shCoeffsRGB[:,2]
chShCoeffsRel = 0.3*shCoeffsRGBRel[:,0] + 0.59*shCoeffsRGBRel[:,1] + 0.11*shCoeffsRGBRel[:,2]

chAmbientSHGT = chShCoeffs.ravel() * chAmbientIntensityGT * clampedCosCoeffs
chAmbientSHGTRel = chShCoeffsRel.ravel() * chAmbientIntensityGT * clampedCosCoeffs
chComponentGT = chAmbientSHGT
chComponentGTRel = chAmbientSHGTRel

# shapeParams = np.random.randn(latentDim)
if useShapeModel:
    shapeParams = np.random.randn(latentDim)
    chShapeParamsGT = ch.Ch(shapeParams)

    chVerticesGT = shape_model.VerticesModel(chShapeParams =chShapeParamsGT,meshLinearTransform=meshLinearTransform,W=W,b=b)
    chVerticesGT.init()

    chVerticesGT = ch.dot(geometry.RotateZ(-np.pi/2)[0:3,0:3],chVerticesGT.T).T
    # chNormalsGT = shape_model.chShapeParamsToNormals(teapotModel['N'], landmarks, teapotModel['linT'])
    # chNormalsGT = shape_model.shapeParamsToNormals(shapeParams, teapotModel)
    chNormalsGT = shape_model.chGetNormals(chVerticesGT, faces)

    smNormalsGT = [chNormalsGT]
    smFacesGT = [[faces]]
    smVColorsGT = [chVColorsGT*np.ones(chVerticesGT.shape)]
    smUVsGT = [ch.Ch(np.zeros([chVerticesGT.shape[0],2]))]
    smHaveTexturesGT = [[False]]
    smTexturesListGT = [[None]]

    smCenterGT = ch.mean(chVerticesGT, axis=0)

    chVerticesGT = chVerticesGT - ch.mean(chVerticesGT, axis=0)
    minZ = ch.min(chVerticesGT[:,2])

    chMinZ = ch.min(chVerticesGT[:,2])

    zeroZVerts = chVerticesGT[:,2]- chMinZ
    chVerticesGT = ch.hstack([chVerticesGT[:,0:2] , zeroZVerts.reshape([-1,1])])


    chVerticesGT = chVerticesGT*0.09
    smCenterGT = ch.array([0,0,0.1])
    smVerticesGT = [chVerticesGT]

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

post = generative_models.layerPosteriorsRobustCh(rendererGT, renderer, np.array([]), 'MASK', globalPrior, variances)[0]

# modelLogLikelihoodRobustRegionCh = -ch.sum(generative_models.LogRobustModelRegion(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances))/numPixels
#
# pixelLikelihoodRobustRegionCh = generative_models.LogRobustModelRegion(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances)

# models = [negLikModel, negLikModelRobust, hogError]
models = [negLikModel, negLikModelRobust]
pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh]
modelsDescr = ["Gaussian Model", "Outlier model" ]

model = 1
pixelErrorFun = pixelModels[model]
errorFun = models[model]


############
# Experiments
############

seed = 1
np.random.seed(seed)

gtPrefix = 'train4_occlusion_shapemodel_photorealistic_10K_test100-1100'
# gtPrefix = 'train4_occlusion_shapemodel'

gtDir = 'groundtruth/' + gtPrefix + '/'
featuresDir = gtDir

experimentPrefix = 'train4_occlusion_shapemodel_10k'
experimentDir = 'experiments/' + experimentPrefix + '/'

ignore = []
if os.path.isfile(gtDir + 'ignore.npy'):
    ignore = np.load(gtDir + 'ignore.npy')

groundTruthFilename = gtDir + 'groundTruth.h5'
gtDataFile = h5py.File(groundTruthFilename, 'r')

rangeTests = np.arange(100,1100)
idsInRange = np.arange(len(rangeTests))
testSet = np.load(experimentDir + 'test.npy')[rangeTests][idsInRange]

shapeGT = gtDataFile[gtPrefix].shape

boolTestSet = np.array([np.any(num == testSet) for num in gtDataFile[gtPrefix]['trainIds']])
dataIds = gtDataFile[gtPrefix][boolTestSet]['trainIds']

dataIdsTestIndices = np.array([np.where(dataIds==num)[0][0] for num in testSet])

groundTruth = gtDataFile[gtPrefix][boolTestSet][dataIdsTestIndices]

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
dataEnvMapPhiOffsets = groundTruth['trainEnvMapPhiOffsets']
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
syntheticGroundtruth = False

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

##### Load Results data:

# testPrefix = 'train4_occlusion_shapemodel_10k_ECCVNEW-JOINED-ALL2018'
testPrefix = 'train4_occlusion_shapemodel_10k_ECCV-PHOTOREALISTIC-JOINT2018'
resultDir = 'results/' + testPrefix + '/'

with open(resultDir + 'experiment.pickle', 'rb') as pfile:
    experimentDic = pickle.load(pfile)
testSet = experimentDic['testSet']
methodsPred = experimentDic['methodsPred']
testOcclusions = experimentDic['testOcclusions']
# testOcclusions = testOcclusions[dataIdsTestIndices] #Change depending on the source of GT.

# import ipdb; ipdb.set_trace()

testPrefixBase = experimentDic[ 'testPrefixBase']
parameterRecognitionModels = experimentDic[ 'parameterRecognitionModels']
azimuths = experimentDic[ 'azimuths']

azimuths = [azimuths[method][dataIdsTestIndices] if  azimuths[method] is not None else None for method in range(len(azimuths))]

elevations = experimentDic[ 'elevations']
elevations = [elevations[method][dataIdsTestIndices] if  elevations[method] is not None else None for method in range(len(elevations))]

vColors = experimentDic[ 'vColors']
vColors = [vColors[method][dataIdsTestIndices] if  vColors[method] is not None else None for method in range(len(vColors))]

lightCoeffs = experimentDic[ 'lightCoeffs']
lightCoeffs = [lightCoeffs[method][dataIdsTestIndices] if  lightCoeffs[method] is not None else None for method in range(len(lightCoeffs))]

likelihoods = []
# likelihoods = experimentDic['likelihoods']
# likelihoods = [likelihoods[method][dataIdsTestIndices] if  likelihoods[method] is not None else None for method in range(len(likelihoods))]

shapeParams = experimentDic['shapeParams']
shapeParams = [shapeParams[method][dataIdsTestIndices] if  shapeParams[method] is not None else None for method in range(len(shapeParams))]

if os.path.isfile(resultDir + 'segmentations.pickle'):
    with open(resultDir + 'segmentations.pickle', 'rb') as pfile:
        segmentationsDic = pickle.load(pfile)
    segmentations = segmentationsDic['segmentations']
    segmentations = [segmentations[method][dataIdsTestIndices] if  segmentations[method] is not None else None for method in range(len(segmentations))]
else:
    segmentations = [None]*len(methodsPred)

testOcclusionsFull = testOcclusions.copy()

loadMask = True
if loadMask:
    masksGT = loadMasks(gtDir + '/masks_occlusion/', dataIds)

#
# with open(resultDir + 'approxProjections.pickle', 'rb') as pfile:
#     approxProjectionsDic = pickle.load(pfile)
#
# approxProjections = approxProjectionsDic['approxProjections']
# approxProjectionsGT = approxProjectionsDic['approxProjectionsGT']
#
# envMapTexture = np.zeros([180,360,3])
# approxProjections = []
# for method in range(len(methodsPred)):
#     print("Approx projection on method " + str(method))
#     approxProjectionsFittedList = []
#     for test_i in range(len(testSet)):
#         pEnvMap = SHProjection(envMapTexture, np.concatenate([lightCoeffs[4][test_i][:,None], lightCoeffs[4][test_i][:,None], lightCoeffs[4][test_i][:,None]], axis=1))
#         approxProjection = np.sum(pEnvMap, axis=(2,3))
#         approxProjectionsFittedList = approxProjectionsFittedList + [approxProjection[None,:]]
#         approxProjections = approxProjections + [np.vstack(approxProjectionsFittedList)]
#
#
# approxProjectionsGTList = []
# for test_i in range(len(testSet)):
#     pEnvMap = SHProjection(envMapTexture, np.concatenate([testLightCoefficientsGTRel[test_i][:,None], testLightCoefficientsGTRel[test_i][:,None], testLightCoefficientsGTRel[test_i][:,None]], axis=1))
#     approxProjectionGT = np.sum(pEnvMap, axis=(2,3))
#     approxProjectionsGTList = approxProjectionsGTList + [approxProjectionGT[None,:]]
# approxProjectionsGT = np.vstack(approxProjectionsGTList)

approxProjections = None
approxProjectionsGT = None

# #
# # ##### Load Results data:
# #
#
# testPrefix2 = 'train4_occlusion_shapemodel_10k_ECCV-PHOTREALISTIC-MEANBASELINE-2530-17944_predict_1000samples__method1errorFun1_std0.05_shapePen0'
# resultDir2 = 'results/' + testPrefix2 + '/'
# with open(resultDir2 + 'experiment.pickle', 'rb') as pfile:
#     experimentDic2 = pickle.load(pfile)
# testSet2 = experimentDic2['testSet']
# methodsPred2 = experimentDic2['methodsPred']
# testOcclusions2 = experimentDic2[ 'testOcclusions']
# testPrefixBase2 = experimentDic2[ 'testPrefixBase']
# parameterRecognitionModels2 = experimentDic2[ 'parameterRecognitionModels']
# azimuths2 = experimentDic2['azimuths']
# elevations2 = experimentDic2[ 'elevations']
# vColors2 = experimentDic2[ 'vColors']
# lightCoeffs2 = experimentDic2[ 'lightCoeffs']
# likelihoods2 = experimentDic2['likelihoods']
# shapeParams2 = experimentDic2['shapeParams']
# if os.path.isfile(resultDir2 + 'segmentations.pickle'):
#     with open(resultDir2 + 'segmentations.pickle', 'rb') as pfile:
#         segmentationsDic2 = pickle.load(pfile)
#     segmentations2 = segmentationsDic2['segmentations']
# else:
#     segmentations2 = [None]*len(methodsPred2)
#
# testOcclusionsFull2 = testOcclusions2.copy()
#
#
# testPrefix3 = 'train4_occlusion_shapemodel_10k_ECCV-SYNTH-FIX-2530-17944_optimize_1000samples__method1errorFun1_std0.03_shapePen0'
# resultDir3 = 'results/' + testPrefix3 + '/'
# with open(resultDir3 + 'experiment.pickle', 'rb') as pfile:
#     experimentDic3 = pickle.load(pfile)
# testSet3 = experimentDic3['testSet']
# methodsPred3 = experimentDic3['methodsPred']
# testOcclusions3 = experimentDic3[ 'testOcclusions']
# testPrefixBase3 = experimentDic3[ 'testPrefixBase']
# parameterRecognitionModels3 = experimentDic3[ 'parameterRecognitionModels']
# azimuths3 = experimentDic3['azimuths']
# elevations3 = experimentDic3[ 'elevations']
# vColors3 = experimentDic3[ 'vColors']
# lightCoeffs3 = experimentDic3[ 'lightCoeffs']
# likelihoods3 = experimentDic3['likelihoods']
# shapeParams3 = experimentDic3['shapeParams']
# if os.path.isfile(resultDir3 + 'segmentations.pickle'):
#     with open(resultDir3 + 'segmentations.pickle', 'rb') as pfile:
#         segmentationsDic3 = pickle.load(pfile)
#     segmentations3 = segmentationsDic3['segmentations']
# else:
#     segmentations3 = [None]*len(methodsPred3)
#
# testOcclusionsFull3 = testOcclusions3.copy()
#
# # with open(resultDir2 + 'approxProjections.pickle', 'rb') as pfile:
# #     approxProjectionsDic2 = pickle.load(pfile)
# #
# # approxProjections2 = approxProjectionsDic2['approxProjections']
# # approxProjectionsGT2 = approxProjectionsDic2['approxProjectionsGT']
#
# approxProjections2 = None
# approxProjectionsGT2 = None

#
# testPrefix3 = 'train4_occlusion_shapemodel_10k_ECCV-SYNTHETIC-GAUSSIAN11664-17944_optimize_300samples__method1errorFun1_std0.05_shapePen0'
# resultDir3 = 'results/' + testPrefix3 + '/'
# with open(resultDir3 + 'experiment.pickle', 'rb') as pfile:
#     experimentDic3 = pickle.load(pfile)
# testSet3 = experimentDic3['testSet']
# methodsPred3 = experimentDic3['methodsPred']
# testOcclusions3 = experimentDic3[ 'testOcclusions']
# testPrefixBase3 = experimentDic3[ 'testPrefixBase']
# parameterRecognitionModels3 = experimentDic3[ 'parameterRecognitionModels']
# azimuths3 = experimentDic3['azimuths']
# elevations3 = experimentDic3[ 'elevations']
# vColors3 = experimentDic3[ 'vColors']
# lightCoeffs3 = experimentDic3[ 'lightCoeffs']
# likelihoods3 = experimentDic3['likelihoods']
# shapeParams3 = experimentDic3['shapeParams']
# if os.path.isfile(resultDir3 + 'segmentations.pickle'):
#     with open(resultDir3 + 'segmentations.pickle', 'rb') as pfile:
#         segmentationsDic3 = pickle.load(pfile)
#     segmentations3 = segmentationsDic3['segmentations']
# else:
#     segmentations3 = [None]*len(methodsPred3)
#
# testOcclusionsFull3 = testOcclusions3.copy()
#
# # with open(resultDir3 + 'approxProjections.pickle', 'rb') as pfile:
# #     approxProjectionsDic3 = pickle.load(pfile)
# #
# # approxProjections3 = approxProjectionsDic3['approxProjections']
# # approxProjectionsGT3 = approxProjectionsDic3['approxProjectionsGT']
#
# approxProjections3 = None
# approxProjectionsGT3 = None
#
# ipdb.set_trace()
#
# range1 = np.arange(len(azimuths2[3]))
# range2 = np.arange(-1000 + len(azimuths2[3]) + len(azimuths3[3]),len(azimuths3[3]))
#
# testSet, parameterRecognitionModels, testPrefixBase, methodsPred, testOcclusions, azimuths, elevations, vColors, lightCoeffs, shapeParams, likelihoods, segmentations, approxProjections, approxProjectionsGT = \
#     joinExperiments(range1, range2, testSet2,methodsPred2,testOcclusions2,testPrefixBase2,parameterRecognitionModels2,azimuths2,elevations2,vColors2,lightCoeffs2,likelihoods2,shapeParams2,segmentations2, approxProjections2, approxProjectionsGT3, testSet3,methodsPred3,testOcclusions3,testPrefixBase3,parameterRecognitionModels3,azimuths3,elevations3,vColors3,lightCoeffs3,likelihoods3,shapeParams3,segmentations3, approxProjections3, approxProjectionsGT3)
#
# testSet = testSet2
# testOcclusions = testOcclusions2
# testPrefix = 'train4_occlusion_shapemodel_10k_ECCV-SYNTHETIC-GAUSSIAN2530-17944_optimize_1000samples__method1errorFun1_std0.05_shapePen0-FINAL'
# resultDir = 'results/' + testPrefix + '/'
#
# experimentDic = {'testSet':testSet, 'methodsPred':methodsPred, 'testOcclusions':testOcclusions, 'likelihoods':likelihoods, 'testPrefixBase':testPrefixBase, 'parameterRecognitionModels':parameterRecognitionModels, 'azimuths':azimuths, 'elevations':elevations, 'vColors':vColors, 'lightCoeffs':lightCoeffs, 'shapeParams':shapeParams}
#
# with open(resultDir + 'experiment.pickle', 'wb') as pfile:
#     pickle.dump(experimentDic, pfile)
# #
# #


#
# methodsPred = methodsPred + ['Gaussian (OpenGL)']
# azimuths = azimuths + [azimuths2[3]]
# elevations = elevations + [elevations2[3]]
# vColors = vColors + [vColors2[3]]
# lightCoeffs = lightCoeffs + [lightCoeffs2[3]]
# likelihoods = likelihoods + [likelihoods2[3]]
# shapeParams = shapeParams + [shapeParams2[3]]
# segmentations = segmentations + [segmentations2[3]]

#

# methodsPred[0] = methodsPred2[0]
# azimuths[0] = azimuths2[0]
# elevations[0] = elevations2[0]
# vColors[0] = vColors2[0]
# lightCoeffs[0] = lightCoeffs2[0]
# likelihoods[0] = likelihoods2[0]
# shapeParams[0] = shapeParams2[0]
# segmentations[0] = segmentations2[0]

rendererGT = None

chPointLightIntensityGT = ch.Ch([1])
light_colorGT = ch.ones(3)*chPointLightIntensityGT
chDisplacementGT = ch.Ch([0.0,0.0,0.0])
chScaleGT = ch.Ch([1, 1.,1.])

replaceableScenesFile = '../databaseFull/fields/scene_replaceables_backup.txt'

# likelihoods = [np.array([]), np.array([])]

from OpenGL import contextdata

# #
# segmentations[2] = np.zeros([len(testSet), 150,150])
# segmentations[3] = np.zeros([len(testSet), 150,150])
# segmentations[4] = np.zeros([len(testSet), 150,150])
# segmentations[5] = np.zeros([len(testSet), 150,150])
#
#
# for test_i in range(len(testSet)):
#
#     # sceneNumber = dataScenes[test_i]
#     # sceneIdx = scene_io_utils.getSceneIdx(sceneNumber, replaceableScenesFile)
#     # sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndicesScene, targetPositions = scene_io_utils.getSceneInformation(sceneIdx, replaceableScenesFile)
#     # # sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndices, targetPositions = scene_io_utils.getSceneInformation(sceneIdx, replaceableScenesFile)
#     #
#     # targetIndex = dataTargetIndices[test_i]
#     # sceneDicFile = 'data/scene' + str(sceneNumber) + '.pickle'
#     # v, f_list, vc, vn, uv, haveTextures_list, textures_list = scene_io_utils.loadSavedScene(sceneDicFile, True)
#     #
#     # removeObjectData(len(v) -1 - targetIndex, v, f_list, vc, vn, uv, haveTextures_list, textures_list)
#     #
#     # addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list,  smVerticesGT, smFacesGT, smVColorsGT, smNormalsGT, smUVsGT, smHaveTexturesGT, smTexturesListGT)
#     #
#     # if rendererGT is not None:
#     #     rendererGT.makeCurrentContext()
#     #     rendererGT.clear()
#     #     contextdata.cleanupContext(contextdata.getContext())
#     #     if glMode == 'glfw':
#     #         glfw.destroy_window(rendererGT.win)
#     #     del rendererGT
#     #
#     # targetPosition = targetPositions[np.where(targetIndex==np.array(targetIndicesScene))[0]]
#     #
#     # rendererGT = createRendererGT(glMode, chAzGT, chObjAzGT, chElGT, chDistGT, center, v, vc, f_list, vn, light_colorGT, chComponentGT, chVColorsGT, targetPosition.copy(), chDisplacementGT, chScaleGT, width,height, uv, haveTextures_list, textures_list, frustum, None )
#     #
#     # for hdrFile, hdrValues in hdritems:
#     #     hdridx = hdrValues[0]
#     #     envMapCoeffs = hdrValues[1]
#     #     if hdridx == dataEnvMaps[test_i]:
#     #         break
#     # envMapFilename = hdrFile
#     #
#     # phiOffset[:] = dataEnvMapPhiOffsets[test_i]
#     # chObjAzGT[:] = testObjAzsGT[test_i]
#     # chAzGT[:] = testAzsGT[test_i]
#     # chElGT[:] = testElevsGT[test_i]
#     # chVColorsGT[:] = testVColorGT[test_i]
#     # envMapCoeffsRotated[:] = np.dot(light_probes.chSphericalHarmonicsZRotation(totalOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]
#     # envMapCoeffsRotatedRel[:] = np.dot(light_probes.chSphericalHarmonicsZRotation(phiOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]
#     # chShapeParamsGT[:] =  dataShapeModelCoeffsGT[test_i]
#
#     im = images[test_i]
#     rendererGT = srgb2lin(im.copy())
#
#     chLightSHCoeffs[:] = testLightCoefficientsGTRel[test_i]
#     chObjAz[:] = 0
#     chAz[:] = testAzsRel[test_i]
#     chEl[:] = testElevsGT[test_i]
#     chVColors[:] = testVColorGT[test_i]
#     chShapeParams[:] =  testShapeParamsGT[test_i]
#
#     stds[:] = 0.05
#
#     negLikModelRobust = -ch.sum(generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances))/numPixels
#
#     # likelihoods[0] = np.append(likelihoods[0], negLikModelRobust.r)
#
#     chLightSHCoeffs[:] = lightCoeffs[3][idsInRange[test_i]]
#     chObjAz[:] = 0
#     chAz[:] = azimuths[3][idsInRange[test_i]]
#     chEl[:] = elevations[3][idsInRange[test_i]]
#     chVColors[:] = vColors[3][idsInRange[test_i]]
#     chShapeParams[:] =  shapeParams[3][idsInRange[test_i]]
#
#     # likelihoods[1] = np.append(likelihoods[1], negLikModelRobust.r)
#
#     #masksGT
#     #render[~mask*vis_im] = np.concatenate([np.ones([1000,1000])[:,:,None],  np.zeros([1000,1000])[:,:,None],np.zeros([1000,1000])[:,:,None]], axis=2)[~mask*vis_im]
#
#     vis_im = np.array(renderer.indices_image==1).copy().astype(np.bool)
#     post = generative_models.layerPosteriorsRobustCh(rendererGT, renderer, vis_im, 'MASK', globalPrior, variances)[0].r>0.5
#     render = ~post.copy()
#     mask = masksGT[test_i]
#     render[~vis_im] = 1
#
#     segmentations[3][test_i] = post
#
#
#     chLightSHCoeffs[:] = lightCoeffs[2][idsInRange[test_i]]
#     chObjAz[:] = 0
#     chAz[:] = azimuths[2][idsInRange[test_i]]
#     chEl[:] = elevations[2][idsInRange[test_i]]
#     chVColors[:] = vColors[2][idsInRange[test_i]]
#     chShapeParams[:] = shapeParams[2][idsInRange[test_i]]
#
#     # likelihoods[2] = np.append(likelihoods[2], negLikModelRobust.r)
#
#     # masksGT
#     # render[~mask*vis_im] = np.concatenate([np.ones([1000,1000])[:,:,None],  np.zeros([1000,1000])[:,:,None],np.zeros([1000,1000])[:,:,None]], axis=2)[~mask*vis_im]
#
#     post = np.array(renderer.indices_image == 1).copy().astype(np.bool)
#
#     segmentations[2][test_i] = post
#
#
#     chLightSHCoeffs[:] = lightCoeffs[5][idsInRange[test_i]]
#     chObjAz[:] = 0
#     chAz[:] = azimuths[5][idsInRange[test_i]]
#     chEl[:] = elevations[5][idsInRange[test_i]]
#     chVColors[:] = vColors[5][idsInRange[test_i]]
#     chShapeParams[:] = shapeParams[5][idsInRange[test_i]]
#
#     # likelihoods[1] = np.append(likelihoods[1], negLikModelRobust.r)
#
#     # masksGT
#     # render[~mask*vis_im] = np.concatenate([np.ones([1000,1000])[:,:,None],  np.zeros([1000,1000])[:,:,None],np.zeros([1000,1000])[:,:,None]], axis=2)[~mask*vis_im]
#
#     post = np.array(renderer.indices_image == 1).copy().astype(np.bool)
#
#     segmentations[5][test_i] = post
# #
#     # chLightSHCoeffs[:] = lightCoeffs[4][idsInRange[test_i]]
#     # chObjAz[:] = 0
#     # chAz[:] = azimuths[4][idsInRange[test_i]]
#     # chEl[:] = elevations[4][idsInRange[test_i]]
#     # chVColors[:] = vColors[4][idsInRange[test_i]]
#     # chShapeParams[:] = shapeParams[4][idsInRange[test_i]]
#
#     # likelihoods[1] = np.append(likelihoods[1], negLikModelRobust.r)
#
#     # masksGT
#     # render[~mask*vis_im] = np.concatenate([np.ones([1000,1000])[:,:,None],  np.zeros([1000,1000])[:,:,None],np.zeros([1000,1000])[:,:,None]], axis=2)[~mask*vis_im]
#
#     # vis_im = np.array(renderer.indices_image == 1).copy().astype(np.bool)
#     # post = generative_models.layerPosteriorsRobustCh(rendererGT, renderer, vis_im, 'MASK', globalPrior, variances)[0].r > 0.5
#     # render = ~post.copy()
#     # mask = masksGT[test_i]
#     # render[~vis_im] = 1
#     #
#     # segmentations[4][test_i] = post
#
#     # renderRGB = np.concatenate([render[:,:,None],  render[:,:,None], render[:,:,None]], axis=2)
#
#     # cv2.imwrite('tmp/SH/renderergt' + str(test_i) + '.jpeg' , 255*lin2srgb(rendererGT[:,:,[2,1,0]]), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#     # plt.imsave('tmp/SH/mask' + str(test_i) + '.jpeg', mask)
#     # cv2.imwrite('tmp/SH/post' + str(test_i) + '.jpeg' , 255*post, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#     # cv2.imwrite('tmp/SH/post' + str(test_i) + '.jpeg' , 255*render, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#     # plt.imsave('tmp/SH/renderer' + str(test_i) + '.png' , lin2srgb(renderer.r.copy()))
#     # ipdb.set_trace()
# # # #
#
# methodsPred[3] = 'Robust Fit'
#
# methodsPred[4] = 'Robust (OpenGL)'
#
# methodsPred[5] = 'Recognition (OpenGL)'
#
# methodsPred[6] = 'Gaussian Fit'
#
# segmentations[4] = segmentations3[3]
#
# errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList, errorsVColorsSList, errorsSegmentationList \
#         = computeErrors(np.arange(len(rangeTests)), azimuths, testAzsRel, elevations, testElevsGT, vColors, testVColorGT, lightCoeffs, testLightCoefficientsGTRel, approxProjections,  approxProjectionsGT, shapeParams, testShapeParamsGT, useShapeModel, chShapeParams, chVertices, segmentations, masksGT)


# envMapTexture = np.zeros([180,360,3])
# approxProjectionsFittedList = []
# for test_i in range(len(testSet)):
#     pEnvMap = SHProjection(envMapTexture, np.concatenate([lightCoeffs[4][test_i][:,None], lightCoeffs[4][test_i][:,None], lightCoeffs[4][test_i][:,None]], axis=1))
#     approxProjection = np.sum(pEnvMap, axis=(2,3))
#     approxProjectionsFittedList = approxProjectionsFittedList + [approxProjection[None,:]]
# approxProjectionsFitted = np.vstack(approxProjectionsFittedList)
#
# envMapTexture = np.zeros([180,360,3])
# approxProjectionsGTList = []
# for test_i in range(len(testSet)):
#     pEnvMap = SHProjection(envMapTexture, np.concatenate([testLightCoefficientsGTRel[test_i][:,None], testLightCoefficientsGTRel[test_i][:,None], testLightCoefficientsGTRel[test_i][:,None]], axis=1))
#     approxProjectionGT = np.sum(pEnvMap, axis=(2,3))
#     approxProjectionsGTList = approxProjectionsGTList + [approxProjectionGT[None,:]]
# approxProjectionsGT = np.vstack(approxProjectionsGTList)


# with open(resultDir + 'experiment_errors.pickle', 'rb') as pfile:
#     experimentErrorsDic = pickle.load(pfile)
#
# errorsPosePredList  = experimentErrorsDic['errorsPosePredList']
# errorsLightCoeffsList  = experimentErrorsDic['errorsLightCoeffsList']
# errorsShapeParamsList  = experimentErrorsDic['errorsShapeParamsLis']
# errorsShapeVerticesList  = experimentErrorsDic['errorsShapeVerticesList']
# errorsEnvMapList  = experimentErrorsDic['errorsEnvMapList']
# errorsLightCoeffsCList  = experimentErrorsDic['errorsLightCoeffsCList']
# errorsVColorsEList  = experimentErrorsDic['errorsVColorsEList']
# errorsVColorsCList  = experimentErrorsDic['errorsVColorsCList']
# errorsVColorsSList  = experimentErrorsDic['errorsVColorsSList']
# errorsSegmentationList = experimentErrorsDic['errorsSegmentationList']


testPrefix = 'train4_occlusion_shapemodel_10k_ECCV-PHOTOREALISTIC-JOINT2018'
resultDir = 'results/' + testPrefix + '/'

# experimentDic = {'testSet':testSet, 'methodsPred':methodsPred, 'testOcclusions':testOcclusions, 'likelihoods':likelihoods, 'testPrefixBase':testPrefixBase, 'parameterRecognitionModels':parameterRecognitionModels, 'azimuths':azimuths, 'elevations':elevations, 'vColors':vColors, 'lightCoeffs':lightCoeffs, 'shapeParams':shapeParams}
#
# with open(resultDir + 'experiment.pickle', 'wb') as pfile:
#     pickle.dump(experimentDic, pfile)
# #
# experimentErrorsDic = {'errorsPosePredList':errorsPosePredList, 'errorsLightCoeffsList':errorsLightCoeffsList, 'errorsShapeParamsLis':errorsShapeParamsList, 'errorsShapeVerticesList':errorsShapeVerticesList, 'errorsEnvMapList':errorsEnvMapList, 'errorsLightCoeffsCList':errorsLightCoeffsCList, 'errorsVColorsEList':errorsVColorsEList, 'errorsVColorsCList':errorsVColorsCList, 'errorsVColorsSList':errorsVColorsSList,'errorsSegmentationList':errorsSegmentationList}
#
#
# with open(resultDir + 'experiment_errors.pickle', 'wb') as pfile:
#     pickle.dump(experimentErrorsDic, pfile)

with open(resultDir + 'experiment_errors.pickle', 'rb') as pfile:
    experimentErrorsDic = pickle.load(pfile)

errorsPosePredList  = experimentErrorsDic['errorsPosePredList']
errorsLightCoeffsList  = experimentErrorsDic['errorsLightCoeffsList']
errorsShapeParamsList  = experimentErrorsDic['errorsShapeParamsLis']
errorsShapeVerticesList  = experimentErrorsDic['errorsShapeVerticesList']
errorsEnvMapList  = experimentErrorsDic['errorsEnvMapList']
errorsLightCoeffsCList  = experimentErrorsDic['errorsLightCoeffsCList']
errorsVColorsEList  = experimentErrorsDic['errorsVColorsEList']
errorsVColorsCList  = experimentErrorsDic['errorsVColorsCList']
errorsVColorsSList  = experimentErrorsDic['errorsVColorsSList']
errorsSegmentationList = experimentErrorsDic['errorsSegmentationList']


meanAbsErrAzsList, meanAbsErrElevsList, meanErrorsLightCoeffsList, meanErrorsShapeParamsList, meanErrorsShapeVerticesList, meanErrorsLightCoeffsCList, meanErrorsEnvMapList, meanErrorsVColorsEList, meanErrorsVColorsCList, meanErrorsVColorsCList, meanErrorsSegmentation \
    = computeErrorAverages(np.mean, np.arange(len(rangeTests)), useShapeModel, errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList,errorsVColorsSList, errorsSegmentationList)

nearestNeighbours = False
if 'Nearest Neighbours' in set(methodsPred):
    nearestNeighbours = True

plotColors = ['k']
if nearestNeighbours:
    # methodsPred = methodsPred + ["Nearest Neighbours"]
    plotColors = plotColors + ['m']

plotColors = plotColors + ['b']

plotColors = plotColors + ['r']

plotColors = plotColors + ['r']

plotColors = plotColors + ['b']

plotColors = plotColors + ['g']

plotColors = plotColors + ['g']

plotStyles = ['solid']
if nearestNeighbours:
    # methodsPred = methodsPred + ["Nearest Neighbours"]
    plotStyles = plotStyles + ['solid']

plotStyles = plotStyles + ['solid']

plotStyles = plotStyles + ['solid']

plotStyles = plotStyles + ['dashed']

plotStyles = plotStyles + ['dashed']

plotStyles = plotStyles + ['solid']

plotStyles = plotStyles + ['dashed']

plotMethodsIndices = [2,5,6, 7, 3,4]
# plotMethodsIndices = [0,2,6,3]

import ipdb; ipdb.set_trace()

recognitionIdx = 2
robustIdx = 3

errorsAzimuthList = [errorsPosePredList[i][0] if errorsPosePredList[i] is not None else None for i in range(len(methodsPred))]
errorsElevationList = [errorsPosePredList[i][1] if errorsPosePredList[i] is not None else None for i in range(len(methodsPred))]

variablesDescr = ['Azimuth', 'Elevation', 'VColor', 'Illumination', 'Shape', 'Segmentation']


errorsList = [errorsAzimuthList,errorsElevationList,  errorsVColorsCList, errorsLightCoeffsCList, errorsShapeVerticesList, errorsSegmentationList]

saveConditionalHistograms(resultDir, testOcclusions, methodsPred, variablesDescr, plotMethodsIndices, errorsList)


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
meanErrorsVColorsSArr = []
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
    meanErrorsVColorsSArr = meanErrorsVColorsSArr + [np.array([])]
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

            if errorsPosePredList[method_i] is not None:
                meanAbsErrAzsArr[method_i] = np.append(meanAbsErrAzsArr[method_i], np.mean(np.abs(errorsPosePredList[method_i][0][setUnderOcclusionLevel])))
                meanAbsErrElevsArr[method_i] = np.append(meanAbsErrElevsArr[method_i], np.mean(np.abs(errorsPosePredList[method_i][1][setUnderOcclusionLevel])))

            if errorsLightCoeffsList[method_i] is not None:
                meanErrorsLightCoeffsArr[method_i] = np.append(meanErrorsLightCoeffsArr[method_i],np.mean(np.mean(errorsLightCoeffsList[method_i][setUnderOcclusionLevel], axis=1), axis=0))
            if errorsLightCoeffsCList[method_i] is not None:
                meanErrorsLightCoeffsCArr[method_i] = np.append(meanErrorsLightCoeffsCArr[method_i],np.mean(np.mean(errorsLightCoeffsCList[method_i][setUnderOcclusionLevel], axis=1), axis=0))

            if useShapeModel:
                if errorsShapeParamsList[method_i] is not None:
                    meanErrorsShapeParamsArr[method_i] = np.append(meanErrorsShapeParamsArr[method_i],np.mean(np.mean(errorsShapeParamsList[method_i][setUnderOcclusionLevel], axis=1), axis=0))

                if errorsShapeVerticesList[method_i] is not None:
                    meanErrorsShapeVerticesArr[method_i] = np.append(meanErrorsShapeVerticesArr[method_i], np.mean(errorsShapeVerticesList[method_i][setUnderOcclusionLevel], axis=0))

            if errorsEnvMapList[method_i] is not None:
                meanErrorsEnvMapArr[method_i] = np.append(meanErrorsEnvMapArr[method_i], np.mean(errorsEnvMapList[method_i][setUnderOcclusionLevel]))

            if errorsVColorsEList[method_i] is not None:
                meanErrorsVColorsEArr[method_i] = np.append(meanErrorsVColorsEArr[method_i], np.mean(errorsVColorsEList[method_i][setUnderOcclusionLevel], axis=0))

            if errorsVColorsCList[method_i] is not None:
                meanErrorsVColorsCArr[method_i] = np.append(meanErrorsVColorsCArr[method_i], np.mean(errorsVColorsCList[method_i][setUnderOcclusionLevel], axis=0))

            if errorsVColorsSList[method_i] is not None:
                meanErrorsVColorsSArr[method_i] = np.append(meanErrorsVColorsSArr[method_i], np.mean(errorsVColorsSList[method_i][setUnderOcclusionLevel], axis=0))

            if errorsSegmentationList[method_i] is not None:
                meanErrorsSegmentationArr[method_i] = np.append(meanErrorsSegmentationArr[method_i], np.mean(errorsSegmentationList[method_i][setUnderOcclusionLevel], axis=0))
            else:
                meanErrorsSegmentationArr[method_i] = None

print("Printing occlusin-error plots - median!")

saveOcclusionPlots(resultDir, 'mean',occlusions, methodsPred, plotColors, plotStyles, plotMethodsIndices, useShapeModel, meanAbsErrAzsArr, meanAbsErrElevsArr, meanErrorsVColorsCArr, meanErrorsVColorsEArr, meanErrorsVColorsSArr, meanErrorsLightCoeffsArr, meanErrorsShapeParamsArr, meanErrorsShapeVerticesArr, meanErrorsLightCoeffsCArr, meanErrorsEnvMapArr, meanErrorsSegmentationArr)



medianAbsErrAzsArr = []
medianAbsErrElevsArr = []
medianErrorsLightCoeffsArr = []
medianErrorsEnvMapArr = []
medianErrorsShapeParamsArr = []
medianErrorsShapeVerticesArr = []
medianErrorsLightCoeffsCArr = []
medianErrorsVColorsEArr = []
medianErrorsVColorsCArr = []
medianErrorsVColorsSArr = []
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
    medianErrorsVColorsSArr = medianErrorsVColorsSArr + [np.array([])]
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

            if errorsPosePredList[method_i] is not None:
                medianAbsErrAzsArr[method_i] = np.append(medianAbsErrAzsArr[method_i], np.median(np.abs(errorsPosePredList[method_i][0][setUnderOcclusionLevel])))
                medianAbsErrElevsArr[method_i] = np.append(medianAbsErrElevsArr[method_i], np.median(np.abs(errorsPosePredList[method_i][1][setUnderOcclusionLevel])))

            if errorsLightCoeffsList[method_i] is not None:
                medianErrorsLightCoeffsArr[method_i] = np.append(medianErrorsLightCoeffsArr[method_i],np.median(np.mean(errorsLightCoeffsList[method_i][setUnderOcclusionLevel], axis=1), axis=0))
            if errorsLightCoeffsCList[method_i] is not None:
                medianErrorsLightCoeffsCArr[method_i] = np.append(medianErrorsLightCoeffsCArr[method_i],np.median(np.mean(errorsLightCoeffsCList[method_i][setUnderOcclusionLevel], axis=1), axis=0))

            if useShapeModel:
                if errorsShapeParamsList[method_i] is not None:
                    medianErrorsShapeParamsArr[method_i] = np.append(medianErrorsShapeParamsArr[method_i],np.median(np.median(errorsShapeParamsList[method_i][setUnderOcclusionLevel], axis=1), axis=0))

                if errorsShapeVerticesList[method_i] is not None:
                    medianErrorsShapeVerticesArr[method_i] = np.append(medianErrorsShapeVerticesArr[method_i], np.median(errorsShapeVerticesList[method_i][setUnderOcclusionLevel], axis=0))

            if errorsEnvMapList[method_i] is not None:
                medianErrorsEnvMapArr[method_i] = np.append(medianErrorsEnvMapArr[method_i], np.median(errorsEnvMapList[method_i][setUnderOcclusionLevel]))

            if errorsVColorsEList[method_i] is not None:
                medianErrorsVColorsEArr[method_i] = np.append(medianErrorsVColorsEArr[method_i], np.median(errorsVColorsEList[method_i][setUnderOcclusionLevel], axis=0))

            if errorsVColorsCList[method_i] is not None:
                medianErrorsVColorsCArr[method_i] = np.append(medianErrorsVColorsCArr[method_i], np.median(errorsVColorsCList[method_i][setUnderOcclusionLevel], axis=0))

            if errorsVColorsSList[method_i] is not None:
                medianErrorsVColorsSArr[method_i] = np.append(medianErrorsVColorsSArr[method_i], np.median(errorsVColorsSList[method_i][setUnderOcclusionLevel], axis=0))

            if errorsSegmentationList[method_i] is not None:
                medianErrorsSegmentationArr[method_i] = np.append(medianErrorsSegmentationArr[method_i], np.median(errorsSegmentationList[method_i][setUnderOcclusionLevel], axis=0))
            else:
                medianErrorsSegmentationArr[method_i] = None



saveOcclusionPlots(resultDir, 'median', occlusions,methodsPred, plotColors, plotStyles, plotMethodsIndices, useShapeModel, medianAbsErrAzsArr, medianAbsErrElevsArr, medianErrorsVColorsCArr, medianErrorsVColorsEArr, medianErrorsVColorsSArr, medianErrorsLightCoeffsArr, medianErrorsShapeParamsArr, medianErrorsShapeVerticesArr, medianErrorsLightCoeffsCArr, medianErrorsEnvMapArr, medianErrorsSegmentationArr)
SHModel = ""


for occlusionLevel in [25,75,100]:


    resultDirOcclusion = 'results/' + testPrefix + '/occlusion' + str(occlusionLevel) + '/'
    if not os.path.exists(resultDirOcclusion):
        os.makedirs(resultDirOcclusion)

    setUnderOcclusionLevel = testOcclusionsFull * 100 < occlusionLevel
    testOcclusions = testOcclusionsFull[setUnderOcclusionLevel]
    #
    #
    errorsPosePred = [errorsPosePredList[recognitionIdx][0][setUnderOcclusionLevel], errorsPosePredList[recognitionIdx][1][setUnderOcclusionLevel]]
    errorsLightCoeffs = errorsLightCoeffsList[recognitionIdx][setUnderOcclusionLevel]
    errorsShapeParams = errorsShapeParamsList[recognitionIdx][setUnderOcclusionLevel]
    errorsShapeVertices= errorsShapeVerticesList[recognitionIdx][setUnderOcclusionLevel]
    if errorsEnvMapList[recognitionIdx] is not None:
        errorsEnvMap= errorsEnvMapList[recognitionIdx][setUnderOcclusionLevel]
    else:
        errorsEnvMap = None
    errorsLightCoeffsC= errorsLightCoeffsCList[recognitionIdx][setUnderOcclusionLevel]
    errorsVColorsE= errorsVColorsEList[recognitionIdx][setUnderOcclusionLevel]
    errorsVColorsC= errorsVColorsCList[recognitionIdx][setUnderOcclusionLevel]
    errorsVColorsS= errorsVColorsSList[recognitionIdx][setUnderOcclusionLevel]
    # errorsSegmentation = errorsSegmentationList[recognitionIdx][setUnderOcclusionLevel]
    errorsSegmentation = None


    errorsPoseFitted =  [errorsPosePredList[robustIdx][0][setUnderOcclusionLevel], errorsPosePredList[robustIdx][1][setUnderOcclusionLevel]]
    errorsFittedLightCoeffs = errorsLightCoeffsList[robustIdx][setUnderOcclusionLevel]
    errorsFittedShapeParams = errorsShapeParamsList[robustIdx][setUnderOcclusionLevel]
    errorsFittedShapeVertices= errorsShapeVerticesList[robustIdx][setUnderOcclusionLevel]
    if errorsEnvMapList[recognitionIdx] is not None:
        errorsFittedEnvMap = errorsEnvMapList[robustIdx][setUnderOcclusionLevel]
    else:
        errorsFittedEnvMap = None
    errorsFittedLightCoeffsC= errorsLightCoeffsCList[robustIdx][setUnderOcclusionLevel]
    errorsFittedVColorsE = errorsVColorsEList[robustIdx][setUnderOcclusionLevel]
    errorsFittedVColorsC = errorsVColorsCList[robustIdx][setUnderOcclusionLevel]
    errorsFittedVColorsS = errorsVColorsSList[robustIdx][setUnderOcclusionLevel]
    # errorsFittedSegmentation = errorsSegmentationList[recognitionIdx][setUnderOcclusionLevel]
    errorsFittedSegmentation = None


    saveScatterPlots(resultDirOcclusion, testOcclusions, useShapeModel, errorsPosePred, errorsPoseFitted,errorsLightCoeffsC,errorsFittedLightCoeffsC,errorsEnvMap,errorsFittedEnvMap,errorsLightCoeffs,errorsFittedLightCoeffs,errorsShapeParams,errorsFittedShapeParams,errorsShapeVertices,errorsFittedShapeVertices,errorsVColorsE,errorsFittedVColorsE,errorsVColorsC,errorsFittedVColorsC, errorsVColorsS,errorsFittedVColorsS)

    errorsPoseFitted =  [errorsPosePredList[5][0][setUnderOcclusionLevel], errorsPosePredList[5][1][setUnderOcclusionLevel]]
    errorsFittedLightCoeffs = errorsLightCoeffsList[5][setUnderOcclusionLevel]
    errorsFittedShapeParams = errorsShapeParamsList[5][setUnderOcclusionLevel]
    errorsFittedShapeVertices= errorsShapeVerticesList[5][setUnderOcclusionLevel]
    if errorsEnvMapList[recognitionIdx] is not None:
        errorsFittedEnvMap = errorsEnvMapList[5][setUnderOcclusionLevel]
    else:
        errorsFittedEnvMap = None
    errorsFittedLightCoeffsC= errorsLightCoeffsCList[5][setUnderOcclusionLevel]
    errorsFittedVColorsE= errorsVColorsEList[5][setUnderOcclusionLevel]
    errorsFittedVColorsC= errorsVColorsCList[5][setUnderOcclusionLevel]
    errorsFittedVColorsS= errorsVColorsSList[5][setUnderOcclusionLevel]
    errorsFittedSegmentation = errorsSegmentationList[5][setUnderOcclusionLevel]

    saveScatterPlotsMethodFit(4, resultDirOcclusion, testOcclusions, useShapeModel, errorsPosePred, errorsPoseFitted,errorsLightCoeffsC,errorsFittedLightCoeffsC,errorsEnvMap,errorsFittedEnvMap,errorsLightCoeffs,errorsFittedLightCoeffs,errorsShapeParams,errorsFittedShapeParams,errorsShapeVertices,errorsFittedShapeVertices,errorsVColorsE,errorsFittedVColorsE,errorsVColorsC,errorsFittedVColorsC, errorsVColorsS,errorsFittedVColorsS)

    # saveLikelihoodScatter(resultDirOcclusion, setUnderOcclusionLevel, testOcclusions,  likelihoods)


    # if len(stdevsFull) > 0:
    #     stdevs = stdevsFull[setUnderOcclusionLevel]

    colors = matplotlib.cm.plasma(testOcclusions)

    meanAbsErrAzsList, meanAbsErrElevsList, meanErrorsLightCoeffsList, meanErrorsShapeParamsList, meanErrorsShapeVerticesList, meanErrorsLightCoeffsCList, meanErrorsEnvMapList, meanErrorsVColorsEList, meanErrorsVColorsCList, meanErrorsVColorsSList, meanErrorsSegmentationList \
        = computeErrorAverages(np.mean, setUnderOcclusionLevel, useShapeModel, errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList, errorsVColorsSList, errorsSegmentationList)

    medianAbsErrAzsList, medianAbsErrElevsList, medianErrorsLightCoeffsList, medianErrorsShapeParamsList, medianErrorsShapeVerticesList, medianErrorsLightCoeffsCList, medianErrorsEnvMapList, medianErrorsVColorsEList, medianErrorsVColorsCList, medianErrorsVColorsSList, medianErrorsSegmentationList  \
        = computeErrorAverages(np.median, setUnderOcclusionLevel, useShapeModel, errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList, errorsVColorsSList, errorsSegmentationList)
    # Write statistics to file.

    import tabulate

    headers = ["Errors"] + methodsPred

    table = [["Azimuth"] +  meanAbsErrAzsList,
             ["Elevation"] + meanAbsErrElevsList,
             ["VColor C"] + meanErrorsVColorsCList,
             ["VColor S"] + meanErrorsVColorsSList,
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
             ["VColor S"] + medianErrorsVColorsSList,
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



