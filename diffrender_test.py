test__author__ = 'pol'

# from damascene import damascene

import matplotlib
# matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
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
from opendr_utils import *
from utils import *
import OpenGL.GL as GL
import light_probes
from OpenGL import contextdata
from light_probes import SHProjection

import theano
theano.config.optimizer='fast_compile'
theano.config.cycle_detection = 'fast'
# theano.sandbox.cuda.use('cpu')
import lasagne
import lasagne_nn

plt.ion()

#########################################
# Test configuration
#########################################

seed = 1
np.random.seed(seed)

parameterRecognitionModels = set(['randForestAzs', 'randForestElevs', 'randForestVColors', 'linearRegressionVColors', 'neuralNetModelSHLight', ])
parameterRecognitionModels = set(['randForestAzs', 'randForestElevs', 'randForestVColors', 'linearRegressionVColors', 'linRegModelSHZernike' ])
parameterRecognitionModels = set(['randForestAzs', 'randForestElevs','linearRegressionVColors','neuralNetModelSHLight' ])
parameterRecognitionModels = set(['neuralNetPose', 'linearRegressionVColors','constantSHLight' ])
parameterRecognitionModels = set(['neuralNetPose', 'neuralNetApperanceAndLight', 'neuralNetVColors' ])
parameterRecognitionModels = set(['neuralNetPose', 'neuralNetModelSHLight', 'neuralNetVColors', 'neuralNetModelShape' ])
# parameterRecognitionModels = set(['neuralNetPose', 'neuralNetApperanceAndLight'])

# parameterRecognitionModels = set(['randForestAzs', 'randForestElevs','randForestVColors','randomForestSHZernike' ])
#

gtPrefix = 'train4_occlusion_shapemodel'
# gtPrefix = 'train4_occlusion_shapemodel_synthetic_10K_test100-1100'
gtPrefix = 'train4_occlusion_shapemodel_photorealistic_10K_test100-1100'
# gtPrefix = 'objectnet3d_teapots'
experimentPrefix = 'train4_occlusion_shapemodel_10k'

# gtPrefix = 'train4_occlusion_multi'
# experimentPrefix = 'train4_occlusion_multi'
trainPrefixPose = 'train4_occlusion_shapemodel_10k'
trainPrefixVColor = 'train4_occlusion_shapemodel_10k'
trainPrefixLightCoeffs = 'train4_occlusion_shapemodel_10k'
trainPrefixShapeParams = 'train4_occlusion_shapemodel_10k'
trainModelsDirAppLight = 'train4_occlusion_shapemodel_10k'

#########################################
# OpenDR Initialization starts here
#########################################

#Main script options:

glModes = ['glfw','mesa']
glMode = glModes[0]

width, height = (150, 150)
win = -1

multiObjects = False
chThError = None
global renderer

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
hdridx = 0
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

# if multiObjects:
chObjDist = ch.Ch([0])
chObjRotation = ch.Ch([0])
chObjAzMug = ch.Ch([0])
chObjDistMug = ch.Ch([0])
chObjRotationMug = ch.Ch([0])

chVColorsMug = ch.Ch([1,0,0])

clampedCosCoeffs = clampedCosineCoefficients()
chComponent = chLightSHCoeffs * clampedCosCoeffs

light_color = ch.ones(3)

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

    vmod, vnmod, _ = transformObject(vmod, vnmod, chScale, chObjAz, ch.Ch([0]), ch.Ch([0]), np.array([0,0,0]))
    renderer = createRendererTarget(glMode, chAz, chEl, chDist, centermod, vmod, vcmod, fmod_list, vnmod, light_color, chComponent, chVColors, 0, chDisplacement,  width,height, uvmod, haveTexturesmod_list, texturesmod_list, frustum, win )
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

if multiObjects:
    mugs = [line.strip() for line in open('mugs.txt')]
    renderMugsList = np.arange(len(teapots))[0:1]

    v_mugs, f_list_mugs, vc_mugs, vn_mugs, uv_mugs, haveTextures_list_mugs, textures_list_mugs, vflat_mugs, varray_mugs, center_mugs = scene_io_utils.loadMugsOpenDRData(
        renderMugsList, False, False, None)

    v_mug = v_mugs[0][0]
    f_list_mug = f_list_mugs[0][0]
    chVColorsMug = ch.Ch([1, 0, 0])
    vc_mug = [chVColorsMug * np.ones(v_mug[0].shape)]
    vn_mug = vn_mugs[0][0]
    uv_mug = uv_mugs[0][0]
    haveTextures_list_mug = haveTextures_list_mugs[0][0]
    textures_list_mug = textures_list_mugs[0][0]


#########################################
# Initialization ends here
#########################################

#########################################
# Generative model set up
#########################################

global rendererGT

rendererGT = ch.Ch(renderer.r.copy())
numPixels = width*height

E_raw = renderer - rendererGT
SE_raw = ch.sum(E_raw*E_raw, axis=2)

SSqE_raw = ch.SumOfSquares(E_raw)/numPixels

initialPixelStdev = 0.1

# finalPixelStdev = 0.05
stds = ch.Ch([initialPixelStdev])
variances = stds ** 2
globalPrior = ch.Ch([0.9])

negLikModel = -ch.sum(generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances))
negLikModelRobust = -ch.sum(generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances))
pixelLikelihoodCh = generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances)
pixelLikelihoodRobustCh = generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances)

post = generative_models.layerPosteriorsRobustCh(rendererGT, renderer, np.array([]), 'MASK', globalPrior, variances)[0]

models = [negLikModel, negLikModelRobust]
pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh]
modelsDescr = ["Gaussian Model", "Outlier model" ]

model = 1
pixelErrorFun = pixelModels[model]
errorFun = models[model]

global iterat
iterat = 0

t = time.time()

#########################################
# Generative model setup ends here.
#########################################

#########################################
# Test code starts here:
#########################################

gtDir = 'groundtruth/' + gtPrefix + '/'
featuresDir = gtDir

experimentDir = 'experiments/' + experimentPrefix + '/'
trainModelsDirPose = 'experiments/' + trainPrefixPose + '/' + gtPrefix
trainModelsDirVColor = 'experiments/' + trainPrefixVColor + '/' + gtPrefix
trainModelsDirLightCoeffs = 'experiments/' + trainPrefixLightCoeffs + '/' + gtPrefix
trainModelsDirShapeParams = 'experiments/' + trainPrefixShapeParams + '/' + gtPrefix
trainModelsDirAppLight = 'experiments/' + trainModelsDirAppLight + '/' + gtPrefix

useCRFOcclusionPred = False
useShapeModel = True
makeVideo = False
reduceVariance = False
getColorFromCRF = False
syntheticGroundtruth = False
evaluateWithGT = True

ignoreGT = True
ignore = []
if os.path.isfile(gtDir + 'ignore.npy'):
    ignore = np.load(gtDir + 'ignore.npy')

testSet = np.load(experimentDir + 'test.npy')

rangeTests = np.arange(len(testSet))

rangeTests = np.arange(100,1100)

# rangeTests = np.arange(100,1100)

#
testSet = testSet[rangeTests]

numTests = len(testSet)

if evaluateWithGT:
    groundTruthFilename = gtDir + 'groundTruth.h5'
    gtDataFile = h5py.File(groundTruthFilename, 'r')

    shapeGT = gtDataFile[gtPrefix].shape

    # boolTestSet = np.zeros(shapeGT).astype(np.bool)
    # boolTestSet[testSet] = True
    boolTestSet = np.array([np.any(num == testSet) for num in gtDataFile[gtPrefix]['trainIds']])

    dataIds = gtDataFile[gtPrefix][boolTestSet]['trainIds']

    dataIdsTestIndices = np.array([np.where(dataIds==num)[0][0] for num in testSet])

    # testGroundTruth = gtDataFile[gtPrefix][boolTestSet]
    # groundTruthTest = np.zeros(shapeGT, dtype=testGroundTruth.dtype)
    # groundTruthTest[boolTestSet] = testGroundTruth
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
    dataAmbientIntensityGT = groundTruth['trainAmbientIntensityGT']
    dataIds = groundTruth['trainIds']

    if multiObjects:
        dataObjDistGT = groundTruth['trainObjDistGT']
        dataObjRotationGT = groundTruth['trainObjRotationGT']
        dataObjDistMug = groundTruth['trainObjDistMug']
        dataObjRotationMug = groundTruth['trainObjRotationMug']
        dataObjAzMug = groundTruth['trainObjAzMug']
        dataVColorsMug = groundTruth['trainVColorsMug']
        dataObjAzMugRel = groundTruth['trainObjAzMugRel']
        dataObjAzGTRel = groundTruth['trainObjAzGTRel']

        dataMugPosOffset = groundTruth['trainMugPosOffset']
        dataTeapotPosOffset = groundTruth['trainTeapotPosOffset']

    if useShapeModel:
        dataShapeModelCoeffsGT = groundTruth['trainShapeModelCoeffsGT']

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

synthPrefix = '_cycles'
if syntheticGroundtruth:
    synthPrefix = ''

if syntheticGroundtruth:
    imagesDir = gtDir + 'images_opendr/'
else:
    imagesDir = gtDir + 'images/'

if evaluateWithGT:
    images = readImages(imagesDir, dataIds, loadFromHdf5)
else:
    import glob

    imageFiles = glob.glob1(imagesDir, "*.png")
    images = np.zeros([len(imageFiles), width, height, 3])
    for imageFile_i, imageFile in enumerate(imageFiles):
        image = skimage.io.imread(imagesDir + imageFile)
        image = skimage.transform.resize(image , [height, width])

        images[imageFile_i] = image

    testSet = np.arange(len(images))

print("Backprojecting and fitting estimates.")
# testSet = np.arange(len(images))[0:10]

if evaluateWithGT:
    testAzsGT = dataAzsGT
    testObjAzsGT = dataObjAzsGT
    testElevsGT = dataElevsGT
    testLightAzsGT = dataLightAzsGT
    testLightElevsGT = dataLightElevsGT
    testLightIntensitiesGT = dataLightIntensitiesGT
    testVColorGT = dataVColorGT
    if useShapeModel:
        testShapeParamsGT = dataShapeModelCoeffsGT

    testLightCoefficientsGTRel = dataLightCoefficientsGTRel * dataAmbientIntensityGT[:,None]

    testAzsRel = np.mod(testAzsGT - testObjAzsGT, 2*np.pi)
    testOcclusions = dataOcclusions

    testIds = dataIds
    if multiObjects:
        testObjAzMugRel = dataObjAzMugRel
        testObjAzGTRel = dataObjAzGTRel

        testMugPosOffset = dataMugPosOffset
        testTeapotPosOffset = dataTeapotPosOffset
        testVColorMug = dataVColorsMug

        testObjDistGT = dataObjDistGT
        testObjRotationGT = dataObjRotationGT
        testObjDistMug = dataObjDistMug
        testObjRotationMug = dataObjRotationMug
        testObjAzMug = dataObjAzMug
        testVColorsMug = dataVColorsMug

else:
    #There is no GT.
    testAzsGT = np.zeros(len(testSet))
    testObjAzsGT = np.zeros(len(testSet))
    testElevsGT = np.zeros(len(testSet))
    testLightAzsGT = np.zeros(len(testSet))
    testLightElevsGT = np.zeros(len(testSet))
    testLightIntensitiesGT = np.zeros(len(testSet))
    testOcclusions = np.zeros(len(testSet))
    testIds = np.zeros(len(testSet))
    testVColorGT = np.zeros(len(testSet))
    if useShapeModel:
        testShapeParamsGT = np.zeros(len(testSet))

    testLightCoefficientsGTRel = np.zeros(len(testSet))

    testAzsRel = np.zeros(len(testSet))

    if multiObjects:
        testObjAzMugRel = np.zeros(len(testSet))
        testObjAzGTRel = np.zeros(len(testSet))

        testMugPosOffset = np.zeros(len(testSet))
        testTeapotPosOffset = np.zeros(len(testSet))
        testVColorMug = np.zeros(len(testSet))

        testObjDistGT = np.zeros(len(testSet))
        testObjRotationGT = np.zeros(len(testSet))
        testObjDistMug = np.zeros(len(testSet))
        testObjRotationMug = np.zeros(len(testSet))
        testObjAzMug = np.zeros(len(testSet))
        testVColorsMug = np.zeros(len(testSet))

##Read Training set labels

# trainGTPrefix = 'train4_occlusion_shapemodel'
#
# trainGTDir = 'groundtruth/' + trainGTPrefix + '/'
# trainGroundTruthFilename = trainGTDir + 'groundTruth.h5'
# trainGTDataFile = h5py.File(trainGroundTruthFilename, 'r')
#
# trainSet = np.load(experimentDir + 'train.npy')
#
# shapeGT = trainGTDataFile[trainGTPrefix].shape
# boolTrainSet = np.zeros(shapeGT).astype(np.bool)
# boolTrainSet[trainSet] = True
# trainGroundTruth = trainGTDataFile[trainGTPrefix][boolTrainSet]
# groundTruthTrain = np.zeros(shapeGT, dtype=trainGroundTruth.dtype)
# groundTruthTrain[boolTrainSet] = trainGroundTruth
# groundTruthTrain = groundTruthTrain[trainSet]
# dataTeapotIdsTrain = groundTruthTrain['trainTeapotIds']
# train = np.arange(len(trainSet))
#
# testSet = testSet[test]
#
# print("Reading experiment.")
# trainAzsGT = groundTruthTrain['trainAzsGT']
# trainObjAzsGT = groundTruthTrain['trainObjAzsGT']
# trainElevsGT = groundTruthTrain['trainElevsGT']
# trainLightAzsGT = groundTruthTrain['trainLightAzsGT']
# trainLightElevsGT = groundTruthTrain['trainLightElevsGT']
# trainLightIntensitiesGT = groundTruthTrain['trainLightIntensitiesGT']
# trainVColorGT = groundTruthTrain['trainVColorGT']
# trainScenes = groundTruthTrain['trainScenes']
# trainTeapotIds = groundTruthTrain['trainTeapotIds']
# trainEnvMaps = groundTruthTrain['trainEnvMaps']
# trainOcclusions = groundTruthTrain['trainOcclusions']
# trainTargetIndices = groundTruthTrain['trainTargetIndices']
# trainLightCoefficientsGT = groundTruthTrain['trainLightCoefficientsGT']
# trainLightCoefficientsGTRel = groundTruthTrain['trainLightCoefficientsGTRel']
# trainAmbientIntensityGT = groundTruthTrain['trainAmbientIntensityGT']
# trainIds = groundTruthTrain['trainIds']
#
# if useShapeModel:
#     trainShapeModelCoeffsGT = groundTruthTrain['trainShapeModelCoeffsGT']
#
# trainLightCoefficientsGTRel = trainLightCoefficientsGTRel * trainAmbientIntensityGT[:,None]
#
# trainAzsRel = np.mod(trainAzsGT - trainObjAzsGT, 2*np.pi)


# latexify(columns=2)
#
# directory = 'tmp/occlusions'
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.hist(trainOcclusions*100, bins=40)
# ax.set_xlabel('Occlusion level (\%)')
# ax.set_ylabel('Counts')
# ax.set_title('Occlusion histogram')
# ax.set_xlim(0,100)
# fig.savefig(directory + '-histogram.pdf', bbox_inches='tight')
# plt.close(fig)

recognitionTypeDescr = ["near", "mean", "sampling"]
recognitionType = 1

optimizationTypeDescr = ["predict", "optimize", "joint"]
optimizationType = 1
computePredErrorFuns = True

method = 5
model = 1
maxiter = 100
numSamples = 1

mintime = time.time()

free_variables = [ chAz, chEl]

boundEl = (-np.pi, np.pi)
boundAz = (-3*np.pi, 3*np.pi)
boundscomponents = (0,None)
bounds = [boundAz,boundEl]
bounds = [(None , None ) for sublist in free_variables for item in sublist]
methods = ['dogleg', 'minimize', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'TNC', 'SGDMom', 'probLineSearch']
# options = {'disp':False, 'maxiter':maxiter, 'lr':0.0001, 'momentum':0.1, 'decay':0.99}
options = {'disp':False, 'maxiter':maxiter}
azVar = 0.1
elVar = 0.1
vColorVar = 0.01
shCoeffsVar = 0.01
df_vars = np.concatenate([azVar*np.ones(chAz.shape), elVar*np.ones(chEl.shape), vColorVar*np.ones(chVColors.r.shape), shCoeffsVar*np.ones(chLightSHCoeffs.r.shape)])
options = {'disp':False, 'maxiter':maxiter, 'df_vars':df_vars}

testRenderer = 0

#Read texture UV mapping
sceneFilename = 'teapot.blend'
with bpy.data.libraries.load(filepath=sceneFilename) as (data_from, data_to):
    for attr in dir(data_to):
        setattr(data_to, attr, getattr(data_from, attr))
teapot = bpy.data.scenes['Teapot'].objects[1]
vmod, fmod_list, vcmod, vnmod, uvmod, haveTexturesmod_list, texturesmod_list = scene_io_utils.unpackBlenderObject(teapot, '', False)
color = np.array([0,0,0])
texturesmod_list[0][0][0] = ch.Ch(np.ones(texturesmod_list[0][0][0].shape)*color[None,None,:])

# testRenderer = 2
if useShapeModel:
    import shape_model
    #%% Load data
    filePath = 'data/teapotModel.pkl'
    teapotModel = shape_model.loadObject(filePath)
    faces = teapotModel['faces']

    #%% Sample random shape Params
    latentDim = np.shape(teapotModel['ppcaW'])[1]
    shapeParams = np.zeros(latentDim)
    chShapeParams = ch.Ch(shapeParams.copy())

    meshLinearTransform=teapotModel['meshLinearTransform']
    W=teapotModel['ppcaW']
    b=teapotModel['ppcaB']

    chVertices = shape_model.VerticesModel(chShapeParams=chShapeParams,meshLinearTransform=meshLinearTransform,W = W,b=b)
    chVertices.init()

    chVertices = ch.dot(geometry.RotateZ(-np.pi/2)[0:3,0:3],chVertices.T).T

    smFaces = [[faces]]
    smVColors = [chVColors*np.ones(chVertices.shape)]
    # smUVs = ch.Ch(np.zeros([chVertices.shape[0],2]))
    smUVs = uvmod[0]
    # smHaveTextures = [[False]]
    smHaveTextures = [[False]]
    smTexturesList = [[None]]
    # smTexturesList = [[texturesmod_list[0][0][0]]]

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

    if useShapeModel:
        center = smCenter
        UVs = smUVs
        v = smVertices
        vn = smNormals
        Faces = smFaces
        VColors = smVColors
        HaveTextures = smHaveTextures
        TexturesList = smTexturesList
    else:
        v, vn = v_teapots[currentTeapotModel][0], vn_teapots[currentTeapotModel][0]
        Faces = f_list_teapots[currentTeapotModel][0]
        VColors = vc_teapots[currentTeapotModel][0]
        UVs = uv_teapots[currentTeapotModel][0]
        HaveTextures = haveTextures_list_teapots[currentTeapotModel][0]
        TexturesList = textures_list_teapots[currentTeapotModel][0]

    v, vn, teapotPosOffset = transformObject(v, vn, chScale, chObjAz, chObjDist, chObjRotation, np.array([0,0,0]))

    verticesCube, facesCube, normalsCube, vColorsCube, texturesListCube, haveTexturesCube = getCubeData()

    uvCube = np.zeros([verticesCube.shape[0],2])

    chCubePosition = ch.Ch([0, 0, 0])
    chCubeScale = ch.Ch([10.0])
    chCubeAzimuth = ch.Ch([0])
    chCubeVCColors = ch.Ch(np.ones_like(vColorsCube) * 1.0)
    v_transf, vn_transf = transformObject2([verticesCube], [normalsCube], chCubeScale, chCubeAzimuth, chCubePosition)

    v_scene = [v]
    f_list_scene = [smFaces]
    vc_scene = [smVColors]
    vn_scene = [vn]
    uv_scene = [smUVs]
    haveTextures_list_scene = [smHaveTextures]
    textures_list_scene = [smTexturesList]

    addObjectData(v_scene, f_list_scene, vc_scene, vn_scene, uv_scene, haveTextures_list_scene, textures_list_scene, v_transf, [[facesCube]],[chCubeVCColors], vn_transf, [uvCube], haveTexturesCube, texturesListCube)

    if multiObjects:
        verticesMug, normalsMug, mugPosOffset = transformObject(v_mug, vn_mug, chScale, chObjAzMug + np.pi / 2, chObjDistMug, chObjRotationMug, np.array([0,0,0]))

        VerticesB = [v] + [verticesMug]
        NormalsB = [vn] + [normalsMug]
        FacesB = [Faces] + [f_list_mug]
        VColorsB = [VColors] + [vc_mug]
        UVsB = [UVs] + [uv_mug]
        HaveTexturesB = [HaveTextures] + [haveTextures_list_mug]
        TexturesListB = [TexturesList] + [textures_list_mug]

        renderer = createRendererTarget(glMode, chAz, chEl, chDist, center, VerticesB, VColorsB, FacesB, NormalsB, light_color,chComponent, chVColors, np.array([0,0,0]), chDisplacement, width, height, UVsB, HaveTexturesB, TexturesListB, frustum, None)
        renderer.initGL()
        renderer.initGLTexture()
    else:
        # renderer = createRendererTarget(glMode, chAz, chEl, chDist, smCenter, [v], [smVColors], [smFaces], [vn], light_color, chComponent, chVColors, 0, chDisplacement, width,height, [smUVs], [smHaveTextures], [smTexturesList], frustum, win )
        renderer = createNewRendererTarget(glMode, chAz, chEl, chDist, smCenter, v_scene, vc_scene, f_list_scene, vn_scene, light_color, chComponent, chVColors, 0, chDisplacement, width,height, uv_scene, haveTextures_list_scene, textures_list_scene, frustum, win )

        renderer.overdraw = True
        renderer.nsamples = 8
        renderer.msaa = True
        renderer.initGL()
        renderer.initGLTexture()
        renderer.initGL_AnalyticRenderer()
        renderer.imageGT = None
        renderer.r

    # chShapeParams[:] = np.zeros([latentDim])
    chVerticesMean = chVertices.r.copy()

else:
    renderer = renderer_teapots[testRenderer]

# plt.imsave('testrender.png', sqeRenderer.render_image)

loadMask = True
if loadMask:
    masksGT = loadMasks(gtDir + '/masks_occlusion/', testSet)

# ### Groundtruth triplets generation
#
# groundTruthFilename = 'groundtruth/' + gtPrefix + '/' '/groundTruth.h5'
# gtDataFileToRender = h5py.File(groundTruthFilename, 'r')
# groundTruthToRender = gtDataFileToRender[gtPrefix]
#
# rangeGT = np.arange(len(groundTruthToRender))
#
# for gtIdx in rangeGT[:]:
#     groundTruthToRender['trainEnvMapPhiOffsets'][gtIdx]
#
#     color = groundTruthToRender['trainVColorGT'][gtIdx]
#     az = groundTruthToRender['trainObjAzsGT'][gtIdx] - groundTruthToRender['trainAzsGT'][gtIdx]
#     el = groundTruthToRender['trainElevsGT'][gtIdx]
#     lightCoefficientsRel = groundTruthToRender['trainLightCoefficientsGTRel'][gtIdx] * groundTruthToRender['trainAmbientIntensityGT'][gtIdx]
#
#     if useShapeModel:
#         shapeParams = groundTruthToRender['trainShapeModelCoeffsGT'][gtIdx]
#
#     chAz[:] = 0
#     chEl[:] = el
#     chObjAz[:] = az
#     chVColors[:] = color
#     chLightSHCoeffs[:] = lightCoefficientsRel
#     if useShapeModel:
#         chShapeParams[:] = shapeParams
#
#     image = renderer.r[:].copy()
#     lin2srgb(image)
#     cv2.imwrite(gtDir + '/backprojections/im' + str(gtIdx) + '.jpeg', 255 * image[:, :, [2, 1, 0]], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#
# ipdb.set_trace()

# vis_im = np.array(renderer.indices_image==1).copy().astype(np.bool)
# im = skimage.io.imread('renderergt539.jpeg').astype(np.float32)/255.
# rendererGT[:] = srgb2lin(im.copy())
# post = generative_models.layerPosteriorsRobustCh(rendererGT, renderer, vis_im, 'MASK', globalPrior, variances)[0].r>0.05
# render = renderer.r.copy()
# render[~mask*vis_im] = np.concatenate([np.ones([1000,1000])[:,:,None],  np.zeros([1000,1000])[:,:,None],np.zeros([1000,1000])[:,:,None]], axis=2)[~mask*vis_im]
#
# render[renderer.boundarybool_image.astype(np.bool)] = renderer.r[renderer.boundarybool_image.astype(np.bool)]
#
# cv2.imwrite('renderer' + '.jpeg' , 255*lin2srgb(render[:,:,[2,1,0]]), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
# cv2.imwrite('rendererGT' + '.jpeg' , 255*lin2srgb(rendererGT.r[:,:,[2,1,0]]), [int(cv2.IMWRITE_JPEG_QUALITY), 100])

# plt.imsave('renderered.png', lin2srgb(render))


##### Tests with optimizing multiple objects in scene and camera settings.
# test_i = 0
# for test_i in range(len(testSet)):
#
#     color = testVColorGT[test_i]
#     az = testAzsGT[test_i]
#     el = testElevsGT[test_i]
#     lightCoefficientsRel = testLightCoefficientsGTRel[test_i]
#
#     if useShapeModel:
#         shapeParams = testShapeParamsGT[test_i]
#
#     chAz[:] = 0
#     chEl[:] = el
#     chVColors[:] = color
#     chLightSHCoeffs[:] = lightCoefficientsRel
#     if useShapeModel:
#         chShapeParams[:] = shapeParams
#
#     chObjAz[:] = testObjAzsGT[test_i] - testAzsGT[test_i]
#     chObjDist[:] = testObjDistGT[test_i]
#     chObjRotation[:] = testObjRotationGT[test_i] - testAzsGT[test_i]
#     chObjAzMug[:] =  testObjAzMug[test_i] - testAzsGT[test_i]
#     chObjDistMug[:] = testObjDistMug[test_i]
#     chObjRotationMug[:] = testObjRotationMug[test_i] - testAzsGT[test_i]
#
#     chVColorsMug[:] = testVColorMug[test_i]
#
#     image = skimage.transform.resize(images[test_i], [height, width])
#     imageSrgb = image.copy()
#     rendererGT[:] = srgb2lin(image)
#
#     masksGT = loadMasks(gtDir + '/masks_occlusion/', testSet)
#     masksMug = loadMasksMug(gtDir + '/masks_occlusion/', testSet)
#
#     maskTeapot = masksGT[test_i]
#     maskMug = masksMug[test_i]
#
#     coords = np.meshgrid(np.arange(width)-width/2, np.arange(height)-height/2)
#
#     coordsMugX = coords[1][maskMug]
#     coordsMugY = coords[0][maskMug]
#
#     bbRendererGT = rendererGT.r.copy()
#
#     bbRendererGT[coordsMugX.min()+width/2, coordsMugY.min()+height/2 : coordsMugY.max() + height/2] = np.array([1,0,0])
#     bbRendererGT[coordsMugX.max()+width/2, coordsMugY.min()+height/2 : coordsMugY.max() + height/2] = np.array([1,0,0])
#     bbRendererGT[coordsMugX.min()+width/2 : coordsMugX.max() + width/2, coordsMugY.min()+height/2] = np.array([1,0,0])
#     bbRendererGT[coordsMugX.min()+width/2:coordsMugX.max()+width/2, coordsMugY.max() + height/2] = np.array([1,0,0])
#
#     coordsTeapotX = coords[1][maskTeapot]
#     coordsTeapotY = coords[0][maskTeapot]
#
#     bbRendererGT[coordsTeapotX.min()+width/2, coordsTeapotY.min()+height/2 : coordsTeapotY.max() + height/2] = np.array([1,0,0])
#     bbRendererGT[coordsTeapotX.max()+width/2, coordsTeapotY.min()+height/2 : coordsTeapotY.max() + height/2] = np.array([1,0,0])
#     bbRendererGT[coordsTeapotX.min()+width/2 : coordsTeapotX.max() + width/2, coordsTeapotY.min()+height/2] = np.array([1,0,0])
#     bbRendererGT[coordsTeapotX.min()+width/2:coordsTeapotX.max()+width/2, coordsTeapotY.max() + height/2] = np.array([1,0,0])
#
#
#     posMug = np.array([(coordsMugY.min() + coordsMugY.max())/2, (coordsMugX.min() + coordsMugX.max())/2])
#     posTeapot = np.array([(coordsTeapotY.min() + coordsTeapotY.max())/2, (coordsTeapotX.min() + coordsTeapotX.max())/2])
#
#     # createRendererTarget(glMode, False, chAz, chEl, chDist, center, VerticesB, VColorsB, FacesB, NormalsB, light_color,chComponent, chVColors, np.array([0,0,0]), chDisplacement, width, height, UVsB, HaveTexturesB, TexturesListB, frustum, None)
#     chAzCam = ch.Ch([0])
#     chElCam = ch.Ch(chEl.r.copy())
#
#     relPosTeapotGT = v[0].r.sum(axis=0)/v[0].r.shape[0]
#     relPosMugGT = verticesMug[0].r.sum(axis=0)/verticesMug[0].r.shape[0]
#
#     bbRendererGT[posMug[1] - 2+width/2:posMug[1] + 2+width/2, posMug[0] - 2 + height/2: posMug[0] + 2 + height/2] = np.array([1,0,0])
#     bbRendererGT[posTeapot[1] - 2+width/2:posTeapot[1] + 2+width/2, posTeapot[0] - 2 + height/2: posTeapot[0] + 2 + height/2] = np.array([1,0,0])
#
#     _, _, camTransfomMatGT = setupCamera(np.array([0,0,0]), chAz.r, chEl.r, chDist, np.array([0,0,0.1]), width, height)
#     camEyeGT = camTransfomMatGT[0:4,0:4].dot(np.array([0,0,0,1]))[0:3]
#
#     vecMugToCamGT = camEyeGT - (mugPosOffset + np.array([0,0,0.1]))
#     mugCamElGT = 2*ch.arctan(ch.norm(ch.array([0,-1,0])*ch.norm(vecMugToCamGT) - vecMugToCamGT*ch.norm(ch.array([0,-1,0])))/ch.norm(ch.array([0,-1,0])*ch.norm(vecMugToCamGT) + ch.norm(ch.array([0,-1,0]))*vecMugToCamGT))
#
#     vecTeapotToCamGT = camEyeGT - (teapotPosOffset + np.array([0,0,0.1]))
#     teapotCamElGT = 2*ch.arctan(ch.norm(ch.array([0,-1,0])*ch.norm(vecTeapotToCamGT) - vecTeapotToCamGT*ch.norm(ch.array([0,-1,0])))/ch.norm(ch.array([0,-1,0])*ch.norm(vecTeapotToCamGT) + ch.norm(ch.array([0,-1,0]))*vecTeapotToCamGT))
#
#     objDisplacementMat = computeHemisphereTransformation(chObjRotationMug, 0, chObjDistMug, np.array([0, 0, 0.05]))
#     pointMug = objDisplacementMat[0:3, 3]
#
#     # pointMug = ch.Ch([0,0,0.1])
#     imPosMugCam, modelRotation, camTransfomMat = setupCamera(pointMug, chAzCam, chElCam, chDist, np.array([0,0,0.1]), width, height)
#     imPosMug = imPosMugCam - np.array([height/2, width/2])
#
#     objDisplacementMat = computeHemisphereTransformation(chObjRotation, 0, chObjDist, np.array([0, 0, 0.1]))
#     pointTeapot = objDisplacementMat[0:3, 3]
#     # pointTeapot = ch.Ch([0,0,0.1])
#
#     imPosTeapotCam, modelRotation, camTransfomMat = setupCamera(pointTeapot, chAzCam, chElCam, chDist, np.array([0,0,0.1]), width, height)
#     imPosTeapot = imPosTeapotCam - np.array([height/2, width/2])
#
#     errMug = ch.sum((imPosMug - posMug)**2)
#     errTeapot = ch.sum((imPosTeapot - posTeapot)**2)
#
#     camEye = camTransfomMat[0:4,0:4].dot(np.array([0,0,0, 1]))[0:3]
#
#     vecMugToCam = camEye - pointMug
#     mugCamEl = 2*ch.arctan(ch.norm(ch.array([0,-1,0])*ch.norm(vecMugToCam) - vecMugToCam*ch.norm(ch.array([0,-1,0])))/ch.norm(ch.array([0,-1,0])*ch.norm(vecMugToCam) + ch.norm(ch.array([0,-1,0]))*vecMugToCam))
#
#     vecTeapotToCam = camEye - pointTeapot
#
#     teapotCamEl = 2*ch.arctan(ch.norm(ch.array([0,-1,0])*ch.norm(vecTeapotToCam) - vecTeapotToCam*ch.norm(ch.array([0,-1,0])))/ch.norm(ch.array([0,-1,0])*ch.norm(vecTeapotToCam) + ch.norm(ch.array([0,-1,0]))*vecTeapotToCam))
#
#     chElPredMug = mugCamElGT
#     chElPredTeapot = teapotCamElGT
#
#     errElMug = ch.sum((chElPredMug*180/np.pi  - mugCamEl*180/np.pi)**2)
#     errElTeapot = ch.sum((chElPredTeapot*180/np.pi - teapotCamEl*180/np.pi)**2)
#
#     # spatialFreeVars = [pointTeapot[0:2], pointMug[0:2], chElCam]
#     spatialFreeVars = [chObjRotationMug, chObjRotation, chObjDistMug, chObjDist, chElCam]
#
#     spatialErrorFun = errMug + errTeapot + errElMug+ errElTeapot
#
#     def cbS(_):
#         pass
#
#     bbRendererGT[imPosTeapot[1] - 2 +width/2:imPosTeapot[1] + 2+width/2, imPosTeapot[0] - 2 + height/2: imPosTeapot[0] + 2 + height/2] = np.array([0,0,1])
#     bbRendererGT[imPosMug[1] - 2+width/2:imPosMug[1] + 2+width/2, imPosMug[0] - 2 + height/2: imPosMug[0] + 2 + height/2] = np.array([0,0,1])
#
#     plt.imsave('tmp/rendererPred' + str(test_i) + '.png', renderer.r)
#
#     #Dumb initial state (center of table).
#     chObjDist[:] = 0
#     chObjRotation[:] = 0
#     chObjDistMug[:] = 0
#     chObjRotationMug[:] = 0
#
#     plt.imsave('tmp/rendererInit' + str(test_i) + '.png', renderer.r)
#
#     # ch.minimize({'raw': spatialErrorFun }, bounds=[(-0.5,0.5), (-0.5,0.5), (0,np.pi/2)], method=methods[1], x0=spatialFreeVars, callback=cbS, options={'disp':False, 'maxiter':10})
#     ch.minimize({'raw': spatialErrorFun }, bounds=[(-2*np.pi,2*np.pi), (-2*np.pi,2*np.pi), (0,0.5), (0,0.5), (0,np.pi/2)], method=methods[2], x0=spatialFreeVars, callback=cbS, options={'disp':True, 'maxiter':50})
#
#     bbRendererGT[imPosTeapot[1] - 2+width/2:imPosTeapot[1] + 2+width/2, imPosTeapot[0] - 2 + height/2: imPosTeapot[0] + 2 + height/2] = np.array([0,1,0])
#     bbRendererGT[imPosMug[1] - 2+width/2:imPosMug[1] + 2+width/2, imPosMug[0] - 2 + height/2: imPosMug[0] + 2 + height/2] = np.array([0,1,0])
#
#     plt.imsave('tmp/bbRendererGT' + str(test_i) + '.png', bbRendererGT)
#
#     chEl[:] = chElCam.r.copy()
#
#     plt.imsave('tmp/rendererOpt' + str(test_i) + '.png', renderer.r)

import skimage.color

nearGTOffsetRelAz = 0.01
nearGTOffsetEl = 0.01
nearGTOffsetLighCoeffs = np.zeros(9)
nearGTOffsetVColor = np.zeros(3)

############ RECOGNITION MDOEL PREDICTIONS (COMPUTE AND SAVE, DON'T USE THIS EVERY TIME)

#Load trained recognition models

nnBatchSize = 100

azsPredictions = np.array([])

recomputeMeans = False
includeMeanBaseline = False

recomputePredictions = False
recomputePredictionsPose = False

if includeMeanBaseline:
    meanTrainLightCoefficientsGTRel = np.repeat(np.mean(trainLightCoefficientsGTRel, axis=0)[None,:], numTests, axis=0)
    meanTrainElevation = np.repeat(np.mean(trainElevsGT, axis=0), numTests,  axis=0)
    meanTrainAzimuthRel = np.repeat(0, numTests, axis=0)
    meanTrainShapeParams = np.repeat(np.zeros([latentDim])[None,:], numTests, axis=0)
    meanTrainVColors = np.repeat(np.mean(trainVColorGT, axis=0)[None,:], numTests, axis=0)
    chShapeParams[:] = np.zeros([10])
    meanTrainShapeVertices = np.repeat(chVertices.r.copy()[None,:], numTests, axis=0)

    # if recomputeMeans or not os.path.isfile(experimentDir + "meanTrainEnvMapProjections.npy"):
    #     envMapTexture = np.zeros([180,360,3])
    #     approxProjectionsPredList = []
    #     for train_i in range(len(trainSet)):
    #         pEnvMap = SHProjection(envMapTexture, np.concatenate([trainLightCoefficientsGTRel[train_i][:,None], trainLightCoefficientsGTRel[train_i][:,None], trainLightCoefficientsGTRel[train_i][:,None]], axis=1))
    #         approxProjectionPred = np.sum(pEnvMap, axis=(2,3))
    #
    #         approxProjectionsPredList = approxProjectionsPredList + [approxProjectionPred[None,:]]
    #     approxProjections = np.vstack(approxProjectionsPredList)
    #     meanTrainEnvMapProjections = np.mean(approxProjections, axis=0)
    #     np.save(experimentDir + 'meanTrainEnvMapProjections.npy', meanTrainEnvMapProjections)
    # else:
    #     meanTrainEnvMapProjections = np.load(experimentDir + 'meanTrainEnvMapProjections.npy')

    # meanTrainEnvMapProjections = np.repeat(meanTrainEnvMapProjections[None,:], numTests, axis=0)
    meanTrainEnvMapProjections = None

rangeTests = np.arange(len(testSet))


if recomputePredictionsPose or not os.path.isfile(trainModelsDirPose + "elevsPred.npy"):
    if 'neuralNetPose' in parameterRecognitionModels:

        poseModel = ""
        with open(experimentDir + 'neuralNetModelPose.pickle', 'rb') as pfile:
            neuralNetModelPose = pickle.load(pfile)

        meanImage = neuralNetModelPose['mean']
        modelType = neuralNetModelPose['type']
        param_values = neuralNetModelPose['params']
        grayTestImages =  0.3*images[:,:,:,0] +  0.59*images[:,:,:,1] + 0.11*images[:,:,:,2]
        grayTestImages = grayTestImages[:,None, :,:]

        grayTestImages = grayTestImages - meanImage.reshape([1,1, grayTestImages.shape[2],grayTestImages.shape[3]])

        network = lasagne_nn.load_network(modelType=modelType, param_values=param_values)
        posePredictionFun = lasagne_nn.get_prediction_fun(network)

        posePredictions = np.zeros([len(grayTestImages), 4])
        for start_idx in range(0, len(grayTestImages), nnBatchSize):
            posePredictions[start_idx:start_idx + nnBatchSize] = posePredictionFun(grayTestImages.astype(np.float32)[start_idx:start_idx + nnBatchSize])
        # posePredictions = posePredictionFun(grayTestImages.astype(np.float32))

        cosAzsPred = posePredictions[:,0]
        sinAzsPred = posePredictions[:,1]
        cosElevsPred = posePredictions[:,2]
        sinElevsPred = posePredictions[:,3]


        elevsPred = np.arctan2(sinElevsPred, cosElevsPred)
        azsPred = np.arctan2(sinAzsPred, cosAzsPred)


        np.save(trainModelsDirPose + 'elevsPred.npy', elevsPred)
        np.save(trainModelsDirPose + 'azsPred.npy', azsPred)

        # ##Get predictions with dropout on to get samples.
        # with open(trainModelsDirPose + 'neuralNetModelPose.pickle', 'rb') as pfile:
        #     neuralNetModelPose = pickle.load(pfile)
        #
        # meanImage = neuralNetModelPose['mean']
        # modelType = neuralNetModelPose['type']
        # param_values = neuralNetModelPose['params']
        #
        # # network = lasagne_nn.load_network(modelType=modelType, param_values=param_values)
        # nonDetPosePredictionFun = lasagne_nn.get_prediction_fun_nondeterministic(network)
        # posePredictionsSamples = []
        # cosAzsPredSamples = []
        # sinAzsPredSamples = []
        # cosElevsPredSamples = []
        # sinElevsPredSamples = []
        # for i in range(100):
        #     posePredictionsSample = np.zeros([len(grayTestImages), 4])
        #     for start_idx in range(0, len(grayTestImages), nnBatchSize):
        #         posePredictionsSample[start_idx:start_idx + nnBatchSize] = nonDetPosePredictionFun(grayTestImages.astype(np.float32)[start_idx:start_idx + nnBatchSize])
        #
        #     cosAzsPredSample = posePredictionsSample[:,0]
        #     sinAzsPredSample = posePredictionsSample[:,1]
        #     cosAzsPredSamples = cosAzsPredSamples + [cosAzsPredSample[:,None]]
        #     sinAzsPredSamples = sinAzsPredSamples + [sinAzsPredSample[:,None]]
        #
        #     cosElevsPredSample = posePredictionsSample[:,2]
        #     sinElevsPredSample = posePredictionsSample[:,3]
        #     cosElevsPredSamples = cosElevsPredSamples + [cosElevsPredSample[:,None]]
        #     sinElevsPredSamples = sinElevsPredSamples + [sinElevsPredSample[:,None]]
        #
        # cosAzsPredSamples = np.hstack(cosAzsPredSamples)
        # sinAzsPredSamples = np.hstack(sinAzsPredSamples)
        #
        # cosElevsPredSamples = np.hstack(cosElevsPredSamples)
        # sinElevsPredSamples = np.hstack(sinElevsPredSamples)
        #
        # azsPredictions = np.arctan2(sinAzsPredSamples, cosAzsPredSamples)
        # elevsPredictions = np.arctan2(sinElevsPredSamples, cosElevsPredSamples)
        #
        # np.save(trainModelsDirPose + 'azsPredictions.npy', azsPredictions)
        # np.save(trainModelsDirPose + 'elevsPredictions.npy', elevsPredictions)

        # ##Get predictions with dropout on to get samples.
        # with open(trainModelsDirPose + 'neuralNetModelPose.pickle', 'rb') as pfile:
        #     neuralNetModelPose = pickle.load(pfile)
        #
else:
    elevsPred = np.load(trainModelsDirPose + 'elevsPred.npy')[rangeTests]
    azsPred = np.load(trainModelsDirPose + 'azsPred.npy')[rangeTests]

    # azsPredictions =  np.load(trainModelsDirPose + 'azsPredictions.npy')[rangeTests]
    azsPredictions =  None
    # elevsPredictions = np.load(trainModelsDirPose + 'elevsPredictions.npy')[rangeTests]
    elevsPredictions = None

if recomputePredictions or not os.path.isfile(trainModelsDirVColor + "vColorsPred.npy"):

    if 'neuralNetVColors' in parameterRecognitionModels:

        import theano
        # theano.sandbox.cuda.use('cpu')
        import lasagne
        import lasagne_nn

        nnModel = ""
        with open(experimentDir + 'neuralNetModelAppearance.pickle', 'rb') as pfile:
            neuralNetModelAppearance = pickle.load(pfile)

        meanImage = neuralNetModelAppearance['mean']
        modelType = neuralNetModelAppearance['type']
        param_values = neuralNetModelAppearance['params']

        testImages = images.reshape([images.shape[0],3,images.shape[1],images.shape[2]]) - meanImage.reshape([1,meanImage.shape[2], meanImage.shape[0],meanImage.shape[1]]).astype(np.float32)

        network = lasagne_nn.load_network(modelType=modelType, param_values=param_values)
        appPredictionFun = lasagne_nn.get_prediction_fun(network)

        appPredictions = np.zeros([len(testSet), 3])
        for start_idx in range(0, len(testSet), nnBatchSize):
            appPredictions[start_idx:start_idx + nnBatchSize] = appPredictionFun(testImages.astype(np.float32)[start_idx:start_idx + nnBatchSize])

        vColorsPred = appPredictions

        np.save(trainModelsDirPose + 'vColorsPred.npy', vColorsPred)


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
else:
    vColorsPred = np.load(trainModelsDirVColor + 'vColorsPred.npy')[rangeTests]

SHModel = ""

import theano
import theano.tensor as T

## Theano NN error function 1.
# import lasagne_nn
# import lasagne
#
# with open(trainModelsDirPose + 'neuralNetModelAppearance.pickle', 'rb') as pfile:
#     neuralNetModelPose = pickle.load(pfile)
#
# meanImage = neuralNetModelPose['mean'].reshape([150,150,3])
#
# modelType = neuralNetModelPose['type']
# param_values = neuralNetModelPose['params']
# network = lasagne_nn.load_network(modelType=modelType, param_values=param_values)
# layer = lasagne.layers.get_all_layers(network)[6]
# inputLayer = lasagne.layers.get_all_layers(network)[0]
# layer_output = lasagne.layers.get_output(layer, deterministic=True)
# dim_output= layer.output_shape[1]
#
# networkGT = lasagne_nn.load_network(modelType=modelType, param_values=param_values)
# layerGT = lasagne.layers.get_all_layers(networkGT)[6]
# inputLayerGT = lasagne.layers.get_all_layers(networkGT)[0]
# layer_outputGT = lasagne.layers.get_output(layerGT, deterministic=True)
#
# rendererGray =  0.3*renderer[:,:,0] +  0.59*renderer[:,:,1] + 0.11*renderer[:,:,2]
# rendererGrayGT =  0.3*rendererGT[:,:,0] +  0.59*rendererGT[:,:,1] + 0.11*rendererGT[:,:,2]
#
# chThError = TheanoFunOnOpenDR(theano_input=inputLayer.input_var, theano_output=layer_output, opendr_input=renderer - meanImage, dim_output = dim_output,
#                               theano_input_gt=inputLayerGT.input_var, theano_output_gt=layer_outputGT, opendr_input_gt=rendererGT - meanImage)
#
# chThError.compileFunctions(layer_output, theano_input=inputLayer.input_var, dim_output=dim_output, theano_input_gt=inputLayerGT.input_var, theano_output_gt=layer_outputGT)
#
# chThError.r

#
## Theano NN error function finite differences.
# with open(trainModelsDirPose + 'neuralNetModelAzimuthTriplet.pickle', 'rb') as pfile:
#     neuralNetModelPose = pickle.load(pfile)

#meanImage = neuralNetModelPose['mean'].reshape([150,150])

# modelType = neuralNetModelPose['type']
# param_values = neuralNetModelPose['params']
#
# network = lasagne_nn.load_network(modelType=modelType, param_values=param_values, imgSize=75)
#
# # layer = lasagne.layers.get_all_layers(network)[-2]
# inputLayer = lasagne.layers.get_all_layers(network)[0]
# layer_output = lasagne.layers.get_output(network, deterministic=True)
# dim_output= network.output_shape[1]
#
# networkGT = lasagne_nn.load_network(modelType=modelType, param_values=param_values, imgSize=75)
# # layerGT = lasagne.layers.get_all_layers(networkGT)[-2]
# inputLayerGT = lasagne.layers.get_all_layers(networkGT)[0]
# # layer_outputGT = lasagne.layers.get_output(layerGT, deterministic=True)
# layer_outputGT = lasagne.layers.get_output(networkGT, deterministic=True)
#
# rendererGray =  0.3*renderer[:,:,0] +  0.59*renderer[:,:,1] + 0.11*renderer[:,:,2]
# rendererGrayGT =  0.3*rendererGT[:,:,0] +  0.59*rendererGT[:,:,1] + 0.11*rendererGT[:,:,2]

# chThError = TheanoFunFiniteDiff(theano_input=inputLayer.input_var, theano_output=layer_output, opendr_input=rendererGray, dim_output = dim_output,
#                               theano_input_gt=inputLayerGT.input_var, theano_output_gt=layer_outputGT, opendr_input_gt=rendererGrayGT, imSize=75)
#
# chThError.compileFunctions(layer_output, theano_input=inputLayer.input_var, dim_output=dim_output, theano_input_gt=inputLayerGT.input_var, theano_output_gt=layer_outputGT)
#
# chThError.r


if recomputePredictions or not os.path.isfile(trainModelsDirLightCoeffs + "relLightCoefficientsPred.npy"):
    if 'neuralNetModelSHLight' in parameterRecognitionModels:

        import theano
        # theano.sandbox.cuda.use('cpu')
        import lasagne
        import lasagne_nn


        nnModel = ""
        with open(experimentDir + 'neuralNetModelLight.pickle', 'rb') as pfile:
            neuralNetModelSHLight = pickle.load(pfile)

        meanImage = neuralNetModelSHLight['mean']
        modelType = neuralNetModelSHLight['type']
        param_values = neuralNetModelSHLight['params']

        testImages = images.reshape([images.shape[0],3,images.shape[1],images.shape[2]]) - meanImage.reshape([1,meanImage.shape[2], meanImage.shape[0],meanImage.shape[1]]).astype(np.float32)

        network = lasagne_nn.load_network(modelType=modelType, param_values=param_values)
        lightPredictionFun = lasagne_nn.get_prediction_fun(network)

        lightPredictions = np.zeros([len(testImages), 9])
        for start_idx in range(0, len(testImages), nnBatchSize):
            lightPredictions[start_idx:start_idx + nnBatchSize] = lightPredictionFun(testImages.astype(np.float32)[start_idx:start_idx + nnBatchSize])

        relLightCoefficientsPred = lightPredictions

        np.save(trainModelsDirLightCoeffs + 'relLightCoefficientsPred.npy', relLightCoefficientsPred)

else:
    relLightCoefficientsPred = np.load(trainModelsDirLightCoeffs + 'relLightCoefficientsPred.npy')[rangeTests]

if recomputePredictions or not os.path.isfile(trainModelsDirShapeParams + "shapeParamsPred.npy"):
    if 'neuralNetModelShape' in parameterRecognitionModels and useShapeModel:

        import theano
        # theano.sandbox.cuda.use('cpu')
        import lasagne
        import lasagne_nn

        nnModel = ""
        with open(experimentDir + 'neuralNetModelShape.pickle', 'rb') as pfile:
            neuralNetModelSHLight = pickle.load(pfile)

        meanImage = neuralNetModelSHLight['mean']
        modelType = neuralNetModelSHLight['type']
        param_values = neuralNetModelSHLight['params']

        grayTestImages =  0.3*images[:,:,:,0] +  0.59*images[:,:,:,1] + 0.11*images[:,:,:,2]
        grayTestImages = grayTestImages[:,None, :,:]
        grayTestImages = grayTestImages - meanImage.reshape([1,1, grayTestImages.shape[2],grayTestImages.shape[3]])

        network = lasagne_nn.load_network(modelType=modelType, param_values=param_values)
        shapePredictionFun = lasagne_nn.get_prediction_fun(network)

        shapePredictions = np.zeros([len(grayTestImages), latentDim])

        for start_idx in range(0, len(grayTestImages), nnBatchSize):
            shapePredictions[start_idx:start_idx + nnBatchSize] = shapePredictionFun(grayTestImages.astype(np.float32)[start_idx:start_idx + nnBatchSize])

        shapeParamsPred = shapePredictions

        np.save(trainModelsDirShapeParams + 'shapeParamsPred.npy', shapeParamsPred)


        # #Samples:
        # shapeParamsPredSamples = []
        # shapeParamsNonDetFun = lasagne_nn.get_prediction_fun_nondeterministic(network)
        #
        # for i in range(100):
        #     shapeParamsPredictionsSample = np.zeros([len(grayTestImages), 10])
        #     for start_idx in range(0, len(grayTestImages), nnBatchSize):
        #         shapeParamsPredictionsSample[start_idx:start_idx + nnBatchSize] = shapeParamsNonDetFun(grayTestImages.astype(np.float32)[start_idx:start_idx + nnBatchSize])
        #
        #     shapeParamsPredSamples = shapeParamsPredSamples + [shapeParamsPredictionsSample[:,:][:,:,None]]
        #
        # shapeParamsPredSamples = np.concatenate(shapeParamsPredSamples, axis=2)
        #
        # np.save(trainModelsDirShapeParams + 'shapeParamsPredSamples.npy', shapeParamsPredSamples)

else:
    shapeParamsPred = np.load(trainModelsDirShapeParams + 'shapeParamsPred.npy')[rangeTests]
    # shapeParamsPredSamples = np.load(trainModelsDirShapeParams + 'shapeParamsPredSamples.npy')[rangeTests]

# if recomputePredictions or not os.path.isfile(trainModelsDirShapeParams + "neuralNetModelMaskLarge.npy"):
#     if 'neuralNetModelMask' in parameterRecognitionModels:
#
#         import theano
#         # theano.sandbox.cuda.use('cpu')
#         import lasagne
#         import lasagne_nn
#
#
#         nnModel = ""
#         with open(trainModelsDirLightCoeffs + 'neuralNetModelMaskLarge.pickle', 'rb') as pfile:
#             neuralNetModelSHLight = pickle.load(pfile)
#
#         meanImage = neuralNetModelSHLight['mean']
#         modelType = neuralNetModelSHLight['type']
#         param_values = neuralNetModelSHLight['params']
#
#         testImages = images.reshape([images.shape[0],3,images.shape[1],images.shape[2]]) - meanImage.reshape([1,meanImage.shape[2], meanImage.shape[0],meanImage.shape[1]]).astype(np.float32)
#
#         network = lasagne_nn.load_network(modelType=modelType, param_values=param_values)
#         maskPredictionFun = lasagne_nn.get_prediction_fun(network)
#
#         maskPredictions = np.zeros([len(testImages), 50*50])
#         for start_idx in range(0, len(testImages), nnBatchSize):
#             maskPredictions[start_idx:start_idx + nnBatchSize] = maskPredictionFun(testImages.astype(np.float32)[start_idx:start_idx + nnBatchSize])
#
#         maskPredictions = np.reshape(maskPredictions, [len(testImages), 50,50])
#
#         np.save(trainModelsDirShapeParams + 'maskPredictions.npy', maskPredictions)
#
#         # # #Samples:
#         # maskSamples = []
#         # maskPredNonDetFun = lasagne_nn.get_prediction_fun_nondeterministic(network)
#         #
#         # for i in range(100):
#         #     maskPredictionsSamples = np.zeros([len(testImages), 50*50])
#         #     for start_idx in range(0, len(testImages),nnBatchSize):
#         #         maskPredictionsSamples[start_idx:start_idx + nnBatchSize] = maskPredNonDetFun(testImages.astype(np.float32)[start_idx:start_idx + nnBatchSize])
#         #
#         #     maskSamples = maskSamples + [maskPredictionsSamples[:,:][:,:,None]]
#         #
#         # maskSamples = np.concatenate(maskSamples, axis=2)
#         # loadMask = True
#         #
#         # gtDirMask = 'groundtruth/train4_occlusion_mask/'
#         #
#         # masksDir =  gtDirMask + 'masks_occlusion/'
#         # if loadMask:
#         #     masksGT = loadMasks(masksDir, testSet)
# else:
#     maskPredictions = np.load(trainModelsDirShapeParams + 'maskPredictions.npy')[rangeTests]

loadMask = False
if loadMask:
    masksGT = loadMasks(gtDir + '/masks_occlusion/', testSet)

print("Finished loading and compiling recognition models")

envMapDic = {}
SHFilename = 'data/LightSHCoefficients.pickle'

with open(SHFilename, 'rb') as pfile:
    envMapDic = pickle.load(pfile)
hdritems = list(envMapDic.items())[:]

analyzeSamples = False

# def analyzeHue(figurePath, rendererGT, renderer, sampleEl, sampleAz, sampleSH, sampleVColorsPredictions=None, sampleStds=0.1):
#     global stds
#     global chAz
#     global test_i
#     global samplingMode
#
#     plt.ioff()
#     fig = plt.figure()
#
#     stds[:] = sampleStds
#     negLikModel = -ch.sum(generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances))
#     negLikModelRobust = -ch.sum(generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances))
#     stds = ch.Ch([initialPixelStdev])
#
#     models = [negLikModel, negLikModelRobust, negLikModelRobust]
#     errorFunRobust = models[1]
#     errorFunGaussian = models[0]
#
#     vColorsPredSamplesHSV = cv2.cvtColor(np.uint8(sampleVColorsPredictions.reshape([1, 100, 3])*255), cv2.COLOR_RGB2HSV)[0,:,0]
#
#     plt.hist(vColorsPredSamplesHSV, bins=30, alpha=0.2)
#
#     hueGT = cv2.cvtColor(np.uint8(testVColorGT[test_i][None,None,:]*255), cv2.COLOR_RGB2HSV)[0,0,0]
#     huePred = cv2.cvtColor(np.uint8(vColorsPred[test_i][None,None,:]*255), cv2.COLOR_RGB2HSV)[0,0,0]
#
#     chAz[:] = sampleAz
#     chEl[:] = sampleEl
#     currentVColors = chVColors.r
#     currentHSV = cv2.cvtColor(np.uint8(currentVColors[None,None,:]*255), cv2.COLOR_RGB2HSV)[0,0]
#
#     chLightSHCoeffs[:] = sampleSH
#
#     trainingTeapots = [0]
#     hueRange = np.arange(0,255,5)
#     # chThErrors = np.zeros([len(trainingTeapots), len(hueRange)])
#
#     robustErrors = np.array([])
#     gaussianErrors = np.array([])
#     hues = np.array([])
#     for hue_i, hue in enumerate(hueRange):
#         hues = np.append(hues, hue)
#
#         color = cv2.cvtColor(np.array([hue, currentHSV[1],currentHSV[2]])[None,None,:].astype(np.uint8), cv2.COLOR_HSV2RGB)/255
#         chVColors[:] = color
#
#         for idx, renderer_idx in enumerate(trainingTeapots):
#             renderer_i = renderer_teapots[renderer_idx]
#             rendererGray =  0.3*renderer_i[:,:,0] +  0.59*renderer_i[:,:,1] + 0.11*renderer_i[:,:,2]
#             # chThError.opendr_input = rendererGray
#             # chThErrors[idx, az_i] = chThError.r
#
#         robustErrors = np.append(robustErrors, errorFunRobust.r)
#         gaussianErrors = np.append(gaussianErrors, errorFunGaussian.r)
#
#     x1,x2,y1,y2 = plt.axis()
#
#     robustErrors = robustErrors - np.min(robustErrors)
#     gaussianErrors = gaussianErrors - np.min(gaussianErrors)
#     # chThErrors = chThErrors - np.min(chThErrors)
#     plt.plot(hues, robustErrors*y2/np.max(robustErrors),  c='brown')
#     plt.plot(hues, gaussianErrors*y2/np.max(gaussianErrors),  c='purple')
#
#     chThError.opendr_input = renderer
#     lineStyles = ['-', '--', '-.', ':']
#
#     # plt.axvline(np.mod(bestAzNormal*180/np.pi,360), linewidth=2, c='purple')
#     # plt.axvline(np.mod(bestAzRobust*180/np.pi,360), linewidth=2, c='brown')
#     # plt.axvline(bestHue, linewidth=2, c='y')
#     plt.axvline(hueGT, linewidth=2,c='g')
#     plt.axvline(huePred, linewidth=2,c='r')
#
#     # plt.axvline(np.mod(currentAz*180/np.pi, 360), linewidth=2, linestyle='--',c='b')
#
#     plt.xlabel('Sample')
#     plt.ylabel('Angular error')
#
#     plt.axis((0,255,y1,y2))
#     plt.title('Neuralnet multiple predictions')
#     fig.savefig(figurePath + 'sample' + '.png', bbox_inches='tight')
#     plt.close(fig)
#
#     chVColors[:] = currentVColors
#     cv2.imwrite(figurePath + '_render.png', cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))
#
# def analyzeAz(figurePath, rendererGT, renderer, sampleEl, sampleVColor, sampleSH, sampleAzsPredictions=None, sampleStds=0.1):
#     global stds
#     global chAz
#     global test_i
#     global samplingMode
#
#     plt.ioff()
#     fig = plt.figure()
#
#     stds[:] = sampleStds
#     negLikModel = -ch.sum(generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances))
#     negLikModelRobust = -ch.sum(generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances))
#     models = [negLikModel, negLikModelRobust, negLikModelRobust]
#     errorFunRobust = models[1]
#     errorFunGaussian = models[0]
#
#     plt.hist(np.mod(sampleAzsPredictions*180/np.pi,360), bins=30, alpha=0.2)
#
#     bestAz = testAzsRel[test_i]
#
#     currentAz = chAz.r.copy()
#     chEl[:] = sampleEl
#     chVColors[:] = sampleVColor
#     chLightSHCoeffs[:] = sampleSH
#
#     trainingTeapots = [0,14,20,25,26,1]
#     trainingTeapots = [0]
#     azRange = np.arange(0,2*np.pi,5*np.pi/180)
#     chThErrors = np.zeros([len(trainingTeapots), len(azRange)])
#
#     robustErrors = np.array([])
#     gaussianErrors = np.array([])
#     angles = np.array([])
#     for az_i, az in enumerate(azRange):
#         angles = np.append(angles, az*180/np.pi)
#         chAz[:] = az
#         # for idx, renderer_idx in enumerate(trainingTeapots):
#         #     renderer_i = renderer_teapots[renderer_idx]
#         #     rendererGray =  0.3*renderer_i[:,:,0] +  0.59*renderer_i[:,:,1] + 0.11*renderer_i[:,:,2]
#         #     chThError.opendr_input = rendererGray
#         chThErrors[0, az_i] = chThError.r
#
#         robustErrors = np.append(robustErrors, errorFunRobust.r)
#         gaussianErrors = np.append(gaussianErrors, errorFunGaussian.r)
#
#     x1,x2,y1,y2 = plt.axis()
#
#     robustErrors = robustErrors - np.min(robustErrors)
#     gaussianErrors = gaussianErrors - np.min(gaussianErrors)
#     chThErrors = chThErrors - np.min(chThErrors)
#     plt.plot(angles, robustErrors*y2/np.max(robustErrors),  c='brown')
#     plt.plot(angles, gaussianErrors*y2/np.max(gaussianErrors),  c='purple')
#
#     # chThError.opendr_input = renderer
#     lineStyles = ['-', '--', '-.', ':']
#     for renderer_idx in range(len(trainingTeapots)):
#         plt.plot(angles, chThErrors[renderer_idx]*y2/np.max(chThErrors[renderer_idx]), linestyle=lineStyles[np.mod(renderer_idx,4)], c='y')
#
#     if len(trainingTeapots) > 1:
#         prodErrors = np.prod(chThErrors, axis=0)
#         plt.plot(angles, prodErrors*y2/np.max(prodErrors), linestyle='-', c='black')
#         meanErrors = np.mean(chThErrors, axis=0)
#         plt.plot(angles, meanErrors*y2/np.max(meanErrors), linestyle='--', c='black')
#         # plt.plot(angles, gaussianErrors*robustErrors*y2/np.max(gaussianErrors*robustErrors), linestyle='--', c='black')
#
#     # plt.axvline(np.mod(bestAzNormal*180/np.pi,360), linewidth=2, c='purple')
#     # plt.axvline(np.mod(bestAzRobust*180/np.pi,360), linewidth=2, c='brown')
#     plt.axvline(np.mod(bestAz*180/np.pi,360), linewidth=2, c='y')
#     plt.axvline(testAzsRel[test_i]*180/np.pi, linewidth=2,c='g')
#     plt.axvline(np.mod(azsPred[test_i]*180/np.pi, 360), linewidth=2,c='r')
#
#     plt.axvline(np.mod(currentAz*180/np.pi, 360), linewidth=2, linestyle='--',c='b')
#
#     plt.xlabel('Sample')
#     plt.ylabel('Angular error')
#
#     # if samplingMode == False:
#     #
#     #     scaleAzSamples = np.array(errorFunAzSamples)
#     #     scaleAzSamples = scaleAzSamples - np.min(scaleAzSamples) + 1
#     #     scaleAzSamples = scaleAzSamples*0.25*y2/np.max(scaleAzSamples)
#     #     for azSample_i, azSample in enumerate(scaleAzSamples):
#     #         plt.plot(np.mod(totalAzSamples[azSample_i]*180/np.pi, 360), azSample, marker='o', ms=20., c='r')
#     #
#     #     scaleAzSamples = np.array(errorFunGaussianAzSamples)
#     #     scaleAzSamples = scaleAzSamples - np.min(scaleAzSamples) + 1
#     #     scaleAzSamples = scaleAzSamples*0.4*y2/np.max(scaleAzSamples)
#     #     for azSample_i, azSample in enumerate(scaleAzSamples):
#     #         plt.plot(np.mod(totalAzSamples[azSample_i]*180/np.pi, 360), azSample, marker='o', ms=20., c='g')
#     #
#     #     scaleAzSamples = np.array(errorFunAzSamplesPred)
#     #     scaleAzSamples = scaleAzSamples - np.min(scaleAzSamples) + 1
#     #     scaleAzSamples = scaleAzSamples*0.65*y2/np.max(scaleAzSamples)
#     #     for azSample_i, azSample in enumerate(scaleAzSamples):
#     #         plt.plot(np.mod(totalAzSamples[azSample_i]*180/np.pi, 360), azSample, marker='o', ms=20., c='b')
#     #
#     #     scaleAzSamples = np.array(errorFunGaussianAzSamplesPred)
#     #     scaleAzSamples = scaleAzSamples - np.min(scaleAzSamples) + 1
#     #     scaleAzSamples = scaleAzSamples*0.75*y2/np.max(scaleAzSamples)
#     #     for azSample_i, azSample in enumerate(scaleAzSamples):
#     #         plt.plot(np.mod(totalAzSamples[azSample_i]*180/np.pi, 360), azSample, marker='o', ms=20., c='y')
#
#     plt.axis((0,360,y1,y2))
#     plt.title('Neuralnet multiple predictions')
#     fig.savefig(figurePath + 'sample' + '.png', bbox_inches='tight')
#     plt.close(fig)
#
#     chAz[:] = currentAz
#     cv2.imwrite(figurePath + '_render.png', cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))

#Fit:
print("Fitting predictions")

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

negLikModel = -ch.sum(generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances))/ numPixels
negLikModelRobust = -ch.sum(generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances))/ numPixels
pixelLikelihoodCh = generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances)
pixelLikelihoodRobustCh = generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances)
# pixelLikelihoodRobustSQErrorCh = generative_models.LogRobustSQErrorModel(sqeRenderer, foregroundPrior=globalPrior, variances=variances)

post = generative_models.layerPosteriorsRobustCh(rendererGT, renderer, np.array([]), 'FULL', globalPrior, variances)[0]
# postSqerror = generative_models.layerPosteriorsRobustSQErrorCh(sqeRenderer, np.array([]), 'MASK', globalPrior, variances)[0].r>0.5

# negLikModelRobustSQError = generative_models.NLLRobustSQErrorModel(sqeRenderer=sqeRenderer, Q=globalPrior.r*np.ones([height, width]),variances=variances) / numPixels

models = [negLikModel, negLikModelRobust]
pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh]
modelsDescr = ["Gaussian Model", "Outlier model" ]

errorFun = models[model]

testRangeStr = str(testSet[0]) + '-' + str(testSet[-1])
testDescription = 'photorealistic-replicateopendr_server-' + testRangeStr
testPrefix = experimentPrefix + '_' + testDescription + '_' + optimizationTypeDescr[optimizationType] + '_' + str(len(testSet)) + 'samples_'

testPrefixBase = testPrefix

runExp = True
shapePenaltyTests = [0,0,0,0]
# shapePenaltyTests = [0,0,0,0]
stdsTests = [0.03]
# stdsTests = [0.03]
modelTests = len(stdsTests)*[1]
# modelTests = [1]
methodTests = len(stdsTests)*[1]

maxOptIters = len(stdsTests)*[40]

if makeVideo:
    plt.ioff()
    import matplotlib.animation as animation
    Writer = animation.writers['ffmpeg']

    writer = Writer(fps=1, metadata=dict(title='Fitting process', artist=''), bitrate=1800)
    figvid, ((vax1, vax2, vax3, vax4), (vax5, vax6, vax7, vax8)) = plt.subplots(2, 4, figsize=(12, 5))

    figvid.delaxes(vax8)

    vax1.axes.get_xaxis().set_visible(False)
    vax1.axes.get_yaxis().set_visible(False)
    vax1.set_title("Ground truth")

    vax2.axes.get_xaxis().set_visible(False)
    vax2.axes.get_yaxis().set_visible(False)
    vax2.set_title("Recognition")

    vax3.axes.get_xaxis().set_visible(False)
    vax3.axes.get_yaxis().set_visible(False)
    vax3.set_title("Fit")

    vax4.axes.get_xaxis().set_visible(False)
    vax4.axes.get_yaxis().set_visible(False)
    vax4.set_title("Posterior")

    vax5.axes.get_xaxis().set_visible(False)
    vax5.axes.get_yaxis().set_visible(False)
    vax5.set_title("Env Map GT")

    vax6.axes.get_xaxis().set_visible(False)
    vax6.axes.get_yaxis().set_visible(False)
    vax6.set_title("Env Map Recognition")

    vax7.axes.get_xaxis().set_visible(False)
    vax7.axes.get_yaxis().set_visible(False)
    vax7.set_title("Env Map Fit")

    plt.tight_layout()
    vidImgs = []

nearestNeighbours = False

if nearestNeighbours:
    trainImages = readImages(imagesDir, trainIds, loadFromHdf5)
    trainImagesR = trainImages.reshape([len(trainImages), -1])

methodsPred = ["Mean Baseline", "Nearest Neighbours", "Recognition", "Fit" ]
plotColors = ['k', 'm', 'b', 'r']

segmentVColorError = np.array([])
useSegmentation = True

segmentVColorsList = []
global annot_t
annot_t = None

def cb(_):
    global t
    global samplingMode
    elapsed_time = time.time() - t
    print("Ended interation in  " + str(elapsed_time))
    # if samplingMode:
    #     analyzeAz(resultDir + 'az_samples/test' + str(test_i) +'/azNum' + str(sampleAzNum) + '_it' + str(iterat)  , rendererGT, renderer, chEl.r, chVColors.r, chLightSHCoeffs.r, azsPredictions[test_i], sampleStds=stds.r)
    # else:
    #     analyzeAz(resultDir + 'az_samples/test' + str(test_i) +'/min_azNum' + str(sampleAzNum) +  '_it' + str(iterat)  , rendererGT, renderer, chEl.r, chVColors.r, chLightSHCoeffs.r, azsPredictions[test_i], sampleStds=stds.r)
    global pixelErrorFun
    global errorFun
    global iterat
    iterat = iterat + 1
    print("Callback! " + str(iterat))
    print("Sq Error: " + str(errorFun.r))
    if chThError is not None:
        print("Theano Error: " + str(chThError.r))

    global negLikModelRobustSmallStd
    global bestShapeParamsSmallStd
    global bestRobustSmallStdError
    # if minimizingShape and useShapeModel:
    #     # if negLikModelRobustSmallStd.r  < bestRobustSmallStdError:
    #     #     bestRobustSmallStdError = negLikModelRobustSmallStd.r.copy()
    #     #     bestShapeParamsSmallStd = chShapeParams.r.copy()
    #     maxShapeSize = 2.5
    #     largeShapeParams = np.abs(chShapeParams.r) > maxShapeSize
    #     if np.any(largeShapeParams):
    #         print("Warning: found large shape parameters to fix!")
    #     chShapeParams[largeShapeParams] = np.sign(chShapeParams.r[largeShapeParams])*maxShapeSize

    if getColorFromCRF:
        global chVColors
        global Q
        global color
        segmentation = np.argmax(Q.r, axis=0).reshape(renderer.r.shape[:2])
        if np.sum(segmentation == 0) == 0:
            vColor = color
        else:
            segmentRegion = segmentation == 0
            vColor = np.median(rendererGT.reshape([-1, 3])[segmentRegion.ravel()], axis=0) * 1.4
            vColor = vColor / max(np.max(vColor), 1.)
            vis_im = np.array(renderer.indices_image == 1).copy().astype(np.bool)
            chVColors[:] = vColor
            colorRegion = np.all(renderer.r != 0,axis=2).ravel() * segmentRegion.ravel() * vis_im.ravel()
            vColor = vColor * np.mean(
                rendererGT.r.reshape([-1, 3])[colorRegion] / renderer.r.reshape([-1, 3])[colorRegion])

        chVColors[:] = vColor

    if makeVideo:
        global im1
        global im2
        global vidImgs
        global writer
        global writer_i
        global rendererRecognition

        plt.figure(figvid.number)
        im1 = vax1.imshow(lin2srgb(rendererGT.r.copy()))
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8)
        im2 = vax2.imshow(lin2srgb(rendererRecognition.copy()))

        im3 = vax3.imshow(lin2srgb(renderer.r.copy()))

        stdsOld = stds.r
        stds[:] = 0.05
        vis_im = np.array(renderer.indices_image==1).copy().astype(np.bool)

        post = generative_models.layerPosteriorsRobustCh(rendererGT, renderer, vis_im, 'MASK', globalPrior, variances)[0].r>0.5
        stds[:] = stdsOld

        im4 = vax4.imshow(post.copy())
        # plt.colorbar(im4, ax=vax4, use_gridspec=True)

        pEnvMap = SHProjection(envMapTexture, np.concatenate([chLightSHCoeffs.r[:,None], chLightSHCoeffs.r[:,None], chLightSHCoeffs.r[:,None]], axis=1))
        approxProjectionFitted = np.sum(pEnvMap, axis=(2,3))
        # approxProjectionFitted[approxProjectionFitted<0] = 0
        #
        # approxProjectionGT[approxProjectionGT<0] = 0
        # approxProjectionPred[approxProjectionPred<0] = 0

        # cv2.imwrite(resultDir + 'approxProjectionGT.jpeg' , 255*np.concatenate([approxProjectionGT[...,None], approxProjectionGT[...,None], approxProjectionGT[...,None]], axis=2)[:,:,[2,1,0]])
        # cv2.imwrite(resultDir + 'approxProjectionPred.jpeg' , 255*np.concatenate([approxProjectionPred[...,None], approxProjectionPred[...,None], approxProjectionPred[...,None]], axis=2)[:,:,[2,1,0]])
        # cv2.imwrite(resultDir + 'approxProjectionFitted.jpeg' , 255*np.concatenate([approxProjectionFitted[...,None], approxProjectionFitted[...,None], approxProjectionFitted[...,None]], axis=2)[:,:,[2,1,0]])

        cv2.imwrite(resultDir + 'approxProjectionGT.jpeg' , 255*np.sum(pEnvMapGT, axis=3)[:,:,[2,1,0]])
        cv2.imwrite(resultDir + 'approxProjectionPred.jpeg' , 255*np.sum(pEnvMapPred, axis=3)[:,:,[2,1,0]])
        cv2.imwrite(resultDir + 'approxProjectionFitted.jpeg' , 255*np.sum(pEnvMap, axis=3)[:,:,[2,1,0]])

        approxProjectionGTlocal = skimage.io.imread(resultDir +'approxProjectionGT.jpeg').astype(np.float32)/255.
        approxProjectionPredlocal = skimage.io.imread(resultDir +'approxProjectionPred.jpeg').astype(np.float32)/255.
        approxProjectionFittedlocal = skimage.io.imread(resultDir +'approxProjectionFitted.jpeg').astype(np.float32)/255.

        approxProjectionGTlocal = skimage.transform.resize(approxProjectionGTlocal, [75,150])
        approxProjectionPredlocal = skimage.transform.resize(approxProjectionPredlocal, [75,150])
        approxProjectionFittedlocal = skimage.transform.resize(approxProjectionFittedlocal, [75,150])

        im5 = vax5.imshow(approxProjectionGTlocal.copy())

        im6 = vax6.imshow(approxProjectionPredlocal.copy())

        im7 = vax7.imshow(approxProjectionFittedlocal.copy())

        # if annot_t is not None:
        #     annot_t.remove()
        annot_t = vax3.annotate("Fitting iter: " + str(iterat), xy=(1, 0), xycoords='axes fraction', fontsize=16,
                     xytext=(-20, 5), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props)


        plt.tight_layout()

        vidImgs.append([im1,im2, im3,im4, im5, im6, im7, annot_t])

        if iterat == 1:
            vidImgs.append([im1,im2, im3,im4, im5, im6, im7, annot_t])
            vidImgs.append([im1,im2, im3,im4, im5, im6, im7, annot_t])

        # figvid.savefig(resultDir + 'videos/' + 'lastFig.png')

        # writer_i.grab_frame()

    global imagegt
    global gradAz
    global gradEl
    global performance
    global azimuths
    global elevations
    global shapeParams

    t = time.time()

replaceExisting = False

for testSetting, model in enumerate(modelTests):
    model = modelTests[testSetting]
    method = methodTests[testSetting]
    shapePenalty = shapePenaltyTests[testSetting]
    stds[:] = stdsTests[testSetting]

    testPrefix = testPrefixBase + '_method' + str(method)  + 'errorFun' + str(model) + '_std' + str(stds.r[0])  + '_shapePen'+ str(shapePenalty)

    resultDir = 'results/' + testPrefix + '/'

    if not os.path.exists(resultDir + 'imgs/'):
        os.makedirs(resultDir + 'results/')

    if not os.path.exists(resultDir + 'imgs/'):
        os.makedirs(resultDir + 'imgs/')
    if not os.path.exists(resultDir +  'imgs/samples/'):
        os.makedirs(resultDir + 'imgs/samples/')

    ## NN individual prediction samples analysis.
    if not os.path.exists(resultDir + 'nn_samples/'):
        os.makedirs(resultDir + 'nn_samples/')

    if not os.path.exists(resultDir + 'az_samples/'):
        os.makedirs(resultDir + 'az_samples/')

    if not os.path.exists(resultDir + 'hue_samples/'):
        os.makedirs(resultDir + 'hue_samples/')

    if makeVideo:
        if not os.path.exists(resultDir + 'videos/'):
            os.makedirs(resultDir + 'videos/')

    azimuths = []
    elevations = []
    vColors = []
    lightCoeffs = []
    approxProjections = []
    likelihoods = []
    shapeParams = []
    posteriors = []

    startTime = time.time()
    samplingMode = False

    fittedVColorsList = []
    fittedRelLightCoeffsList = []

    approxProjectionsFittedList = []
    approxProjectionsGTList = []
    approxProjectionsPredList = []

    fittedAzs = np.array([])
    fittedElevs = np.array([])
    fittedRelLightCoeffs = []
    fittedShapeParamsList = []
    fittedVColors = []
    fittedPosteriorsList = []

    revertedSamples = np.array([])

    errorFunAzSamples = []
    errorFunAzSamplesPred = []
    errorFunGaussianAzSamplesPred = []
    errorFunGaussianAzSamples = []

    if nearestNeighbours:
        azimuthNearestNeighboursList = []
        elevationNearestNeighboursList = []
        vColorNearestNeighboursList = []
        shapeParamsNearestNeighboursList = []
        lightCoeffsNearestNeighboursList = []
        nearestNeighboursErrorFuns = np.array([])
        approxProjectionsNearestNeighbourList = []

        nearestNeighboursPosteriorsList = []

    errorsShapeVerticesSoFar = np.array([])

    predictedErrorFuns = np.array([])
    predictedPosteriorsList = []

    if includeMeanBaseline:
        meanBaselineErrorFuns = np.array([])
        meanBaselinePosteriorList = []

    errorsFittedShapeVertices = np.array([])
    fittedErrorFuns = np.array([])
    fittedShapeParams = np.array([])

    if (computePredErrorFuns and optimizationType == 0) or optimizationType != 0:
        for test_i in range(len(testAzsRel)):
            print("************** Minimizing loss of prediction " + str(test_i) + "of " + str(len(testAzsRel)))

            resultDir = 'results/' + testPrefix + '/results/'
            testDir = resultDir + str(test_i) + '/'

            if not os.path.exists(testDir):
                os.makedirs(testDir)

            if not replaceExisting and os.path.isfile(testDir + 'fitted_' + 'shapeParams'+ 'npy.npy'):
                continue

            bestFittedAz = chAz.r
            bestFittedEl = chEl.r
            bestModelLik = np.finfo('f').max
            bestVColors = chVColors.r
            bestLightSHCoeffs = chLightSHCoeffs.r
            if useShapeModel:
                bestShapeParams = chShapeParams.r

            testId = testIds[test_i]


            image = skimage.transform.resize(images[test_i], [height,width])
            imageSrgb = image.copy()
            rendererGT[:] = srgb2lin(image)

            # sqeRenderer.imageGT = ch.Ch(image)

            negLikModel = -ch.sum(generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances))/numPixels
            # negLikModelRobust = -ch.sum(generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances))/numPixels
            # negLikModelRobust = generative_models.NLLRobustModel(renderer=renderer, groundtruth=rendererGT, Q=globalPrior.r*np.ones([height, width]),variances=variances) / numPixels

            negLikModelRobust = -ch.sum(generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances))/ numPixels

            # negLikModelRobustSQError = generative_models.NLLRobustSQErrorModel(sqeRenderer=sqeRenderer, Q=globalPrior.r*np.ones([height, width]),
            #                                                                     variances=variances) / numPixels

            models = [negLikModel, negLikModelRobust]

            stds[:] = stdsTests[testSetting]

            if makeVideo:
                writer_i = Writer(fps=1, metadata=dict(title='', artist=''), bitrate=1800)
                writer_i.setup(figvid, resultDir + 'videos/vid_'+ str(test_i) + '.mp4', dpi=70)
                vidImgs = []


            if nearestNeighbours:
                onenn_i = one_nn(trainImagesR, imageSrgb.ravel())
                nearesTrainImage = trainImages[onenn_i]
                cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/id' + str(testId) +'_nearestneighbour' + '.png', cv2.cvtColor(np.uint8(nearesTrainImage[0].copy()*255), cv2.COLOR_RGB2BGR))
                azimuthNearestNeighbour = trainAzsRel[onenn_i]
                elevationNearestNeighbour = trainElevsGT[onenn_i]
                vColorNearestNeighbour = trainVColorGT[onenn_i]
                if useShapeModel:
                    shapeParamsNearestNeighbour = trainShapeModelCoeffsGT[onenn_i]
                lightCoeffsNearestNeighbour = trainLightCoefficientsGTRel[onenn_i]

                azimuthNearestNeighboursList = azimuthNearestNeighboursList + [azimuthNearestNeighbour]
                elevationNearestNeighboursList = elevationNearestNeighboursList + [elevationNearestNeighbour]
                vColorNearestNeighboursList = vColorNearestNeighboursList + [vColorNearestNeighbour]
                shapeParamsNearestNeighboursList = shapeParamsNearestNeighboursList + [shapeParamsNearestNeighbour]
                lightCoeffsNearestNeighboursList = lightCoeffsNearestNeighboursList + [lightCoeffsNearestNeighbour]

                chAz[:] = azimuthNearestNeighbour
                chEl[:] = elevationNearestNeighbour
                chVColors[:] = vColorNearestNeighbour
                chLightSHCoeffs[:] = lightCoeffsNearestNeighbour
                if useShapeModel:
                    chShapeParams[:] = shapeParamsNearestNeighbour

                nearestNeighboursErrorFuns = np.append(nearestNeighboursErrorFuns, errorFun.r)

                # nearestNeighboursPosteriorsList = nearestNeighboursPosteriorsList + [np.array(renderer.indices_image==1).copy().astype(np.bool)]

            if includeMeanBaseline:
                chAz[:] = meanTrainAzimuthRel[test_i]
                chEl[:] = meanTrainElevation[test_i]
                chVColors[:] = meanTrainVColors[test_i]
                chLightSHCoeffs[:] = meanTrainLightCoefficientsGTRel[test_i]
                if useShapeModel:
                    chShapeParams[:] = meanTrainShapeParams[test_i]

                # meanBaselinePosteriorList = meanBaselinePosteriorList + [np.array(renderer.indices_image==1).copy().astype(np.bool)[None,:]]
                meanBaselineErrorFuns = np.append(meanBaselineErrorFuns, errorFun.r)


            stdsSmall = ch.Ch([0.01])
            variancesSmall = stdsSmall ** 2
            negLikModelRobustSmallStd = -ch.sum(generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variancesSmall))/numPixels

            if not os.path.exists(resultDir + 'imgs/test'+ str(test_i) + '/'):
                os.makedirs(resultDir + 'imgs/test'+ str(test_i) + '/')

            if not os.path.exists(resultDir + 'imgs/test'+ str(test_i) + '/crf/'):
                os.makedirs(resultDir + 'imgs/test'+ str(test_i) + '/crf/')

            if not os.path.exists(resultDir + 'imgs/crf/'):
                os.makedirs(resultDir + 'imgs/crf/')

            if not os.path.exists(resultDir + 'imgs/test'+ str(test_i) + '/SH/'):
                os.makedirs(resultDir + 'imgs/test'+ str(test_i) + '/SH/')

            cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/id' + str(testId) +'_groundtruth' + '.png', cv2.cvtColor(np.uint8(lin2srgb(rendererGT.r.copy())*255), cv2.COLOR_RGB2BGR))

            for sample in range(testSamples):

                if recognitionType == 0:
                    #Prediction from (near) ground truth.

                    color = testVColorGT[test_i] + nearGTOffsetVColor
                    az = testAzsRel[test_i] + nearGTOffsetRelAz
                    el = testElevsGT[test_i] + nearGTOffsetEl
                    lightCoefficientsRel = testLightCoefficientsGTRel[test_i]

                    if useShapeModel:
                        shapeParams = testShapeParamsGT[test_i]
                elif recognitionType == 1 or recognitionType == 2:

                    #Recognition estimate:
                    az = azsPred[test_i]
                    el = min(max(elevsPred[test_i],radians(1)), np.pi/2-radians(1))
                    color = vColorsPred[test_i]
                    lightCoefficientsRel = relLightCoefficientsPred[test_i]
                    if useShapeModel:
                        shapeParams = shapeParamsPred[test_i]

                chAz[:] = az
                chEl[:] = el
                chVColors[:] = color
                chLightSHCoeffs[:] = lightCoefficientsRel
                if useShapeModel:
                    chShapeParams[:] = shapeParams


                rendererRecognition = renderer.r.copy()

                cv2.imwrite(resultDir + 'imgs/test' + str(test_i) + '/sample' + str(sample) + '_predicted' + '.png',  cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy()) * 255), cv2.COLOR_RGB2BGR))

                np.save(testDir + 'pred_' + 'az' + 'npy', az)
                np.save(testDir + 'pred_' + 'el'+ 'npy', el)
                np.save(testDir + 'pred_' + 'color'+ 'npy', color)
                np.save(testDir + 'pred_' + 'lightCoefficientsRel'+ 'npy', lightCoefficientsRel)
                np.save(testDir + 'pred_' + 'shapeParams'+ 'npy', shapeParams)

                if not os.path.exists(resultDir + 'az_samples/'):
                    os.makedirs(resultDir + 'az_samples/')
                #
                # analyzeAz(resultDir + 'az_samples/test' + str(test_i), rendererGT, renderer, chEl.r, chVColors.r, chLightSHCoeffs.r, azsPredictions[test_i], sampleStds=stds.r)


                # Dense CRF prediction of labels:

                vis_im = np.array(renderer.indices_image==1).copy().astype(np.bool)
                bound_im = renderer.boundarybool_image.astype(np.bool)

                # if optimizationTypeDescr[optimizationType] != 'optimize':
                #     import densecrf_model
                #     #
                #     # plt.imsave(resultDir + 'imgs/test'+ str(test_i) + '/crf/renderer', renderer.r)
                #     #
                #     sample_i = test_i
                #     segmentation, Q = densecrf_model.crfInference(rendererGT.r, vis_im, bound_im, [0.75,0.25,0.01], resultDir + 'imgs/crf/Q_' + str(sample_i))
                # if np.sum(segmentation==0) == 0:
                #     segmentVColorsList = segmentVColorsList + [color]
                # else:
                #
                #     segmentRegion = segmentation==0
                #     segmentColor = np.median(rendererGT.reshape([-1,3])[segmentRegion.ravel()], axis=0)
                #     segmentVColorsList = segmentVColorsList + [segmentColor]
                #

                # optLightSHCoeffs = chLightSHCoeffs.r

                if evaluateWithGT:
                    hdridx = dataEnvMaps[test_i]

                    envMapTexture = np.zeros([360,180,3])
                    envMapFound = False
                    for hdrFile, hdrValues in hdritems:
                        if hdridx == hdrValues[0]:

                            envMapCoeffsGT = hdrValues[1]
                            envMapFilename = hdrFile

                            try:
                                envMapTexture = np.array(imageio.imread(envMapFilename))[:,:,0:3]
                            except:
                                envMapTexture = np.zeros([360,180,3])

                            envMapTexture = cv2.resize(src=envMapTexture, dsize=(360,180))
                            envMapFound = True
                            break

                    if not envMapFound:
                        ipdb.set_trace()

                    pEnvMapGT = SHProjection(envMapTexture, np.concatenate([testLightCoefficientsGTRel[test_i][:,None], testLightCoefficientsGTRel[test_i][:,None], testLightCoefficientsGTRel[test_i][:,None]], axis=1))
                    approxProjectionGT = np.sum(pEnvMapGT, axis=(2,3))
                    approxProjectionsGTList = approxProjectionsGTList + [approxProjectionGT[None,:]]

                    cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/SH/' + str(hdridx) + '_GT.jpeg' , 255*np.sum(pEnvMapGT, axis=3)[:,:,[2,1,0]])

                    pEnvMapPred = SHProjection(envMapTexture, np.concatenate([relLightCoefficientsPred[test_i][:,None], relLightCoefficientsPred[test_i][:,None], relLightCoefficientsPred[test_i][:,None]], axis=1))
                    approxProjectionPred = np.sum(pEnvMapPred, axis=(2,3))

                    approxProjectionsPredList = approxProjectionsPredList + [approxProjectionPred[None,:]]

                    predictedPosteriorsList = predictedPosteriorsList + [np.array(renderer.indices_image==1).copy().astype(np.bool)[None,:]]

                    if nearestNeighbours:
                        pEnvMap = SHProjection(envMapTexture, np.concatenate([lightCoeffsNearestNeighbour.ravel()[:,None], lightCoeffsNearestNeighbour.ravel()[:,None], lightCoeffsNearestNeighbour.ravel()[:,None]], axis=1))
                        approxProjectionNearestNeighbour = np.sum(pEnvMap, axis=(2,3))
                        approxProjectionsNearestNeighbourList = approxProjectionsNearestNeighbourList + [approxProjectionNearestNeighbour[None,:]]
                        cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/SH/' + str(hdridx) + '_NearestNeighbour.jpeg' , 255*np.sum(pEnvMap, axis=3)[:,:,[2,1,0]])

                    cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/SH/' + str(hdridx) + '_Pred.jpeg' , 255*np.sum(pEnvMapPred, axis=3)[:,:,[2,1,0]])

                ## RecognitionType =2 : Use samples from neural net to explore the space better.
                if recognitionType == 2:
                    errorFunAzSamples = []
                    errorFunAzSamplesPred = []
                    errorFunGaussianAzSamplesPred = []
                    errorFunGaussianAzSamples = []

                    if not os.path.exists(resultDir + 'az_samples/test' + str(test_i) + '/'):
                        os.makedirs(resultDir + 'az_samples/test' + str(test_i) + '/')
                    if not os.path.exists(resultDir + 'hue_samples/test' + str(test_i) + '/'):
                        os.makedirs(resultDir + 'hue_samples/test' + str(test_i) + '/')
                    samplingMode = True
                    cv2.imwrite(resultDir + 'az_samples/test' + str(test_i)  + '_gt.png', cv2.cvtColor(np.uint8(lin2srgb(rendererGT.r.copy())*255), cv2.COLOR_RGB2BGR))
                    # stds[:] = 0.1
                    #
                    # azSampleStdev = np.std(azsPredictions[test_i])
                    azSampleStdev = np.sqrt(-np.log(np.min([np.mean(sinAzsPredSamples[test_i])**2 + np.mean(cosAzsPredSamples[test_i])**2,1])))
                    predAz = chAz.r
                    numSamples = max(int(np.ceil(azSampleStdev*180./(np.pi*25.))),1)
                    azSamples = np.linspace(0, azSampleStdev, numSamples)
                    totalAzSamples = predAz + np.concatenate([azSamples, -azSamples[1:]])
                    sampleAzNum  = 0

                    # model = 1
                    # errorFun = models[model]

                    bestPredAz = chAz.r
                    # bestPredEl = chEl.r
                    bestPredEl = min(max(chEl.r.copy(),radians(1)), np.pi/2-radians(1))
                    bestPredVColors = chVColors.r.copy()
                    bestPredLightSHCoeffs = chLightSHCoeffs.r.copy()
                    if useShapeModel:
                        bestPredShapeParams = chShapeParams.r.copy()
                    bestModelLik = np.finfo('f').max
                    bestPredModelLik = np.finfo('f').max

                    # analyzeAz(resultDir + 'az_samples/test' + str(test_i) +'/pre'  , rendererGT, renderer, chEl.r, chVColors.r, chLightSHCoeffs.r, azsPredictions[test_i], sampleStds=stds.r)

                    # analyzeHue(resultDir + 'hue_samples/test' + str(test_i) +'/pre', rendererGT, renderer, chEl.r, chAz.r, chLightSHCoeffs.r, vColorsPredSamples[test_i], sampleStds=stds.r)

                    for sampleAz in totalAzSamples:
                        iterat = 0
                        sampleAzNum += 1

                        chAz[:] = sampleAz
                        # chEl[:] = elsample
                        print("Minimizing first step")
                        model = 1
                        errorFun = models[model]
                        # method = 5

                        chLightSHCoeffs[:] = lightCoefficientsRel
                        chVColors[:] = color
                        #Todo test with adding chEl.
                        free_variables = [chLightSHCoeffs]
                        options={'disp':False, 'maxiter':2}

                        errorFunAzSamplesPred = errorFunAzSamplesPred + [errorFun.r]
                        errorFunGaussianAzSamplesPred = errorFunGaussianAzSamplesPred + [models[0].r]

                        # analyzeAz(resultDir + 'az_samples/test' + str(test_i) +'/azNum' + str(sampleAzNum), rendererGT, renderer, chEl.r, chVColors.r, chLightSHCoeffs.r, azsPredictions[test_i], sampleStds=stds.r)

                        if models[1].r.copy() < bestPredModelLik:

                            print("Found best angle!")
                            # bestPredModelLik = errorFun.r.copy()
                            bestPredModelLik = models[1].r.copy()
                            bestPredAz = sampleAz
                            bestPredEl = min(max(chEl.r.copy(),radians(1)), np.pi/2-radians(1))
                            bestPredVColors = chVColors.r.copy()
                            bestPredLightSHCoeffs = chLightSHCoeffs.r.copy()
                            if useShapeModel:
                                bestPredShapeParams = chShapeParams.r.copy()
                            bestModelLik = errorFun.r.copy()

                            # cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/best_predSample' + str(numPredSamples) + '.png', cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))

                        errorFunAzSamples = errorFunAzSamples + [errorFun.r]
                        errorFunGaussianAzSamples = errorFunGaussianAzSamples + [models[0].r]

                    color = bestPredVColors
                    lightCoefficientsRel = bestPredLightSHCoeffs
                    az = bestPredAz
                    if useShapeModel:
                        shapeParams = bestShapeParams

                    # previousAngles = np.vstack([previousAngles, np.array([[azsample, elsample],[chAz.r.copy(), chEl.r.copy()]])])

                    samplingMode = False
                    chAz[:] = az


                chAz[:] = az

                chEl[:] = min(max(el,radians(1)), np.pi/2-radians(1))
                chVColors[:] = color.copy()
                chLightSHCoeffs[:] = lightCoefficientsRel.copy()
                if useShapeModel:
                    chShapeParams[:] = shapeParams.copy()

                cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/best_sample' + '.png', cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))

                predictedErrorFuns = np.append(predictedErrorFuns, errorFun.r)

                # analyzeAz(resultDir + 'az_samples/test' + str(test_i) + '_samples', rendererGT, renderer, chEl.r, color, lightCoefficientsRel,
                #           azsPredictions[test_i], sampleStds=stds.r)

                iterat = 0

                sampleAzNum = 0

                sys.stdout.flush()

                ## Finally: Do Fitting!
                ignore = False
                reverted = False

                if optimizationTypeDescr[optimizationType] == 'optimize':

                    #Get VCOLOR using CRF:

                    print("** Minimizing from initial predicted parameters. **")
                    globalPrior[:] = 0.9

                    errorFun = models[model]
                    vColor = color
                    if useCRFOcclusionPred:
                        priorProbs = [0.75, 0.25, 0.01]
                        test_sample = test_i
                        Q = recognition_models.segmentCRFModel(renderer=renderer, groundtruth=rendererGT,
                                                               priorProbs=priorProbs, resultDir=resultDir,
                                                               test_i=test_sample)

                        getColorFromCRF = False
                        segmentation = np.argmax(Q.r, axis=0).reshape(renderer.r.shape[:2])
                        if np.sum(segmentation == 0) == 0:
                            ignore = True
                            vColor = color
                        else:
                            segmentRegion = segmentation == 0
                            vColor = np.median(rendererGT.reshape([-1, 3])[segmentRegion.ravel()], axis=0) * 1.4
                            vColor = vColor / max(np.max(vColor), 1.)

                            vis_im = np.array(renderer.indices_image == 1).copy().astype(np.bool)
                            chVColors[:] = vColor
                            colorRegion = np.all(renderer.r != 0,axis=2).ravel() * segmentRegion.ravel() * vis_im.ravel()
                            if colorRegion.sum() == 0:
                                ignore = True
                            else:
                                vColor = vColor * np.median(rendererGT.r.reshape([-1, 3])[colorRegion] / renderer.r.reshape([-1, 3])[colorRegion])

                    if not ignore:
                        color = vColor
                        chVColors[:] = vColor

                        ## Bayes Optimization

                        ### Bayesian Optimization tests.
                        if useCRFOcclusionPred:
                            free_variables = [ chAz, chEl]
                            free_variables_app_light = [ chLightSHCoeffs]

                            azSampleStdev = np.sqrt(-np.log(np.min([np.mean(np.sin(azsPredictions)[test_i])**2 + np.mean(np.cos(azsPredictions)[test_i])**2,1])))

                            azMean = azsPred[test_i]
                            elMean = elevsPred[test_i]

                            elSampleStdev = np.sqrt(-np.log(np.min([np.mean(np.sin(elevsPredictions)[test_i])**2 + np.mean(np.cos(elevsPredictions)[test_i])**2,1])))
                            elBound = np.min(elSampleStdev, np.pi/4)

                            elMean = min(max(elevsPred[test_i],radians(1)), np.pi/2-radians(1))

                            shapeStdevs = np.min(np.vstack([2*np.sqrt(np.cov(shapeParamsPredSamples[test_i]).diagonal()), np.ones([latentDim])*3.5]), 0)

                            azBound = min(azSampleStdev, np.pi)
                            optBounds = [(azMean - azBound, azMean + azBound), (min(max(elMean - elBound,0), np.pi/2), min(max(elMean + elBound,0), np.pi/2))] + [(max(shapeParams[shape_i] - shapeStdevs[shape_i],-3.5), min(shapeParams[shape_i] + shapeStdevs[shape_i],3.5)) for shape_i in range(len(shapeStdevs))]
                            optBounds = [(azMean - azBound, azMean + azBound), (min(max(elMean - elBound,0), np.pi/2), min(max(elMean + elBound,0), np.pi/2))]

                            numSamples = max(int(np.ceil(2*azSampleStdev*180./(np.pi*20.))),1)
                            # azSamples = np.linspace(0, azSampleStdev, numSamples)
                            azSamples = np.linspace(0, np.pi, 10)

                            numSamples = max(int(np.ceil(2*elSampleStdev*180./(np.pi*10.))),1)
                            elSamples = np.linspace(0, 2*elSampleStdev, numSamples)

                            elSamples = chEl.r + np.concatenate([elSamples, -elSamples[1:]])

                            totalAzSamples = chAz.r + np.concatenate([azSamples, -azSamples[1:]])
                            if len(totalAzSamples) == 0:
                                totalAzSamples = np.array([chAz.r])

                            if azSampleStdev < 20*np.pi/180:
                                totalAzSamples = np.array([chAz.r])

                            totalElSamples = np.array([elSample for elSample in elSamples if elSample > 0 and elSample < np.pi/2])
                            if len(totalElSamples) == 0:
                                totalElSamples = np.array([chEl.r])

                            totalElSamples = np.array([chEl.r])


                            # totalElSamples = np.array([testElevsGT[test_i]])
                            # chShapeParams[:] = testShapeParamsGT[test_i]
                            # chLightSHCoeffs[:] = testLightCoefficientsGTRel[test_i]
                            # chVColors[:] = testVColorGT[test_i]
                            # vColor = testVColorGT[test_i]
                            # lightCoefficientsRel = testLightCoefficientsGTRel[test_i]

                            totalSamples = np.meshgrid(totalAzSamples, totalElSamples)
                            totalSamples = np.hstack([totalSamples[0].reshape([-1,1]),totalSamples[1].reshape([-1,1])])

                            import diffrender_opt
                            # objFun = diffrender_opt.opendrObjectiveFunction(errorFun, free_variables)

                            # res = diffrender_opt.bayesOpt(objFun, objBounds)
                            methodOpt = methods[method]
                            stds[:] = 0.1

                            # crfObjFun = diffrender_opt.opendrObjectiveFunctionCRF(free_variables, rendererGT, renderer, vColor, chVColors, chLightSHCoeffs, lightCoefficientsRel, free_variables_app_light, resultDir, test_i, stds.r, methodOpt, False, False)

                            cv2.imwrite(
                                resultDir + 'imgs/test' + str(test_i) + '/sample' + str(sample) + '_predictedCRF' + '.png',
                                cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy()) * 255), cv2.COLOR_RGB2BGR))


                            # errorFunCRF = -ch.sum(generative_models.LogCRFModel(renderer=renderer, groundtruth=rendererGT, Q=Q.r.reshape([3,height, width]),
                            #                                                     variances=variances)) / numPixels
                            errorFunNLLCRF = generative_models.NLLCRFModel(renderer=renderer, groundtruth=rendererGT, Q=Q.r.reshape([3,height, width]),
                                                                                variances=variances) / numPixels

                            # errorFun = models[1]
                            # samplesEvals = []
                            # for azSample in totalSamples[:,0]:
                            #     chAz[:] = azSample
                            #     samplesEvals = samplesEvals + [errorFunNLLCRF.r]
                            #
                            # samplesEvals = np.array(samplesEvals)
                            #
                            # # samplesEvals = crfObjFun(totalSamples)
                            #
                            # #Start Optimiaztion
                            #
                            # # res = diffrender_opt.bayesOpt(crfObjFun, totalSamples, samplesEvals, optBounds)
                            # #
                            # # optAz = res.x_opt[0]
                            # # optEl = res.x_opt[1]
                            # # # optShapeParams = res.x_opt[2:]
                            # bestSample = np.argmin(samplesEvals)
                            # optAz = totalSamples[bestSample][0]
                            # optEl = totalSamples[bestSample][1]
                            #
                            # azfig = plt.figure()
                            # ax = azfig.add_subplot(111)
                            # ax.plot(np.mod(totalSamples[:,0], 2*np.pi), samplesEvals, 'ro')
                            # y1, y2 = ax.get_ylim()
                            # ax.vlines(testAzsRel[test_i], y1, y2, 'g', label = 'Groundtruth Az')
                            # ax.vlines(np.mod(optAz, 2*np.pi), y1, y2, 'r', label = 'Optimum CRF Az')
                            # ax.vlines(np.mod(az, 2*np.pi), y1, y2, 'b', label = 'Recognition Az')
                            # ax.set_xlim((0, np.pi*2))
                            # legend = ax.legend()
                            # azfig.savefig(resultDir + 'imgs/test' + str(test_i) + '/samplesAzPlot_' + str(int(azSampleStdev*180/np.pi)) + '.png')
                            # plt.close(azfig)
                            #
                            # legend = ax.legend()
                            #
                            # chAz[:] = optAz
                            # chEl[:] = optEl
                            # #### Local search:
                            #
                            # cv2.imwrite(
                            #     resultDir + 'imgs/test' + str(test_i) + '/sample' + str(sample) + '_optimizedCRFPose' + '.png',
                            #     cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy()) * 255), cv2.COLOR_RGB2BGR))

                            free_variables = [chShapeParams, chAz, chEl, chLightSHCoeffs, chVColors]
                            #
                            # stds[:] = 0.03
                            # shapePenalty = 0.0001

                            stds[:] = stdsTests[testSetting]
                            shapePenalty = 0.0001

                            options={'disp':False, 'maxiter':100}
                            # options={'disp':False, 'maxiter':2}

                            minimizingShape = True

                            # method = 5

                            ch.minimize({'raw': errorFun }, bounds=None, method=methods[method], x0=free_variables, callback=cb, options=options)

                            maxShapeSize = 4
                            largeShapeParams = np.abs(chShapeParams.r) > maxShapeSize
                            if np.any(largeShapeParams) or chEl.r > np.pi/2 + radians(10) or chEl.r < -radians(15) or np.linalg.norm(chVertices.r - chVerticesMean) >= 3.5:
                                print("Warning: found large shape parameters to fix!")
                                reverted = True
                            # chShapeParams[largeShapeParams] = np.sign(chShapeParams.r[largeShapeParams])*maxShapeSize
                            cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/it1'+ '.png', cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))

                            # if reverted:
                            #     cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/reverted1'+ '.png', cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))
                            #     reverted = False
                            #     chAz[:] = az
                            #     chEl[:] = min(max(el,radians(1)), np.pi/2-radians(1))
                            #     chVColors[:] = color.copy()
                            #     chLightSHCoeffs[:] = lightCoefficientsRel.copy()
                            #     if useShapeModel:
                            #         chShapeParams[:] = shapeParams.copy()
                            #
                            # cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/it1'+ '.png', cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))
                            #
                            # free_variables = [chVColors, chLightSHCoeffs]
                            # stds[:] = 0.01
                            # shapePenalty = 0.0
                            # options={'disp':False, 'maxiter':50}
                            # # options={'disp':False, 'maxiter':2}
                            # # free_variables = [chShapeParams ]
                            # minimizingShape = False
                            # getColorFromCRF = False
                            # ch.minimize({'raw': errorFunNLLCRF }, bounds=None, method=methods[method], x0=free_variables, callback=cb, options=options)

                            # largeShapeParams = np.abs(chShapeParams.r) > maxShapeSize
                            # if np.any(largeShapeParams) or chEl.r > np.pi/2 + radians(10) or chEl.r < -radians(15) or np.linalg.norm(chVertices.r - chVerticesMean) >= 3.5:
                            #     print("Warning: found large shape parameters to fix!")
                            #     reverted = True
                            #     cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/reverted2'+ '.png', cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))
                            #
                            # cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/it2'+ '.png', cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))
                        else:

                            ## *** If not using CRF ***

                            free_variables = [chShapeParams, chAz, chEl, chLightSHCoeffs, chVColors]
                            #
                            # shapePenalty = 0.0001

                            stds[:] = stdsTests[testSetting]

                            shapePenalty = shapePenaltyTests[testSetting]

                            options = {'disp': False, 'maxiter': maxOptIters[testSetting]}
                            # options={'disp':False, 'maxiter':2}

                            minimizingShape = True


                            errorFun = models[1]

                            # errorFunFast = generative_models.NLLRobustModel(renderer=renderer, groundtruth=rendererGT, Q=globalPrior.r*np.ones([height, width]),
                            #                                                      variances=variances) / numPixels
                            #
                            # plt.imsave('errorscolors1.png', sqeRenderer.render_image)
                            #
                            # chVColors[:] = testVColorGT[test_i] + nearGTOffsetVColor
                            # chAz[:]= testAzsRel[test_i] + nearGTOffsetRelAz
                            # chEl[:] = testElevsGT[test_i] + nearGTOffsetEl
                            # chLightSHCoeffs[:] = testLightCoefficientsGTRel[test_i]
                            # chShapeParams[:] = testShapeParamsGT[test_i]
                            # # chLightSHCoeffs[:] = 0
                            # # chLightSHCoeffs[2] = 0.5
                            # # chLightSHCoeffs[0] = 1
                            # plt.imsave('render.png', renderer.r)
                            #
                            # imageGT = renderer.r.copy()
                            #
                            # chEl[:] = chEl[:].r + 0.2
                            #
                            # SQError = (rendererGT.r - renderer)**2
                            #
                            # sqeRenderer.imageGT = ch.Ch(imageGT)
                            #
                            # plt.imsave('errors.png', sqeRenderer.r)
                            # plt.imsave('errorscolors.png', sqeRenderer.render_image)
                            # # plt.imsave('renderer2.png', sqeRenderer.render_dedx)
                            # # plt.imsave('errorsdx.png', sqeRenderer.render_dedx, cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
                            # drAz = np.sum(sqeRenderer.dr_wrt(chEl).toarray().reshape([height,width, 3]), axis=2)
                            #
                            # drAzErrorFun = -pixelLikelihoodRobustSQErrorCh.dr_wrt(chEl).toarray().reshape([height, width])
                            #
                            # plt.ioff()
                            # fig = plt.figure()
                            # ax = fig.add_subplot(111, aspect='equal')
                            # ims = ax.imshow(drAz, cmap=matplotlib.cm.coolwarm, vmin=-4, vmax=4)
                            # fig.colorbar(ims)
                            # fig.savefig('drAzfig.png', bbox_inches='tight')
                            # plt.close(fig)
                            # plt.ioff()
                            #
                            # fig = plt.figure()
                            # ax = fig.add_subplot(111, aspect='equal')
                            # ims = ax.imshow(drAzErrorFun, cmap=matplotlib.cm.coolwarm, vmin=-300, vmax=300)
                            # fig.colorbar(ims)
                            # fig.savefig('drAzfigErrFun.png', bbox_inches='tight')
                            # plt.close(fig)
                            #
                            #
                            # plt.imsave('errorsdr_wrtaz.png', drAz, cmap=matplotlib.cm.coolwarm, vmin=drAz.min(), vmax=drAz.max())
                            #
                            # plt.imsave('SQerrorsdr_wrtaz.png', SQError.dr_wrt(chEl).toarray().reshape([height,width, 3]), cmap=matplotlib.cm.coolwarm, vmin=-4, vmax=4)
                            # plt.imsave('errorsgt.png', sqeRenderer.imageGT.r)
                            #
                            # sys.exit()

                            ch.minimize({'raw': errorFun}, bounds=None, method=methods[method], x0=free_variables, callback=cb, options=options)

                            maxShapeSize = 4
                            largeShapeParams = np.abs(chShapeParams.r) > maxShapeSize
                            if np.any(largeShapeParams) or chEl.r > np.pi / 2 + radians(10) or chEl.r < -radians(15) or np.linalg.norm(
                                            chVertices.r - chVerticesMean) >= 3.5:
                                print("Warning: found large shape parameters to fix!")
                                reverted = True
                            # chShapeParams[largeShapeParams] = np.sign(chShapeParams.r[largeShapeParams])*maxShapeSize
                            cv2.imwrite(resultDir + 'imgs/test' + str(test_i) + '/it1' + '.png',
                                        cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy()) * 255), cv2.COLOR_RGB2BGR))

                            # if reverted:
                            #     cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/reverted1'+ '.png', cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))
                            #     reverted = False
                            #     chAz[:] = az
                            #     chEl[:] = min(max(el,radians(1)), np.pi/2-radians(1))
                            #     chVColors[:] = color.copy()
                            #     chLightSHCoeffs[:] = lightCoefficientsRel.copy()
                            #     if useShapeModel:
                            #         chShapeParams[:] = shapeParams.copy()
                            # #
                            # cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/it1'+ '.png', cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))
                            #
                            # free_variables = [chVColors]
                            # stds[:] = 0.01
                            # shapePenalty = 0.0
                            # options = {'disp': False, 'maxiter': 50}
                            # # options={'disp':False, 'maxiter':2}
                            # # free_variables = [chShapeParams ]
                            # minimizingShape = False
                            # getColorFromCRF = False
                            # ch.minimize({'raw': errorFun}, bounds=None, method=methods[method], x0=free_variables, callback=cb, options=options)
                            #
                            # largeShapeParams = np.abs(chShapeParams.r) > maxShapeSize
                            # if np.any(largeShapeParams) or chEl.r > np.pi/2 + radians(10) or chEl.r < -radians(15) or np.linalg.norm(chVertices.r - chVerticesMean) >= 3.5:
                            #     print("Warning: found large shape parameters to fix!")
                            #     reverted = True
                            #     cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/reverted2'+ '.png', cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))
                            #
                            # cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/it2'+ '.png', cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))

                if not reverted:
                    bestFittedAz = chAz.r.copy()
                    bestFittedEl = min(max(chEl.r.copy(),radians(1)), np.pi/2-radians(1))
                    bestVColors = chVColors.r.copy()
                    bestLightSHCoeffs = chLightSHCoeffs.r.copy()
                    if useShapeModel:
                        bestShapeParams = chShapeParams.r.copy()
                else:
                    revertedSamples = np.append(revertedSamples, test_i)
                    bestFittedAz =  az
                    bestFittedEl =  min(max(el,radians(1)), np.pi/2-radians(1))
                    bestVColors =  color.copy()
                    bestLightSHCoeffs =  lightCoefficientsRel.copy()
                    if useShapeModel:
                        bestShapeParams =  shapeParams.copy()

                chAz[:] = bestFittedAz
                chEl[:] = min(max(bestFittedEl,radians(1)), np.pi/2-radians(1))
                chVColors[:] = bestVColors
                chLightSHCoeffs[:] = bestLightSHCoeffs
                if useShapeModel:
                    chShapeParams[:] = bestShapeParams

                if makeVideo:
                    if len(vidImgs) > 0:
                        im_ani = animation.ArtistAnimation(figvid, vidImgs, interval=2000, repeat_delay=5000, repeat=True, blit=False)
                        im_ani.save(resultDir + 'videos/fitting_'+ str(testSet[test_i]) + '.mp4', fps=None, writer=writer, codec='mp4')
                        vidImgs[-1][7].remove()

                    writer_i.finish()

                cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/fitted'+ '.png',cv2.cvtColor(np.uint8(lin2srgb(renderer.r.copy())*255), cv2.COLOR_RGB2BGR))

                np.save(testDir + 'fitted_' + 'az' + 'npy', chAz.r)
                np.save(testDir + 'fitted_' + 'el'+ 'npy', chEl.r)
                np.save(testDir + 'fitted_' + 'color'+ 'npy', chVColors.r)
                np.save(testDir + 'fitted_' + 'lightCoefficientsRel'+ 'npy', chLightSHCoeffs.r)
                np.save(testDir + 'fitted_' + 'shapeParams'+ 'npy', chShapeParams.r)

            if optimizationTypeDescr[optimizationType] != 'predict':
                if evaluateWithGT:
                    fittedErrorFuns = np.append(fittedErrorFuns, bestModelLik)
                    fittedAzs = np.append(fittedAzs, bestFittedAz)
                    fittedElevs = np.append(fittedElevs, bestFittedEl)
                    fittedVColorsList = fittedVColorsList + [bestVColors]
                    fittedRelLightCoeffsList = fittedRelLightCoeffsList + [bestLightSHCoeffs]
                    if useShapeModel:
                        fittedShapeParamsList = fittedShapeParamsList + [bestShapeParams]

                    pEnvMap = SHProjection(envMapTexture, np.concatenate([bestLightSHCoeffs[:,None], bestLightSHCoeffs[:,None], bestLightSHCoeffs[:,None]], axis=1))
                    approxProjectionFitted = np.sum(pEnvMap, axis=(2,3))
                    approxProjectionsFittedList = approxProjectionsFittedList + [approxProjectionFitted[None,:]]
                    cv2.imwrite(resultDir + 'imgs/test'+ str(test_i) + '/SH/' + str(hdridx) + '_Fitted.jpeg' , 255*np.sum(pEnvMap, axis=3)[:,:,[2,1,0]])

                #Best std for posterior recognition.
                stds[:] = 0.1

                vis_im = np.array(renderer.indices_image==1).copy().astype(np.bool)
                post = generative_models.layerPosteriorsRobustCh(rendererGT, renderer, vis_im, 'MASK', globalPrior, variances)[0].r>0.5

                # postSqerror = generative_models.layerPosteriorsRobustSQErrorCh(sqeRenderer, np.array([]), 'MASK', globalPrior, variances)[0].r>0.5

                fittedPosteriorsList = fittedPosteriorsList + [post[None,:]]

                stds[:] = stdsTests[testSetting]

                plt.imsave(resultDir + 'imgs/test'+ str(test_i) + '/' + str(hdridx) + '_Outlier.jpeg', np.tile(post.reshape(shapeIm[0],shapeIm[1],1), [1,1,3]).astype(np.float32))


            #Every now and then (or after the final test case), produce plots to keep track of work accross different levels of occlusion.
            experimentDic = {'model':model, 'method':method, 'shapePenalty':shapePenalty, 'stds':stds.r, 'dataIds': testIds, 'gtPrefix': gtPrefix, 'trainPrefixPose': trainPrefixPose, 'trainPrefixVColor': trainPrefixVColor,
                             'trainPrefixLightCoeffs': trainPrefixLightCoeffs, 'trainPrefixShapeParams': trainPrefixShapeParams,
                             'trainModelsDirAppLight': trainModelsDirAppLight, 'experimentDir': experimentDir, 'testSet': testSet,
                             'methodsPred': methodsPred, 'testOcclusions': testOcclusions, 'likelihoods': likelihoods,
                             'testPrefixBase': testPrefixBase, 'parameterRecognitionModels': parameterRecognitionModels, 'azimuths': azimuths,
                             'elevations': elevations, 'vColors': vColors, 'lightCoeffs': lightCoeffs, 'shapeParams': shapeParams, 'evaluateWithGT':evaluateWithGT}


            with open(resultDir + 'experiment.pickle', 'wb') as pfile:
                pickle.dump(experimentDic, pfile)

            if evaluateWithGT:
                if np.mod(test_i+1,100) == 0 or test_i + 1 >= len(testSet):
                    if approxProjectionsPredList:
                        approxProjectionsPred = np.vstack(approxProjectionsPredList)
                    if approxProjectionsGTList:
                        approxProjectionsGT = np.vstack(approxProjectionsGTList)

                    if predictedPosteriorsList:
                        predictedPosteriors = np.vstack(predictedPosteriorsList)

                    if nearestNeighbours:
                        approxProjectionsNearestNeighbours = np.vstack(approxProjectionsNearestNeighbourList)

                    if optimizationTypeDescr[optimizationType] != 'predict':
                        if fittedVColorsList:
                            fittedVColors = np.vstack(fittedVColorsList)
                        if fittedRelLightCoeffsList:
                            fittedRelLightCoeffs = np.vstack(fittedRelLightCoeffsList)
                        if approxProjectionsFittedList:
                            approxProjectionsFitted = np.vstack(approxProjectionsFittedList)

                        if fittedShapeParamsList and useShapeModel:
                            fittedShapeParams = np.vstack(fittedShapeParamsList)
                        if fittedPosteriorsList:
                            fittedPosteriors = np.vstack(fittedPosteriorsList)

                    if nearestNeighbours:
                        azimuthNearestNeighbours = np.concatenate(azimuthNearestNeighboursList)
                        elevationNearestNeighbours = np.concatenate(elevationNearestNeighboursList)
                        vColorNearestNeighbours = np.vstack(vColorNearestNeighboursList)
                        if useShapeModel:
                            shapeParamsNearestNeighbours = np.vstack(shapeParamsNearestNeighboursList)
                        lightCoeffsNearestNeighbours = np.vstack(lightCoeffsNearestNeighboursList)
                        # nearestNeighboursPosteriors = np.vstack(nearestNeighboursPosteriorsList)

                    if optimizationTypeDescr[optimizationType] != 'predict':
                        numFitted = range(len(fittedAzs))
                    else:
                        numFitted = range(test_i+1)

                    if includeMeanBaseline:
                        # meanBaselinePosteriors = np.vstack(meanBaselinePosteriorList)
                        azimuths = [meanTrainAzimuthRel]
                        elevations= [meanTrainElevation]
                        vColors= [meanTrainVColors]
                        lightCoeffs= [meanTrainLightCoefficientsGTRel]
                        if useShapeModel:
                            shapeParams = [meanTrainShapeParams]
                        else:
                            shapeParams = [None]
                        approxProjections= [meanTrainEnvMapProjections]
                        likelihoods = [meanBaselineErrorFuns]
                        segmentations = [None]
                        ipdb.set_trace()
                    else:
                        azimuths =  [None]
                        elevations=  [None]
                        vColors=  [None]
                        lightCoeffs=  [None]
                        approxProjections=  [None]
                        shapeParams =  [None]
                        likelihoods =  [None]
                        segmentations =  [None]

                    if nearestNeighbours:
                        azimuths = azimuths + [azimuthNearestNeighbours]
                        elevations = elevations + [elevationNearestNeighbours]
                        vColors = vColors + [vColorNearestNeighbours]
                        lightCoeffs = lightCoeffs + [lightCoeffsNearestNeighbours]
                        approxProjections = approxProjections + [approxProjectionsNearestNeighbours]
                        if useShapeModel:
                            shapeParams  = shapeParams + [shapeParamsNearestNeighbours]
                        else:
                            shapeParams = shapeParams + [None]

                        segmentations = segmentations + [None]
                    else:
                        azimuths = azimuths + [None]
                        elevations= elevations + [None]
                        vColors= vColors + [None]
                        lightCoeffs= lightCoeffs + [None]
                        approxProjections = approxProjections + [None]
                        likelihoods = likelihoods + [None]
                        shapeParams = shapeParams + [None]
                        segmentations = segmentations + [None]

                    azimuths = azimuths + [azsPred]
                    elevations = elevations + [elevsPred]
                    vColors = vColors + [vColorsPred]
                    lightCoeffs = lightCoeffs + [relLightCoefficientsPred]
                    approxProjections = approxProjections + [approxProjectionsPred]
                    likelihoods = likelihoods + [predictedErrorFuns]
                    segmentations = segmentations + [predictedPosteriors]

                    if useShapeModel:
                        shapeParams  = shapeParams + [shapeParamsPred]
                    else:
                        shapeParams = shapeParams + [None]

                    if optimizationTypeDescr[optimizationType] != 'predict':
                        azimuths = azimuths + [fittedAzs]
                        elevations = elevations + [fittedElevs]
                        vColors = vColors + [fittedVColors]
                        lightCoeffs = lightCoeffs + [fittedRelLightCoeffs]
                        approxProjections = approxProjections + [approxProjectionsFitted]
                        if useShapeModel:
                            shapeParams  = shapeParams + [fittedShapeParams]
                        else:
                            shapeParams = shapeParams + [None]

                        likelihoods = likelihoods + [fittedErrorFuns]
                        segmentations = segmentations + [fittedPosteriors]
                    else:
                        azimuths = azimuths + [None]
                        elevations = elevations +  [None]
                        vColors = vColors + [None]
                        lightCoeffs = lightCoeffs + [None]
                        approxProjections = approxProjections +  [None]
                        likelihoods = likelihoods + [None]
                        segmentations = segmentations + [None]
                        shapeParams = shapeParams + [None]


                    errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList, errorsVColorsSList, errorsSegmentationList \
                        = computeErrors(numFitted, azimuths, testAzsRel, elevations, testElevsGT, vColors, testVColorGT, lightCoeffs, testLightCoefficientsGTRel, approxProjections,  approxProjectionsGT, shapeParams, testShapeParamsGT, useShapeModel, chShapeParams, chVertices, segmentations, masksGT)

                    meanAbsErrAzsList, meanAbsErrElevsList, meanErrorsLightCoeffsList, meanErrorsShapeParamsList, meanErrorsShapeVerticesList, meanErrorsLightCoeffsCList, meanErrorsEnvMapList, meanErrorsVColorsEList, meanErrorsVColorsCList, meanErrorsVColorsCList, meanErrorsSegmentationList \
                        = computeErrorAverages(np.mean, numFitted, useShapeModel, errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList, errorsVColorsSList, errorsSegmentationList)

                    medianAbsErrAzsList, medianAbsErrElevsList,medianErrorsLightCoeffsList, medianErrorsShapeParamsList, medianErrorsShapeVerticesList, medianErrorsLightCoeffsCList, medianErrorsEnvMapList, medianErrorsVColorsEList, medianErrorsVColorsCList, medianErrorsVColorsCList, medianErrorsSegmentationList \
                        = computeErrorAverages(np.median, numFitted, useShapeModel, errorsPosePredList, errorsLightCoeffsList, errorsShapeParamsList, errorsShapeVerticesList, errorsEnvMapList, errorsLightCoeffsCList, errorsVColorsEList, errorsVColorsCList, errorsVColorsSList, errorsSegmentationList)

                    #Write statistics to file.
                    with open(resultDir + 'performance.txt', 'w') as expfile:
                        for method_i in range(len(azimuths)):
                            # expfile.write(str(z))
                            expfile.write("Mean Azimuth Error " + methodsPred[method_i] + " " +  str(meanAbsErrAzsList) + '\n')
                            expfile.write("Mean Elevation Error " + methodsPred[method_i] + " " +  str(meanAbsErrElevsList[method_i])+ '\n')
                            expfile.write("Mean SH Components Error " + methodsPred[method_i] + " " +  str(meanErrorsLightCoeffsList[method_i])+ '\n')
                            expfile.write("Mean SH Components Error " + methodsPred[method_i] + " " +  str(meanErrorsLightCoeffsCList[method_i])+ '\n')
                            expfile.write("Mean Vertex Colors Error E " + methodsPred[method_i] + " " +  str(meanErrorsVColorsEList[method_i])+ '\n')
                            expfile.write("Mean Vertex Colors Error C " + methodsPred[method_i] + " " +  str(meanErrorsVColorsCList[method_i])+ '\n')
                            expfile.write("Mean Vertex Colors Error S " + methodsPred[method_i] + " " + str(meanErrorsVColorsCList[method_i]) + '\n')
                            expfile.write("Mean Shape Error " + methodsPred[method_i] + " " +  str(meanErrorsShapeParamsList[method_i])+ '\n')
                            expfile.write("Mean Shape Vertices Error " + methodsPred[method_i] + " " +  str(meanErrorsShapeVerticesList[method_i])+ '\n')
                            expfile.write("Mean Segmentation Error " + methodsPred[method_i] + " " +  str(meanErrorsSegmentationList[method_i])+ '\n\n')

                    #Write statistics to file.
                    with open(resultDir + 'median-performance.txt', 'w') as expfile:
                        for method_i in range(len(azimuths)):
                            # expfile.write(str(z))
                            expfile.write("Median Azimuth Error " + methodsPred[method_i] + " " +  str(medianAbsErrAzsList) + '\n')
                            expfile.write("Median Elevation Error " + methodsPred[method_i] + " " +  str(medianAbsErrElevsList[method_i])+ '\n')
                            expfile.write("Median SH Components Error " + methodsPred[method_i] + " " +  str(medianErrorsLightCoeffsList[method_i])+ '\n')
                            expfile.write("Median SH Components Error " + methodsPred[method_i] + " " +  str(medianErrorsLightCoeffsCList[method_i])+ '\n')
                            expfile.write("Median Vertex Colors Error E " + methodsPred[method_i] + " " +  str(medianErrorsVColorsEList[method_i])+ '\n')
                            expfile.write("Median Vertex Colors Error C " + methodsPred[method_i] + " " +  str(medianErrorsVColorsCList[method_i])+ '\n')
                            expfile.write("Mean Vertex Colors Error S " + methodsPred[method_i] + " " + str(meanErrorsVColorsCList[method_i]) + '\n')
                            expfile.write("Median Shape Error " + methodsPred[method_i] + " " +  str(medianErrorsShapeParamsList[method_i])+ '\n')
                            expfile.write("Median Shape Vertices Error " + methodsPred[method_i] + " " +  str(medianErrorsShapeVerticesList[method_i])+ '\n')
                            expfile.write("Median Segmentation Error " + methodsPred[method_i] + " " +  str(medianErrorsSegmentationList[method_i])+ '\n\n')

                    if not os.path.exists(resultDir + 'stats/'):
                        os.makedirs(resultDir + 'stats/')

                    with open(resultDir + 'stats/' + 'performance' + str(test_i) + '.txt', 'w') as expfile:

                        for method_i in range(len(azimuths)):
                            expfile.write("Mean Azimuth Error " + methodsPred[method_i] + " " +  str(meanAbsErrAzsList) + '\n')
                            expfile.write("Mean Elevation Error " + methodsPred[method_i] + " " +  str(meanAbsErrElevsList[method_i])+ '\n')
                            expfile.write("Mean SH Components Error " + methodsPred[method_i] + " " +  str(meanErrorsLightCoeffsList[method_i])+ '\n')
                            expfile.write("Mean SH Components Error " + methodsPred[method_i] + " " +  str(meanErrorsLightCoeffsCList[method_i])+ '\n')
                            expfile.write("Mean Vertex Colors Error E " + methodsPred[method_i] + " " +  str(meanErrorsVColorsEList[method_i])+ '\n')
                            expfile.write("Mean Vertex Colors Error C " + methodsPred[method_i] + " " +  str(meanErrorsVColorsCList[method_i])+ '\n')
                            expfile.write("Mean Shape Error " + methodsPred[method_i] + " " +  str(meanErrorsShapeParamsList[method_i])+ '\n')
                            expfile.write("Mean Shape Vertices Error " + methodsPred[method_i] + " " +  str(meanErrorsShapeVerticesList[method_i])+ '\n')
                            expfile.write("Mean Segmentation Error " + methodsPred[method_i] + " " +  str(meanErrorsSegmentationList[method_i])+ '\n\n')

                    segmentationsDic = {'segmentations':segmentations}
                    with open(resultDir + 'segmentations.pickle', 'wb') as pfile:
                        pickle.dump(segmentationsDic, pfile)

                    np.save(resultDir +  'reverted.npy', revertedSamples)

                    envMapDic = {'approxProjections':approxProjections, 'approxProjectionsGT':approxProjectionGT}
                    with open(resultDir + 'approxProjections.pickle', 'wb') as pfile:
                        pickle.dump(envMapDic, pfile)

                    # ipdb.set_trace()
                    totalTime = time.time() - startTime
                    print("Took " + str(totalTime/test_i) + " time per instance.")

                    experimentErrorsDic = {'errorsPosePredList':errorsPosePredList, 'errorsLightCoeffsList':errorsLightCoeffsList, 'errorsShapeParamsLis':errorsShapeParamsList, 'errorsShapeVerticesList':errorsShapeVerticesList, 'errorsEnvMapList':errorsEnvMapList, 'errorsLightCoeffsCList':errorsLightCoeffsCList, 'errorsVColorsEList':errorsVColorsEList, 'errorsVColorsCList':errorsVColorsCList, 'errorsVColorsCList':errorsVColorsCList, 'errorsSegmentationList':errorsSegmentationList}
                    #
                    with open(resultDir + 'experiment_errors.pickle', 'wb') as pfile:
                        pickle.dump(experimentErrorsDic, pfile)

