__author__ = 'pol'

import matplotlib
matplotlib.use('Qt4Agg')
import bpy
import sceneimport
import mathutils
from math import radians
import timeit
import time
import opendr
import chumpy as ch
import geometry
import imageproc
import numpy as np
import cv2
from utils import *
import glfw
import score_image
import matplotlib.pyplot as plt
from opendr_utils import *
import OpenGL.GL as GL
import light_probes
import imageio
plt.ion()
import recognition_models

#########################################
# Initialization starts here
#########################################

#Main script options:
useBlender = False
loadBlenderSceneFile = True
groundTruthBlender = False
useCycles = True
demoMode = True
unpackModelsFromBlender = False
unpackSceneFromBlender = False
loadSavedSH = False
useGTasBackground = False
glModes = ['glfw','mesa']
glMode = glModes[0]
sphericalMap = False

np.random.seed(1)
width, height = (200, 200)
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
camDistance = 0.75

teapots = [line.strip() for line in open('teapots.txt')]
renderTeapotsList = np.arange(len(teapots))
sceneIdx = 0
replaceableScenesFile = '../databaseFull/fields/scene_replaceables.txt'
sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndices, targetPositions = sceneimport.getSceneInformation(sceneIdx, replaceableScenesFile)
sceneDicFile = 'data/scene' + str(sceneNumber) + '.pickle'
targetParentIdx = 0
targetIndex = targetIndices[targetParentIdx]
targetParentPosition = targetPositions[targetParentIdx]
targetPosition = targetParentPosition

if useBlender and not loadBlenderSceneFile:
    scene = sceneimport.loadBlenderScene(sceneIdx, replaceableScenesFile)
    sceneimport.setupScene(scene, roomInstanceNum, scene.world, scene.camera, width, height, 16, useCycles, False)
    scene.update()
    scene.render.filepath = 'opendr_blender.png'
    targetPosition = np.array(targetPosition)
    #Save barebones scene.

elif useBlender and loadBlenderSceneFile:
    sceneimport.loadSceneBlendData(sceneIdx, replaceableScenesFile)
    scene = bpy.data.scenes['Main Scene']
    scene.render.resolution_x = width #perhaps set resolution in code
    scene.render.resolution_y = height
    scene.render.tile_x = height/2
    scene.render.tile_y = width/2
    bpy.context.screen.scene = scene

if unpackSceneFromBlender:
    v, f_list, vc, vn, uv, haveTextures_list, textures_list = sceneimport.unpackBlenderScene(scene, sceneDicFile, targetPosition, True)
else:
    v, f_list, vc, vn, uv, haveTextures_list, textures_list = sceneimport.loadSavedScene(sceneDicFile)

removeObjectData(int(targetIndex), v, f_list, vc, vn, uv, haveTextures_list, textures_list)

targetModels = []
if useBlender and not loadBlenderSceneFile:
    [targetScenes, targetModels, transformations] = sceneimport.loadTargetModels(renderTeapotsList)
elif useBlender:
    teapots = [line.strip() for line in open('teapots.txt')]
    selection = [ teapots[i] for i in renderTeapotsList]
    sceneimport.loadTargetsBlendData()
    for teapotIdx, teapotName in enumerate(selection):
        targetModels = targetModels + [bpy.data.scenes[teapotName[0:63]].objects['teapotInstance' + str(renderTeapotsList[teapotIdx])]]

v_teapots, f_list_teapots, vc_teapots, vn_teapots, uv_teapots, haveTextures_list_teapots, textures_list_teapots, vflat, varray, center_teapots, blender_teapots = sceneimport.loadTeapotsOpenDRData(renderTeapotsList, useBlender, unpackModelsFromBlender, targetModels)

chAz = ch.Ch([1.1693706])
chEl =  ch.Ch([0.95993109])
chDist = ch.Ch([camDistance])

chAzGT = ch.Ch([1.1693706])
chElGT = ch.Ch([0.95993109])
chDistGT = ch.Ch([camDistance])
chComponentGT = ch.Ch(np.array([2, 0.25, 0.25, 0.12,-0.17,0.36,0.1,0.,0.]))
chComponent = ch.Ch(np.array([2, 0.25, 0.25, 0.12,-0.17,0.36,0.1,0.,0.]))

chPointLightIntensity = ch.Ch([1])
chPointLightIntensityGT = ch.Ch([1])
chLightAz = ch.Ch([0.0])
chLightEl = ch.Ch([np.pi/2])
chLightDist = ch.Ch([0.5])
chLightDistGT = ch.Ch([0.5])
chLightAzGT = ch.Ch([0.0])
chLightElGT = ch.Ch([np.pi/4])

ligthTransf = computeHemisphereTransformation(chLightAz, chLightEl, chLightDist, targetPosition)
ligthTransfGT = computeHemisphereTransformation(chLightAzGT, chLightElGT, chLightDistGT, targetPosition)

lightPos = ch.dot(ligthTransf, ch.Ch([0.,0.,0.,1.]))[0:3]
lightPos = ch.Ch([targetPosition[0]+0.5,targetPosition[1],targetPosition[2] + 0.5])
lightPosGT = ch.dot(ligthTransfGT, ch.Ch([0.,0.,0.,1.]))[0:3]

chGlobalConstant = ch.Ch([0.5])
chGlobalConstantGT = ch.Ch([0.5])
light_color = ch.ones(3)*chPointLightIntensity
light_colorGT = ch.ones(3)*chPointLightIntensityGT
chVColors = ch.Ch([0.4,0.4,0.4])
chVColorsGT = ch.Ch([0.4,0.4,0.4])

shCoefficientsFile = 'data/sceneSH' + str(sceneIdx) + '.pickle'

chAmbientIntensityGT = ch.Ch([2])
clampedCosCoeffs = clampedCosineCoefficients()
chAmbientSHGT = ch.zeros([9])

envMapFilename = 'data/hdr/dataset/studio_land.hdr'
envMapTexture = np.array(imageio.imread(envMapFilename))[:,:,0:3]
phiOffset = 0

if sphericalMap:
    envMapTexture, envMapMean = light_probes.processSphericalEnvironmentMap(envMapTexture)
    envMapCoeffs = light_probes.getEnvironmentMapCoefficients(envMapTexture, envMapMean,  -phiOffset, 'spherical')
else:
    envMapMean = envMapTexture.mean()

    envMapCoeffs = light_probes.getEnvironmentMapCoefficients(envMapTexture, envMapMean, -phiOffset, 'equirectangular')

if useBlender:
    addEnvironmentMapWorld(envMapFilename, scene)
    setEnviornmentMapStrength(0.3/envMapMean, scene)
    rotateEnviornmentMap(phiOffset, scene)

shCoeffsRGB = ch.Ch(envMapCoeffs)
chShCoeffs = 0.3*shCoeffsRGB[:,0] + 0.59*shCoeffsRGB[:,1] + 0.11*shCoeffsRGB[:,2]
chAmbientSHGT = chShCoeffs.ravel() * chAmbientIntensityGT * clampedCosCoeffs

if loadSavedSH:
    if os.path.isfile(shCoefficientsFile):
        with open(shCoefficientsFile, 'rb') as pfile:
            shCoeffsDic = pickle.load(pfile)
            shCoeffs = shCoeffsDic['shCoeffs']
            chAmbientSHGT = shCoeffs.ravel()* chAmbientIntensityGT * clampedCosCoeffs

chLightRadGT = ch.Ch([0.1])
chLightDistGT = ch.Ch([0.5])
chLightIntensityGT = ch.Ch([0])
chLightAzGT = ch.Ch([np.pi*3/2])
chLightElGT = ch.Ch([np.pi/4])
angle = ch.arcsin(chLightRadGT/chLightDistGT)
zGT = chZonalHarmonics(angle)
shDirLightGT = chZonalToSphericalHarmonics(zGT, np.pi/2 - chLightElGT, chLightAzGT - np.pi/2) * clampedCosCoeffs
chComponentGT = chAmbientSHGT + shDirLightGT*chLightIntensityGT
# chComponentGT = chAmbientSHGT.r[:] + shDirLightGT.r[:]*chLightIntensityGT.r[:]

chLightAzGT = ch.Ch([np.pi/2])
chLightElGT = ch.Ch([np.pi/4])
shDirLight = chZonalToSphericalHarmonics(zGT, np.pi/2 - chLightElGT, chLightAzGT - np.pi/2) * clampedCosCoeffs
chComponentStuff = chAmbientSHGT + shDirLight*chLightIntensityGT
chComponent[:] = chComponentStuff.r[:]

chDisplacement = ch.Ch([0.0, 0.0,0.0])
chDisplacementGT = ch.Ch([0.0,0.0,0.0])
chScale = ch.Ch([1.0,1.0,1.0])
chScaleGT = ch.Ch([1, 1.,1.])
scaleMat = geometry.Scale(x=chScale[0], y=chScale[1],z=chScale[2])[0:3,0:3]
scaleMatGT = geometry.Scale(x=chScaleGT[0], y=chScaleGT[1],z=chScaleGT[2])[0:3,0:3]
invTranspModel = ch.transpose(ch.inv(scaleMat))
invTranspModelGT = ch.transpose(ch.inv(scaleMatGT))

# vcch[0] = np.ones_like(vcflat[0])*chVColorsGT.reshape([1,3])
renderer_teapots = []

for teapot_i in range(len(renderTeapotsList)):
    if useBlender:
        teapot = blender_teapots[teapot_i]
        teapot.matrix_world = mathutils.Matrix.Translation(targetPosition)

    vmod = v_teapots[teapot_i]
    fmod_list = f_list_teapots[teapot_i]
    vcmod = vc_teapots[teapot_i]
    vnmod = vn_teapots[teapot_i]
    uvmod = uv_teapots[teapot_i]
    haveTexturesmod_list = haveTextures_list_teapots[teapot_i]
    texturesmod_list = textures_list_teapots[teapot_i]
    centermod = center_teapots[teapot_i]

    #Add targetPosition to vertex coordinates.
    # for obj_i in range(len(vmod)):
    #     for mesh_i in range(len(vmod[obj_i])):
    #         vmod[obj_i][mesh_i] = vmod[obj_i][mesh_i]

    renderer = createRendererTarget(glMode, chAz,chEl, chDist, centermod, vmod, vcmod, fmod_list, vnmod, light_color, chComponent, chVColors, targetPosition, chDisplacement, scaleMat, invTranspModel, width,height, uvmod, haveTexturesmod_list, texturesmod_list, frustum, win )

    renderer.r
    renderer_teapots = renderer_teapots + [renderer]

currentTeapotModel = 0
renderer = renderer_teapots[currentTeapotModel]

addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list,  v_teapots[currentTeapotModel][0], f_list_teapots[currentTeapotModel][0], vc_teapots[currentTeapotModel][0], vn_teapots[currentTeapotModel][0], uv_teapots[currentTeapotModel][0], haveTextures_list_teapots[currentTeapotModel][0], textures_list_teapots[currentTeapotModel][0])

center = center_teapots[currentTeapotModel]
rendererGT = createRendererGT(glMode, chAzGT,chElGT, chDistGT, center, v, vc, f_list, vn, light_colorGT, chComponentGT, chVColorsGT, targetPosition, chDisplacementGT, scaleMatGT, invTranspModelGT, width,height, uv, haveTextures_list, textures_list, frustum, win )

if useGTasBackground:
    for teapot_i in range(len(renderTeapotsList)):
        renderer = renderer_teapots[teapot_i]
        renderer.set(background_image=rendererGT.r)

currentTeapotModel = 0
renderer = renderer_teapots[currentTeapotModel]
# ipdb.set_trace()

vis_gt = np.array(rendererGT.indices_image!=1).copy().astype(np.bool)
vis_mask = np.array(rendererGT.indices_image==1).copy().astype(np.bool)
vis_im = np.array(renderer.indices_image!=1).copy().astype(np.bool)

oldChAz = chAz[0].r
oldChEl = chEl[0].r

# Show it
shapeIm = vis_gt.shape
numPixels = shapeIm[0] * shapeIm[1]
shapeIm3D = [vis_im.shape[0], vis_im.shape[1], 3]

#########################################
# Initialization ends here
#########################################



print("Training recognition models.")
trainData = {}

with open(trainDataName, 'rb') as pfile:
    trainData = pickle.load(pfile)

trainAzsGT = trainData['trainAzsGT']
trainElevsGT = trainData['trainElevsGT']
trainLightAzsGT = trainData['trainLightAzsGT']
trainLightElevsGT = trainData['trainLightElevsGT']
trainLightIntensitiesGT = trainData['trainLightIntensitiesGT']
trainVColorGT = trainData['trainVColorGT']
# trainTeapots  = trainData['trainTeapots']

chAzOld = chAz.r[0]
chElOld = chEl.r[0]
chAzGTOld = chAzGT.r[0]
chElGTOld = chElGT.r[0]

images = []
occlusions = np.array([])
hogs = []
# vcolorsfeats = []
illumfeats = []
occludedInstances = []
# split = 0.8
# setTrain = np.arange(np.floor(trainSize*split)).astype(np.uint8)
print("Generating renders")
for train_i in range(len(trainAzsGT)):
    azi = trainAzsGT[train_i]
    eli = trainElevsGT[train_i]
    chAzGT[:] = azi
    chElGT[:] = eli
    chLightAzGT[:] = trainLightAzsGT[train_i]
    chLightElGT[:] = trainLightElevsGT[train_i]
    chLightIntensityGT[:] = trainLightIntensitiesGT[train_i]
    chVColorsGT[:] = trainVColorGT[train_i]
    image = rendererGT.r.copy()

    occlusion = getOcclusionFraction(rendererGT)
    if occlusion < 0.9:
        images = images + [image]
        occlusions = np.append(occlusions, occlusion)
        hogs = hogs + [imageproc.computeHoG(image).reshape([1,-1])]

        # vcolorsfeats = vcolorsfeats +  [imageproc.medianColor(image,40)]
        illumfeats = illumfeats + [imageproc.featuresIlluminationDirection(image,20)]
    else:
        occludedInstances = occludedInstances + [train_i]

trainAzsGT = np.delete(trainAzsGT, occludedInstances)
trainElevsGT = np.delete(trainElevsGT, occludedInstances)
trainLightAzsGT = np.delete(trainLightAzsGT, occludedInstances)
trainLightElevsGT = np.delete(trainLightElevsGT, occludedInstances)
trainLightIntensitiesGT = np.delete(trainLightIntensitiesGT, occludedInstances)
trainVColorGT = np.delete(trainVColorGT, occludedInstances, 0)

hogfeats = np.vstack(hogs)
illumfeats = np.vstack(illumfeats)

print("Training RFs")
randForestModelCosAzs = recognition_models.trainRandomForest(hogfeats, np.cos(trainAzsGT))
randForestModelSinAzs = recognition_models.trainRandomForest(hogfeats, np.sin(trainAzsGT))
randForestModelCosElevs = recognition_models.trainRandomForest(hogfeats, np.cos(trainElevsGT))
randForestModelSinElevs = recognition_models.trainRandomForest(hogfeats, np.sin(trainElevsGT))

randForestModelLightCosAzs = recognition_models.trainRandomForest(illumfeats, np.cos(trainLightAzsGT))
randForestModelLightSinAzs = recognition_models.trainRandomForest(illumfeats, np.sin(trainLightAzsGT))
randForestModelLightCosElevs = recognition_models.trainRandomForest(illumfeats, np.cos(trainLightElevsGT))
randForestModelLightSinElevs = recognition_models.trainRandomForest(illumfeats, np.sin(trainLightElevsGT))

imagesStack = np.vstack([image.reshape([1,-1]) for image in images])
randForestModelLightIntensity = recognition_models.trainRandomForest(imagesStack, trainLightIntensitiesGT)

trainedModels = {'randForestModelCosAzs':randForestModelCosAzs,'randForestModelSinAzs':randForestModelSinAzs,'randForestModelCosElevs':randForestModelCosElevs,'randForestModelSinElevs':randForestModelSinElevs,'randForestModelLightCosAzs':randForestModelLightCosAzs,'randForestModelLightSinAzs':randForestModelLightSinAzs,'randForestModelLightCosElevs':randForestModelLightCosElevs,'randForestModelLightSinElevs':randForestModelLightSinElevs,'randForestModelLightIntensity':randForestModelLightIntensity}
with open('experiments/' + trainprefix + 'models.pickle', 'wb') as pfile:
    pickle.dump(trainedModels, pfile)


# print("Training LR")
# linRegModelCosAzs = recognition_models.trainLinearRegression(hogfeats, np.cos(trainAzsGT))
# linRegModelSinAzs = recognition_models.trainLinearRegression(hogfeats, np.sin(trainAzsGT))
# linRegModelCosElevs = recognition_models.trainLinearRegression(hogfeats, np.cos(trainElevsGT))
# linRegModelSinElevs = recognition_models.trainLinearRegression(hogfeats, np.sin(trainElevsGT))

chAz[:] = chAzOld
chEl[:] = chElOld
chAzGT[:] = chAzGTOld
chElGT[:] = chElGTOld

print("Finished training recognition models.")
beginTraining = False

def imageGT():
    global groundTruthBlender
    global rendererGT
    global blenderRender

    if groundTruthBlender:
        return blenderRender
    else:
        return np.copy(np.array(rendererGT.r)).astype(np.float64)

imagegt = imageGT()
chImage = ch.array(imagegt)
# E_raw_simple = renderer - rendererGT
negVisGT = ~vis_gt
imageWhiteMask = imagegt.copy()
imageWhiteMask[np.tile(negVisGT.reshape([shapeIm[0],shapeIm[1],1]),[1,1,3]).astype(np.bool)] = 1

chImageWhite = ch.Ch(imageWhiteMask)
E_raw = renderer - rendererGT
SE_raw = ch.sum(E_raw*E_raw, axis=2)

SSqE_raw = ch.SumOfSquares(E_raw)/numPixels

initialPixelStdev = 0.075
reduceVariance = False
# finalPixelStdev = 0.05
stds = ch.Ch([initialPixelStdev])
variances = stds ** 2
globalPrior = ch.Ch([0.8])

negLikModel = -score_image.modelLogLikelihoodCh(rendererGT, renderer, vis_im, 'FULL', variances)/numPixels

negLikModelRobust = -score_image.modelLogLikelihoodRobustCh(rendererGT, renderer, vis_im, 'FULL', globalPrior, variances)/numPixels

pixelLikelihoodCh = score_image.logPixelLikelihoodCh(rendererGT, renderer, vis_im, 'FULL', variances)

pixelLikelihoodRobustCh = ch.log(score_image.pixelLikelihoodRobustCh(rendererGT, renderer, vis_im, 'FULL', globalPrior, variances))

post = score_image.layerPosteriorsRobustCh(rendererGT, renderer, vis_im, 'FULL', globalPrior, variances)[0]

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

testImages = []
testHogs = []
testIllumfeats = []
testPredVColors = []
testPredVColorGMMs = []
testPredPoseGMMs = []
occludedInstances = []

print("Generating renders")
for test_i in range(len(testAzsGT)):
    azi = testAzsGT[test_i]
    eli = testElevsGT[test_i]
    chAzGT[:] = azi
    chElGT[:] = eli
    chLightAzGT[:] = testLightAzsGT[test_i]
    chLightElGT[:] = testLightElevsGT[test_i]
    chLightIntensityGT[:] = testLightIntensitiesGT[test_i]
    chVColorsGT[:] = testVColorGT[test_i]
    testImage = rendererGT.r.copy()
    occlusion = getOcclusionFraction(rendererGT)
    if occlusion < 0.95:
        testImages = testImages + [testImage]
        testIllumfeats = testIllumfeats + [imageproc.featuresIlluminationDirection(testImage,20)]
        testHogs = testHogs + [imageproc.computeHoG(testImage).reshape([1,-1])]
        testPredVColors = testPredVColors + [recognition_models.meanColor(testImage, 20)]
        testPredVColorGMMs = testPredVColorGMMs + [recognition_models.colorGMM(testImage, 20)]
    else:
        occludedInstances = occludedInstances + [test_i]

testAzsGT = np.delete(testAzsGT, occludedInstances)
testElevsGT = np.delete(testElevsGT, occludedInstances)
testLightAzsGT = np.delete(testLightAzsGT, occludedInstances)
testLightElevsGT = np.delete(testLightElevsGT, occludedInstances)
testLightIntensitiesGT = np.delete(testLightIntensitiesGT, occludedInstances)
testVColorGT = np.delete(testVColorGT, occludedInstances, 0)

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
