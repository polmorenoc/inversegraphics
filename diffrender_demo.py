__author__ = 'pol'

import matplotlib
matplotlib.use('Qt4Agg')
import bpy
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
from blender_utils import *
import glfw
import generative_models
import matplotlib.pyplot as plt
from opendr_utils import *
import OpenGL.GL as GL
import light_probes
import imageio
plt.ion()

#__GL_THREADED_OPTIMIZATIONS

#Main script options:
useBlender = True
loadBlenderSceneFile = True
groundTruthBlender = False
useCycles = True
demoMode = True
showSubplots = True
unpackModelsFromBlender = False
unpackSceneFromBlender = False
loadSavedSH = False
useGTasBackground = False
refreshWhileMinimizing = False
computePerformance = True
savePerformance = True
glModes = ['glfw','mesa']
glMode = glModes[0]
sphericalMap = False

np.random.seed(1)
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

gtPrefix = 'train3'
gtDir = 'groundtruth/' + gtPrefix + '/'
groundTruthFilename = gtDir + 'groundTruth.h5'
gtDataFile = h5py.File(groundTruthFilename, 'r')
groundTruth = gtDataFile[gtPrefix]
dataAzsGT = groundTruth['trainAzsGT']
dataObjAzsGT = groundTruth['trainObjAzsGT']
dataElevsGT = groundTruth['trainElevsGT']
# dataLightAzsGT = groundTruth['trainLightAzsGT']
# dataLightElevsGT = groundTruth['trainLightElevsGT']
# dataLightIntensitiesGT = groundTruth['trainLightIntensities']
dataVColorGT = groundTruth['trainVColorGT']
dataScenes = groundTruth['trainScenes']
dataTeapotIds = groundTruth['trainTeapotIds']
dataEnvMaps = groundTruth['trainEnvMaps']
dataOcclusions = groundTruth['trainOcclusions']
dataTargetIndices = groundTruth['trainTargetIndices']
dataComponentsGT = groundTruth['trainComponentsGT']
dataComponentsGTRel = groundTruth['trainComponentsGTRel']
dataIds = groundTruth['trainIds']
dataLightCoefficientsGT = groundTruth['trainLightCoefficientsGT']
dataLightCoefficientsGTRel = groundTruth['trainLightCoefficientsGTRel']
dataAmbientIntensityGT = groundTruth['trainAmbientIntensityGT']
dataEnvMapPhiOffsets = groundTruth['trainEnvMapPhiOffsets']

readDataId = 1

teapots = [line.strip() for line in open('teapots.txt')]
renderTeapotsList = np.arange(len(teapots))[0:1]

sceneNumber = dataScenes[readDataId]

replaceableScenesFile = '../databaseFull/fields/scene_replaceables_backup.txt'

sceneIdx = scene_io_utils.getSceneIdx(sceneNumber, replaceableScenesFile)
sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndices, targetPositions = scene_io_utils.getSceneInformation(sceneIdx, replaceableScenesFile)
sceneDicFile = 'data/scene' + str(sceneNumber) + '.pickle'
targetParentIdx = 0
targetIndex = targetIndices[targetParentIdx]
targetIndex = dataTargetIndices[readDataId]

for targetParentIdx, targetParentIndex in enumerate(targetIndices):
    if targetParentIndex == targetIndex:
        #Now targetParentIdx has the right idx of the list of parent indices.
        break

targetParentPosition = targetPositions[targetParentIdx]
targetPosition = targetParentPosition

if useBlender and not loadBlenderSceneFile:
    scene = scene_io_utils.loadBlenderScene(sceneIdx, replaceableScenesFile)
    scene_io_utils.setupScene(scene, roomInstanceNum, scene.world, scene.camera, width, height, 16, useCycles, False)
    scene.update()
    targetPosition = np.array(targetPosition)
    #Save barebones scene.

elif useBlender and loadBlenderSceneFile:
    scene_io_utils.loadSceneBlendData(sceneIdx, replaceableScenesFile)
    scene = bpy.data.scenes['Main Scene']
    scene.render.resolution_x = width #perhaps set resolution in code
    scene.render.resolution_y = height
    scene.render.tile_x = height/2
    scene.render.tile_y = width/2
    scene.cycles.samples = 1024
    scene.sequencer_colorspace_settings.name = 'Linear'
    scene.display_settings.display_device = 'None'
    bpy.context.screen.scene = scene

tex_srgb2lin =  True
if unpackSceneFromBlender:
    v, f_list, vc, vn, uv, haveTextures_list, textures_list = scene_io_utils.unpackBlenderScene(scene, sceneDicFile, True)
else:
    v, f_list, vc, vn, uv, haveTextures_list, textures_list = scene_io_utils.loadSavedScene(sceneDicFile, tex_srgb2lin)

removeObjectData(len(v) -1 - targetIndex, v, f_list, vc, vn, uv, haveTextures_list, textures_list)

targetModels = []
if useBlender and not loadBlenderSceneFile:
    [targetScenes, targetModels, transformations] = scene_io_utils.loadTargetModels(renderTeapotsList)
elif useBlender:
    teapots = [line.strip() for line in open('teapots.txt')]
    selection = [ teapots[i] for i in renderTeapotsList]
    scene_io_utils.loadTargetsBlendData()
    for teapotIdx, teapotName in enumerate(selection):
        targetModels = targetModels + [bpy.data.scenes[teapotName[0:63]].objects['teapotInstance' + str(renderTeapotsList[teapotIdx])]]

v_teapots, f_list_teapots, vc_teapots, vn_teapots, uv_teapots, haveTextures_list_teapots, textures_list_teapots, vflat, varray, center_teapots = scene_io_utils.loadTeapotsOpenDRData(renderTeapotsList, useBlender, unpackModelsFromBlender, targetModels)

azimuth = np.pi
chCosAz = ch.Ch([np.cos(azimuth)])
chSinAz = ch.Ch([np.sin(azimuth)])

chAz = 2*ch.arctan(chSinAz/(ch.sqrt(chCosAz**2 + chSinAz**2) + chCosAz))

elevation = 0
chLogCosEl = ch.Ch(np.log(np.cos(elevation)))
chLogSinEl = ch.Ch(np.log(np.sin(elevation)))
chEl = 2*ch.arctan(ch.exp(chLogSinEl)/(ch.sqrt(ch.exp(chLogCosEl)**2 + ch.exp(chLogSinEl)**2) + ch.exp(chLogCosEl)))

chDist = ch.Ch([camDistance])

chPointLightIntensity = ch.Ch([1])
chPointLightIntensityGT = ch.Ch([1])
chLightAz = ch.Ch([0.0])
chLightEl = ch.Ch([0])
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

chVColors = ch.Ch(dataVColorGT[readDataId])
chVColorsGT = ch.Ch(dataVColorGT[readDataId])
 
shCoefficientsFile = 'data/sceneSH' + str(sceneIdx) + '.pickle'

clampedCosCoeffs = clampedCosineCoefficients()

# envMapFilename = 'data/hdr/dataset/TropicalRuins_3k.hdr'

SHFilename = 'data/LightSHCoefficients.pickle'
with open(SHFilename, 'rb') as pfile:
    envMapDic = pickle.load(pfile)
hdritems = list(envMapDic.items())
hdrstorender = []
phiOffsets = [0, np.pi/2, np.pi, 3*np.pi/2]
for hdrFile, hdrValues in hdritems:
    hdridx = hdrValues[0]
    envMapCoeffs = hdrValues[1]
    if hdridx == dataEnvMaps[readDataId]:
        break
envMapFilename = hdrFile

envMapTexture = np.array(imageio.imread(envMapFilename))[:,:,0:3]
envMapGray = 0.3*envMapTexture[:,:,0] + 0.59*envMapTexture[:,:,1] + 0.11*envMapTexture[:,:,2]
envMapGrayMean = np.mean(envMapGray, axis=(0,1))

# if sphericalMap:
#     envMapTexture, envMapMean = light_probes.processSphericalEnvironmentMap(envMapTexture)
#     envMapCoeffsGT = light_probes.getEnvironmentMapCoefficients(envMapTexture, 1,  0, 'spherical')
# else:
#     envMapMean = np.mean(envMapTexture,axis=(0,1))[None,None,:]
#     envMapGray = 0.3*envMapTexture[:,:,0] + 0.59*envMapTexture[:,:,1] + 0.11*envMapTexture[:,:,2]
#     envMapGrayMean = np.mean(envMapGray, axis=(0,1))
#     envMapTexture = envMapTexture/envMapGrayMean
#
#     # envMapTexture = 4*np.pi*envMapTexture/np.sum(envMapTexture, axis=(0,1))
#     envMapCoeffsGT = light_probes.getEnvironmentMapCoefficients(envMapTexture, 1, 0, 'equirectangular')
#     pEnvMap = SHProjection(envMapTexture, envMapCoeffsGT)
#     approxProjection = np.sum(pEnvMap, axis=3)

    # imageio.imwrite("tmp.exr", approxProjection)

envMapCoeffsGT = ch.Ch(envMapCoeffs)

# rotation = ch.Ch([0.0])
phiOffsetGT = ch.Ch(dataEnvMapPhiOffsets[readDataId])
phiOffset = ch.Ch(dataEnvMapPhiOffsets[readDataId])

chObjAzGT = ch.Ch(dataObjAzsGT[readDataId])
chAzGT = ch.Ch(dataAzsGT[readDataId])
chAzRelGT = chAzGT - chObjAzGT
chElGT = ch.Ch(dataElevsGT[readDataId])
chDistGT = ch.Ch([camDistance])

totalOffsetGT = phiOffsetGT + chObjAzGT

chAmbientIntensityGT = ch.Ch(dataAmbientIntensityGT[readDataId])
# chAmbientIntensityGT = ch.Ch([0.125])
shCoeffsRGBGT = ch.dot(light_probes.chSphericalHarmonicsZRotation(totalOffsetGT), envMapCoeffsGT[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]
shCoeffsRGBGTRel = ch.dot(light_probes.chSphericalHarmonicsZRotation(phiOffsetGT), envMapCoeffsGT[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]

chShCoeffsGT = 0.3*shCoeffsRGBGT[:,0] + 0.59*shCoeffsRGBGT[:,1] + 0.11*shCoeffsRGBGT[:,2]
chShCoeffsGTRel = 0.3*shCoeffsRGBGTRel[:,0] + 0.59*shCoeffsRGBGTRel[:,1] + 0.11*shCoeffsRGBGTRel[:,2]
chAmbientSHGT = chShCoeffsGT.ravel() * chAmbientIntensityGT * clampedCosCoeffs
chAmbientSHGTRel = chShCoeffsGTRel.ravel() * chAmbientIntensityGT * clampedCosCoeffs

chLightRadGT = ch.Ch([0.1])
chLightDistGT = ch.Ch([0.5])
chLightIntensityGT = ch.Ch([0])
chLightAzGT = ch.Ch([0])
chLightElGT = ch.Ch([0])
angleGT = ch.arcsin(chLightRadGT/chLightDistGT)
zGT = chZonalHarmonics(angleGT)
shDirLightGTOriginal = np.array(chZonalToSphericalHarmonics(zGT, np.pi/2 - chLightElGT, chLightAzGT - np.pi/2).r[:]).copy()
shDirLightGT = ch.Ch(shDirLightGTOriginal.copy())
chComponentGTOriginal = ch.array(np.array(chAmbientSHGT + shDirLightGT*chLightIntensityGT * clampedCosCoeffs).copy())
# chComponentGT = chAmbientSHGT + shDirLightGT*chLightIntensityGT * clampedCosCoeffs
chComponentGT = chAmbientSHGT
# chComponentGT = ch.Ch([0.2,0,0,0,0,0,0,0,0])

chAz = ch.Ch(dataAzsGT[readDataId])
chObjAz = ch.Ch(dataObjAzsGT[readDataId])
chEl =  ch.Ch(dataElevsGT[readDataId])
chAzRel = chAz - chObjAz

totalOffset = phiOffset + chObjAz

chAmbientIntensity = ch.Ch(dataAmbientIntensityGT[readDataId])
shCoeffsRGB = ch.dot(light_probes.chSphericalHarmonicsZRotation(totalOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]
shCoeffsRGBRel = ch.dot(light_probes.chSphericalHarmonicsZRotation(phiOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]

chShCoeffs = 0.3*shCoeffsRGB[:,0] + 0.59*shCoeffsRGB[:,1] + 0.11*shCoeffsRGB[:,2]
chShCoeffs = ch.Ch(0.3*shCoeffsRGB.r[:,0] + 0.59*shCoeffsRGB.r[:,1] + 0.11*shCoeffsRGB.r[:,2])
chShCoeffsRel = 0.3*shCoeffsRGBRel[:,0] + 0.59*shCoeffsRGBRel[:,1] + 0.11*shCoeffsRGBRel[:,2]
chAmbientSH = chShCoeffs.ravel() * chAmbientIntensity * clampedCosCoeffs

chLightRad = ch.Ch([0.1])
chLightDist = ch.Ch([0.5])
chLightIntensity = ch.Ch([0])
chLightAz = ch.Ch([np.pi/2])
chLightEl = ch.Ch([0])
angle = ch.arcsin(chLightRad/chLightDist)
z = chZonalHarmonics(angle)
shDirLight = chZonalToSphericalHarmonics(z, np.pi/2 - chLightEl, chLightAz - np.pi/2) * clampedCosCoeffs
chComponent = chAmbientSH + shDirLight*chLightIntensity
# chComponent = chComponentGT

if useBlender:
    addEnvironmentMapWorld(scene)
    updateEnviornmentMap(envMapFilename, scene)
    setEnviornmentMapStrength(1./envMapGrayMean, scene)
    rotateEnviornmentMap(-totalOffset, scene)


chDisplacement = ch.Ch([0.0, 0.0,0.0])
chDisplacementGT = ch.Ch([0.0,0.0,0.0])
chScale = ch.Ch([1.0,1.0,1.0])
chScaleGT = ch.Ch([1, 1.,1.])

# vcch[0] = np.ones_like(vcflat[0])*chVColorsGT.reshape([1,3])
renderer_teapots = []
blender_teapots = []
teapots = [line.strip() for line in open('teapots.txt')]
selection = [ teapots[i] for i in renderTeapotsList]
scene_io_utils.loadTargetsBlendData()
for teapotIdx, teapotName in enumerate(selection):
    teapot = bpy.data.scenes[teapotName[0:63]].objects['teapotInstance' + str(renderTeapotsList[teapotIdx])]
    teapot.layers[1] = True
    teapot.layers[2] = True
    targetModels = targetModels + [teapot]
    blender_teapots = blender_teapots + [teapot]
    setObjectDiffuseColor(teapot, chVColorsGT.r.copy())

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

    renderer = createRendererTarget(glMode, chAz, chObjAz, chEl, chDist, centermod, vmod, vcmod, fmod_list, vnmod, light_color, chComponent, chVColors, targetPosition, chDisplacement, chScale, width,height, uvmod, haveTexturesmod_list, texturesmod_list, frustum, win )
    renderer.msaa = True
    renderer.overdraw = True
    renderer.r
    renderer_teapots = renderer_teapots + [renderer]


# # # Funky theano stuff
# import lasagne_nn
# import lasagne
# import theano
# import theano.tensor as T
# with open('experiments/train4/neuralNetModelRelSHLight.pickle', 'rb') as pfile:
#     neuralNetModelSHLight = pickle.load(pfile)
# meanImage = neuralNetModelSHLight['mean']
# modelType = neuralNetModelSHLight['type']
# param_values = neuralNetModelSHLight['params']
# rendererGray =  0.3*renderer[:,:,0] +  0.59*renderer[:,:,1] + 0.11*renderer[:,:,2]
# input = rendererGray.r[None,None, :,:]
# input_var = T.tensor4('inputs')
# network = lasagne_nn.build_cnn(input_var)
# network_small = lasagne_nn.build_cnn_small(input_var)
# lasagne.layers.set_all_param_values(network, param_values)
# prediction = lasagne.layers.get_output(network)
# chThFun = TheanoFunOnOpenDR(theano_input=input_var, theano_output=prediction, opendr_input=renderer, dim_output = 9)
# sys.exit(0)

currentTeapotModel = 0
renderer = renderer_teapots[currentTeapotModel]

addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list,  v_teapots[currentTeapotModel][0], f_list_teapots[currentTeapotModel][0], vc_teapots[currentTeapotModel][0], vn_teapots[currentTeapotModel][0], uv_teapots[currentTeapotModel][0], haveTextures_list_teapots[currentTeapotModel][0], textures_list_teapots[currentTeapotModel][0])

center = center_teapots[currentTeapotModel]
rendererGT = createRendererGT(glMode, chAzGT, chObjAzGT, chElGT, chDistGT, center, v, vc, f_list, vn, light_colorGT, chComponentGT, chVColorsGT, targetPosition, chDisplacementGT, chScaleGT, width,height, uv, haveTextures_list, textures_list, frustum, win )
rendererGT.msaa = True
rendererGT.overdraw = True
if useGTasBackground:
    for teapot_i in range(len(renderTeapotsList)):
        renderer = renderer_teapots[teapot_i]
        renderer.set(background_image=rendererGT.r)

currentTeapotModel = 0
renderer = renderer_teapots[currentTeapotModel]

import differentiable_renderer
paramsList = [chAz, chEl]

# diffRenderer = differentiable_renderer.DifferentiableRenderer(renderer=renderer, params_list=paramsList, params=ch.concatenate(paramsList))
diffRenderer = renderer

vis_gt = np.array(rendererGT.indices_image!=1).copy().astype(np.bool)
vis_mask = np.array(rendererGT.indices_image==1).copy().astype(np.bool)
vis_im = np.array(renderer.indices_image!=1).copy().astype(np.bool)

oldChAz = chAz[0].r
oldChEl = chEl[0].r

# Show it
shapeIm = vis_gt.shape
numPixels = shapeIm[0] * shapeIm[1]
shapeIm3D = [vis_im.shape[0], vis_im.shape[1], 3]

if useBlender:

    center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)
    # addLamp(scene, center, chLightAzGT.r, chLightElGT.r, chLightDistGT, chLightIntensityGT.r)
    #Add ambient lighting to scene (rectangular lights at even intervals).
    # addAmbientLightingScene(scene, useCycles)

    teapot = blender_teapots[currentTeapotModel]
    teapotGT = blender_teapots[currentTeapotModel]
    placeNewTarget(scene, teapot, targetPosition)
    teapot.layers[1]=True
    # scene.layers[0] = False
    # scene.layers[1] = True
    scene.objects.unlink(scene.objects[str(targetIndex)])

    placeCamera(scene.camera, -chAzGT.r[:].copy()*180/np.pi, chElGT.r[:].copy()*180/np.pi, chDistGT, center)
    azimuthRot = mathutils.Matrix.Rotation(chObjAzGT.r[:].copy(), 4, 'Z')
    original_matrix_world = teapot.matrix_world.copy()
    teapot.matrix_world = mathutils.Matrix.Translation(original_matrix_world.to_translation()) * azimuthRot * (mathutils.Matrix.Translation(-original_matrix_world.to_translation())) * original_matrix_world

    scene.update()

    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.filepath = 'opendr_blender.exr'
    # bpy.ops.file.pack_all()
    # bpy.ops.wm.save_as_mainfile(filepath='data/scene' + str(sceneIdx) + '_complete.blend')
    # scene.render.filepath = 'blender_envmap_render.exr'

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

initialPixelStdev = 0.01
reduceVariance = False
# finalPixelStdev = 0.05
stds = ch.Ch([initialPixelStdev])
variances = stds ** 2
globalPrior = ch.Ch([0.8])

negLikModel = -ch.sum(generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances))/numPixels

negLikModelRobust = -ch.sum(generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances))/numPixels

pixelLikelihoodCh = generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances)

pixelLikelihoodRobustCh = generative_models.LogRobustModel(renderer=renderer, groundtruth=rendererGT, foregroundPrior=globalPrior, variances=variances)

post = generative_models.layerPosteriorsRobustCh(rendererGT, renderer, vis_im, 'FULL', globalPrior, variances)[0]

# hogGT, hogImGT, drconv = image_processing.diffHog(rendererGT)
# hogRenderer, hogImRenderer, _ = image_processing.diffHog(renderer, drconv)
#
# hogE_raw = hogGT - hogRenderer
# hogCellErrors = ch.sum(hogE_raw*hogE_raw, axis=2)
# hogError = ch.SumOfSquares(hogE_raw)
# hogError = ch.SumOfSquares(hogE_raw)

# hogError = negLikModelRobust
# hogCellErrors = pixelLikelihoodRobustCh

models = [negLikModel, negLikModelRobust]
pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh]
modelsDescr = ["Gaussian Model", "Outlier model"]
# , negLikModelPyr, negLikModelRobustPyr, SSqE_raw


# negLikModel2 = -generative_models.modelLogLikelihoodCh(rendererGT, diffRenderer, vis_im, 'FULL', variances)/numPixels
#
# negLikModelRobust2 = -generative_models.modelLogLikelihoodRobustCh(rendererGT, diffRenderer, vis_im, 'FULL', globalPrior, variances)/numPixels
#
# pixelLikelihoodCh2 = generative_models.logPixelLikelihoodCh(rendererGT, diffRenderer, vis_im, 'FULL', variances)
#
# pixelLikelihoodRobustCh2 = ch.log(generative_models.pixelLikelihoodRobustCh(rendererGT, diffRenderer, vis_im, 'FULL', globalPrior, variances))
#
# post2 = generative_models.layerPosteriorsRobustCh(rendererGT, diffRenderer, vis_im, 'FULL', globalPrior, variances)[0]
# pixelModels2 = [pixelLikelihoodCh2, pixelLikelihoodRobustCh2]
# models2 = [negLikModel2, negLikModelRobust2]

model = 1

pixelErrorFun = pixelModels[model]
errorFun = models[model]

# pixelErrorFun2 = pixelModels2[model]
# errorFun2 = models2[model]

# zpolys = image_processing.zernikePolynomials(image=rendererGT.r.copy(), numCoeffs=20)

iterat = 0
changedGT = False
refresh = True
drawSurf = False
makeVideo = False
updateErrorFunctions = False
pendingCyclesRender = True
performance = {}
elevations = {}
azimuths = {}
gradEl = {}
gradAz = {}
performanceSurf = {}
elevationsSurf = {}
azimuthsSurf = {}
gradElSurf = {}
gradAzSurf = {}
gradFinElSurf = {}
gradFinAzSurf = {}
ims = []

# free_variables = [chCosAz, chSinAz, chLogCosEl, chLogSinEl]
free_variables = [chAz, chEl, chVColors, chShCoeffs]

mintime = time.time()
boundEl = (0, np.pi/2.0)
boundAz = (0, None)
boundscomponents = (0,None)
bounds = [boundAz,boundEl]
bounds = [(None , None ) for sublist in free_variables for item in sublist]

methods=['dogleg', 'minimize', 'BFGS', 'L-BFGS-B', 'Nelder-Mead']
methods=['dogleg', 'minimize', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'probLineSearch']
method = 5
exit = False
minimize = False
plotMinimization = False
changeRenderer = False
printStatsBool = False
beginTraining = False
createGroundTruth = False
beginTesting = False
exploreSurfaceBool = False
newTeapotAsGT = False

global chAzSaved
global chElSaved
global chComponentSaved
chAzSaved = chAz.r[0]
chElSaved = chEl.r[0]
chComponentSaved = chComponent.r[0]

if showSubplots:
    f, ((ax1, ax2), (ax3, ax4), (ax5,ax6)) = plt.subplots(3, 2, subplot_kw={'aspect':'equal'}, figsize=(9, 12))
    pos1 = ax1.get_position()
    pos5 = ax5.get_position()
    pos5.x0 = pos1.x0
    ax5.set_position(pos5)

    f.tight_layout()

    ax1.set_title("Ground Truth")

    ax2.set_title("Backprojection")
    rendererIm = lin2srgb(renderer.r.copy())
    pim2 = ax2.imshow(rendererIm)


    edges = renderer.boundarybool_image
    gtoverlay = imageGT().copy()
    gtoverlay = lin2srgb(gtoverlay)
    gtoverlay[np.tile(edges.reshape([shapeIm[0],shapeIm[1],1]),[1,1,3]).astype(np.bool)] = 1
    pim1 = ax1.imshow(gtoverlay)

    # ax3.set_title("Pixel negative log probabilities")
    # pim3 = ax3.imshow(-pixelErrorFun.r)
    # cb3 = plt.colorbar(pim3, ax=ax3,use_gridspec=True)
    # cb3.mappable = pim3
    #
    # ax4.set_title("Posterior probabilities")
    # pim4 = ax4.imshow(np.tile(post.reshape(shapeIm[0],shapeIm[1],1), [1,1,3]))
    # cb4 = plt.colorbar(pim4, ax=ax4,use_gridspec=True)

    diffAz = -ch.optimization.gradCheckSimple(pixelErrorFun, chAz, 0.01745)
    diffEl = -ch.optimization.gradCheckSimple(pixelErrorFun, chEl, 0.01745)

    ax3.set_title("Dr wrt. Azimuth Checkgrad")


    drazsum = np.sign(-diffAz.reshape(shapeIm[0],shapeIm[1],1))*pixelErrorFun.dr_wrt(chAz).reshape(shapeIm[0],shapeIm[1],1)

    drazsumnobnd = -pixelErrorFun.dr_wrt(chAz).reshape(shapeIm[0],shapeIm[1],1)*(1-renderer.boundarybool_image.reshape(shapeIm[0],shapeIm[1],1))
    drazsumbnd = -pixelErrorFun.dr_wrt(chAz).reshape(shapeIm[0],shapeIm[1],1)*(renderer.boundarybool_image.reshape(shapeIm[0],shapeIm[1],1))

    drazsumnobnddiff = diffAz.reshape(shapeIm[0],shapeIm[1],1)*(1-renderer.boundarybool_image.reshape(shapeIm[0],shapeIm[1],1))
    drazsumbnddiff = diffAz.reshape(shapeIm[0],shapeIm[1],1)*(renderer.boundarybool_image.reshape(shapeIm[0],shapeIm[1],1))

    img3 = ax3.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
    cb3 = plt.colorbar(img3, ax=ax3,use_gridspec=True)
    cb3.mappable = img3

    ax4.set_title("Dr wrt. Elevation Checkgrad")
    drazsum = np.sign(-diffEl.reshape(shapeIm[0],shapeIm[1],1))*pixelErrorFun.dr_wrt(chEl).reshape(shapeIm[0],shapeIm[1],1)
    img4 = ax4.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
    cb4 = plt.colorbar(img4, ax=ax4,use_gridspec=True)
    cb4.mappable = img4

    ax5.set_title("Dr wrt. Azimuth")
    drazsum = -pixelErrorFun.dr_wrt(chAz).reshape(shapeIm[0],shapeIm[1],1).reshape(shapeIm[0],shapeIm[1],1)
    img5 = ax5.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
    cb5 = plt.colorbar(img5, ax=ax5,use_gridspec=True)
    cb5.mappable = img5

    ax6.set_title("Dr wrt. Elevation")
    drazsum = -pixelErrorFun.dr_wrt(chEl).reshape(shapeIm[0],shapeIm[1],1).reshape(shapeIm[0],shapeIm[1],1)
    img6 = ax6.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
    cb6 = plt.colorbar(img6, ax=ax6,use_gridspec=True)
    cb6.mappable = img6

    pos1 = ax1.get_position()
    pos5 = ax5.get_position()
    pos5.x0 = pos1.x0
    ax5.set_position(pos5)

    pos1 = ax1.get_position()
    pos5 = ax5.get_position()
    pos5.x0 = pos1.x0
    ax5.set_position(pos5)

    plt.show()
    plt.pause(0.01)

t = time.time()


if makeVideo:
    import matplotlib.animation as animation
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=1, metadata=dict(title='', artist=''), bitrate=1800)
    figvid, (vax1, vax2) = plt.subplots(1, 2, sharey=True, subplot_kw={'aspect':'equal'}, figsize=(12, 6))
    vax1.axes.get_xaxis().set_visible(False)
    vax1.axes.get_yaxis().set_visible(False)
    vax1.set_title("Ground truth")
    vax2.axes.get_xaxis().set_visible(False)
    vax2.axes.get_yaxis().set_visible(False)
    vax2.set_title("Backprojection")

    plt.tight_layout()

if computePerformance:
    from mpl_toolkits.mplot3d import Axes3D
    global figperf
    figperf = plt.figure()
    global axperf
    axperf = figperf.add_subplot(111, projection='3d')
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')
    # x1,x2,y1,y2 = plt.axis()
    # plt.axis((0,360,0,90))
    performance[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    azimuths[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    elevations[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    gradAz[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    gradEl[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

    performanceSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

    gradElSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

    gradFinAzSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
    gradFinElSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

def refreshSubplots():
    #Other subplots visualizing renders and its pixel derivatives

    edges = renderer.boundarybool_image
    imagegt = imageGT()
    gtoverlay = imageGT().copy()
    gtoverlay = lin2srgb(gtoverlay)
    gtoverlay[np.tile(edges.reshape([shapeIm[0],shapeIm[1],1]),[1,1,3]).astype(np.bool)] = 1
    pim1.set_data(gtoverlay)
    rendererIm = lin2srgb(renderer.r.copy())
    pim2.set_data(rendererIm)

    diffAz = -ch.optimization.gradCheckSimple(pixelErrorFun, chAz, 0.01745)
    diffEl = -ch.optimization.gradCheckSimple(pixelErrorFun, chEl, 0.01745)


    if model != 2:
        # ax3.set_title("Pixel negative log probabilities")
        # pim3 = ax3.imshow(-pixelErrorFun.r)
        # cb3.mappable = pim3
        # cb3.update_normal(pim3)
        ax3.set_title("Dr wrt. Azimuth Checkgrad")
        drazsum = -np.sign(diffAz.reshape(shapeIm[0],shapeIm[1],1))*pixelErrorFun.dr_wrt(chAz).reshape(shapeIm[0],shapeIm[1],1)
        # drazsum = drazsum*(renderer.boundarybool_image.reshape(shapeIm[0],shapeIm[1],1))
        img3 = ax3.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
        cb3.mappable = img3
    else:
        ax3.set_title("HoG Image GT")
        pim3 = ax3.imshow(hogImGT.r)
        cb3.mappable = pim3
        cb3.update_normal(pim3)

    if model != 2:
        # ax4.set_title("Posterior probabilities")
        # ax4.imshow(np.tile(post.reshape(shapeIm[0],shapeIm[1],1), [1,1,3]))
        ax4.set_title("Dr wrt. Elevation Checkgrad")
        drazsum = -np.sign(diffEl.reshape(shapeIm[0],shapeIm[1],1))*pixelErrorFun.dr_wrt(chEl).reshape(shapeIm[0],shapeIm[1],1).reshape(shapeIm[0],shapeIm[1],1)
        img4 = ax4.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
        cb4.mappable = img4
    else:

        ax4.set_title("HoG image renderer")
        ax4.set_title("HoG Image GT")
        pim4 = ax4.imshow(hogImRenderer.r)
        cb4.mappable = pim4
        cb4.update_normal(pim4)

    sdy, sdx = pixelErrorFun.shape
    drazsum = -pixelErrorFun.dr_wrt(chAz).reshape(sdy,sdx,1).reshape(sdy,sdx,1)
    # drazsum = drazsum*(renderer.boundarybool_image.reshape(shapeIm[0],shapeIm[1],1))
    img5 = ax5.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
    cb5.mappable = img5
    cb5.update_normal(img5)
    drazsum = -pixelErrorFun.dr_wrt(chEl).reshape(sdy, sdx,1).reshape(sdy,sdx,1)
    img6 = ax6.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
    cb6.mappable = img6
    cb6.update_normal(img6)
    f.canvas.draw()
    plt.pause(0.01)

def plotSurface(model):
    global figperf
    global axperf
    global surf
    global line
    global drawSurf
    global computePerformance
    global plotMinimization
    global chAz
    global chEl
    global chDist
    global chAzGT
    global chElGT
    global chDistGT
    global scene
    if not plotMinimization and not drawSurf:
        figperf.clear()
        global axperf
        axperf = figperf.add_subplot(111, projection='3d')

    plt.figure(figperf.number)
    axperf.clear()
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')
    scaleSurfGrads = 0
    if drawSurf:
        print("Drawing gardient surface.")
        from scipy.interpolate import griddata
        x1 = np.linspace((azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi).min(), (azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi).max(), len((azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi)))
        y1 = np.linspace((elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi).min(), (elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi).max(), len((elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi)))
        x2, y2 = np.meshgrid(x1, y1)
        z2 = griddata(((azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi), (elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi)), performanceSurf[(model, chAzGT.r[0], chElGT.r[0])], (x2, y2), method='cubic')
        from matplotlib import cm, colors
        surf = axperf.plot_surface(x2, y2, z2, rstride=3, cstride=3, cmap=cm.coolwarm, linewidth=0.1, alpha=0.85)

        # scaleSurfGrads = 5./avgSurfGradMagnitudes
        # for point in range(len(performanceSurf[(model, chAzGT.r[0], chElGT.r[0])])):
        #     perfi = performanceSurf[(model, chAzGT.r[0], chElGT.r[0])][point]
        #     azi = azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])][point]
        #     eli = elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])][point]
        #     gradAzi = -gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])][point]
        #     gradEli = -gradElSurf[(model, chAzGT.r[0], chElGT.r[0])][point]
        #     scaleGrad = np.sqrt(gradAzi**2+gradEli**2) / 5
        #
        #     arrowGrad = Arrow3D([azi*180./np.pi, azi*180./np.pi + gradAzi/scaleGrad], [eli*180./np.pi, eli*180./np.pi + gradEli/scaleGrad], [perfi, perfi], mutation_scale=10, lw=1, arrowstyle="-|>", color="b")
        #     axperf.add_artist(arrowGrad)
        #
        #     diffAzi = -gradFinAzSurf[(model, chAzGT.r[0], chElGT.r[0])][point]
        #     diffEli = -gradFinElSurf[(model, chAzGT.r[0], chElGT.r[0])][point]
        #     scaleDiff = np.sqrt(diffAzi**2+diffEli**2) / 5
        #     colorArrow = 'g'
        #     if diffAzi * gradAzi + diffEli * gradEli < 0:
        #         colorArrow = 'r'
        #     arrowGradDiff = Arrow3D([azi*180./np.pi, azi*180./np.pi + diffAzi/scaleDiff], [eli*180./np.pi, eli*180./np.pi + diffEli/scaleDiff], [perfi, perfi], mutation_scale=10, lw=1, arrowstyle="-|>", color=colorArrow)
        #     axperf.add_artist(arrowGradDiff)

        axperf.plot([chAzGT.r[0]*180./np.pi, chAzGT.r[0]*180./np.pi], [chElGT.r[0]*180./np.pi,chElGT.r[0]*180./np.pi], [z2.min(), z2.max()], 'b--', linewidth=1)

        errorFun = models[model]

        # axperf.plot(chAz.r*180./np.pi, chEl.r*180./np.pi, errorFun.r[0], 'yD')
        #
        # import scipy.sparse as sp
        # if sp.issparse(errorFun.dr_wrt(chAz)):
        #     drAz = -errorFun.dr_wrt(chAz).toarray()[0][0]
        # else:
        #     drAz = -errorFun.dr_wrt(chAz)[0][0]
        # if sp.issparse(errorFun.dr_wrt(chEl)):
        #     drEl = -errorFun.dr_wrt(chEl).toarray()[0][0]
        # else:
        #     drEl = -errorFun.dr_wrt(chEl)[0][0]
        # scaleDr = np.sqrt(drAz**2+drEl**2) / 5
        # chAzOldi = chAz.r[0]
        # chElOldi = chEl.r[0]
        # diffAz = -ch.optimization.gradCheckSimple(errorFun, chAz, 0.01745)
        # diffEl = -ch.optimization.gradCheckSimple(errorFun, chEl, 0.01745)
        # scaleDiff = np.sqrt(diffAz**2+diffEl**2) / 5
        # chAz[0] = chAzOldi
        # chEl[0] = chElOldi
        #
        # arrowGrad = Arrow3D([chAz.r[0]*180./np.pi, chAz.r[0]*180./np.pi + drAz/scaleDr], [chEl.r[0]*180./np.pi, chEl.r[0]*180./np.pi + drEl/scaleDr], [errorFun.r[0], errorFun.r[0]], mutation_scale=10, lw=1, arrowstyle="-|>", color="b")
        # axperf.add_artist(arrowGrad)
        # colorArrow = 'g'
        # if diffAz * drAz + diffEl * drEl < 0:
        #     colorArrow = 'r'
        #
        # arrowGradDiff = Arrow3D([chAz.r[0]*180./np.pi, chAz.r[0]*180./np.pi + diffAz/scaleDiff], [chEl.r[0]*180./np.pi, chEl.r[0]*180./np.pi + diffEl/scaleDiff], [errorFun.r[0], errorFun.r[0]], mutation_scale=10, lw=1, arrowstyle="-|>", color=colorArrow)
        # axperf.add_artist(arrowGradDiff)

    if plotMinimization:
        if azimuths.get((model, chAzGT.r[0], chElGT.r[0])) != None:
            axperf.plot(azimuths[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi, elevations[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi, performance[(model, chAzGT.r[0], chElGT.r[0])], color='g', linewidth=1.5)
            axperf.plot(azimuths[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi, elevations[(model, chAzGT.r[0], chElGT.r[0])]*180./np.pi, performance[(model, chAzGT.r[0], chElGT.r[0])], 'rD')

    axperf.set_xlabel('Azimuth (degrees)')
    axperf.set_ylabel('Elevation (degrees)')
    if model == 2:
        axperf.set_zlabel('Squared Error')
    plt.title('Model: ' + modelsDescr[model])

    plt.pause(0.01)
    plt.draw()

def printStats():
    print("**** Statistics ****" )
    print("Relative Azimuth: " + str(chAzRel))
    print("GT Relative Azimuth: " + str(chAzRelGT))
    print("GT Cam Azimuth: " + str(chAzGT))
    print("Cam Azimuth: " + str(chAz))
    print("GT Cam Elevation: " + str(chElGT))
    print("Cam Elevation: " + str(chEl))

    print("Dr wrt cam Azimuth: " + str(errorFun.dr_wrt(chAz)))
    print("Dr wrt cam Elevation: " + str(errorFun.dr_wrt(chEl)))
    # print("Dr wrt Distance: " + str(errorFun.dr_wrt(chDist)))
    print("Occlusion is " + str(getOcclusionFraction(rendererGT)*100) + " %")

    if drawSurf:
        avgError = np.mean(np.sqrt((gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])] - gradFinAzSurf[(model, chAzGT.r[0], chElGT.r[0])])**2 + (gradElSurf[(model, chAzGT.r[0], chElGT.r[0])] - gradFinElSurf[(model, chAzGT.r[0], chElGT.r[0])])**2))
        print("** Approx gradients - finite differenes." )
        print("Avg Eucl. distance :: " + str(avgError))
        norm2Grad = np.sqrt((gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])])**2 + (gradElSurf[(model, chAzGT.r[0], chElGT.r[0])])**2)
        norm2Diff = np.sqrt((gradFinAzSurf[(model, chAzGT.r[0], chElGT.r[0])])**2 + (gradFinElSurf[(model, chAzGT.r[0], chElGT.r[0])])**2)
        avgAngle = np.arccos((gradFinAzSurf[(model, chAzGT.r[0], chElGT.r[0])]*gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])] + gradFinElSurf[(model, chAzGT.r[0], chElGT.r[0])]*gradElSurf[(model, chAzGT.r[0], chElGT.r[0])])/(norm2Grad*norm2Diff))
        print("Avg Angle.: " + str(np.mean(avgAngle)))
        print("Num opposite (red) gradients: " + str(np.sum((gradFinAzSurf[(model, chAzGT.r[0], chElGT.r[0])]*gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])] + gradFinElSurf[(model, chAzGT.r[0], chElGT.r[0])]*gradElSurf[(model, chAzGT.r[0], chElGT.r[0])]) < 0)))
        idxmin = np.argmin(performanceSurf[(model, chAzGT.r[0], chElGT.r[0])])
        azDiff = np.arctan2(np.arcsin(chAzGT - azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])][idxmin]), np.arccos(chAzGT - azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])][idxmin]))
        elDiff = np.arctan2(np.arcsin(chElGT - elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])][idxmin]), np.arccos(chElGT - elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])][idxmin]))
        print("Minimum Azimuth difference of " + str(azDiff*180/np.pi))
        print("Minimum Elevation difference of " + str(elDiff*180/np.pi))

    azDiff = np.arctan2(np.arcsin(chAzGT - chAz.r[0]), np.arccos(chAzGT - chAz.r[0]))
    elDiff = np.arctan2(np.arcsin(chElGT - chEl.r[0]), np.arccos(chElGT - chEl.r[0]))
    print("Current Azimuth difference of " + str(azDiff*180/np.pi))
    print("Current Elevation difference of " + str(elDiff*180/np.pi))

    global printStatsBool
    printStatsBool = False

def cb(_):
    global t
    elapsed_time = time.time() - t
    print("Ended interation in  " + str(elapsed_time))

    global pixelErrorFun
    global errorFun
    global iterat
    iterat = iterat + 1
    print("Callback! " + str(iterat))
    print("Model Log Likelihood: " + str(errorFun.r))
    global imagegt
    global renderer
    global gradAz
    global gradEl
    global performance
    global azimuths
    global elevations

    t = time.time()

def cb2(_):
    global t
    elapsed_time = time.time() - t
    print("Ended interation in  " + str(elapsed_time))

    global pixelErrorFun
    global errorFun
    global iterat
    iterat = iterat + 1
    print("Callback! " + str(iterat))
    print("Model Log Likelihood: " + str(errorFun.r))
    global imagegt
    global renderer
    global gradAz
    global gradEl
    global performance
    global azimuths
    global elevations

    if reduceVariance:
        stds[:] = stds.r[:]*0.9

    if demoMode and refreshWhileMinimizing:
        refreshSubplots()

    if makeVideo:
        plt.figure(figvid.number)
        im1 = vax1.imshow(gtoverlay)

        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8)

        t = vax1.annotate("Minimization iteration: " + str(iterat), xy=(1, 0), xycoords='axes fraction', fontsize=16,
                    xytext=(-20, 5), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props)
        im2 = vax2.imshow(renderer.r)
        ims.append([im1, im2, t])

    if computePerformance and demoMode:
        if performance.get((model, chAzGT.r[0], chElGT.r[0])) == None:
            performance[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            azimuths[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            elevations[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            gradAz[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            gradEl[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
        performance[(model, chAzGT.r[0], chElGT.r[0])] = numpy.append(performance[(model, chAzGT.r[0], chElGT.r[0])], errorFun.r)
        azimuths[(model, chAzGT.r[0], chElGT.r[0])] = numpy.append(azimuths[(model, chAzGT.r[0], chElGT.r[0])], chAz.r)
        elevations[(model, chAzGT.r[0], chElGT.r[0])] = numpy.append(elevations[(model, chAzGT.r[0], chElGT.r[0])], chEl.r)

        import scipy.sparse as sp
        if sp.issparse(errorFun.dr_wrt(chAz)):
            drAz = errorFun.dr_wrt(chAz).toarray()[0][0]
        else:
            drAz = errorFun.dr_wrt(chAz)[0][0]
        if sp.issparse(errorFun.dr_wrt(chEl)):
            drEl = errorFun.dr_wrt(chEl).toarray()[0][0]
        else:
            drEl = errorFun.dr_wrt(chEl)[0][0]

        gradAz[(model, chAzGT.r[0], chElGT.r[0])] = numpy.append(gradAz[(model, chAzGT.r[0], chElGT.r[0])], drAz)
        gradEl[(model, chAzGT.r[0], chElGT.r[0])] = numpy.append(gradEl[(model, chAzGT.r[0], chElGT.r[0])], drEl)

        plotSurface(model)

    if drawSurf and demoMode and refreshWhileMinimizing:
        # plt.pause(0.1)
        plt.show()
        plt.draw()
        plt.pause(0.01)
    t = time.time()

def readKeys(window, key, scancode, action, mods):
    print("Reading keys...")
    global exit
    global refresh
    global chAz
    global chEl
    global phiOffsetGT
    global chComponent
    global changedGT
    refresh = False
    if mods!=glfw.MOD_SHIFT and key == glfw.KEY_ESCAPE and action == glfw.RELEASE:
        glfw.set_window_should_close(window, True)
        exit = True
    if mods!=glfw.MOD_SHIFT and key == glfw.KEY_LEFT and action == glfw.RELEASE:
        refresh = True
        chAz[:] = chAz.r[0] - radians(5)
        azimuth = chAz.r[0] - radians(5)
        # chCosAz[:] = np.cos(azimuth)
        # chSinAz[:] = np.sin(azimuth)

    if mods!=glfw.MOD_SHIFT and key == glfw.KEY_RIGHT and action == glfw.RELEASE:
        refresh = True
        chAz[:]  = chAz.r[0] + radians(5)
        azimuth = chAz.r[0] + radians(5)
        # chCosAz[:] = np.cos(azimuth)
        # chSinAz[:] = np.sin(azimuth)

    if mods!=glfw.MOD_SHIFT and key == glfw.KEY_DOWN and action == glfw.RELEASE:
        refresh = True
        chEl[:] = chEl[0].r - radians(5)
        elevation = chEl[0].r - radians(5)
        if elevation <= 0:
            elevation = 0.0001

        chLogCosEl[:] = np.log(np.cos(elevation))
        chLogSinEl[:] = np.log(np.sin(elevation))

        refresh = True
    if mods!=glfw.MOD_SHIFT and key == glfw.KEY_UP and action == glfw.RELEASE:
        refresh = True
        chEl[:] = chEl[0].r + radians(5)
        elevation = chEl[0].r + radians(5)
        if elevation >= np.pi/2 - 0.0001:
            elevation = np.pi/2 - 0.0002
        chLogCosEl[:] = np.log(np.cos(elevation))
        chLogSinEl[:] = np.log(np.sin(elevation))

    if mods==glfw.MOD_SHIFT and key == glfw.KEY_LEFT and action == glfw.RELEASE:
        refresh = True
        chAz[:] = chAz.r[0] - radians(1)
        # azimuth = chAz.r[0] - radians(1)
        # chCosAz[:] = np.cos(azimuth)
        # chSinAz[:] = np.sin(azimuth)

    if mods==glfw.MOD_SHIFT and key == glfw.KEY_RIGHT and action == glfw.RELEASE:
        refresh = True
        chAz[:]  = chAz.r[0] + radians(1)
        # azimuth = chAz.r[0] + radians(1)
        # chCosAz[:] = np.cos(azimuth)
        # chSinAz[:] = np.sin(azimuth)

    # if mods==glfw.MOD_SHIFT and key == glfw.KEY_LEFT and action == glfw.RELEASE:
    #     print("Left modifier!")
    #     refresh = True
    #     chAzGT[0] = chAzGT[0].r - radians(1)
    # if mods==glfw.MOD_SHIFT and key == glfw.KEY_RIGHT and action == glfw.RELEASE:
    #     refresh = True
    #     # chAz[0] = chAz[0].r + radians(1)
    #     rotation[:] = rotation.r[0] + np.pi/4
    #     # rotation[:] = np.pi/2
    #     shCoeffsRGBGT[:] = np.dot(light_probes.sphericalHarmonicsZRotation(totalOffsetGT.r[:]), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]
    #     chAzGT[:] = rotation.r[:]
    #     # ipdb.set_trace()
    #     shOriginal = chComponentGTOriginal[[0,3,2,1,4,5,6,7,8]]
    #     shOriginalDir = shDirLightGTOriginal[[0,3,2,1,4,5,6,7,8]]
    #     # chComponentGT[:] = np.dot(light_probes.sphericalHarmonicsZRotation(rotation), shOriginal)[[0,3,2,1,4,5,6,7,8]]
    #     # shDirLightGT[:] = np.dot(light_probes.sphericalHarmonicsZRotation(rotation), shOriginalDir)[[0,3,2,1,4,5,6,7,8]]
    #     # shDirLightGT[:] = np.dot(shDirLightGTOriginal.T, light_probes.sphericalHarmonicsZRotation(rotation)).T[:]
    #     # shDirLightGT[:] = np.sum(np.array(light_probes.sphericalHarmonicsZRotation(rotation) * shDirLightGTOriginal[:,None]), axis=1)
    #     # shDirLightGT[:] = chZonalToSphericalHarmonics(zGT, np.pi/2 - chLightElGT, chLightAzGT + rotation[:] - np.pi/2).r[:]
    #     print("Original: " + str(shDirLightGTOriginal))
    #     print(str(shCoeffsRGBGT.r))
    #     print(str(shDirLightGT.r))
    #     print(str(rendererGT.tn[0]))

    if mods==glfw.MOD_SHIFT and key == glfw.KEY_DOWN and action == glfw.RELEASE:
        refresh = True
        chEl[0] = chEl[0].r - radians(1)
        refresh = True
    if mods==glfw.MOD_SHIFT and key == glfw.KEY_UP and action == glfw.RELEASE:
        refresh = True
        chEl[0] = chEl[0].r + radians(1)

    if mods!=glfw.MOD_SHIFT and key == glfw.KEY_X and action == glfw.RELEASE:
        refresh = True
        chScale[0] = chScale[0].r + 0.05

    if mods==glfw.MOD_SHIFT and key == glfw.KEY_X and action == glfw.RELEASE:
        refresh = True
        chScale[0] = chScale[0].r - 0.05
    if mods!=glfw.MOD_SHIFT and key == glfw.KEY_Y and action == glfw.RELEASE:
        refresh = True
        chScale[1] = chScale[1].r + 0.05

    if mods==glfw.MOD_SHIFT and key == glfw.KEY_Y and action == glfw.RELEASE:
        refresh = True
        chScale[1] = chScale[1].r - 0.05
    if mods!=glfw.MOD_SHIFT and key == glfw.KEY_Z and action == glfw.RELEASE:
        refresh = True
        chScale[2] = chScale[2].r + 0.05

    if mods==glfw.MOD_SHIFT and key == glfw.KEY_Z and action == glfw.RELEASE:
        refresh = True
        chScale[2] = chScale[2].r - 0.05
    global errorFun
    if key != glfw.MOD_SHIFT and key == glfw.KEY_C and action == glfw.RELEASE:
        print("Azimuth grad check: ")
        jacs, approxjacs, check = ch.optimization.gradCheck(errorFun, [chAz], [1.49e-08])
        print("Grad check jacs: " + "%.2f" % jacs)
        print("Grad check fin jacs: " + "%.2f" % approxjacs)
        print("Grad check check: " + "%.2f" % check)
        # print("Scipy grad check: " + "%.2f" % ch.optimization.scipyGradCheck({'raw': errorFun}, [chAz]))

        print("Elevation grad check: ")
        jacs, approxjacs, check = ch.optimization.gradCheck(errorFun, [chEl], [1])
        print("Grad check jacs: " + "%.2f" % jacs)
        print("Grad check fin jacs: " + "%.2f" % approxjacs)
        print("Grad check check: " + "%.2f" % check)
        # print("Scipy grad check: " + "%.2f" % ch.optimization.scipyGradCheck({'raw': errorFun}, [chEl]))

        print("Red VColor grad check: ")
        jacs, approxjacs, check = ch.optimization.gradCheck(errorFun, [chVColors[0]], [0.01])
        print("Grad check jacs: " + "%.2f" % jacs)
        print("Grad check fin jacs: " + "%.2f" % approxjacs)
        print("Grad check check: " + "%.2f" % check)
        # print("Scipy grad check: " + "%.2f" % ch.optimization.scipyGradCheck({'raw': errorFun}, [chVColors]))

    if key == glfw.KEY_D:
        refresh = True
        # chComponent[0] = chComponent[0].r + 0.1
    if key == glfw.MOD_SHIFT and glfw.KEY_D:
        refresh = True
        # chComponent[0] = chComponent[0].r - 0.1
    global drawSurf
    global model
    global models
    global updateErrorFunctions
    if key == glfw.KEY_G and action == glfw.RELEASE:
        refresh = True
        changedGT = True
        updateErrorFunctions = True

    global targetPosition
    global center

    global cameraGT
    global rendererGT
    global renderer
    global teapotGT
    global teapot

    global newTeapotAsGT
    if mods==glfw.MOD_SHIFT and key == glfw.KEY_G and action == glfw.RELEASE:

        newTeapotAsGT = True

    global groundTruthBlender
    global blenderRender
    if key != glfw.MOD_SHIFT and key == glfw.KEY_B and action == glfw.RELEASE:
        if useBlender:
            updateErrorFunctions = True
            groundTruthBlender = not groundTruthBlender
            # changedGT = True
            # if groundTruthBlender:
            #         bpy.ops.render.render( write_still=True )
            #         image = cv2.imread(scene.render.filepath)
            #         image = np.float64(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))/255.0
            #         blenderRender = image
            refresh = True

    #Compute in order to plot the surface neighouring the azimuth/el of the gradients and error function.
    global exploreSurfaceBool
    if key == glfw.KEY_E and action == glfw.RELEASE:
        print("Key E pressed?")
        exploreSurfaceBool = True

    if key == glfw.KEY_P and action == glfw.RELEASE:
        ipdb.set_trace()
        refresh = True

    if key == glfw.KEY_N and action == glfw.RELEASE:
        print("Back to GT!")
        chAz[:] = chAzGT.r[:]
        chEl[:] = chElGT.r[:]
        # chComponent[:] = chComponentGT.r[:]
        refresh = True

    global chAzSaved
    global chElSaved
    global chComponentSaved

    if key == glfw.KEY_Z and action == glfw.RELEASE:
        print("Saved!")
        chAzSaved = chAz.r[0]
        chElSaved = chEl.r[0]
        chComponentSaved = chComponent.r[0]

    if key == glfw.KEY_X and action == glfw.RELEASE:
        print("Back to Saved!")
        chAz[0] = chAzSaved
        chEl[0] = chElSaved
        chComponent[0] = chComponentSaved
        refresh = True

    global printStatsBool
    if mods!=glfw.MOD_SHIFT and key == glfw.KEY_S and action == glfw.RELEASE:
        printStatsBool = True

    if key == glfw.KEY_V and action == glfw.RELEASE:
        global ims
        if makeVideo:
            im_ani = animation.ArtistAnimation(figvid, ims, interval=2000, repeat_delay=3000, repeat=False, blit=True)
            im_ani.save('minimization_demo.mp4', fps=None, writer=writer, codec='mp4')
            ims = []

    global stds
    global globalPrior
    global plotMinimization
    if key == glfw.KEY_KP_1 and action == glfw.RELEASE:
        stds[:] = stds.r[0]/1.5
        print("New standard devs of " + str(stds.r))
        refresh = True
        drawSurf = False
        plotMinimization = False
    if key == glfw.KEY_KP_2 and action == glfw.RELEASE:
        stds[:] = stds.r[0]*1.5
        print("New standard devs of " + str(stds.r))
        refresh = True
        drawSurf = False
        plotMinimization = False

    if key == glfw.KEY_KP_4 and action == glfw.RELEASE:
        globalPrior[0] = globalPrior.r[0] - 0.05
        print("New foreground prior of" + str(globalPrior.r))
        refresh = True
        drawSurf = False
        plotMinimization = False
    if key == glfw.KEY_KP_5 and action == glfw.RELEASE:
        globalPrior[0] = globalPrior.r[0] + 0.05
        print("New foreground prior of " + str(globalPrior.r))
        refresh = True
        drawSurf = False
        plotMinimization = False

    global changeRenderer
    global currentTeapotModel
    changeRenderer = False
    if key == glfw.KEY_KP_7 and action == glfw.RELEASE:
        currentTeapotModel = (currentTeapotModel - 1) % len(renderTeapotsList)
        changeRenderer = True
    if key == glfw.KEY_KP_8 and action == glfw.RELEASE:
        currentTeapotModel = (currentTeapotModel + 1) % len(renderTeapotsList)
        changeRenderer = True

    global renderer


    if key == glfw.KEY_R and action == glfw.RELEASE:
        refresh = True

    global pixelErrorFun
    global pixelErrorFun2
    global errorFun2
    global models2
    global pixelModels2
    global model
    global models
    global modelsDescr
    global pixelModels
    global reduceVariance
    if key == glfw.KEY_O and action == glfw.RELEASE:
        # drawSurf = False
        model = (model + 1) % len(models)
        print("Using " + modelsDescr[model])
        errorFun = models[model]
        pixelErrorFun = pixelModels[model]

        # errorFun2 = models2[model]
        # pixelErrorFun2 = pixelModels2[model]

        # if model == 2:
        #     reduceVariance = True
        # else:
        #     reduceVariance = False

        refresh = True

    global method
    global methods
    if key == glfw.KEY_1 and action == glfw.RELEASE:
        method = 0
        print("Changed to minimizer: " + methods[method])
    if key == glfw.KEY_2 and action == glfw.RELEASE:
        method = 1
        print("Changed to minimizer: " + methods[method])
    if key == glfw.KEY_3 and action == glfw.RELEASE:
        method = 2
        print("Changed to minimizer: " + methods[method])
    if key == glfw.KEY_4 and action == glfw.RELEASE:
        print("Changed to minimizer: " + methods[method])
        method = 3
    if key == glfw.KEY_5 and action == glfw.RELEASE:
        method = 4
        print("Changed to minimizer: " + methods[method])

    global minimize
    global free_variables
    if mods==glfw.MOD_SHIFT and key == glfw.KEY_M and action == glfw.RELEASE:
        free_variables = [renderer.v.a.a]
        minimize = True
    if mods!=glfw.MOD_SHIFT and key == glfw.KEY_M and action == glfw.RELEASE:
        free_variables = [chAz, chEl]
        minimize = True

def timeRendering(iterations):
    t = time.time()
    for i in range(iterations):
        chAz[:]  = chAz.r[0] + radians(0.001)
        renderer.r
    print("Per iteration time of " + str((time.time() - t)/iterations))


def timeGradients(iterations):
    t = time.time()
    for i in range(iterations):
            chAz[:] = chAz.r[0] + radians(0.001)
            errorFun.dr_wrt(chAz)
    print("Per iteration time of " + str((time.time() - t)/iterations))

def exploreSurface():
    global drawSurf
    global errorFun
    global refresh
    global model

    if computePerformance:
        print("Estimating cost function surface and gradients...")
        drawSurf = True
        chAzOld = chAz.r[0]
        chElOld = chEl.r[0]

        for model_num, errorFun in enumerate(models):
            performanceSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = np.array([])
            azimuthsSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = np.array([])
            elevationsSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = np.array([])

            gradAzSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = np.array([])
            gradElSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = np.array([])

            gradFinAzSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = np.array([])
            gradFinElSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = np.array([])

        for chAzi in np.linspace(max(chAzGT.r[0]-np.pi/3.,0), min(chAzGT.r[0] + np.pi/3., 2.*np.pi), num=20):
            for chEli in np.linspace(max(chElGT.r[0]-np.pi/2,0), min(chElGT.r[0]+np.pi/2, np.pi/2), num=10):
                for model_num, errorFun in enumerate(models):
                    chAz[:] = chAzi
                    chEl[:] = chEli

                    performanceSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = numpy.append(performanceSurf[(model_num, chAzGT.r[0], chElGT.r[0])], errorFun.r)
                    azimuthsSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = numpy.append(azimuthsSurf[(model_num, chAzGT.r[0], chElGT.r[0])], chAzi)
                    elevationsSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = numpy.append(elevationsSurf[(model_num, chAzGT.r[0], chElGT.r[0])], chEli)
                    import scipy.sparse as sp
                    if sp.issparse(errorFun.dr_wrt(chAz)):
                        drAz = errorFun.dr_wrt(chAz).toarray()[0][0]
                    else:
                        drAz = errorFun.dr_wrt(chAz)[0][0]
                    if sp.issparse(errorFun.dr_wrt(chEl)):
                        drEl = errorFun.dr_wrt(chEl).toarray()[0][0]
                    else:
                        drEl = errorFun.dr_wrt(chEl)[0][0]

                    gradAzSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = numpy.append(gradAzSurf[(model_num, chAzGT.r[0], chElGT.r[0])], drAz)
                    gradElSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = numpy.append(gradElSurf[(model_num, chAzGT.r[0], chElGT.r[0])], drEl)
                    chAzOldi = chAz.r[0]
                    chElOldi = chEl.r[0]
                    diffAz = ch.optimization.gradCheckSimple(errorFun, chAz, 0.01745)
                    diffEl = ch.optimization.gradCheckSimple(errorFun, chEl, 0.01745)
                    chAz[:] = chAzOldi
                    chEl[:] = chElOldi
                    gradFinAzSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = numpy.append(gradFinAzSurf[(model_num, chAzGT.r[0], chElGT.r[0])], diffAz)
                    gradFinElSurf[(model_num, chAzGT.r[0], chElGT.r[0])] = numpy.append(gradFinElSurf[(model_num, chAzGT.r[0], chElGT.r[0])], diffEl)
        errorFun = models[model]

        chAz[:] = chAzOld
        chEl[:] = chElOld

        refresh = True

        if savePerformance:
            def writeStats(model):
                with open('stats/statistics.txt', 'a') as statsFile:

                    statsFile.write("**** Statistics for " + modelsDescr[model] + " ****"  + '\n')
                    if drawSurf:
                        avgError = np.mean(np.sqrt((gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])] - gradFinAzSurf[(model, chAzGT.r[0], chElGT.r[0])])**2 + (gradElSurf[(model, chAzGT.r[0], chElGT.r[0])] - gradFinElSurf[(model, chAzGT.r[0], chElGT.r[0])])**2))
                        statsFile.write("** Approx gradients - finite differenes."  + '\n')
                        statsFile.write("Avg Eucl. distance :: " + str(avgError) + '\n')
                        norm2Grad = np.sqrt((gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])])**2 + (gradElSurf[(model, chAzGT.r[0], chElGT.r[0])])**2)
                        norm2Diff = np.sqrt((gradFinAzSurf[(model, chAzGT.r[0], chElGT.r[0])])**2 + (gradFinElSurf[(model, chAzGT.r[0], chElGT.r[0])])**2)
                        avgAngle = np.arccos((gradFinAzSurf[(model, chAzGT.r[0], chElGT.r[0])]*gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])] + gradFinElSurf[(model, chAzGT.r[0], chElGT.r[0])]*gradElSurf[(model, chAzGT.r[0], chElGT.r[0])])/(norm2Grad*norm2Diff))
                        statsFile.write("Avg Angle.: " + str(np.mean(avgAngle)) + '\n')
                        statsFile.write("Num opposite (red) gradients: " + str(np.sum((gradFinAzSurf[(model, chAzGT.r[0], chElGT.r[0])]*gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])] + gradFinElSurf[(model, chAzGT.r[0], chElGT.r[0])]*gradElSurf[(model, chAzGT.r[0], chElGT.r[0])]) < 0)) + '\n')
                        idxmin = np.argmin(performanceSurf[(model, chAzGT.r[0], chElGT.r[0])])
                        azDiff = np.arctan2(np.arcsin(chAzGT - azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])][idxmin]), np.arccos(chAzGT - azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])][idxmin]))
                        elDiff = np.arctan2(np.arcsin(chElGT - elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])][idxmin]), np.arccos(chElGT - elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])][idxmin]))
                        statsFile.write("Minimum Azimuth difference of " + str(azDiff*180/np.pi) + '\n')
                        statsFile.write("Minimum Elevation difference of " + str(elDiff*180/np.pi) + '\n')

                    azDiff = np.arctan2(np.arcsin(chAzGT - chAz.r[0]), np.arccos(chAzGT - chAz.r[0]))
                    elDiff = np.arctan2(np.arcsin(chElGT - chEl.r[0]), np.arccos(chElGT - chEl.r[0]))
                    statsFile.write("Current Azimuth difference of " + str(azDiff*180/np.pi) + '\n')
                    statsFile.write("Current Elevation difference of " + str(elDiff*180/np.pi) + '\n\n')

            for model_idx, model_i in enumerate(models):
                writeStats(model_idx)
                plotSurface(model_idx)
                plt.savefig('stats/surfaceModel' + modelsDescr[model_idx] + '.png')
        print("Finshed estimating.")


if demoMode:
    glfw.set_key_callback(win, readKeys)
    while not exit:
        # Poll for and process events

        glfw.make_context_current(win)
        glfw.poll_events()

        if newTeapotAsGT:

            rendererGT.makeCurrentContext()

            rendererGT.clear()
            del rendererGT

            removeObjectData(len(v) - targetIndex - 1, v, f_list, vc, vn, uv, haveTextures_list, textures_list)
            addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list,  v_teapots[currentTeapotModel][0], f_list_teapots[currentTeapotModel][0], vc_teapots[currentTeapotModel][0], vn_teapots[currentTeapotModel][0], uv_teapots[currentTeapotModel][0], haveTextures_list_teapots[currentTeapotModel][0], textures_list_teapots[currentTeapotModel][0])

            rendererGT = createRendererGT(glMode, chAzGT, chObjAzGT, chElGT, chDistGT, center, v, vc, f_list, vn, light_colorGT, chComponentGT, chVColorsGT, targetPosition, chDisplacementGT, chScaleGT, width,height, uv, haveTextures_list, textures_list, frustum, win )

            updateErrorFunctions = True
            refresh = True
            changedGT = True

            #Unlink and place the new teapot for Blender.
            if useBlender:
                scene.objects.unlink(teapotGT)
                teapot.matrix_world = mathutils.Matrix.Translation(targetPosition)
                teapotGT = blender_teapots[currentTeapotModel]
                placeNewTarget(scene, teapotGT, targetPosition)
                placeCamera(scene.camera, -chAzGT.r[:].copy()*180/np.pi, chElGT.r[:].copy()*180/np.pi, chDistGT, center)
                scene.update()

            newTeapotAsGT = False

        if printStatsBool:
            printStats()

        if changedGT:
            drawSurf = False
            plotMinimization = False
            imagegt = imageGT()
            chImage[:,:,:] = imagegt[:,:,:]

            chAzGT[:] = chAz.r[:]
            chElGT[:] = chEl.r[:]
            chDistGT[:] = chDist.r[:]
            # chComponentGT[:] = chComponent.r[:]
            # chVColorsGT[:] = chVColors.r[:]

            if makeVideo:
                ims = []

            performance[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            azimuths[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            elevations[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            gradAz[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            gradEl[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

            performanceSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            azimuthsSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            elevationsSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

            gradElSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            gradAzSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

            gradFinAzSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])
            gradFinElSurf[(model, chAzGT.r[0], chElGT.r[0])] = np.array([])

            if useBlender:
                print("Updating Ground Truth blender camera!")
                scene.update()
                center = centerOfGeometry(teapotGT.dupli_group.objects, teapotGT.matrix_world)
                placeCamera(scene.camera, -chAzGT.r[:].copy()*180/np.pi, chElGT.r[:].copy()*180/np.pi, chDistGT, center)
                scene.update()

            if useBlender:
                pendingCyclesRender = True

            if useGTasBackground:
                for teapot_i in range(len(renderTeapotsList)):
                    renderer_i = renderer_teapots[teapot_i]
                    renderer_i.set(background_image=rendererGT.r)

            changedGT = False

        if exploreSurfaceBool:
            print("Explores surface is true??")
            exploreSurface()

            exploreSurfaceBool = False

        if groundTruthBlender and pendingCyclesRender:
            scene.update()
            # scene.layers[0] = False
            # scene.layers[1] = True
            bpy.ops.render.render( write_still=True )

            # image = cv2.imread(scene.render.filepath)
            # image = np.float64(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))/255.0
            image = np.array(imageio.imread(scene.render.filepath))[:,:,0:3]
            image[image>1]=1
            blenderRender = image

            # blenderRenderGray = 0.3*blenderRender[:,:,0] + 0.59*blenderRender[:,:,1] + 0.11*blenderRender[:,:,2]
            # rendererGTGray = 0.3*rendererGT[:,:,0] + 0.59*rendererGT[:,:,1] + 0.11*rendererGT[:,:,2]
            # chAmbientIntensityGT[:] = chAmbientIntensityGT.r*(np.mean(blenderRenderGray,axis=(0,1))/np.mean(rendererGTGray.r,axis=(0,1)))

            pendingCyclesRender = False

        if changeRenderer:
            print("New teapot model " + str(currentTeapotModel))
            drawSurf = False
            plotMinimization = False
            refresh = True
            renderer = renderer_teapots[currentTeapotModel]
            updateErrorFunctions = True
            if useBlender:
                teapot = blender_teapots[currentTeapotModel]
            changeRenderer = False

        if updateErrorFunctions:
            currentGT = rendererGT
            if useBlender and groundTruthBlender:
                image = np.array(imageio.imread(scene.render.filepath))[:,:,0:3]
                image[image>1]=1
                blenderRender = image
                currentGT = blenderRender

            negLikModel = -ch.sum(generative_models.LogGaussianModel(renderer=renderer, groundtruth=currentGT, variances=variances))/numPixels
            negLikModelRobust = -ch.sum(generative_models.LogRobustModel(renderer=renderer, groundtruth=currentGT, foregroundPrior=globalPrior, variances=variances))/numPixels
            pixelLikelihoodCh = generative_models.LogGaussianModel(renderer=renderer, groundtruth=currentGT, variances=variances)
            pixelLikelihoodRobustCh = generative_models.LogRobustModel(renderer=renderer, groundtruth=currentGT, foregroundPrior=globalPrior, variances=variances)

            post = generative_models.layerPosteriorsRobustCh(currentGT, renderer, vis_im, 'FULL', globalPrior, variances)[0]

            # hogGT, hogImGT, drconv = image_processing.diffHog(currentGT, drconv)
            # hogRenderer, hogImRenderer, _ = image_processing.diffHog(renderer, drconv)
            # hogRenderer.dr_wrt(chAz)
            #
            # hogE_raw = hogGT - hogRenderer
            # hogCellErrors = ch.sum(hogE_raw*hogE_raw, axis=2)
            # hogError = ch.SumOfSquares(hogE_raw)

            # models = [negLikModel, negLikModelRobust, hogError]
            models = [negLikModel, negLikModelRobust]
            # pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh, hogCellErrors]
            pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh]
            # pixelModels = [pixelLikelihoodCh, pixelLikelihoodRobustCh]

            pixelErrorFun = pixelModels[model]
            errorFun = models[model]

            updateErrorFunctions = False

        if minimize:
            iterat = 0
            maxiter = 100
            print("Minimizing with method " + methods[method])
            ch.minimize({'raw': errorFun}, bounds=bounds, method=methods[method], x0=free_variables, callback=cb2, options={'disp':False, 'maxiter':maxiter})
            plotMinimization = True
            minimize = False

        if refresh:

            print("Model Log Likelihood: " + str(errorFun.r))

            if showSubplots:
                refreshSubplots()

            if computePerformance and drawSurf:
                plotSurface(model)



        # if demoMode or drawSurf:
        #     plt.pause(0.1)
        #     plt.draw()

            refresh = False

    refreshSubplots()
