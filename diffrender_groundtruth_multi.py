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
import generative_models
import matplotlib.pyplot as plt
from opendr_utils import *
import OpenGL.GL as GL
import light_probes
import imageio
from OpenGL import contextdata
from light_probes import SHProjection
import collision
import copy

plt.ion()

#########################################
# Initialization starts here
#########################################
prefix = 'train4_occlusion_multi'
previousGTPrefix = 'train4_occlusion_shapemodel'

#Main script options:

renderFromPreviousGT = False
useShapeModel = True
renderOcclusions = False
useOpenDR = True
useBlender = True
renderBlender = False
captureEnvMapFromBlender = False
parseSceneInstantiations = True
loadBlenderSceneFile = True
groundTruthBlender = True
useCycles = True
unpackModelsFromBlender = False
unpackSceneFromBlender = False
loadSavedSH = False
glModes = ['glfw','mesa']
glMode = glModes[0]

width, height = (200, 200)
win = -1

if useOpenDR:
    if glMode == 'glfw':
        import glfw

        #Initialize base GLFW context for the Demo and to share context among all renderers.
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.DEPTH_BITS,32)
        glfw.window_hint(glfw.VISIBLE, GL.GL_FALSE)
        # win = glfw.create_window(width, height, "Demo",  None, None)
        # glfw.make_context_current(win)

angle = 60 * 180 / numpy.pi
clip_start = 0.01
clip_end = 10
frustum = {'near': clip_start, 'far': clip_end, 'width': width, 'height': height}
camDistance = 0.6

teapots = [line.strip() for line in open('teapots.txt')]
renderTeapotsList = np.arange(len(teapots))[0:1]
mugs = [line.strip() for line in open('mugs.txt')]
renderMugsList = np.arange(len(teapots))[0:1]

sceneIdx = 0
replaceableScenesFile = '../databaseFull/fields/scene_replaceables_backup.txt'
sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndices, targetPositions = scene_io_utils.getSceneInformation(sceneIdx, replaceableScenesFile)
sceneDicFile = 'data/scene' + str(sceneNumber) + '.pickle'
targetParentIdx = 0
targetIndex = targetIndices[targetParentIdx]
targetParentPosition = targetPositions[targetParentIdx]
targetPosition = targetParentPosition

tex_srgb2lin =  True

v, f_list, vc, vn, uv, haveTextures_list, textures_list = scene_io_utils.loadSavedScene(sceneDicFile, tex_srgb2lin)

removeObjectData(int(targetIndex), v, f_list, vc, vn, uv, haveTextures_list, textures_list)

targetModels = []
mugModels = []
blender_teapots = []
selection = [ teapots[i] for i in renderTeapotsList]
scene_io_utils.loadTargetsBlendData()
for teapotIdx, teapotName in enumerate(selection):
    teapot = bpy.data.scenes[teapotName[0:63]].objects['teapotInstance' + str(renderTeapotsList[teapotIdx])]
    teapot.layers[1] = True
    teapot.layers[2] = True
    targetModels = targetModels + [teapot]
    blender_teapots = blender_teapots + [teapot]

v_teapots, f_list_teapots, vc_teapots, vn_teapots, uv_teapots, haveTextures_list_teapots, textures_list_teapots, vflat, varray, center_teapots = scene_io_utils.loadTeapotsOpenDRData(renderTeapotsList, useBlender, unpackModelsFromBlender, targetModels)

blender_mugs = []
selectionMugs = [ mugs[i] for i in renderMugsList]
scene_io_utils.loadMugsBlendData()
for mugIdx, mugName in enumerate(selectionMugs):
    mug = bpy.data.scenes[mugName[0:63]].objects['mugInstance' + str(renderMugsList[mugIdx])]
    mug.layers[1] = True
    mug.layers[2] = True
    mugModels = mugModels + [mug]
    blender_mugs = blender_mugs + [mug]
v_mugs, f_list_mugs, vc_mugs, vn_mugs, uv_mugs, haveTextures_list_mugs, textures_list_mugs, vflat_mugs, varray_mugs, center_mugs = scene_io_utils.loadMugsOpenDRData(renderMugsList, useBlender, unpackModelsFromBlender, mugModels)

v_mug = v_mugs[0][0]
f_list_mug = f_list_mugs[0][0]
chVColorsMug = ch.Ch([1,0,0])
vc_mug = [chVColorsMug * np.ones(v_mug[0].shape)]
vn_mug = vn_mugs[0][0]
uv_mug = uv_mugs[0][0]
haveTextures_list_mug = haveTextures_list_mugs[0][0]
textures_list_mug = textures_list_mugs[0][0]

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
chEl =  ch.Ch([0.0])
chDist = ch.Ch([camDistance])

chObjAzGT = ch.Ch([0])
chAzGT = ch.Ch([0])
chAzRelGT = chAzGT - chObjAzGT
chElGT = ch.Ch(chEl.r[0])
chDistGT = ch.Ch([camDistance])
chComponentGT = ch.Ch(np.array([2, 0.25, 0.25, 0.12,-0.17,0.36,0.1,0.,0.]))
chComponent = ch.Ch(np.array([2, 0.25, 0.25, 0.12,-0.17,0.36,0.1,0.,0.]))

chObjDistGT = ch.Ch([0])
chObjRotationGT = ch.Ch([0])
chObjAzMug = ch.Ch([0])
chObjDistMug = ch.Ch([0])
chObjRotationMug = ch.Ch([0])


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
chVColors = ch.Ch([0.8,0.8,0.8])
chVColorsGT = ch.Ch([0.8,0.8,0.8])

shCoefficientsFile = 'data/sceneSH' + str(sceneIdx) + '.pickle'

chAmbientIntensityGT = ch.Ch([0.1])
clampedCosCoeffs = clampedCosineCoefficients()
chAmbientSHGT = ch.zeros([9])

envMapDic = {}
SHFilename = 'data/LightSHCoefficients.pickle'

with open(SHFilename, 'rb') as pfile:
    envMapDic = pickle.load(pfile)

phiOffset = ch.Ch([0])
totalOffset = phiOffset + chObjAzGT
envMapCoeffs = ch.Ch(list(envMapDic.items())[0][1][1])

envMapCoeffsRotated = ch.Ch(np.dot(light_probes.chSphericalHarmonicsZRotation(totalOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]])
envMapCoeffsRotatedRel = ch.Ch(np.dot(light_probes.chSphericalHarmonicsZRotation(phiOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]])

shCoeffsRGB = envMapCoeffsRotated
shCoeffsRGBRel = envMapCoeffsRotatedRel
chShCoeffs = 0.3*shCoeffsRGB[:,0] + 0.59*shCoeffsRGB[:,1] + 0.11*shCoeffsRGB[:,2]
chShCoeffsRel = 0.3*shCoeffsRGBRel[:,0] + 0.59*shCoeffsRGBRel[:,1] + 0.11*shCoeffsRGBRel[:,2]
chAmbientSHGT = chShCoeffs.ravel() * chAmbientIntensityGT * clampedCosCoeffs
chAmbientSHGTRel = chShCoeffsRel.ravel() * chAmbientIntensityGT * clampedCosCoeffs

chLightRadGT = ch.Ch([0.1])
chLightDistGT = ch.Ch([0.5])
chLightIntensityGT = ch.Ch([0])
chLightAzGT = ch.Ch([np.pi*3/2])
chLightElGT = ch.Ch([np.pi/4])
angle = ch.arcsin(chLightRadGT/chLightDistGT)
zGT = chZonalHarmonics(angle)
shDirLightGT = chZonalToSphericalHarmonics(zGT, np.pi/2 - chLightElGT, chLightAzGT + chObjAzGT - np.pi/2) * clampedCosCoeffs
shDirLightGTRel = chZonalToSphericalHarmonics(zGT, np.pi/2 - chLightElGT, chLightAzGT - np.pi/2) * clampedCosCoeffs
chComponentGT = chAmbientSHGT
# chComponentGT = ch.Ch(chAmbientSHGT.r[:].copy())
 # + shDirLightGT*chLightIntensityGT
chComponentGTRel = chAmbientSHGTRel
# chComponentGTRel = ch.Ch(chAmbientSHGTRel.r[:].copy())
# chComponentGT = chAmbientSHGT.r[:] + shDirLightGT.r[:]*chLightIntensityGT.r[:]

chDisplacement = ch.Ch([0.0, 0.0,0.0])
chDisplacementGT = ch.Ch([0.0,0.0,0.0])
chScale = ch.Ch([1.0,1.0,1.0])
chScaleGT = ch.Ch([1, 1.,1.])

currentTeapotModel = 0
currentMugModel = 0

addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list,  v_teapots[currentTeapotModel][0], f_list_teapots[currentTeapotModel][0], vc_teapots[currentTeapotModel][0], vn_teapots[currentTeapotModel][0], uv_teapots[currentTeapotModel][0], haveTextures_list_teapots[currentTeapotModel][0], textures_list_teapots[currentTeapotModel][0])

addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list,  v_mugs[currentMugModel][0], f_list_mugs[currentMugModel][0], vc_mugs[currentMugModel][0], vn_mugs[currentMugModel][0], uv_mugs[currentMugModel][0], haveTextures_list_mugs[currentMugModel][0], textures_list_mugs[currentMugModel][0])

center = center_teapots[currentTeapotModel]

if useOpenDR:

    rendererGT = createRendererGT(glMode, chAzGT, chElGT, chDistGT, center, v, vc, f_list, vn, light_colorGT, chComponentGT, chVColorsGT, targetPosition[:].copy(), chDisplacementGT, width,height, uv, haveTextures_list, textures_list, frustum, None )

    vis_gt = np.array(rendererGT.indices_image!=1).copy().astype(np.bool)
    vis_mask = np.array(rendererGT.indices_image==1).copy().astype(np.bool)

    shapeIm = vis_gt.shape

numPixels = height * width

def imageGT():
    global groundTruthBlender
    global rendererGT
    global blenderRender

    if groundTruthBlender:
        return blenderRender
    else:
        return np.copy(np.array(rendererGT.r)).astype(np.float64)

import multiprocessing
numTileAxis = np.ceil(np.sqrt(multiprocessing.cpu_count())/2)
numTileAxis = 3

#########################################
# Initialization ends here
#########################################

if useShapeModel:
    teapot_i = -1

    import shape_model
    #%% Load data
    filePath = 'data/teapotModel.pkl'
    teapotModel = shape_model.loadObject(filePath)
    faces = teapotModel['faces']

    #%% Sample random shape Params
    latentDim = np.shape(teapotModel['ppcaW'])[1]
    shapeParams = np.random.randn(latentDim)
    chShapeParamsGT = ch.Ch(shapeParams)

    meshLinearTransform=teapotModel['meshLinearTransform']
    W=teapotModel['ppcaW']
    b=teapotModel['ppcaB']

    chVerticesGT = shape_model.VerticesModel(chShapeParams=chShapeParamsGT,meshLinearTransform=meshLinearTransform,W = W,b=b)
    chVerticesGT.init()

    chVerticesGT = ch.dot(geometry.RotateZ(-np.pi/2)[0:3,0:3],chVerticesGT.T).T

    chNormalsGT = shape_model.chGetNormals(chVerticesGT, faces)

    smNormalsGT = [chNormalsGT]
    smFacesGT = [[faces]]
    smVColorsGT = [chVColorsGT*np.ones(chVerticesGT.shape)]
    smUVsGT = [ch.Ch(np.zeros([chVerticesGT.shape[0],2]))]
    smHaveTexturesGT = [[False]]
    smTexturesListGT = [[None]]

    chVerticesGT = chVerticesGT - ch.mean(chVerticesGT, axis=0)
    minZ = ch.min(chVerticesGT[:,2])

    chMinZ = ch.min(chVerticesGT[:,2])

    zeroZVerts = chVerticesGT[:,2]- chMinZ
    chVerticesGT = ch.hstack([chVerticesGT[:,0:2] , zeroZVerts.reshape([-1,1])])

    chVerticesGT = chVerticesGT*0.09
    smCenterGT = ch.array([0,0,0.1])

    smVerticesGT = [chVerticesGT]
    chNormalsGT = shape_model.chGetNormals(chVerticesGT, faces)
    smNormalsGT = [chNormalsGT]

else:
    latentDim = 1
    chShapeParamsGT = ch.array([0])

print("Creating Ground Truth")

trainAzsGT = np.array([])
trainObjAzsGT = np.array([])
trainElevsGT = np.array([])
trainLightAzsGT = np.array([])
trainLightElevsGT = np.array([])
trainLightIntensitiesGT = np.array([])
trainVColorGT = np.array([])
trainScenes = np.array([], dtype=np.uint8)
trainTeapotIds = np.array([], dtype=np.uint8)
trainEnvMaps = np.array([], dtype=np.uint8)
trainOcclusions = np.array([])
trainTargetIndices = np.array([], dtype=np.uint8)
trainIds = np.array([], dtype=np.uint32)
#zeros

trainLightCoefficientsGT = np.array([]).reshape([0,9])
trainLightCoefficientsGTRel = np.array([]).reshape([0,9])
trainAmbientIntensityGT = np.array([])
trainEnvMapPhiOffsets = np.array([])
trainShapeModelCoeffsGT = np.array([]).reshape([0,latentDim])

gtDir = 'groundtruth/' + prefix + '/'
if not os.path.exists(gtDir + 'images/'):
    os.makedirs(gtDir + 'images/')

if not os.path.exists(gtDir + 'sphericalharmonics/'):
    os.makedirs(gtDir + 'sphericalharmonics/')

if not os.path.exists(gtDir + 'images_opendr/'):
    os.makedirs(gtDir + 'images_opendr/')

if not os.path.exists(gtDir + 'masks_occlusion/'):
    os.makedirs(gtDir + 'masks_occlusion/')

print("Generating renders")

sceneLines = [line.strip() for line in open(replaceableScenesFile)]
scenesToRender = range(len(sceneLines))[:]

trainSize = 20000

renderTeapotsList = np.arange(len(teapots))[0:1]

# for hdrit, hdri in enumerate(list(envMapDic.items())):
#     if hdri[0] == 'data/hdr/dataset/canada_montreal_nad_photorealism.exr':
#         hdrtorenderi = hdrit

ignoreEnvMaps = np.loadtxt('data/bad_envmaps.txt')

hdritems = list(envMapDic.items())[:]
hdrstorender = []
phiOffsets = [0, np.pi/2, np.pi, 3*np.pi/2]

for hdrFile, hdrValues in hdritems:
    hdridx = hdrValues[0]
    envMapCoeffs = hdrValues[1]
    if hdridx not in ignoreEnvMaps:
        hdrstorender = hdrstorender + [(hdrFile,hdrValues)]


    # if not os.path.exists('light_probes/envMap' + str(hdridx)):
    #     os.makedirs('light_probes/envMap' + str(hdridx))
    #
    # for phiOffset in phiOffsets:
    #
    #     # phiOffset = np.random.uniform(0,2*np.pi, 1)
    #     from numpy.random import choice
    #     objAzGT = np.pi/2
    #     chObjAzGT[:] = 0
    #     totalOffset = phiOffset + chObjAzGT.r
    #     envMapCoeffsRotated = np.dot(light_probes.sphericalHarmonicsZRotation(totalOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]].copy()
    #     envMapCoeffsRotatedRel = np.dot(light_probes.sphericalHarmonicsZRotation(phiOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]].copy()
    #     shCoeffsRGB = envMapCoeffsRotated.copy()
    #     shCoeffsRGBRel = envMapCoeffsRotatedRel.copy()
    #     chShCoeffs = 0.3*shCoeffsRGB[:,0] + 0.59*shCoeffsRGB[:,1] + 0.11*shCoeffsRGB[:,2]
    #     chShCoeffsRel = 0.3*shCoeffsRGBRel[:,0] + 0.59*shCoeffsRGBRel[:,1] + 0.11*shCoeffsRGBRel[:,2]
    #     chAmbientSHGT = chShCoeffs * chAmbientIntensityGT * clampedCosCoeffs
    #     chAmbientSHGTRel = chShCoeffsRel * chAmbientIntensityGT * clampedCosCoeffs
    #     chComponentGT[:] = chAmbientSHGT.r[:].copy()
    #     chComponentGTRel[:] = chAmbientSHGTRel.r[:].copy()
    #     cv2.imwrite('light_probes/envMap' + str(hdridx) + '/opendr_' + str(np.int(180*phiOffset/np.pi)) + '.png' , 255*rendererGT.r[:,:,[2,1,0]])
# sys.exit("")

gtDtype = [('trainIds', trainIds.dtype.name), ('trainAzsGT', trainAzsGT.dtype.name),('trainObjAzsGT', trainObjAzsGT.dtype.name),('trainElevsGT', trainElevsGT.dtype.name),('trainLightAzsGT', trainLightAzsGT.dtype.name),('trainLightElevsGT', trainLightElevsGT.dtype.name),('trainLightIntensitiesGT', trainLightIntensitiesGT.dtype.name),('trainVColorGT', trainVColorGT.dtype.name, (3,) ),('trainScenes', trainScenes.dtype.name),('trainTeapotIds', trainTeapotIds.dtype.name),('trainEnvMaps', trainEnvMaps.dtype.name),('trainOcclusions', trainOcclusions.dtype.name),('trainTargetIndices', trainTargetIndices.dtype.name), ('trainLightCoefficientsGT',trainLightCoefficientsGT.dtype, (9,)), ('trainLightCoefficientsGTRel', trainLightCoefficientsGTRel.dtype, (9,)), ('trainAmbientIntensityGT', trainAmbientIntensityGT.dtype), ('trainEnvMapPhiOffsets', trainEnvMapPhiOffsets.dtype), ('trainShapeModelCoeffsGT', trainShapeModelCoeffsGT.dtype, (latentDim,))]

groundTruth = np.array([], dtype = gtDtype)
groundTruthFilename = gtDir + 'groundTruth.h5'
gtDataFile = h5py.File(groundTruthFilename, 'a')

gtDataFileToRender = h5py.File(gtDir + 'groundTruthToRender.h5', 'w')
gtDatasetToRender = gtDataFileToRender.create_dataset(prefix, data=groundTruth, maxshape=(None,))

nextId = 0
try:
    gtDataset = gtDataFile[prefix]
    if gtDataset.size > 0:
        nextId = gtDataset['trainIds'][-1] + 1
except:
    gtDataset = gtDataFile.create_dataset(prefix, data=groundTruth, maxshape=(None,))

train_i = nextId

#Re-producible groundtruth generation.

if train_i == 0:
    np.random.seed(1)
unlinkedObj = None

scenesToRenderOcclusions = []
scenes = []
lenScenes = 0

#Compute how many different locations can the teapot be instantiated across all scenes.
for sceneIdx in scenesToRender:

    sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndices, targetPositions = scene_io_utils.getSceneInformation(sceneIdx, replaceableScenesFile)
    sceneDicFile = 'data/scene' + str(sceneNumber) + '.pickle'

    if renderOcclusions:
        targetIndicesNew = []
        occlusionSceneFile = 'data/occlusions/occlusionScene' + str(sceneNumber) + '.pickle'
        with open(occlusionSceneFile, 'rb') as pfile:
            occlusions = pickle.load(pfile)

        for targetidx, targetIndex in enumerate(targetIndices):
            if not occlusions[targetIndex][1]:
                print("Scene idx " + str(sceneIdx) + " at index " + str(targetIndex) + " has no proper occlusion.")
            else:
                targetIndicesNew = targetIndicesNew + [targetIndex]
        targetIndices = targetIndicesNew

    collisionSceneFile = 'data/collisions/collisionScene' + str(sceneNumber) + '.pickle'
    scenes = scenes + [targetIndices]
    with open(collisionSceneFile, 'rb') as pfile:
        collisions = pickle.load(pfile)

    for targetidx, targetIndex in enumerate(targetIndices):
        if not collisions[targetIndex][1]:
            print("Scene idx " + str(sceneIdx) + " at index " + str(targetIndex) + " collides everywhere.")

    lenScenes += len(targetIndices)

#Generate GT labels before rendering them.
if not renderFromPreviousGT:

    for scene_i, sceneIdx in enumerate(scenesToRender):

        print("Generating groundtruth for scene: " + str(sceneIdx))
        sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndicesScene, targetPositions = scene_io_utils.getSceneInformation(sceneIdx, replaceableScenesFile)

        targetIndices = scenes[scene_i]
        if not targetIndices:
            continue

        sceneDicFile = 'data/scene' + str(sceneNumber) + '.pickle'

        collisionSceneFile = 'data/collisions/collisionScene' + str(sceneNumber) + '.pickle'
        with open(collisionSceneFile, 'rb') as pfile:
            collisions = pickle.load(pfile)

        if renderOcclusions:
            occlusionSceneFile = 'data/occlusions/occlusionScene' + str(sceneNumber) + '.pickle'
            with open(occlusionSceneFile, 'rb') as pfile:
                occlusions = pickle.load(pfile)

        v2, f_list2, vc2, vn2, uv2, haveTextures_list2, textures_list2 = scene_io_utils.loadSavedScene(sceneDicFile, tex_srgb2lin)
        if useBlender and not loadBlenderSceneFile:
            bpy.ops.wm.read_factory_settings()
            scene = scene_io_utils.loadBlenderScene(sceneIdx, replaceableScenesFile)
            scene_io_utils.setupScene(scene, roomInstanceNum, scene.world, scene.camera, width, height, 16, useCycles, True)
            scene.update()
            # Save barebones scene.
        elif useBlender and loadBlenderSceneFile:
            bpy.ops.wm.read_factory_settings()
            scene_io_utils.loadSceneBlendData(sceneIdx, replaceableScenesFile)
            scene = bpy.data.scenes['Main Scene']

        # Configure scene
        if useBlender:
            targetModels = []
            blender_teapots = []
            teapots = [line.strip() for line in open('teapots.txt')]
            selection = [teapots[i] for i in renderTeapotsList]
            scene_io_utils.loadTargetsBlendData()
            for teapotIdx, teapotName in enumerate(selection):
                teapot = bpy.data.scenes[teapotName[0:63]].objects[
                    'teapotInstance' + str(renderTeapotsList[teapotIdx])]
                teapot.layers[1] = True
                teapot.layers[2] = True
                targetModels = targetModels + [teapotIdx]
                blender_teapots = blender_teapots + [teapot]

            blender_mugs = []
            selectionMugs = [mugs[i] for i in renderMugsList]
            scene_io_utils.loadMugsBlendData()
            for mugIdx, mugName in enumerate(selectionMugs):
                mug = bpy.data.scenes[mugName[0:63]].objects['mugInstance' + str(renderMugsList[mugIdx])]
                mug.layers[1] = True
                mug.layers[2] = True
                mugModels = mugModels + [mug]
                blender_mugs = blender_mugs + [mug]

            setupSceneGroundtruth(scene, width, height, clip_start, 1000, 'CUDA', 'CUDA_0')

            treeNodes = scene.world.node_tree

            links = treeNodes.links

            cubeScene = createCubeScene(scene)
            setEnviornmentMapStrength(20, cubeScene)
            # cubeScene.render.engine = 'BLENDER_RENDER'


        unlinkedObj = None
        unlinkedCubeObj = None
        envMapFilename = None

        for targetidx, targetIndex in enumerate(targetIndices):
            targetPosition = targetPositions[np.where(targetIndex==np.array(targetIndicesScene))[0]]

            # if sceneIdx != currentScene or targetIndex != currentTargetIndex:
        #     targetPosition = targetPositions[np.where(targetIndex == np.array(targetIndicesScene))[0]]

            v, f_list, vc, vn, uv, haveTextures_list, textures_list = copy.deepcopy(v2), copy.deepcopy(f_list2), copy.deepcopy(vc2), copy.deepcopy(vn2)\
                , copy.deepcopy(uv2), copy.deepcopy(haveTextures_list2), copy.deepcopy(textures_list2)

            removeObjectData(len(v) - 1 - targetIndex, v, f_list, vc, vn, uv, haveTextures_list, textures_list)

        # if sceneIdx != currentScene or targetIndex != currentTargetIndex:
            if useBlender:
                if unlinkedObj != None:
                    scene.objects.link(unlinkedObj)
                    cubeScene.objects.link(unlinkedCubeObj)
                unlinkedObj = scene.objects[str(targetIndex)]
                unlinkedCubeObj = cubeScene.objects['cube' + str(targetIndex)]
                scene.objects.unlink(unlinkedObj)
                cubeScene.objects.unlink(unlinkedCubeObj)

                parentIdx = instances[targetIndex][1]
                parentSupportObj = scene.objects[str(parentIdx)]
                supportWidthMax, supportWidthMin = modelWidth(parentSupportObj.dupli_group.objects,
                                                              parentSupportObj.matrix_world)
                supportDepthMax, supportDepthMin = modelDepth(parentSupportObj.dupli_group.objects,
                                                              parentSupportObj.matrix_world)
                # supportRad = min(0.5 * np.sqrt((supportDepthMax - supportDepthMin) ** 2),0.5 * np.min((supportWidthMax - supportWidthMin) ** 2))
                supportRad = np.sqrt(
                    0.5 * (supportDepthMax - supportDepthMin) ** 2 + 0.5 * (supportWidthMax - supportWidthMin) ** 2)

                sceneCollisions = {}
                collisionsFile = 'data/collisions/discreteCollisions_scene' + str(scene_i) + '_targetIdx' + str(targetIndex) + '.pickle'

                if not parseSceneInstantiations:
                    with open(collisionsFile, 'rb') as pfile:
                        sceneCollisions = pickle.load(pfile)

                    supportRad = sceneCollisions['supportRad']
                    instantiationBins = sceneCollisions['instantiationBins']
                    cubeRoomObj = sceneCollisions['cubeRoomObj']
                    distRange = sceneCollisions['distRange']
                    rotationRange = sceneCollisions['rotationRange']
                    distInterval = sceneCollisions['distInterval']
                    instantiationBinsMug = sceneCollisions['instantiationBinsMug']
                    totalBinsTeapot = sceneCollisions['totalBinsTeapot']
                    totalBinsMug = sceneCollisions['totalBinsMug']

                else:
                    placeCamera(cubeScene.camera, 0,
                                45, chDistGT.r[0].copy(),
                                center[:].copy() + targetPosition[:].copy())

                    scaleZ = 0.2
                    scaleY = 0.2
                    scaleX = 0.2

                    cubeTeapot = createCube(scaleX, scaleY, scaleZ, 'cubeTeapot')
                    cubeTeapot.matrix_world = mathutils.Matrix.Translation(targetPosition)

                    # cubeTeapot = getCubeObj(teapot)
                    # cubeMug = getCubeObj(mug)

                    scaleZ = 0.1
                    scaleY = 0.1
                    scaleX = 0.1

                    cubeMug = createCube(scaleX, scaleY, scaleZ, 'cubeMug')
                    cubeMug.matrix_world = mathutils.Matrix.Translation(targetPosition)

                    cubeScene.objects.link(cubeTeapot)

                    cubeParentSupportObj = cubeScene.objects['cube' + str(parentIdx)]
                    cubeRoomObj = cubeScene.objects['cube' + str(roomInstanceNum)]
                    cubeScene.update()

                    distRange = min(supportRad, 0.5)
                    distInterval = 0.05
                    rotationRange = 2 * np.pi
                    rotationInterval = 10 * np.pi / 180

                    objDisplacementMat = computeHemisphereTransformation(chObjRotationGT, 0, chObjDistGT,
                                                                         np.array([0, 0, 0]))
                    objOffset = objDisplacementMat[0:3, 3]

                    instantiationBinsTeapot, totalBinsTeapot = collision.parseSceneCollisions(gtDir, scene_i, targetIndex, cubeTeapot,
                                                                                  cubeScene, objOffset, chObjDistGT,
                                                                                  chObjRotationGT, cubeParentSupportObj,
                                                                                  cubeRoomObj, distRange, rotationRange,
                                                                                  distInterval, rotationInterval)
                    cubeScene.objects.unlink(cubeTeapot)

                    cubeScene.objects.link(cubeMug)

                    instantiationBinsMug, totalBinsMug = collision.parseSceneCollisions(gtDir, scene_i, targetIndex, cubeMug,
                                                                                        cubeScene, objOffset,
                                                                                        chObjDistGT, chObjRotationGT,
                                                                                        cubeParentSupportObj, cubeRoomObj,
                                                                                        distRange,
                                                                                        rotationRange, distInterval,
                                                                                        rotationInterval)
                    cubeScene.objects.unlink(cubeMug)

                    deleteObject(cubeTeapot)
                    deleteObject(cubeMug)

                    sceneCollisions = {'totalBinsTeapot':totalBinsTeapot,'totalBinsMug':totalBinsMug, 'supportRad':supportRad,'instantiationBinsTeapot':instantiationBinsTeapot,'cubeRoomObj':cubeRoomObj,'distRange':distRange,'rotationRange':rotationRange,'distInterval':distInterval,'instantiationBinsMug':instantiationBinsMug}
                    with open(collisionsFile, 'wb') as pfile:
                        pickle.dump(sceneCollisions, pfile)

                #Go to next target Index or scene if there are no plausible place instantiations for teapot or mug.
                if instantiationBinsTeapot.sum() < 4 or instantiationBinsMug.sum() < 4:
                    print("This scene position has not place to instantiate the objects.")
                    pass

                        # if useShapeModel
            for teapot_i in renderTeapotsList:

                if useShapeModel:
                    teapot_i = -1
                else:
                    currentTeapotModel = teapot_i
                    center = center_teapots[teapot_i]

                print("Ground truth on new teapot" + str(teapot_i))

                ##Destroy and create renderer
                if useOpenDR:
                    rendererGT.makeCurrentContext()
                    rendererGT.clear()
                    contextdata.cleanupContext(contextdata.getContext())
                    if glMode == 'glfw':
                        glfw.destroy_window(rendererGT.win)
                    del rendererGT

                    currentTeapotModel = teapot_i
                    center = center_teapots[teapot_i]

                    removeObjectData(0, v, f_list, vc, vn, uv, haveTextures_list, textures_list)

                    if useShapeModel:
                        center = smCenterGT
                        UVs = smUVsGT
                        vGT = smVerticesGT
                        vnGT = smNormalsGT
                        Faces = smFacesGT
                        VColors = smVColorsGT
                        UVs = smUVsGT
                        HaveTextures = smHaveTexturesGT
                        TexturesList = smTexturesListGT
                    else:
                        vGT, vnGT = v_teapots[currentTeapotModel][0], vn_teapots[currentTeapotModel][0]
                        Faces = f_list_teapots[currentTeapotModel][0]
                        VColors = vc_teapots[currentTeapotModel][0]
                        UVs = uv_teapots[currentTeapotModel][0]
                        HaveTextures = haveTextures_list_teapots[currentTeapotModel][0]
                        TexturesList = textures_list_teapots[currentTeapotModel][0]

                    vGT, vnGT, teapotPosOffset = transformObject(vGT, vnGT, chScaleGT, chObjAzGT, chObjDistGT, chObjRotationGT, targetPosition)

                    verticesMug, normalsMug, mugPosOffset = transformObject(v_mug, vn_mug, chScale, chObjAzMug - np.pi / 2, chObjDistMug, chObjRotationMug, targetPosition)

                    VerticesB = vGT + verticesMug
                    NormalsB = vnGT + normalsMug
                    FacesB = Faces + f_list_mug
                    VColorsB = VColors + vc_mug
                    UVsB = UVs + uv_mug
                    HaveTexturesB = HaveTextures + haveTextures_list_mug
                    TexturesListB = TexturesList + textures_list_mug
                    addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list, verticesMug, f_list_mug, vc_mug,  normalsMug, uv_mug, haveTextures_list_mug, textures_list_mug)
                    addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list, vGT, Faces, VColors, vnGT, UVs, HaveTextures, TexturesList)

                    rendererGT = createRendererGT(glMode, chAzGT, chElGT, chDistGT, center, v, vc, f_list, vn, light_colorGT, chComponentGT, chVColorsGT, targetPosition.copy(), chDisplacementGT, width, height, uv, haveTextures_list, textures_list, frustum, None)
                ## Blender: Unlink and link new teapot.
                if useBlender:
                    # if currentScene != -1 and currentTargetIndex != -1 and currentTeapot != -1 and teapot != None:
                    if teapot.name in scene.objects:
                        scene.objects.unlink(teapot)

                        if useShapeModel:
                            deleteInstance(teapot)

                    if mug.name in scene.objects:
                        scene.objects.unlink(mug)
                    #     deleteInstance(mug)

                    if not useShapeModel:
                        teapot = blender_teapots[currentTeapotModel]
                    else:
                        teapotMesh = createMeshFromData('teapotShapeModelMesh', chVerticesGT.r.tolist(),
                                                        faces.astype(np.int32).tolist())
                        teapotMesh.layers[0] = True
                        teapotMesh.layers[1] = True
                        teapotMesh.pass_index = 1

                        targetGroup = bpy.data.groups.new('teapotShapeModelGroup')
                        targetGroup.objects.link(teapotMesh)
                        teapot = bpy.data.objects.new('teapotShapeModel', None)
                        teapot.dupli_type = 'GROUP'
                        teapot.dupli_group = targetGroup
                        teapot.pass_index = 1

                        mat = makeMaterial('teapotMat', (0, 0, 0), (0, 0, 0), 1)
                        setMaterial(teapotMesh, mat)

                    mug = blender_mugs[currentTeapotModel]
                    # center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)

                    placeNewTarget(scene, teapot, targetPosition[:].copy())
                    placeNewTarget(scene, mug, targetPosition[:].copy())

                    teapot.layers[1] = True
                    teapot.layers[0] = True
                    mug.layers[1] = True
                    mug.layers[0] = True
                    original_matrix_world = teapot.matrix_world.copy()

                    # mug.matrix_world = mug.matrix_world * mathutils.Matrix.Rotation(-np.pi/2, 4, 'Y')
                    # mug.matrix_world = mug.matrix_world

                    original_matrix_world_mug = mug.matrix_world.copy()


                for hdrFile, hdrValues in hdrstorender:
                    hdridx = hdrValues[0]
                    envMapCoeffsVals = hdrValues[1]

                    envMapFilename = hdrFile
                    envMapTexture = np.array(imageio.imread(envMapFilename))[:,:,0:3]

                    for numTeapotTrain in range(max(int(trainSize/(lenScenes*len(hdrstorender)*len(renderTeapotsList))),1)):

                        ignore = False
                        chAmbientIntensityGTVals = 0.75/(0.3*envMapCoeffs[0,0] + 0.59*envMapCoeffs[0,1]+ 0.11*envMapCoeffs[0,2])
                        phiOffsetVals = np.random.uniform(0,2*np.pi, 1)

                        # phiOffset[:] = 0
                        from numpy.random import choice
                        # objAzInterval = choice(len(collisionsProbs), size=1, p=collisionsProbs)
                        # objAzGT = np.random.uniform(0,1)*(collisions[targetIndex][1][objAzInterval][1] - collisions[targetIndex][1][objAzInterval][0]) + collisions[targetIndex][1][objAzInterval][0]
                        #
                        # objAzGT = objAzGT*np.pi/180
                        # chObjAzGTVals = objAzGT.copy()
                        chObjAzGTVals = np.random.uniform(0,np.pi*2)

                        chObjDistGTVals = np.random.uniform(0, np.min(supportRad, 0.4))
                        chObjAzMugVals = np.random.uniform(0,np.pi*2)
                        chObjDistMugVals = np.random.uniform(0,np.min(supportRad, 0.4))
                        chObjRotationMugVals = np.random.uniform(0,np.pi*2)
                        chObjRotationGTVals = np.random.uniform(0,np.pi*2)

                        # if renderOcclusions:
                        #     # azInterval = choice(len(occlusionProbs), size=1, p=occlusionProbs)
                        #     # azGT = np.random.uniform(0,1)*(occlusions[targetIndex][1][azInterval][1] - occlusions[targetIndex][1][azInterval][0]) + occlusions[targetIndex][1][azInterval][0]
                        #     chAzGTVals = azGT*np.pi/180
                        # else:
                        chAzGTVals = np.mod(np.random.uniform(0,np.pi, 1) - np.pi/2, 2*np.pi)

                        chElGTVals = np.random.uniform(0.05,np.pi/2, 1)

                        chLightAzGTVals = np.random.uniform(0,2*np.pi, 1)
                        chLightElGTVals = np.random.uniform(0,np.pi/3, 1)

                        chLightIntensityGTVals = 0

                        chVColorsGTVals =  np.random.uniform(0.2,0.8, [1, 3])
                        chVColorsMugVals = np.random.uniform(0.2, 0.8, [1, 3])

                        envMapCoeffsRotatedVals = np.dot(light_probes.chSphericalHarmonicsZRotation(totalOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]
                        envMapCoeffsRotatedRelVals = np.dot(light_probes.chSphericalHarmonicsZRotation(phiOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]

                        shapeParams = np.random.randn(latentDim)
                        chShapeParamsGTVals = shapeParams

                        ## Update renderer scene latent variables.

                        ignore = False

                        chAmbientIntensityGT[:] = chAmbientIntensityGTVals

                        phiOffset[:] = phiOffsetVals

                        chObjAzGT[:] = chObjAzGTVals

                        chAzGT[:] = chAzGTVals

                        chElGT[:] = chElGTVals

                        chLightAzGT[:] = chLightAzGTVals
                        chLightElGT[:] = chLightElGTVals

                        chVColorsGT[:] = chVColorsGTVals
                        try:
                            chShapeParamsGT[:] = shapeParams
                        except:
                            chShapeParamsGT[:] = np.random.randn(latentDim)

                        teapotPlacement = np.random.choice(instantiationBinsTeapot.sum())

                        mugPlacement = np.random.choice(instantiationBinsMug.sum())

                        chObjDistGTVals = totalBinsTeapot[0].ravel()[instantiationBinsTeapot][teapotPlacement]
                        chObjRotationGTVals = totalBinsTeapot[1].ravel()[instantiationBinsTeapot][teapotPlacement]
                        chObjDistMugVals = totalBinsMug[0].ravel()[instantiationBinsMug][mugPlacement]
                        chObjRotationMugVals = totalBinsMug[1].ravel()[instantiationBinsMug][mugPlacement]

                        chObjAzMug[:] = chObjAzMugVals

                        chObjDistGT[:] = chObjDistGTVals
                        chObjRotationGT[:] = chObjRotationGTVals
                        chObjDistMug[:] = chObjDistMugVals
                        chObjRotationMug[:] = chObjRotationMugVals

                        chVColorsMug[:] = chVColorsMugVals

                        if useBlender and not ignore:

                            azimuthRot = mathutils.Matrix.Rotation(chObjAzGT.r[:].copy(), 4, 'Z')

                            teapot.matrix_world = mathutils.Matrix.Translation(original_matrix_world.to_translation() + mathutils.Vector(teapotPosOffset.r)) * azimuthRot * (mathutils.Matrix.Translation(-original_matrix_world.to_translation())) * original_matrix_world

                            placeCamera(scene.camera, -chAzGT.r[:].copy() * 180 / np.pi,
                                        chElGT.r[:].copy() * 180 / np.pi, chDistGT.r[0].copy(),
                                        center[:].copy() + targetPosition[:].copy())

                            setObjectDiffuseColor(teapot, chVColorsGT.r.copy())

                            setObjectDiffuseColor(mug, chVColorsMug.r.copy())

                            azimuthRotMug = mathutils.Matrix.Rotation(chObjAzMug.r[:].copy() - np.pi / 2, 4, 'Z')

                            mug.matrix_world = mathutils.Matrix.Translation(original_matrix_world_mug.to_translation() + mathutils.Vector(mugPosOffset.r)) * azimuthRotMug * (mathutils.Matrix.Translation(-original_matrix_world_mug.to_translation())) * original_matrix_world_mug

                            if useShapeModel:
                                mesh = teapot.dupli_group.objects[0]
                                for vertex_i, vertex in enumerate(mesh.data.vertices):
                                    vertex.co = mathutils.Vector(chVerticesGT.r[vertex_i])

                            scene.update()


                        ## Some validation checks:

                        if useOpenDR:
                            occlusion = getOcclusionFraction(rendererGT, id=0)
                            occlusionMug = getOcclusionFraction(rendererGT, id=1)

                            vis_occluded = np.array(rendererGT.indices_image == 1).copy().astype(np.bool)
                            vis_im = np.array(rendererGT.image_mesh_bool([0])).copy().astype(np.bool)

                            vis_occluded_mug = np.array(rendererGT.indices_image == 2).copy().astype(np.bool)
                            vis_im_mug = np.array(rendererGT.image_mesh_bool([1])).copy().astype(np.bool)


                        if occlusion > 0.9 or occlusionMug > 0.9 or np.isnan(occlusion) or np.isnan(occlusionMug):
                            ignore = True

                        if vis_occluded.sum() < 10 or vis_occluded_mug.sum() < 10:
                            ignore = True

                        #Check that objects are not only partly in the viewing plane of the camera:
                        if np.sum(vis_im_mug[:,0]) > 1 or np.sum(vis_im_mug[0,:]) > 1 or np.sum(vis_im_mug[:,-1]) > 1 or np.sum(vis_im_mug[-1,:]) > 1:
                            ignore = True

                        if np.sum(vis_im[:,0]) > 1 or np.sum(vis_im[0,:]) > 1 or np.sum(vis_im[:,-1]) > 1 or np.sum(vis_im[-1,:]) > 1:
                            ignore = True


                        if not ignore:
                            # Ignore if camera collides with occluding object as there are inconsistencies with OpenDR and Blender.
                            cameraEye = np.linalg.inv(np.r_[rendererGT.camera.view_mtx, np.array([[0, 0, 0, 1]])])[0:3,3]
                            vDists = rendererGT.v.r[rendererGT.f[rendererGT.visibility_image[
                                rendererGT.visibility_image != 4294967295].ravel()].ravel()] - cameraEye

                            vDistsTeapot = rendererGT.v.r[rendererGT.f[rendererGT.visibility_image[vis_occluded].ravel()].ravel()] - cameraEye

                            vDistsMug = rendererGT.v.r[rendererGT.f[rendererGT.visibility_image[vis_occluded_mug].ravel()].ravel()] - cameraEye

                            #Ignore when teapot or mug is up to 10 cm to the camera eye, or too far (more than 1 meter).
                            minDistToObjects = 0.3
                            maxDistToObjects = 0.75
                            if np.min(np.linalg.norm(vDists, axis=1) <= clip_start) or np.min(np.linalg.norm(vDistsTeapot, axis=1) <= minDistToObjects) or np.min(np.linalg.norm(vDistsMug, axis=1) <= minDistToObjects):
                                ignore = True

                            if np.min(np.linalg.norm(vDistsTeapot, axis=1) > maxDistToObjects) or np.min(np.linalg.norm(vDistsMug, axis=1) > maxDistToObjects):
                                ignore = True

                            if useBlender:


                                cubeTeapot = getCubeObj(teapot)
                                cubeMug = getCubeObj(mug)

                                cubeScene.objects.link(cubeTeapot)
                                cubeScene.objects.link(cubeMug)

                                cubeParentSupportObj = cubeScene.objects['cube'+str(parentIdx)]
                                cubeRoomObj = cubeScene.objects['cube' + str(roomInstanceNum)]
                                cubeScene.update()

                                if collision.targetCubeSceneCollision(cubeTeapot, cubeScene, 'cube'+str(roomInstanceNum), cubeParentSupportObj):
                                    print("Teapot intersects with an object.")
                                    ignore = True
                                    # pass

                                if not ignore and collision.targetCubeSceneCollision(cubeMug, cubeScene, 'cube'+str(roomInstanceNum), cubeParentSupportObj):
                                    print("Mug intersects with an object.")
                                    ignore = True
                                    # pass

                                if not collision.instancesIntersect(mathutils.Matrix.Translation(mathutils.Vector((0,0,-0.01))), [cubeTeapot], mathutils.Matrix.Identity(4), [cubeParentSupportObj]):
                                    print("Teapot not on table.")
                                    ignore = True

                                if not collision.instancesIntersect(mathutils.Matrix.Translation(mathutils.Vector((0, 0, -0.01))), [cubeMug], mathutils.Matrix.Identity(4), [cubeParentSupportObj]):
                                    print("Mug not on table.")
                                    ignore = True

                                if collision.instancesIntersect(mathutils.Matrix.Translation(mathutils.Vector((0, 0, +0.01))), [cubeTeapot], mathutils.Matrix.Identity(4), [cubeParentSupportObj]):
                                    print("Teapot interesects supporting object.")
                                    ignore = True

                                if collision.instancesIntersect(mathutils.Matrix.Translation(mathutils.Vector((0, 0, +0.01))), [cubeMug], mathutils.Matrix.Identity(4), [cubeParentSupportObj]):
                                    print("Mug intersects supporting object")
                                    ignore = True

                                # ipdb.set_trace()
                                if collision.instancesIntersect(mathutils.Matrix.Identity(4), [cubeTeapot], mathutils.Matrix.Identity(4), [cubeRoomObj]):
                                    print("Teapot intersects room")
                                    ignore = True

                                if collision.instancesIntersect(mathutils.Matrix.Identity(4), [cubeMug], mathutils.Matrix.Identity(4), [cubeRoomObj]):
                                    print("Mug intersects room")
                                    ignore = True

                                if not ignore:
                                    print("No collision issues")

                                # setEnviornmentMapStrength(10, cubeScene)

                                # bpy.ops.render.render(write_still=True)

                                # image = np.array(imageio.imread(cubeScene.render.filepath))[:, :, 0:3]
                                # image[image > 1] = 1
                                # blenderRender = image

                                # lin2srgb(blenderRender)
                                #
                                # cv2.imwrite(gtDir + 'images/im' + str(train_i) + '.jpeg', 255 * blenderRender[:, :, [2, 1, 0]], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                                cubeScene.objects.unlink(cubeTeapot)
                                cubeScene.objects.unlink(cubeMug)
                                deleteObject(cubeTeapot)
                                deleteObject(cubeMug)
                                # if not collision.instancesIntersect(mathutils.Matrix.Translation(mathutils.Vector((0,0,-0.01)))*teapot.matrix_world, teapot.dupli_group.objects, parentSupportObj.matrix_world, parentSupportObj.dupli_group.objects):
                                #     ignore = True
                                #
                                # if not collision.instancesIntersect(mathutils.Matrix.Translation(mathutils.Vector((0,0,-0.01)))*mug.matrix_world, mug.dupli_group.objects, parentSupportObj.matrix_world, parentSupportObj.dupli_group.objects):
                                #     ignore = True


                        ## Environment map update if using Cycles.

                        if not ignore and useBlender:
                            if captureEnvMapFromBlender:
                                envMapCoeffs = captureSceneEnvMap(scene, envMapTexture, roomInstanceNum,
                                                                  totalOffset.r.copy(), links, treeNodes, teapot, center,
                                                                  targetPosition, width, height, 1000, gtDir, train_i)

                        if useBlender:
                            envMapCoeffsRotated[:] = np.dot(light_probes.chSphericalHarmonicsZRotation(0),
                                                            envMapCoeffs[[0, 3, 2, 1, 4, 5, 6, 7, 8]])[[0, 3, 2, 1, 4, 5, 6, 7, 8]]
                            envMapCoeffsRotatedRel[:] = np.dot(light_probes.chSphericalHarmonicsZRotation(-chObjAzGT.r),
                                                               envMapCoeffs[[0, 3, 2, 1, 4, 5, 6, 7, 8]])[[0, 3, 2, 1, 4, 5, 6, 7, 8]]
                        else:
                            envMapCoeffsRotated[:] = np.dot(light_probes.chSphericalHarmonicsZRotation(totalOffset),
                                                            envMapCoeffs[[0, 3, 2, 1, 4, 5, 6, 7, 8]])[[0, 3, 2, 1, 4, 5, 6, 7, 8]]
                            envMapCoeffsRotatedRel[:] = np.dot(light_probes.chSphericalHarmonicsZRotation(phiOffset),
                                                               envMapCoeffs[[0, 3, 2, 1, 4, 5, 6, 7, 8]])[[0, 3, 2, 1, 4, 5, 6, 7, 8]]


                        # ipdb.set_trace()

                        if renderBlender and not ignore:
                            bpy.context.screen.scene = scene

                            bpy.ops.render.render(write_still=True)

                            image = np.array(imageio.imread(scene.render.filepath))[:, :, 0:3]
                            image[image > 1] = 1
                            blenderRender = image

                            lin2srgb(blenderRender)

                            cv2.imwrite(gtDir + 'images/im' + str(train_i) + '.jpeg', 255 * blenderRender[:, :, [2, 1, 0]], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                            if captureEnvMapFromBlender:
                                blenderRenderGray = 0.3 * blenderRender[:, :, 0] + 0.59 * blenderRender[:, :,
                                                                                          1] + 0.11 * blenderRender[:, :, 2]

                                # For some reason I need to correct average intensity in OpenDR a few times before it gets it right:

                                rendererGTGray = 0.3 * rendererGT[:, :, 0].r[:] + 0.59 * rendererGT[:, :, 1].r[
                                                                                         :] + 0.11 * rendererGT[:, :, 2].r[
                                                                                                     :]
                                meanIntensityScale = np.mean(blenderRenderGray[vis_occluded]) / np.mean(
                                    rendererGTGray[vis_occluded]).copy()
                                chAmbientIntensityGT[:] = chAmbientIntensityGT.r[:].copy() * meanIntensityScale

                                rendererGTGray = 0.3 * rendererGT[:, :, 0].r[:] + 0.59 * rendererGT[:, :, 1].r[
                                                                                         :] + 0.11 * rendererGT[:, :, 2].r[
                                                                                                     :]
                                meanIntensityScale2 = np.mean(blenderRenderGray[vis_occluded]) / np.mean(
                                    rendererGTGray[vis_occluded]).copy()
                                chAmbientIntensityGT[:] = chAmbientIntensityGT.r[:].copy() * meanIntensityScale2

                                rendererGTGray = 0.3 * rendererGT[:, :, 0].r[:] + 0.59 * rendererGT[:, :, 1].r[
                                                                                         :] + 0.11 * rendererGT[:, :, 2].r[
                                                                                                     :]
                                meanIntensityScale3 = np.mean(blenderRenderGray[vis_occluded]) / np.mean(
                                    rendererGTGray[vis_occluded]).copy()
                                chAmbientIntensityGT[:] = chAmbientIntensityGT.r[:].copy() * meanIntensityScale3

                                rendererGTGray = 0.3 * rendererGT[:, :, 0].r[:] + 0.59 * rendererGT[:, :, 1].r[
                                                                                         :] + 0.11 * rendererGT[:, :, 2].r[
                                                                                                     :]
                                meanIntensityScale4 = np.mean(blenderRenderGray[vis_occluded]) / np.mean(
                                    rendererGTGray[vis_occluded]).copy()
                                chAmbientIntensityGT[:] = chAmbientIntensityGT.r[:].copy() * meanIntensityScale4

                                rendererGTGray = 0.3 * rendererGT[:, :, 0].r[:] + 0.59 * rendererGT[:, :, 1].r[
                                                                                         :] + 0.11 * rendererGT[:, :, 2].r[
                                                                                                     :]
                                meanIntensityScale5 = np.mean(blenderRenderGray[vis_occluded]) / np.mean(
                                    rendererGTGray[vis_occluded]).copy()
                                chAmbientIntensityGT[:] = chAmbientIntensityGT.r[:].copy() * meanIntensityScale5

                                if np.mean(rendererGTGray, axis=(0, 1)) < 0.01:
                                    ignore = True

                        if useOpenDR:
                            image = rendererGT.r[:].copy()
                            lin2srgb(image)

                        if not ignore:

                            if useOpenDR:
                                cv2.imwrite(gtDir + 'images_opendr/im' + str(train_i) + '.jpeg', 255 * image[:, :, [2, 1, 0]], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                            if useOpenDR:
                                np.save(gtDir + 'masks_occlusion/mask' + str(train_i) + '.npy', vis_occluded)

                            #Add groundtruth to arrays
                            trainAzsGT = chAzGTVals
                            trainObjAzsGT = chObjAzGTVals
                            trainElevsGT = chElGTVals
                            trainLightAzsGT = chLightAzGTVals
                            trainLightElevsGT = chLightElGTVals
                            trainLightIntensitiesGT = chLightIntensityGTVals
                            trainVColorGT = chVColorsGTVals
                            lightCoeffs = envMapCoeffsRotatedVals[None, :].copy().squeeze()
                            lightCoeffs = 0.3*lightCoeffs[:,0] + 0.59*lightCoeffs[:,1] + 0.11*lightCoeffs[:,2]
                            trainLightCoefficientsGT = lightCoeffs
                            lightCoeffsRel = envMapCoeffsRotatedRelVals[None, :].copy().squeeze()
                            lightCoeffsRel = 0.3*lightCoeffsRel[:,0] + 0.59*lightCoeffsRel[:,1] + 0.11*lightCoeffsRel[:,2]
                            trainLightCoefficientsGTRel = lightCoeffsRel
                            trainAmbientIntensityGT = chAmbientIntensityGTVals
                            trainEnvMapPhiOffsets = phiOffset
                            trainScenes = sceneNumber
                            trainTeapotIds = teapot_i
                            trainEnvMaps = hdridx
                            trainShapeModelCoeffsGT = chShapeParamsGTVals.copy()
                            trainOcclusions = -1
                            trainIds = train_i
                            trainTargetIndices = targetIndex

                            gtDatasetToRender.resize(gtDatasetToRender.shape[0]+1, axis=0)
                            gtDatasetToRender[-1] = np.array([(trainIds, trainAzsGT,trainObjAzsGT,trainElevsGT,trainLightAzsGT,trainLightElevsGT,trainLightIntensitiesGT,trainVColorGT,trainScenes,trainTeapotIds,trainEnvMaps,trainOcclusions,trainTargetIndices, trainLightCoefficientsGT, trainLightCoefficientsGTRel, trainAmbientIntensityGT, phiOffsetVals, trainShapeModelCoeffsGT)],dtype=gtDtype)

                            train_i = train_i + 1

                            if np.mod(train_i, 100) == 0:
                                print("Generated " + str(train_i) + " GT instances.")
                                print("Generating groundtruth. Iteration of " + str(range(int(trainSize/(lenScenes*len(hdrstorender)*len(renderTeapotsList))))) + " teapots")

if renderFromPreviousGT:
    groundTruthFilename = 'groundtruth/' + previousGTPrefix + '/groundTruth.h5'
    gtDataFileToRender = h5py.File(groundTruthFilename, 'r')
    groundTruthToRender = gtDataFileToRender[previousGTPrefix]
else:
    groundTruthToRender = gtDataFileToRender[prefix]

train_i = nextId

currentScene = -1
currentTeapot = -1
currentTargetIndex = -1

teapot = None

if renderFromPreviousGT:
    rangeGT = np.arange(0, len(groundTruthToRender))
else:
    rangeGT = np.arange(len(groundTruthToRender))

teapot_i = 0

if useShapeModel:
    teapot_i = -1
    # addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list,  v_teapots[currentTeapotModel][0], f_list_teapots[currentTeapotModel][0], vc_teapots[currentTeapotModel][0], vn_teapots[currentTeapotModel][0], uv_teapots[currentTeapotModel][0], haveTextures_list_teapots[currentTeapotModel][0], textures_list_teapots[currentTeapotModel][0])

if renderFromPreviousGT:
    for gtIdx in rangeGT[:]:

        sceneNumber = groundTruthToRender['trainScenes'][gtIdx]

        sceneIdx = scene_io_utils.getSceneIdx(sceneNumber, replaceableScenesFile)

        print("Rendering scene: " + str(sceneIdx))
        sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndicesScene, targetPositions = scene_io_utils.getSceneInformation(sceneIdx, replaceableScenesFile)

        sceneDicFile = 'data/scene' + str(sceneNumber) + '.pickle'

        if sceneIdx != currentScene:
            # v, f_list, vc, vn, uv, haveTextures_list, textures_list = sceneimport.loadSavedScene(sceneDicFile)
            import copy
            v2, f_list2, vc2, vn2, uv2, haveTextures_list2, textures_list2 = scene_io_utils.loadSavedScene(sceneDicFile, tex_srgb2lin)

        if sceneIdx != currentScene:
            if useBlender and not loadBlenderSceneFile:
                bpy.ops.wm.read_factory_settings()
                scene = scene_io_utils.loadBlenderScene(sceneIdx, replaceableScenesFile)
                scene_io_utils.setupScene(scene, roomInstanceNum, scene.world, scene.camera, width, height, 16, useCycles, True)
                scene.update()
                #Save barebones scene.
            elif useBlender and loadBlenderSceneFile:
                bpy.ops.wm.read_factory_settings()
                scene_io_utils.loadSceneBlendData(sceneIdx, replaceableScenesFile)
                scene = bpy.data.scenes['Main Scene']

            #Configure scene
            if useBlender:
                targetModels = []
                blender_teapots = []
                teapots = [line.strip() for line in open('teapots.txt')]
                selection = [ teapots[i] for i in renderTeapotsList]
                scene_io_utils.loadTargetsBlendData()
                for teapotIdx, teapotName in enumerate(selection):
                    teapot = bpy.data.scenes[teapotName[0:63]].objects['teapotInstance' + str(renderTeapotsList[teapotIdx])]
                    teapot.layers[1] = True
                    teapot.layers[2] = True
                    targetModels = targetModels + [teapotIdx]
                    blender_teapots = blender_teapots + [teapot]

                setupSceneGroundtruth(scene, width, height, clip_start, 200, 'CUDA', 'CUDA_0')

                treeNodes = scene.world.node_tree

                links = treeNodes.links

        unlinkedObj = None
        envMapFilename = None

        targetIndex = groundTruthToRender['trainTargetIndices'][gtIdx]

        if sceneIdx != currentScene or targetIndex != currentTargetIndex:
            targetPosition = targetPositions[np.where(targetIndex==np.array(targetIndicesScene))[0]]
            import copy
            v, f_list, vc, vn, uv, haveTextures_list, textures_list = copy.deepcopy(v2), copy.deepcopy(f_list2), copy.deepcopy(vc2), copy.deepcopy(vn2), copy.deepcopy(uv2), copy.deepcopy(haveTextures_list2),  copy.deepcopy(textures_list2)

            removeObjectData(len(v) -1 - targetIndex, v, f_list, vc, vn, uv, haveTextures_list, textures_list)

        if sceneIdx != currentScene or targetIndex != currentTargetIndex:
            if useBlender:
                if unlinkedObj != None:
                    scene.objects.link(unlinkedObj)
                unlinkedObj = scene.objects[str(targetIndex)]
                scene.objects.unlink(unlinkedObj)

        teapot_i = groundTruthToRender['trainTeapotIds'][gtIdx]
        teapot_i = 0
        if useShapeModel:
            teapot_i = -1

        if sceneIdx != currentScene or targetIndex != currentTargetIndex or teapot_i != currentTeapot:

            if useOpenDR:
                rendererGT.makeCurrentContext()
                rendererGT.clear()
                contextdata.cleanupContext(contextdata.getContext())
                if glMode == 'glfw':
                    glfw.destroy_window(rendererGT.win)
                del rendererGT

            currentTeapotModel = teapot_i
            center = center_teapots[teapot_i]

            if currentScene != -1 and  currentTargetIndex != -1 and currentTeapot != -1 and (targetIndex != currentTargetIndex or teapot_i != currentTeapot):
                removeObjectData(0, v, f_list, vc, vn, uv, haveTextures_list, textures_list)

            if useShapeModel:
                center = smCenterGT
                vGT, vnGT, teapostPosOffset = transformObject(smVerticesGT, smNormalsGT, chScaleGT, chObjAzGT, ch.Ch([0]), ch.Ch([0]),np.array([0, 0, 0]))

                addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list, vGT, smFacesGT, smVColorsGT, vnGT, smUVsGT, smHaveTexturesGT, smTexturesListGT)
            else:
                vGT, vnGT, teapotPosOffset = transformObject(v_teapots[currentTeapotModel][0], vn_teapots[currentTeapotModel][0], chScaleGT, chObjAzGT, ch.Ch([0]), ch.Ch([0]), targetPosition)
                addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list, vGT,
                              f_list_teapots[currentTeapotModel][0], vc_teapots[currentTeapotModel][0], vnGT,
                              uv_teapots[currentTeapotModel][0], haveTextures_list_teapots[currentTeapotModel][0],
                              textures_list_teapots[currentTeapotModel][0])

            if useOpenDR:
                rendererGT = createRendererGT(glMode, chAzGT, chElGT, chDistGT, center, v, vc, f_list, vn, light_colorGT,
                                              chComponentGT, chVColorsGT, targetPosition.copy(), chDisplacementGT, width,
                                              height, uv, haveTextures_list, textures_list, frustum, None)
            print("Ground truth on new teapot" + str(teapot_i))

            if useBlender:
                if currentScene != -1 and  currentTargetIndex != -1 and currentTeapot != -1 and teapot != None:
                    if teapot.name in scene.objects:
                        scene.objects.unlink(teapot)

                        if useShapeModel:
                            deleteInstance(teapot)

                if not useShapeModel:
                    teapot = blender_teapots[currentTeapotModel]
                else:
                    teapotMesh = createMeshFromData('teapotShapeModelMesh', chVerticesGT.r.tolist(), faces.astype(np.int32).tolist())
                    teapotMesh.layers[0] = True
                    teapotMesh.layers[1] = True
                    teapotMesh.pass_index = 1

                    targetGroup = bpy.data.groups.new('teapotShapeModelGroup')
                    targetGroup.objects.link(teapotMesh)
                    teapot = bpy.data.objects.new('teapotShapeModel', None)
                    teapot.dupli_type = 'GROUP'
                    teapot.dupli_group = targetGroup
                    teapot.pass_index = 1

                    mat = makeMaterial('teapotMat', (0,0,0), (0,0,0), 1)
                    setMaterial(teapotMesh, mat)

                # center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)

                placeNewTarget(scene, teapot, targetPosition[:].copy())
                teapot.layers[1]=True
                teapot.layers[0]=True
                original_matrix_world = teapot.matrix_world.copy()

        hdridx = groundTruthToRender['trainEnvMaps'][gtIdx]

        envMapFilename = ""
        for hdrFile, hdrValues in hdritems:
            if hdridx == hdrValues[0]:

                envMapCoeffs[:] = hdrValues[1]
                envMapFilename = hdrFile

                # updateEnviornmentMap(envMapFilename, scene)
                envMapTexture = np.array(imageio.imread(envMapFilename))[:,:,0:3]
                break

        if envMapFilename == "":
            ipdb.set_trace()

        print("Render " + str(gtIdx) + "of " + str(len(groundTruthToRender)))
        ignore = False
        # chAmbientIntensityGT[:] = groundTruthToRender['trainAmbientIntensityGT'][gtIdx]
        chAmbientIntensityGT[:] = 1

        phiOffset[:] = groundTruthToRender['trainEnvMapPhiOffsets'][gtIdx]

        chObjAzGT[:] = groundTruthToRender['trainObjAzsGT'][gtIdx]

        chAzGT[:] = groundTruthToRender['trainAzsGT'][gtIdx]

        chElGT[:] = groundTruthToRender['trainElevsGT'][gtIdx]

        chLightAzGT[:] = groundTruthToRender['trainLightAzsGT'][gtIdx]
        chLightElGT[:] = groundTruthToRender['trainLightElevsGT'][gtIdx]

        # chLightIntensityGT[:] = np.random.uniform(5,10, 1)

        chVColorsGT[:] = groundTruthToRender['trainVColorGT'][gtIdx]
        try:
            chShapeParamsGT[:] =  groundTruthToRender['trainShapeModelCoeffsGT'][gtIdx]
        except:
            chShapeParamsGT[:] = np.random.randn(latentDim)

        if useOpenDR:
            occlusion = getOcclusionFraction(rendererGT)

            vis_occluded = np.array(rendererGT.indices_image==1).copy().astype(np.bool)
            vis_im = np.array(rendererGT.image_mesh_bool([0])).copy().astype(np.bool)

        if occlusion > 0.9:
            ignore = True

        if not ignore:
            #Ignore if camera collides with occluding object as there are inconsistencies with OpenDR and Blender.
            cameraEye = np.linalg.inv(np.r_[rendererGT.camera.view_mtx, np.array([[0,0,0,1]])])[0:3,3]
            vDists = rendererGT.v.r[rendererGT.f[rendererGT.visibility_image[rendererGT.visibility_image != 4294967295].ravel()].ravel()] - cameraEye
            if np.min(np.linalg.norm(vDists,axis=1) <= clip_start):
                ignore = True

        if captureEnvMapFromBlender and not ignore and useBlender:
            envMapCoeffs = captureSceneEnvMap(scene, envMapTexture, roomInstanceNum, totalOffset.r.copy(), links, treeNodes, teapot, center, targetPosition, width, height, gtDir, train_i)

        if useBlender:
            envMapCoeffsRotated[:] = np.dot(light_probes.chSphericalHarmonicsZRotation(0), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]
            envMapCoeffsRotatedRel[:] = np.dot(light_probes.chSphericalHarmonicsZRotation(-chObjAzGT.r), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]
        else:
            envMapCoeffsRotated[:] = np.dot(light_probes.chSphericalHarmonicsZRotation(totalOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]
            envMapCoeffsRotatedRel[:] = np.dot(light_probes.chSphericalHarmonicsZRotation(phiOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]

        if useBlender and not ignore:

            azimuthRot = mathutils.Matrix.Rotation(chObjAzGT.r[:].copy(), 4, 'Z')

            teapot.matrix_world = mathutils.Matrix.Translation(original_matrix_world.to_translation()) * azimuthRot * (mathutils.Matrix.Translation(-original_matrix_world.to_translation())) * original_matrix_world
            placeCamera(scene.camera, -chAzGT.r[:].copy()*180/np.pi, chElGT.r[:].copy()*180/np.pi, chDistGT.r[0].copy(), center[:].copy() + targetPosition[:].copy())

            setObjectDiffuseColor(teapot, chVColorsGT.r.copy())

            if useShapeModel:
                mesh = teapot.dupli_group.objects[0]
                for vertex_i, vertex in enumerate(mesh.data.vertices):
                    vertex.co = mathutils.Vector(chVerticesGT.r[vertex_i])
            # ipdb.set_trace()

            scene.update()

            bpy.ops.render.render( write_still=True )

            image = np.array(imageio.imread(scene.render.filepath))[:,:,0:3]
            image[image>1]=1
            blenderRender = image

            blenderRenderGray = 0.3*blenderRender[:,:,0] + 0.59*blenderRender[:,:,1] + 0.11*blenderRender[:,:,2]

            #For some reason I need to correct average intensity in OpenDR a few times before it gets it right:

            rendererGTGray = 0.3*rendererGT[:,:,0].r[:] + 0.59*rendererGT[:,:,1].r[:] + 0.11*rendererGT[:,:,2].r[:]
            meanIntensityScale = np.mean(blenderRenderGray[vis_occluded])/np.mean(rendererGTGray[vis_occluded]).copy()
            chAmbientIntensityGT[:] = chAmbientIntensityGT.r[:].copy()*meanIntensityScale

            rendererGTGray = 0.3*rendererGT[:,:,0].r[:] + 0.59*rendererGT[:,:,1].r[:] + 0.11*rendererGT[:,:,2].r[:]
            meanIntensityScale2 = np.mean(blenderRenderGray[vis_occluded])/np.mean(rendererGTGray[vis_occluded]).copy()
            chAmbientIntensityGT[:] = chAmbientIntensityGT.r[:].copy()*meanIntensityScale2

            rendererGTGray = 0.3*rendererGT[:,:,0].r[:] + 0.59*rendererGT[:,:,1].r[:] + 0.11*rendererGT[:,:,2].r[:]
            meanIntensityScale3 = np.mean(blenderRenderGray[vis_occluded])/np.mean(rendererGTGray[vis_occluded]).copy()
            chAmbientIntensityGT[:] = chAmbientIntensityGT.r[:].copy()*meanIntensityScale3

            rendererGTGray = 0.3*rendererGT[:,:,0].r[:] + 0.59*rendererGT[:,:,1].r[:] + 0.11*rendererGT[:,:,2].r[:]
            meanIntensityScale4 = np.mean(blenderRenderGray[vis_occluded])/np.mean(rendererGTGray[vis_occluded]).copy()
            chAmbientIntensityGT[:] = chAmbientIntensityGT.r[:].copy()*meanIntensityScale4

            rendererGTGray = 0.3*rendererGT[:,:,0].r[:] + 0.59*rendererGT[:,:,1].r[:] + 0.11*rendererGT[:,:,2].r[:]
            meanIntensityScale5 = np.mean(blenderRenderGray[vis_occluded])/np.mean(rendererGTGray[vis_occluded]).copy()
            chAmbientIntensityGT[:] = chAmbientIntensityGT.r[:].copy()*meanIntensityScale5

            lin2srgb(blenderRender)

        if useOpenDR:
            image = rendererGT.r[:].copy()
            lin2srgb(image)

        if useBlender and not ignore and useOpenDR and np.mean(rendererGTGray,axis=(0,1)) < 0.01:
            ignore = True

        if not ignore:
            # hogs = hogs + [imageproc.computeHoG(image).reshape([1,-1])]
            # illumfeats = illumfeats + [imageproc.featuresIlluminationDirection(image,20)]
            if useBlender:
                cv2.imwrite(gtDir + 'images/im' + str(train_i) + '.jpeg' , 255*blenderRender[:,:,[2,1,0]], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            if useOpenDR:
                cv2.imwrite(gtDir + 'images_opendr/im' + str(train_i) + '.jpeg' , 255*image[:,:,[2,1,0]], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            # cv2.imwrite(gtDir + 'images_opendr/im' + str(train_i) + '.jpeg' , 255*image[:,:,[2,1,0]], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            if useOpenDR:
                np.save(gtDir + 'masks_occlusion/mask' + str(train_i)+ '.npy', vis_occluded)

            #Add groundtruth to arrays
            trainAzsGT = chAzGT.r
            trainObjAzsGT = chObjAzGT.r
            trainElevsGT = chElGT.r
            trainLightAzsGT = chLightAzGT.r
            trainLightElevsGT = chLightElGT.r
            trainLightIntensitiesGT = groundTruthToRender['trainLightIntensitiesGT'][gtIdx]
            trainVColorGT = chVColorsGT.r
            lightCoeffs = envMapCoeffsRotated.r[None, :].copy().squeeze()
            lightCoeffs = 0.3*lightCoeffs[:,0] + 0.59*lightCoeffs[:,1] + 0.11*lightCoeffs[:,2]
            trainLightCoefficientsGT = lightCoeffs
            lightCoeffsRel = envMapCoeffsRotatedRel.r[None, :].copy().squeeze()
            lightCoeffsRel = 0.3*lightCoeffsRel[:,0] + 0.59*lightCoeffsRel[:,1] + 0.11*lightCoeffsRel[:,2]
            trainLightCoefficientsGTRel = lightCoeffsRel
            trainAmbientIntensityGT = chAmbientIntensityGT.r
            trainEnvMapPhiOffsets = phiOffset
            trainScenes = sceneNumber
            trainTeapotIds = teapot_i
            trainEnvMaps = hdridx
            trainShapeModelCoeffsGT = chShapeParamsGT.r.copy()
            trainOcclusions = occlusion
            trainIds = train_i
            trainTargetIndices = targetIndex

            gtDataset.resize(gtDataset.shape[0]+1, axis=0)
            gtDataset[-1] = np.array([(trainIds, trainAzsGT,trainObjAzsGT,trainElevsGT,trainLightAzsGT,trainLightElevsGT,trainLightIntensitiesGT,trainVColorGT,trainScenes,trainTeapotIds,trainEnvMaps,trainOcclusions,trainTargetIndices, trainLightCoefficientsGT, trainLightCoefficientsGTRel, trainAmbientIntensityGT, phiOffset.r, trainShapeModelCoeffsGT)],dtype=gtDtype)
            gtDataFile.flush()
            train_i = train_i + 1

        currentScene  = sceneIdx
        currentTargetIndex = targetIndex
        currentTeapot = teapot_i

    # np.savetxt(gtDir + 'data.txt',np.array(np.hstack([trainIds[:,None], trainAzsGT[:,None], trainObjAzsGT[:,None], trainElevsGT[:,None], phiOffsets[:,None], trainOcclusions[:,None]])), fmt="%g")
    gtDataFile.close()

    gtDataFileToRender.close()