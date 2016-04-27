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

plt.ion()

#########################################
# Initialization starts here
#########################################
prefix = 'train4_occlusion_shapemodel_cycles'
previousGTPrefix = 'train4_occlusion_shapemodel'

#Main script options:
renderFromPreviousGT = True
useShapeModel = True
renderOcclusions = True
useOpenDR = True
useBlender = True
loadBlenderSceneFile = True
groundTruthBlender = True
useCycles = True
unpackModelsFromBlender = False
unpackSceneFromBlender = False
loadSavedSH = False
glModes = ['glfw','mesa']
glMode = glModes[0]

width, height = (150, 150)
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
camDistance = 0.4

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
blender_mugs = []
selection = [ teapots[i] for i in renderTeapotsList]
selectionMugs = [ mugs[i] for i in renderMugsList]
scene_io_utils.loadTargetsBlendData()
for teapotIdx, teapotName in enumerate(selection):
    teapot = bpy.data.scenes[teapotName[0:63]].objects['teapotInstance' + str(renderTeapotsList[teapotIdx])]
    teapot.layers[1] = True
    teapot.layers[2] = True
    targetModels = targetModels + [teapot]
    blender_teapots = blender_teapots + [teapot]

v_teapots, f_list_teapots, vc_teapots, vn_teapots, uv_teapots, haveTextures_list_teapots, textures_list_teapots, vflat, varray, center_teapots = scene_io_utils.loadTeapotsOpenDRData(renderTeapotsList, useBlender, unpackModelsFromBlender, targetModels)

for mugIdx, mugName in enumerate(selectionMugs):
    mug = bpy.data.scenes[mugName[0:63]].objects['mugInstance' + str(renderMugsList[mugIdx])]
    mug.layers[1] = True
    mug.layers[2] = True
    mugModels = mugModels + [mug]
    blender_mugs = blender_mugs + [mug]
v_mugs, f_list_mugs, vc_mugs, vn_mugs, uv_mugs, haveTextures_list_mugs, textures_list_mugs, vflat_mugs, varray_mugs, center_mugs = scene_io_utils.loadMugsOpenDRData(renderMugsList, useBlender, unpackModelsFromBlender, mugModels)

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
    rendererGT = createRendererGT(glMode, chAzGT, chObjAzGT, chElGT, chDistGT, center, v, vc, f_list, vn, light_colorGT, chComponentGT, chVColorsGT, targetPosition[:].copy(), chDisplacementGT, chScaleGT, width,height, uv, haveTextures_list, textures_list, frustum, None )


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

        unlinkedObj = None
        envMapFilename = None

        for targetidx, targetIndex in enumerate(targetIndices):
            targetPosition = targetPositions[np.where(targetIndex==np.array(targetIndicesScene))[0]]

            if not collisions[targetIndex][1]:
                continue

            collisionProbs = np.zeros(len(collisions[targetIndex][1]))

            # removeObjectData(int(targetIndex-1), v, f_list, vc, vn, uv, haveTextures_list, textures_list)

            for intervalIdx, interval in enumerate(collisions[targetIndex][1]):
                collisionProbs[intervalIdx] = collisions[targetIndex][1][intervalIdx][1] - collisions[targetIndex][1][intervalIdx][0]

            collisionsProbs = collisionProbs / np.sum(collisionProbs)

            if renderOcclusions:
                occlusionProbs = np.zeros(len(occlusions[targetIndex][1]))

                for intervalIdx, interval in enumerate(occlusions[targetIndex][1]):
                    occlusionProbs[intervalIdx] = abs(occlusions[targetIndex][1][intervalIdx][1] - occlusions[targetIndex][1][intervalIdx][0])

                occlusionProbs = occlusionProbs / np.sum(occlusionProbs)

            # if useShapeModel
            for teapot_i in renderTeapotsList:

                if useShapeModel:
                    teapot_i = -1
                else:
                    currentTeapotModel = teapot_i
                    center = center_teapots[teapot_i]

                print("Ground truth on new teapot" + str(teapot_i))

                for hdrFile, hdrValues in hdrstorender:
                    hdridx = hdrValues[0]
                    envMapCoeffsVals = hdrValues[1]
                    # envMapCoeffs[:] = np.array([[0.5,0,0.0,1,0,0,0,0,0], [0.5,0,0.0,1,0,0,0,0,0],[0.5,0,0.0,1,0,0,0,0,0]]).T

                    envMapFilename = hdrFile
                    # updateEnviornmentMap(envMapFilename, scene)
                    envMapTexture = np.array(imageio.imread(envMapFilename))[:,:,0:3]

                    for numTeapotTrain in range(max(int(trainSize/(lenScenes*len(hdrstorender)*len(renderTeapotsList))),1)):


                        ignore = False
                        chAmbientIntensityGTVals = 0.75/(0.3*envMapCoeffs[0,0] + 0.59*envMapCoeffs[0,1]+ 0.11*envMapCoeffs[0,2])
                        phiOffsetVals = np.random.uniform(0,2*np.pi, 1)

                        # phiOffset[:] = 0
                        from numpy.random import choice
                        objAzInterval = choice(len(collisionsProbs), size=1, p=collisionsProbs)
                        objAzGT = np.random.uniform(0,1)*(collisions[targetIndex][1][objAzInterval][1] - collisions[targetIndex][1][objAzInterval][0]) + collisions[targetIndex][1][objAzInterval][0]

                        objAzGT = objAzGT*np.pi/180
                        chObjAzGTVals = objAzGT.copy()

                        if renderOcclusions:
                            azInterval = choice(len(occlusionProbs), size=1, p=occlusionProbs)
                            azGT = np.random.uniform(0,1)*(occlusions[targetIndex][1][azInterval][1] - occlusions[targetIndex][1][azInterval][0]) + occlusions[targetIndex][1][azInterval][0]
                            chAzGTVals = azGT*np.pi/180
                        else:
                            chAzGTVals = np.mod(np.random.uniform(0,np.pi, 1) - np.pi/2, 2*np.pi)

                        chElGTVals = np.random.uniform(0.05,np.pi/2, 1)

                        chLightAzGTVals = np.random.uniform(0,2*np.pi, 1)
                        chLightElGTVals = np.random.uniform(0,np.pi/3, 1)

                        chLightIntensityGTVals = 0

                        chVColorsGTVals =  np.random.uniform(0.2,0.8, [1, 3])

                        envMapCoeffsRotatedVals = np.dot(light_probes.chSphericalHarmonicsZRotation(totalOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]
                        envMapCoeffsRotatedRelVals = np.dot(light_probes.chSphericalHarmonicsZRotation(phiOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]


                        shapeParams = np.random.randn(latentDim)
                        chShapeParamsGTVals = shapeParams
                        # pEnvMap = SHProjection(envMapTexture, envMapCoeffsRotated)
                        # approxProjection = np.sum(pEnvMap, axis=3)
                        # cv2.imwrite(gtDir + 'sphericalharmonics/envMapProjectionRot' + str(hdridx) + '_rot' + str(int(totalOffset*180/np.pi)) + '_' + str(str(train_i)) + '.jpeg' , 255*approxProjection[:,:,[2,1,0]])

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

        if useBlender:
            scene.render.resolution_x = width #perhaps set resolution in code
            scene.render.resolution_y = height
            scene.render.tile_x = height
            scene.render.tile_y = width
            scene.cycles.samples = 3000
            bpy.context.screen.scene = scene
            addEnvironmentMapWorld(scene)
            scene.render.image_settings.file_format = 'OPEN_EXR'
            scene.render.filepath = 'opendr_blender.exr'
            scene.sequencer_colorspace_settings.name = 'Linear'
            scene.display_settings.display_device = 'None'
            bpy.context.user_preferences.filepaths.render_cache_directory = '/disk/scratch1/pol/.cache/'
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

            scene.cycles.device = 'GPU'
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'

            bpy.context.user_preferences.system.compute_device = 'CUDA_MULTI_2'
            bpy.context.user_preferences.system.compute_device = 'CUDA_0'
            bpy.ops.wm.save_userpref()

            scene.world.horizon_color = mathutils.Color((1.0,1.0,1.0))
            scene.camera.data.clip_start = clip_start
            treeNodes=scene.world.node_tree
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
            addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list,  smVerticesGT, smFacesGT, smVColorsGT, smNormalsGT, smUVsGT, smHaveTexturesGT, smTexturesListGT)

        else:
            addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list,  v_teapots[currentTeapotModel][0], f_list_teapots[currentTeapotModel][0], vc_teapots[currentTeapotModel][0], vn_teapots[currentTeapotModel][0], uv_teapots[currentTeapotModel][0], haveTextures_list_teapots[currentTeapotModel][0], textures_list_teapots[currentTeapotModel][0])

        if useOpenDR:
            rendererGT = createRendererGT(glMode, chAzGT, chObjAzGT, chElGT, chDistGT, center, v, vc, f_list, vn, light_colorGT, chComponentGT, chVColorsGT, targetPosition.copy(), chDisplacementGT, chScaleGT, width,height, uv, haveTextures_list, textures_list, frustum, None )

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

        # envMapCoeffs[:] = np.array([[0.5,0,0.0,1,0,0,0,0,0], [0.5,0,0.0,1,0,0,0,0,0],[0.5,0,0.0,1,0,0,0,0,0]]).T

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

    if not ignore and useBlender:

        envMapTexture = cv2.resize(src=envMapTexture, dsize=(360,180))
        # envMapTexture = skimage.transform.resize(images[test_i], [height,width])

        envMapGray = 0.3*envMapTexture[:,:,0] + 0.59*envMapTexture[:,:,1] + 0.11*envMapTexture[:,:,2]
        envMapGrayMean = np.mean(envMapGray, axis=(0,1))

        envMapGrayRGB = np.concatenate([envMapGray[...,None], envMapGray[...,None], envMapGray[...,None]], axis=2)/envMapGrayMean

        envMapCoeffsNew = light_probes.getEnvironmentMapCoefficients(envMapGrayRGB, 1, 0, 'equirectangular')
        pEnvMap = SHProjection(envMapTexture, envMapCoeffsNew)
        # pEnvMap = SHProjection(envMapGrayRGB, envMapCoeffs)
        approxProjection = np.sum(pEnvMap, axis=3).astype(np.float32)

        # envMapCoeffsNewRE = light_probes.getEnvironmentMapCoefficients(approxProjectionRE, 1, 0, 'equirectangular')
        # pEnvMapRE = SHProjection(envMapTexture, envMapCoeffsNewRE)
        # # pEnvMap = SHProjection(envMapGrayRGB, envMapCoeffs)
        # approxProjectionRE = np.sum(pEnvMapRE, axis=3).astype(np.float32)

        approxProjection[approxProjection<0] = 0

        cv2.imwrite(gtDir + 'im.exr',approxProjection)

        # updateEnviornmentMap(envMapFilename, scene)
        updateEnviornmentMap(gtDir + 'im.exr', scene)

        rotateEnviornmentMap(totalOffset.r.copy(), scene)

        cv2.imwrite(gtDir + 'sphericalharmonics/envMapProjOr' + str(train_i) + '.jpeg' , 255*approxProjection[:,:,[2,1,0]])
        cv2.imwrite(gtDir + 'sphericalharmonics/envMapGrayOr' + str(train_i) + '.jpeg' , 255*envMapGrayRGB[:,:,[2,1,0]])

        links.remove(treeNodes.nodes['lightPathNode'].outputs[0].links[0])

        scene.world.cycles_visibility.camera = True
        scene.camera.data.type ='PANO'
        scene.camera.data.cycles.panorama_type = 'EQUIRECTANGULAR'
        scene.render.resolution_x = 360#perhaps set resolution in code
        scene.render.resolution_y = 180
        roomInstance = scene.objects[str(roomInstanceNum)]
        roomInstance.cycles_visibility.camera = False
        roomInstance.cycles_visibility.shadow = False
        teapot.cycles_visibility.camera = False
        teapot.cycles_visibility.shadow = True

        # image = cv2.imread(scene.render.filepath)
        # image = np.float64(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))/255.0

        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.filepath = gtDir + 'sphericalharmonics/envMap' + str(train_i) + '.exr'

        # bpy.context.user_preferences.system.compute_device_type = 'NONE'
        # bpy.context.user_preferences.system.compute_device = 'CPU'

        scene.cycles.samples = 1000
        scene.camera.up_axis = 'Z'
        # placeCamera(scene.camera, 0, 0, 1, )

        scene.camera.location =  center[:].copy() + targetPosition[:].copy()
        look_at(scene.camera, center[:].copy() + targetPosition[:].copy() + mathutils.Vector((1,0,0)))

        scene.update()
        bpy.ops.render.render( write_still=True )

        imageEnvMap = np.array(imageio.imread(scene.render.filepath))[:,:,0:3]

        cv2.imwrite(gtDir + 'sphericalharmonics/envMapCycles' + str(train_i) + '.jpeg' , 255*imageEnvMap[:,:,[2,1,0]])

        envMapCoeffs = light_probes.getEnvironmentMapCoefficients(imageEnvMap, 1, 0, 'equirectangular')
        pEnvMap = SHProjection(envMapTexture, envMapCoeffs)
        approxProjection = np.sum(pEnvMap, axis=3)
        cv2.imwrite(gtDir + 'sphericalharmonics/envMapCyclesProjection' + str(train_i) + '.jpeg' , 255*approxProjection[:,:,[2,1,0]])

        links.new(treeNodes.nodes['lightPathNode'].outputs[0], treeNodes.nodes['mixShaderNode'].inputs[0])
        scene.cycles.samples = 3000
        scene.render.filepath = 'opendr_blender.exr'
        roomInstance.cycles_visibility.camera = True
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.resolution_x = width#perhaps set resolution in code
        scene.render.resolution_y = height
        scene.camera.data.type ='PERSP'
        scene.world.cycles_visibility.camera = True
        scene.camera.data.cycles.panorama_type = 'FISHEYE_EQUISOLID'
        teapot.cycles_visibility.camera = True
        teapot.cycles_visibility.shadow = True
        # updateEnviornmentMap(envMapFilename, scene)

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
        rendererGTGray = 0.3*rendererGT[:,:,0].r[:] + 0.59*rendererGT[:,:,1].r[:] + 0.11*rendererGT[:,:,2].r[:]

        #For some unkown (yet) reason I need to correct average intensity in OpenDR a few times before it gets it right:
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