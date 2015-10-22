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
from OpenGL import contextdata

plt.ion()

#########################################
# Initialization starts here
#########################################

#Main script options:
useBlender = False
loadBlenderSceneFile = True
groundTruthBlender = False
useCycles = True
unpackModelsFromBlender = False
unpackSceneFromBlender = False
loadSavedSH = False
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
    # win = glfw.create_window(width, height, "Demo",  None, None)
    # glfw.make_context_current(win)

angle = 60 * 180 / numpy.pi
clip_start = 0.05
clip_end = 10
frustum = {'near': clip_start, 'far': clip_end, 'width': width, 'height': height}
camDistance = 0.4

teapots = [line.strip() for line in open('teapots.txt')]
renderTeapotsList = np.arange(len(teapots))
sceneIdx = 0
replaceableScenesFile = '../databaseFull/fields/scene_replaceables_backup.txt'
sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndices, targetPositions = scene_io_utils.getSceneInformation(sceneIdx, replaceableScenesFile)
sceneDicFile = 'data/scene' + str(sceneNumber) + '.pickle'
targetParentIdx = 0
targetIndex = targetIndices[targetParentIdx]
targetParentPosition = targetPositions[targetParentIdx]
targetPosition = targetParentPosition

if useBlender and not loadBlenderSceneFile:
    scene = scene_io_utils.loadBlenderScene(sceneIdx, replaceableScenesFile)
    scene_io_utils.setupScene(scene, roomInstanceNum, scene.world, scene.camera, width, height, 16, useCycles, False)
    scene.update()
    scene.render.filepath = 'opendr_blender.png'
    targetPosition = np.array(targetPosition)
    #Save barebones scene.

elif useBlender and loadBlenderSceneFile:
    scene_io_utils.loadSceneBlendData(sceneIdx, replaceableScenesFile)
    scene = bpy.data.scenes['Main Scene']
    scene.render.resolution_x = width #perhaps set resolution in code
    scene.render.resolution_y = height
    scene.render.tile_x = height/2
    scene.render.tile_y = width/2
    bpy.context.screen.scene = scene

if unpackSceneFromBlender:
    v, f_list, vc, vn, uv, haveTextures_list, textures_list = scene_io_utils.unpackBlenderScene(scene, sceneDicFile, True)
else:
    v, f_list, vc, vn, uv, haveTextures_list, textures_list = scene_io_utils.loadSavedScene(sceneDicFile)

removeObjectData(int(targetIndex), v, f_list, vc, vn, uv, haveTextures_list, textures_list)

targetModels = []
if useBlender and not loadBlenderSceneFile:
    [targetScenes, targetModels, transformations] = scene_io_utils.loadTargetModels(renderTeapotsList)
elif useBlender:
    teapots = [line.strip() for line in open('teapots.txt')]
    selection = [ teapots[i] for i in renderTeapotsList]
    scene_io_utils.loadTargetsBlendData()
    for teapotIdx, teapotName in enumerate(selection):
        targetModels = targetModels + [bpy.data.scenes[teapotName[0:63]].objects['teapotInstance' + str(renderTeapotsList[teapotIdx])]]

v_teapots, f_list_teapots, vc_teapots, vn_teapots, uv_teapots, haveTextures_list_teapots, textures_list_teapots, vflat, varray, center_teapots, blender_teapots = scene_io_utils.loadTeapotsOpenDRData(renderTeapotsList, useBlender, unpackModelsFromBlender, targetModels)

azimuth = np.pi
chCosAz = ch.Ch([np.cos(azimuth)])
chSinAz = ch.Ch([np.sin(azimuth)])

chAz = 2*ch.arctan(chSinAz/(ch.sqrt(chCosAz**2 + chSinAz**2) + chCosAz))
chAz = ch.Ch([np.pi/4])
chObjAz = ch.Ch([np.pi/4])
chAzRel = chAz - chObjAz

elevation = 0
chLogCosEl = ch.Ch(np.log(np.cos(elevation)))
chLogSinEl = ch.Ch(np.log(np.sin(elevation)))
chEl = 2*ch.arctan(ch.exp(chLogSinEl)/(ch.sqrt(ch.exp(chLogCosEl)**2 + ch.exp(chLogSinEl)**2) + ch.exp(chLogCosEl)))
chEl =  ch.Ch([0.95993109])
chDist = ch.Ch([camDistance])

chObjAzGT = ch.Ch([np.pi*3/2])
chAzGT = ch.Ch([np.pi*3/2])
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
chVColors = ch.Ch([0.4,0.4,0.4])
chVColorsGT = ch.Ch([0.4,0.4,0.4])

shCoefficientsFile = 'data/sceneSH' + str(sceneIdx) + '.pickle'

chAmbientIntensityGT = ch.Ch([2])
clampedCosCoeffs = clampedCosineCoefficients()
chAmbientSHGT = ch.zeros([9])

envMapDic = {}
SHFilename = 'data/LightSHCoefficients.pickle'

with open(SHFilename, 'rb') as pfile:
    envMapDic = pickle.load(pfile)

phiOffset = 0
totalOffset = phiOffset + chObjAzGT.r
envMapCoeffs = list(envMapDic.items())[0][1][1]
envMapCoeffsRotated = np.dot(envMapCoeffs.T,light_probes.sphericalHarmonicsZRotation(totalOffset)).T
envMapCoeffsRotatedRel = np.dot(envMapCoeffs.T,light_probes.sphericalHarmonicsZRotation(phiOffset)).T

shCoeffsRGB = ch.Ch(envMapCoeffs.copy())
shCoeffsRGBRel = ch.Ch(envMapCoeffs.copy())
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
chComponentGT = ch.Ch(chAmbientSHGT.r[:].copy())
 # + shDirLightGT*chLightIntensityGT
chComponentGTRel = ch.Ch(chAmbientSHGTRel.r[:].copy())
# chComponentGT = chAmbientSHGT.r[:] + shDirLightGT.r[:]*chLightIntensityGT.r[:]


chDisplacement = ch.Ch([0.0, 0.0,0.0])
chDisplacementGT = ch.Ch([0.0,0.0,0.0])
chScale = ch.Ch([1.0,1.0,1.0])
chScaleGT = ch.Ch([1, 1.,1.])

currentTeapotModel = 0

addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list,  v_teapots[currentTeapotModel][0], f_list_teapots[currentTeapotModel][0], vc_teapots[currentTeapotModel][0], vn_teapots[currentTeapotModel][0], uv_teapots[currentTeapotModel][0], haveTextures_list_teapots[currentTeapotModel][0], textures_list_teapots[currentTeapotModel][0])

center = center_teapots[currentTeapotModel]
rendererGT = createRendererGT(glMode, chAzGT, chObjAzGT, chElGT, chDistGT, center, v, vc, f_list, vn, light_colorGT, chComponentGT, chVColorsGT, targetPosition, chDisplacementGT, chScaleGT, width,height, uv, haveTextures_list, textures_list, frustum, None )

vis_gt = np.array(rendererGT.indices_image!=1).copy().astype(np.bool)
vis_mask = np.array(rendererGT.indices_image==1).copy().astype(np.bool)

shapeIm = vis_gt.shape
numPixels = shapeIm[0] * shapeIm[1]

def imageGT():
    global groundTruthBlender
    global rendererGT
    global blenderRender

    if groundTruthBlender:
        return blenderRender
    else:
        return np.copy(np.array(rendererGT.r)).astype(np.float64)


#########################################
# Initialization ends here
#########################################

prefix = 'train1'

print("Creating Ground Truth")
trainSize = 50000

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
trainComponentsGT = np.array([]).reshape([0,9])
trainComponentsGTRel = np.array([]).reshape([0,9])
trainEnvMapPhiOffsets = np.array([])

gtDir = 'groundtruth/' + prefix + '/'
if not os.path.exists(gtDir + 'images/'):
    os.makedirs(gtDir + 'images/')

print("Generating renders")

replaceableScenesFile = '../databaseFull/fields/scene_replaceables_backup.txt'
sceneLines = [line.strip() for line in open(replaceableScenesFile)]
scenesToRender = range(len(sceneLines))[:]
lenScenes = 0
for sceneIdx in scenesToRender:
    sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndices, targetPositions = scene_io_utils.getSceneInformation(sceneIdx, replaceableScenesFile)
    sceneDicFile = 'data/scene' + str(sceneNumber) + '.pickle'

    lenScenes += len(targetIndices)
    collisionSceneFile = 'data/collisions/collisionScene' + str(sceneNumber) + '.pickle'
    with open(collisionSceneFile, 'rb') as pfile:
        collisions = pickle.load(pfile)

    for targetidx, targetIndex in enumerate(targetIndices):
        if not collisions[targetIndex][1]:
            print("Scene idx " + str(sceneIdx) + " at index " + str(targetIndex) + " collides everywhere.")

renderTeapotsList = np.arange(len(teapots))[0:1]

# for hdrit, hdri in enumerate(list(envMapDic.items())):
#     if hdri[0] == 'data/hdr/dataset/canada_montreal_nad_photorealism.exr':
#         hdrtorenderi = hdrit

hdrstorender = list(envMapDic.items())[:]

gtDtype = [('trainIds', trainIds.dtype.name), ('trainAzsGT', trainAzsGT.dtype.name),('trainObjAzsGT', trainObjAzsGT.dtype.name),('trainElevsGT', trainElevsGT.dtype.name),('trainLightAzsGT', trainLightAzsGT.dtype.name),('trainLightElevsGT', trainLightElevsGT.dtype.name),('trainLightIntensitiesGT', trainLightIntensitiesGT.dtype.name),('trainVColorGT', trainVColorGT.dtype.name, (3,) ),('trainScenes', trainScenes.dtype.name),('trainTeapotIds', trainTeapotIds.dtype.name),('trainEnvMaps', trainEnvMaps.dtype.name),('trainOcclusions', trainOcclusions.dtype.name),('trainTargetIndices', trainTargetIndices.dtype.name), ('trainComponentsGT', trainComponentsGT.dtype, (9,)),('trainComponentsGTRel', trainComponentsGTRel.dtype, (9,)), ('trainEnvMapPhiOffsets', trainEnvMapPhiOffsets.dtype)]

groundTruth = np.array([], dtype = gtDtype)
groundTruthFilename = gtDir + 'groundTruth.h5'
gtDataFile = h5py.File(groundTruthFilename, 'a')

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

for sceneIdx in scenesToRender:

    print("Rendering scene: " + str(sceneIdx))
    sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndices, targetPositions = scene_io_utils.getSceneInformation(sceneIdx, replaceableScenesFile)

    sceneDicFile = 'data/scene' + str(sceneNumber) + '.pickle'
    # v, f_list, vc, vn, uv, haveTextures_list, textures_list = sceneimport.loadSavedScene(sceneDicFile)
    import copy
    v2, f_list2, vc2, vn2, uv2, haveTextures_list2, textures_list2 = scene_io_utils.loadSavedScene(sceneDicFile)

    collisionSceneFile = 'data/collisions/collisionScene' + str(sceneNumber) + '.pickle'
    with open(collisionSceneFile, 'rb') as pfile:
        collisions = pickle.load(pfile)

    for targetidx, targetIndex in enumerate(targetIndices):
        targetPosition = targetPositions[targetidx]
        collisionProbs = np.zeros(len(collisions[targetIndex][1]))
        import copy
        v, f_list, vc, vn, uv, haveTextures_list, textures_list = copy.deepcopy(v2), copy.deepcopy(f_list2), copy.deepcopy(vc2), copy.deepcopy(vn2), copy.deepcopy(uv2), copy.deepcopy(haveTextures_list2),  copy.deepcopy(textures_list2)

        removeObjectData(len(v) -1 - targetIndex, v, f_list, vc, vn, uv, haveTextures_list, textures_list)

        # removeObjectData(int(targetIndex-1), v, f_list, vc, vn, uv, haveTextures_list, textures_list)

        for intervalIdx, interval in enumerate(collisions[targetIndex][1]):
            collisionProbs[intervalIdx] = collisions[targetIndex][1][intervalIdx][1] - collisions[targetIndex][1][intervalIdx][0]

        collisionsProbs = collisionProbs / np.sum(collisionProbs)
        for teapot_i in renderTeapotsList:

            rendererGT.makeCurrentContext()
            rendererGT.clear()
            contextdata.cleanupContext(contextdata.getContext())
            glfw.destroy_window(rendererGT.win)
            del rendererGT

            currentTeapotModel = teapot_i
            center = center_teapots[teapot_i]

            addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list,  v_teapots[currentTeapotModel][0], f_list_teapots[currentTeapotModel][0], vc_teapots[currentTeapotModel][0], vn_teapots[currentTeapotModel][0], uv_teapots[currentTeapotModel][0], haveTextures_list_teapots[currentTeapotModel][0], textures_list_teapots[currentTeapotModel][0])

            rendererGT = createRendererGT(glMode, chAzGT, chObjAzGT, chElGT, chDistGT, center, v, vc, f_list, vn, light_colorGT, chComponentGT, chVColorsGT, targetPosition, chDisplacementGT, chScaleGT, width,height, uv, haveTextures_list, textures_list, frustum, None )

            for hdrFile, hdrValues in hdrstorender:
                hdridx = hdrValues[0]
                envMapCoeffs = hdrValues[1]
                print("Ground truth on new teapot" + str(teapot_i))

                for numTeapotTrain in range(int(trainSize/(lenScenes*len(hdrstorender)*len(renderTeapotsList)))):

                    phiOffset = np.random.uniform(0,2*np.pi, 1)
                    from numpy.random import choice
                    objAzInterval = choice(len(collisionsProbs), size=1, p=collisionsProbs)
                    objAzGT = np.random.uniform(0,1)*(collisions[targetIndex][1][objAzInterval][1] - collisions[targetIndex][1][objAzInterval][0]) + collisions[targetIndex][1][objAzInterval][0]
                    objAzGT = objAzGT*np.pi/180
                    chObjAzGT[:] = objAzGT.copy()
                    totalOffset = phiOffset + chObjAzGT.r
                    envMapCoeffsRotated = np.dot(light_probes.sphericalHarmonicsZRotation(totalOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]].copy()
                    envMapCoeffsRotatedRel = np.dot(light_probes.sphericalHarmonicsZRotation(phiOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]].copy()
                    shCoeffsRGB = envMapCoeffsRotated.copy()
                    shCoeffsRGBRel = envMapCoeffsRotatedRel.copy()
                    chShCoeffs = 0.3*shCoeffsRGB[:,0] + 0.59*shCoeffsRGB[:,1] + 0.11*shCoeffsRGB[:,2]
                    chShCoeffsRel = 0.3*shCoeffsRGBRel[:,0] + 0.59*shCoeffsRGBRel[:,1] + 0.11*shCoeffsRGBRel[:,2]
                    chAmbientSHGT = chShCoeffs * chAmbientIntensityGT * clampedCosCoeffs
                    chAmbientSHGTRel = chShCoeffsRel * chAmbientIntensityGT * clampedCosCoeffs
                    chComponentGT[:] = chAmbientSHGT.r[:].copy()
                    chComponentGTRel[:] = chAmbientSHGTRel.r[:].copy()

                    chAzGT[:] = np.mod(np.random.uniform(0,np.pi, 1) - np.pi/2, 2*np.pi)
                    chElGT[:] = np.random.uniform(0,np.pi/2, 1)

                    chLightAzGT[:] = np.random.uniform(0,2*np.pi, 1)
                    chLightElGT[:] = np.random.uniform(0,np.pi/3, 1)
                    chLightIntensityGT[:] = 0
                    # chLightIntensityGT[:] = np.random.uniform(5,10, 1)

                    chVColorsGT[:] =  np.random.uniform(0,0.7, [1, 3])

                    image = rendererGT.r.copy()

                    occlusion = getOcclusionFraction(rendererGT)
                    if occlusion < 0.9:
                        # hogs = hogs + [imageproc.computeHoG(image).reshape([1,-1])]
                        # illumfeats = illumfeats + [imageproc.featuresIlluminationDirection(image,20)]

                        cv2.imwrite(gtDir + 'images/im' + str(train_i) + '.jpeg' , 255*image[:,:,[2,1,0]], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                        #Add groundtruth to arrays
                        trainAzsGT = chAzGT.r
                        trainObjAzsGT = chObjAzGT.r
                        trainElevsGT = chElGT.r
                        trainLightAzsGT = chLightAzGT.r
                        trainLightElevsGT = chLightElGT.r
                        trainLightIntensitiesGT = chLightIntensityGT.r
                        trainVColorGT = chVColorsGT.r
                        trainComponentsGT = chComponentGT.r[None, :].copy()
                        trainComponentsGTRel = chComponentGTRel.r[None, :].copy()
                        trainEnvMapPhiOffsets = phiOffset
                        trainScenes = sceneNumber
                        trainTeapotIds = teapot_i
                        trainEnvMaps = hdridx
                        trainOcclusions = occlusion
                        trainIds = train_i
                        trainTargetIndices = targetIndex

                        gtDataset.resize(gtDataset.shape[0]+1, axis=0)
                        gtDataset[-1] = np.array([(trainIds, trainAzsGT,trainObjAzsGT,trainElevsGT,trainLightAzsGT,trainLightElevsGT,trainLightIntensitiesGT,trainVColorGT,trainScenes,trainTeapotIds,trainEnvMaps,trainOcclusions,trainTargetIndices, trainComponentsGT,trainComponentsGTRel, phiOffset)],dtype=gtDtype)
                        train_i = train_i + 1

            removeObjectData(0, v, f_list, vc, vn, uv, haveTextures_list, textures_list)

# np.savetxt(gtDir + 'data.txt',np.array(np.hstack([trainIds[:,None], trainAzsGT[:,None], trainObjAzsGT[:,None], trainElevsGT[:,None], phiOffsets[:,None], trainOcclusions[:,None]])), fmt="%g")
gtDataFile.close()
# trainData = {'hdrs':hdrs, 'trainIndices':trainIndices, 'trainAzsGT':trainAzsGT, 'trainObjAzsGT':trainObjAzsGT, 'trainElevsGT':trainElevsGT, 'trainLightAzsGT':trainLightAzsGT, 'trainLightElevsGT':trainLightElevsGT, 'trainLightIntensitiesGT':trainLightIntensitiesGT, 'trainVColorGT':trainVColorGT, 'trainComponentsGT':trainComponentsGT, 'trainComponentsGTRel':trainComponentsGTRel, 'phiOffsets':phiOffsets, 'trainScenes':trainScenes, 'trainTeapotIds':trainTeapotIds, 'trainEnvMaps':trainEnvMaps, 'trainOcclusions':trainOcclusions}

# with open(gtDir + 'annotations.pickle', 'wb') as pfile:
#     pickle.dump(gtDir + 'annotations.pickle', pfile)
