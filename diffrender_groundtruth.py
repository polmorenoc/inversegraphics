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

print("Creating Ground Truth")
trainSize = 1000
testSize = 20

trainAzsGT = numpy.random.uniform(0,2*np.pi, trainSize)
trainElevsGT = numpy.random.uniform(0,np.pi/2, trainSize)
trainLightAzsGT = numpy.random.uniform(0,2*np.pi, trainSize)
trainLightElevsGT = numpy.random.uniform(0,np.pi/3, trainSize)
trainLightIntensitiesGT = numpy.random.uniform(5,10, trainSize)
trainVColorGT = numpy.random.uniform(0,0.7, [trainSize, 3])

trainData = {'trainAzsGT':trainAzsGT,'trainElevsGT':trainElevsGT,'trainLightAzsGT':trainLightAzsGT,'trainLightElevsGT':trainLightElevsGT,'trainLightIntensitiesGT':trainLightIntensitiesGT, 'trainVColorGT':trainVColorGT}

# testAzsGT = numpy.random.uniform(4.742895587179587 - np.pi/4,4.742895587179587 + np.pi/4, testSize)
testAzsGT = numpy.random.uniform(0,2*np.pi, testSize)
testElevsGT = numpy.random.uniform(0,np.pi/3, testSize)
testLightAzsGT = numpy.random.uniform(0,2*np.pi, testSize)
testLightElevsGT = numpy.random.uniform(0,np.pi/3, testSize)
testLightIntensitiesGT = numpy.random.uniform(5,10, testSize)
testVColorGT = numpy.random.uniform(0,0.7, [testSize, 3])
testData = {'testAzsGT':testAzsGT,'testElevsGT':testElevsGT,'testLightAzsGT':testLightAzsGT,'testLightElevsGT':testLightElevsGT,'testLightIntensitiesGT':testLightIntensitiesGT, 'testVColorGT':testVColorGT}

with open(trainDataName, 'wb') as pfile:
    pickle.dump(trainData, pfile)

with open(testDataName, 'wb') as pfile:
    pickle.dump(testData, pfile)

#Render things now!!!