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
from opendr.lighting import LambertianPointLight
from opendr.lighting import SphericalHarmonics
from opendr.filters import gaussian_pyramid
import geometry
import imageproc
import recognition_models
import numpy as np
import cv2
from utils import *
import glfw
import score_image
import matplotlib.pyplot as plt
from opendr_utils import *
from OpenGL.arrays import vbo
import OpenGL.GL as GL
import light_probes
import curses


numpy.random.seed(1)
trainprefix = 'train2/'
testGTprefix = 'test2/'
testprefix = 'test2-robust/'
if not os.path.exists('experiments/' + trainprefix):
    os.makedirs('experiments/' + trainprefix)
if not os.path.exists('experiments/' + testGTprefix):
    os.makedirs('experiments/' + testGTprefix)
trainDataName = 'experiments/' + trainprefix + 'groundtruth.pickle'
testDataName = 'experiments/' + testGTprefix +  'groundtruth.pickle'
trainedModels = {}

width, height = (100, 100)
glModes = ['glfw','mesa']
glMode = glModes[0]
win = -1
demoMode = False

if glMode == 'glfw':
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

useBlender = False
groundTruthBlender = False
useCycles = True

angle = 60 * 180 / numpy.pi
clip_start = 0.05
clip_end = 10

camDistance = 0.4

teapots = [line.strip() for line in open('teapots.txt')]
renderTeapotsList = np.arange(len(teapots))

v_teapots = []
f_list_teapots = []
vc_teapots = []
vn_teapots = []
uv_teapots = []
haveTextures_list_teapots = []
textures_list_teapots = []
renderer_teapots = []
blender_teapots = []
center_teapots = []


unpackModelsFromBlender = False
if useBlender:
    [targetScenes, targetModels, transformations] = sceneimport.loadTargetModels(renderTeapotsList)
for teapotIdx in renderTeapotsList:
    teapotNum = renderTeapotsList[teapotIdx]
    objectDicFile = 'data/target' + str(teapotNum) + '.pickle'
    if useBlender:
        teapot = targetModels[teapotNum]
        teapot.layers[1] = True
        teapot.layers[2] = True
        blender_teapots = blender_teapots + [teapot]
    if unpackModelsFromBlender:
        vmod, fmod_list, vcmod, vnmod, uvmod, haveTexturesmod_list, texturesmod_list = sceneimport.unpackBlenderObject(teapot, objectDicFile, True)
    else:
        vmod, fmod_list, vcmod, vnmod, uvmod, haveTexturesmod_list, texturesmod_list = sceneimport.loadSavedObject(objectDicFile)
    v_teapots = v_teapots + [[vmod]]
    f_list_teapots = f_list_teapots + [[fmod_list]]
    vc_teapots = vc_teapots + [[vcmod]]
    vn_teapots = vn_teapots + [[vnmod]]
    uv_teapots = uv_teapots + [[uvmod]]
    haveTextures_list_teapots = haveTextures_list_teapots + [[haveTexturesmod_list]]
    textures_list_teapots = textures_list_teapots + [[texturesmod_list]]
    vflat = [item for sublist in vmod for item in sublist]
    varray = np.vstack(vflat)
    center_teapots = center_teapots + [np.sum(varray, axis=0)/len(varray)]

sceneIdx = 0
sceneDicFile = 'data/scene' + str(sceneIdx) + '.pickle'

unpackSceneFromBlender = False
if useBlender:
    scene, targetPosition = sceneimport.loadBlenderScene(sceneIdx, width, height, useCycles)
    targetPosition = np.array(targetPosition)
if unpackSceneFromBlender:
    v, f_list, vc, vn, uv, haveTextures_list, textures_list = sceneimport.unpackBlenderScene(scene, sceneDicFile, targetPosition, True)
else:
    v, f_list, vc, vn, uv, haveTextures_list, textures_list, targetPosition = sceneimport.loadSavedScene(sceneDicFile)


# 1 Prepare each teapot renderer.
# 2 Add first teapot to blender scene and GT renderer.

# chAz = ch.Ch([4.742895587179587])
# chEl = ch.Ch([0.22173048])
# chDist = ch.Ch([camDistance])

chAz = ch.Ch([1.1693706])
chEl =  ch.Ch([0.95993109])
chDist = ch.Ch([camDistance])

chAzGT = ch.Ch([1.1693706])
chElGT = ch.Ch([0.95993109])
chDistGT = ch.Ch([camDistance])
chComponentGT = ch.Ch(np.array([2, 0.25, 0.25, 0.12,-0.17,0.36,0.1,0.,0.]))
chComponent = ch.Ch(np.array([2, 0.25, 0.25, 0.12,-0.17,0.36,0.1,0.,0.]))
frustum = {'near': clip_start, 'far': clip_end, 'width': width, 'height': height}

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
 
loadSavedSH = True
shCoefficientsFile = 'sceneSH' + str(sceneIdx) + '.pickle'

chAmbientIntensityGT = ch.Ch([7])
clampedCosCoeffs = clampedCosineCoefficients()
chAmbientSHGT = ch.zeros([9])
if useBlender:
    if not loadSavedSH:
        #Spherical harmonics
        bpy.context.scene.render.engine = 'CYCLES'
        scene.sequencer_colorspace_settings.name = 'Linear'
        for item in bpy.context.selectable_objects:
            item.select = False
        light_probes.lightProbeOp(bpy.context)
        lightProbe = bpy.context.scene.objects[0]
        lightProbe.select = True
        bpy.context.scene.objects.active = lightProbe
        lightProbe.location = mathutils.Vector((targetPosition[0], targetPosition[1],targetPosition[2] + 0.15))
        scene.update()
        lp_data = light_probes.bakeOp(bpy.context)
        scene.objects.unlink(lightProbe)
        scene.update()

        shCoeffsList = [lp_data[0]['coeffs']['0']['0']]
        shCoeffsList = shCoeffsList + [lp_data[0]['coeffs']['1']['1']]
        shCoeffsList = shCoeffsList + [lp_data[0]['coeffs']['1']['0']]
        shCoeffsList = shCoeffsList + [lp_data[0]['coeffs']['1']['-1']]
        shCoeffsList = shCoeffsList + [lp_data[0]['coeffs']['2']['-2']]
        shCoeffsList = shCoeffsList + [lp_data[0]['coeffs']['2']['-1']]
        shCoeffsList = shCoeffsList + [lp_data[0]['coeffs']['2']['0']]
        shCoeffsList = shCoeffsList + [lp_data[0]['coeffs']['2']['1']]
        shCoeffsList = shCoeffsList + [lp_data[0]['coeffs']['2']['2']]
        shCoeffsRGB = np.vstack(shCoeffsList)
        shCoeffs = 0.3*shCoeffsRGB[:,0] + 0.59*shCoeffsRGB[:,1] + 0.11*shCoeffsRGB[:,2]
        with open(shCoefficientsFile, 'wb') as pfile:
            shCoeffsDic = {'shCoeffs':shCoeffs}
            pickle.dump(shCoeffsDic, pfile)
        chAmbientSHGT = shCoeffs.ravel() * chAmbientIntensityGT * clampedCosCoeffs

if loadSavedSH:
    if os.path.isfile(shCoefficientsFile):
        with open(shCoefficientsFile, 'rb') as pfile:
            shCoeffsDic = pickle.load(pfile)
            shCoeffs = shCoeffsDic['shCoeffs']
            chAmbientSHGT = shCoeffs.ravel()* chAmbientIntensityGT * clampedCosCoeffs


chLightRadGT = ch.Ch([0.1])
chLightDistGT = ch.Ch([0.5])
chLightIntensityGT = ch.Ch([10])
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

# chComponentGT = ch.Ch(np.array([2, 0., 0., 0.,0.,0.,0.,0.,0.]))
# chComponent = ch.Ch(np.array([2, 0., 0., 0.,0.,0.,0.,0.,0.]))

chDisplacement = ch.Ch([0.0, 0.0,0.0])
chDisplacementGT = ch.Ch([0.0,0.0,0.0])
chScale = ch.Ch([1.0,1.0,1.0])
chScaleGT = ch.Ch([1, 1.,1.])
scaleMat = geometry.Scale(x=chScale[0], y=chScale[1],z=chScale[2])[0:3,0:3]
scaleMatGT = geometry.Scale(x=chScaleGT[0], y=chScaleGT[1],z=chScaleGT[2])[0:3,0:3]
invTranspModel = ch.transpose(ch.inv(scaleMat))
invTranspModelGT = ch.transpose(ch.inv(scaleMatGT))

rendererGT = TexturedRenderer()
rendererGT.set(glMode=glMode)

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

    #Setup backrpojection renderer
    vmodflat = [item.copy() for sublist in vmod for item in sublist]
    rangeMeshes = range(len(vmodflat))
    # ipdb.set_trace()
    vchmod = [ch.dot(ch.array(vmodflat[mesh]),scaleMat) + targetPosition for mesh in rangeMeshes]
    if len(vchmod)==1:
        vstackmod = vchmod[0]
    else:
        vstackmod = ch.vstack(vchmod)
    camera, modelRotation = setupCamera(vstackmod, chAz, chEl, chDist, centermod + targetPosition + chDisplacement, width, height)
    vnmodflat = [item.copy() for sublist in vnmod for item in sublist]
    vnchmod = [ch.dot(ch.array(vnmodflat[mesh]),invTranspModel) for mesh in rangeMeshes]
    vnchmodnorm = [vnchmod[mesh]/ch.sqrt(vnchmod[mesh][:,0]**2 + vnchmod[mesh][:,1]**2 + vnchmod[mesh][:,2]**2).reshape([-1,1]) for mesh in rangeMeshes]
    vcmodflat = [item.copy() for sublist in vcmod for item in sublist]
    vcchmod = [np.ones_like(vcmodflat[mesh])*chVColors.reshape([1,3]) for mesh in rangeMeshes]
    # vcchmod = [ch.array(vcmodflat[mesh]) for mesh in rangeMeshes]
    vcmod_list = computeSphericalHarmonics(vnchmodnorm, vcchmod, light_color, chComponent)
    # vcmod_list =  computeGlobalAndPointLighting(vchmod, vnchmod, vcchmod, lightPos, chGlobalConstant, light_color)
    renderer = TexturedRenderer()
    renderer.set(glMode = glMode)
    setupTexturedRenderer(renderer, vstackmod, vchmod, fmod_list, vcmod_list, vnchmodnorm,  uvmod, haveTexturesmod_list, texturesmod_list, camera, frustum, win)
    renderer.r
    renderer_teapots = renderer_teapots + [renderer]

currentTeapotModel = 0
renderer = renderer_teapots[currentTeapotModel]

if useBlender:
    #Add directional light to match spherical harmonics
    lamp_data = bpy.data.lamps.new(name="point", type='POINT')
    lamp = bpy.data.objects.new(name="point", object_data=lamp_data)
    lamp.layers[1] = True
    lamp.layers[2] = True
    center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)
    lampLoc = getRelativeLocation(chLightAzGT.r, chLightElGT.r, chLightDistGT, center)
    lamp.location = mathutils.Vector((lampLoc[0],lampLoc[1],lampLoc[2]))
    lamp.data.cycles.use_multiple_importance_sampling = True
    lamp.data.use_nodes = True
    lamp.data.node_tree.nodes['Emission'].inputs[1].default_value = 7
    scene.objects.link(lamp)

    teapot = blender_teapots[currentTeapotModel]
    teapotGT = blender_teapots[currentTeapotModel]
    placeNewTarget(scene, teapot, targetPosition)

    placeCamera(scene.camera, -chAzGT[0].r*180/np.pi, chElGT[0].r*180/np.pi, chDistGT, center)
    scene.update()

addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list,  v_teapots[currentTeapotModel][0], f_list_teapots[currentTeapotModel][0], vc_teapots[currentTeapotModel][0], vn_teapots[currentTeapotModel][0], uv_teapots[currentTeapotModel][0], haveTextures_list_teapots[currentTeapotModel][0], textures_list_teapots[currentTeapotModel][0])

#Setup ground truth renderer
vflat = [item for sublist in v for item in sublist]
rangeMeshes = range(len(vflat))
vch = [ch.array(vflat[mesh]) for mesh in rangeMeshes]
vch[0] = ch.dot(vch[0], scaleMatGT) + targetPosition
if len(vch)==1:
    vstack = vch[0]
else:
    vstack = ch.vstack(vch)

center = center_teapots[currentTeapotModel]
cameraGT, modelRotationGT = setupCamera(vstack, chAzGT, chElGT, chDistGT, center + targetPosition + chDisplacementGT, width, height)
vnflat = [item for sublist in vn for item in sublist]
vnch = [ch.array(vnflat[mesh]) for mesh in rangeMeshes]
vnch[0] = ch.dot(vnch[0], invTranspModelGT)
vnchnorm = [vnch[mesh]/ch.sqrt(vnch[mesh][:,0]**2 + vnch[mesh][:,1]**2 + vnch[mesh][:,2]**2).reshape([-1,1]) for mesh in rangeMeshes]
vcflat = [item for sublist in vc for item in sublist]
vcch = [ch.array(vcflat[mesh]) for mesh in rangeMeshes]
vcch[0] = np.ones_like(vcflat[0])*chVColorsGT.reshape([1,3])
vc_list = computeSphericalHarmonics(vnchnorm, vcch, light_colorGT, chComponentGT)
# vc_list =  computeGlobalAndPointLighting(vch, vnch, vcch, lightPosGT, chGlobalConstantGT, light_colorGT)

setupTexturedRenderer(rendererGT, vstack, vch, f_list, vc_list, vnchnorm,  uv, haveTextures_list, textures_list, cameraGT, frustum, win)
rendererGT.r

useGTasBackground = False
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

changedGT = False
refresh = True
updateErrorFunctions = False
pendingCyclesRender = True


computePerformance = False
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
changeRenderer = False
printStats = False
beginTraining = False
createGroundTruth = False
beginTesting = False
exploreSurface = False
newTeapotAsGT = False

global chAzSaved
global chElSaved
global chComponentSaved
chAzSaved = chAz.r[0]
chElSaved = chEl.r[0]
chComponentSaved = chComponent.r[0]


plt.imsave('opendr_opengl_final.png', rendererGT.r)

