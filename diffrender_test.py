__author__ = 'pol'
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
import recognition_models
import numpy as np
import cv2
from utils import *
import glfw
import score_image
import matplotlib.pyplot as plt
from opendr_utils import *
import OpenGL.GL as GL
import light_probes

plt.ion()

#Main script options:
useBlender = True
loadBlenderSceneFile = True
groundTruthBlender = False
useCycles = True
demoMode = False
unpackModelsFromBlender = False
unpackSceneFromBlender = False
loadSavedSH = False
useGTasBackground = False
refreshWhileMinimizing = False
computePerformance = False
glModes = ['glfw','mesa']
glMode = glModes[1]
sphericalMap = False

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

width, height = (400, 400)

win = -1

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


angle = 60 * 180 / numpy.pi
clip_start = 0.05
clip_end = 10

camDistance = 0.75

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

sceneIdx = 0
sceneDicFile = 'data/scene' + str(sceneIdx) + '.pickle'

if useBlender and not loadBlenderSceneFile:
    scene, targetPosition = sceneimport.loadBlenderScene(sceneIdx, width, height, useCycles)
    targetPosition = np.array(targetPosition)
    #Save barebones scene.

elif useBlender and loadBlenderSceneFile:
    bpy.ops.wm.open_mainfile(filepath='data/scene' + str(sceneIdx) + '.blend')
    scene = bpy.data.scenes['Main Scene']
    scene.render.resolution_x = width #perhaps set resolution in code
    scene.render.resolution_y = height
    scene.render.tile_x = height/2
    scene.render.tile_y = width/2
    bpy.context.screen.scene = scene
    targetPosition = np.array(sceneimport.getSceneTargetParentPosition(sceneIdx))
if unpackSceneFromBlender:
    v, f_list, vc, vn, uv, haveTextures_list, textures_list = sceneimport.unpackBlenderScene(scene, sceneDicFile, targetPosition, True)
else:
    v, f_list, vc, vn, uv, haveTextures_list, textures_list, targetPosition = sceneimport.loadSavedScene(sceneDicFile)

targetModels = []
if useBlender and not loadBlenderSceneFile:
    [targetScenes, targetModels, transformations] = sceneimport.loadTargetModels(renderTeapotsList)
elif useBlender:
    teapots = [line.strip() for line in open('teapots.txt')]
    selection = [ teapots[i] for i in renderTeapotsList]
    for teapotIdx, teapotName in enumerate(selection):
        targetModels = targetModels + [bpy.data.scenes[teapotName[0:63]].objects['teapotInstance' + str(renderTeapotsList[teapotIdx])]]

if useBlender and not loadBlenderSceneFile:
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath='data/scene' + str(sceneIdx) + '.blend')

    # bpy.ops.wm.save_as_mainfile(filepath='data/targets.blend')
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

shCoefficientsFile = 'data/sceneSH' + str(sceneIdx) + '.pickle'

chAmbientIntensityGT = ch.Ch([2])
clampedCosCoeffs = clampedCosineCoefficients()
chAmbientSHGT = ch.zeros([9])

if useBlender:
    if not loadSavedSH:
        #Spherical harmonics
        bpy.context.scene.render.engine = 'CYCLES'

        #Option 1: Sph Harmonic from Light probes using Cycles:
        # scene.sequencer_colorspace_settings.name = 'Linear'
        # for item in bpy.context.selectable_objects:
        #     item.select = False
        # light_probes.lightProbeOp(bpy.context)
        # lightProbe = bpy.context.scene.objects[0]
        # lightProbe.select = True
        # bpy.context.scene.objects.active = lightProbe
        # lightProbe.location = mathutils.Vector((targetPosition[0], targetPosition[1],targetPosition[2] + 0.15))
        # scene.update()
        # lp_data = light_probes.bakeOp(bpy.context)
        # coeffs = lp_data[0]['coeffs']
        # scene.objects.unlink(lightProbe)
        # scene.update()
        # shCoeffsList = [coeffs[0]['coeffs']['0']['0']]
        # shCoeffsList = shCoeffsList + [coeffs['1']['1']]
        # shCoeffsList = shCoeffsList + [coeffs['1']['0']]
        # shCoeffsList = shCoeffsList + [coeffs['1']['-1']]
        # shCoeffsList = shCoeffsList + [coeffs['2']['-2']]
        # shCoeffsList = shCoeffsList + [coeffs['2']['-1']]
        # shCoeffsList = shCoeffsList + [coeffs['2']['0']]
        # shCoeffsList = shCoeffsList + [coeffs['2']['1']]
        # shCoeffsList = shCoeffsList + [coeffs['2']['2']]
        # shCoeffsRGB = np.vstack(shCoeffsList)
        #Option 2:

        # with open(shCoefficientsFile, 'wb') as pfile:
        #     shCoeffsDic = {'shCoeffs':shCoeffs}
        #     pickle.dump(shCoeffsDic, pfile)

import imageio

envMapFilename = 'data/hdr/dataset/studio_land.hdr'
envMapTexture = np.array(imageio.imread(envMapFilename))[:,:,0:3]
phiOffset = 0
if sphericalMap:
    mask = np.ones([envMapTexture.shape[0],envMapTexture.shape[1]]).astype(np.uint8)
    mask[np.int(mask.shape[0]/2), np.int(mask.shape[1]/2)] = 0
    distMask = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    envMapTexture[distMask > mask.shape[0]/2,:] = 0
    envMapMean = envMapTexture[distMask <= mask.shape[0]/2].mean()
    envMapTexture[distMask <= mask.shape[0]/2, :] = envMapTexture[distMask <= mask.shape[0]/2]
    envMapCoeffs = light_probes.getEnvironmentMapCoefficients(envMapTexture, envMapMean,  -phiOffset, 'spherical')
else:
    envMapMean = envMapTexture.mean()

    envMapCoeffs = light_probes.getEnvironmentMapCoefficients(envMapTexture, envMapMean, phiOffset, 'equirectangular')

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
    # vcchmod = [np.ones_like(vcmodflat[mesh])*chVColors.reshape([1,3]) for mesh in rangeMeshes]
    vcchmod = [ch.array(vcmodflat[mesh]) for mesh in rangeMeshes]
    vcmod_list = computeSphericalHarmonics(vnchmodnorm, vcchmod, light_color, chComponent)
    # vcmod_list =  computeGlobalAndPointLighting(vchmod, vnchmod, vcchmod, lightPos, chGlobalConstant, light_color)
    renderer = TexturedRenderer()
    renderer.set(glMode = glMode)
    setupTexturedRenderer(renderer, vstackmod, vchmod, fmod_list, vcmod_list, vnchmodnorm,  uvmod, haveTexturesmod_list, texturesmod_list, camera, frustum, win)
    renderer.r
    renderer_teapots = renderer_teapots + [renderer]

currentTeapotModel = 0
renderer = renderer_teapots[currentTeapotModel]

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
# vcch[0] = np.ones_like(vcflat[0])*chVColorsGT.reshape([1,3])
vcch[0] = vcflat[0]
vc_list = computeSphericalHarmonics(vnchnorm, vcch, light_colorGT, chComponentGT)
# vc_list =  computeGlobalAndPointLighting(vch, vnch, vcch, lightPosGT, chGlobalConstantGT, light_colorGT)

setupTexturedRenderer(rendererGT, vstack, vch, f_list, vc_list, vnchnorm,  uv, haveTextures_list, textures_list, cameraGT, frustum, win)
rendererGT.r

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

if useBlender:

    #Add ambient lighting to scene (rectangular lights at even intervals).
    # addAmbientLightingScene(scene, useCycles)

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
    lamp.data.node_tree.nodes['Emission'].inputs[1].default_value = chLightIntensityGT.r
    scene.objects.link(lamp)

    teapot = blender_teapots[currentTeapotModel]
    teapotGT = blender_teapots[currentTeapotModel]
    placeNewTarget(scene, teapot, targetPosition)

    placeCamera(scene.camera, -chAzGT[0].r*180/np.pi, chElGT[0].r*180/np.pi, chDistGT, center)
    scene.update()
    # bpy.ops.file.pack_all()
    # bpy.ops.wm.save_as_mainfile(filepath='data/scene' + str(sceneIdx) + '_complete.blend')
    scene.render.filepath = 'blender_envmap_render.png'
    # bpy.ops.render.render(write_still=True)

import glob
for hdridx, hdrFile in enumerate(glob.glob("data/hdr/dataset/*")):

    envMapFilename = hdrFile
    envMapTexture = np.array(imageio.imread(envMapFilename))[:,:,0:3]
    phiOffset = 0

    phiOffsets = [0, np.pi/2, np.pi, 3*np.pi/2]

    if not os.path.exists('light_probes/envMap' + str(hdridx)):
        os.makedirs('light_probes/envMap' + str(hdridx))

    cv2.imwrite('light_probes/envMap' + str(hdridx) + '/texture.png' , 255*envMapTexture[:,:,[2,1,0]])

    for phiOffset in phiOffsets:
        envMapMean = envMapTexture.mean()
        envMapCoeffs = light_probes.getEnvironmentMapCoefficients(envMapTexture, envMapMean, -phiOffset, 'equirectangular')
        shCoeffsRGB[:] = envMapCoeffs
        if useBlender:
            updateEnviornmentMap(envMapFilename, scene)
            setEnviornmentMapStrength(0.3/envMapMean, scene)
            rotateEnviornmentMap(phiOffset, scene)

        scene.render.filepath = 'light_probes/envMap' + str(hdridx) + '/blender_' + str(np.int(180*phiOffset/np.pi)) + '.png'
        bpy.ops.render.render(write_still=True)
        cv2.imwrite('light_probes/envMap' + str(hdridx) + '/opendr_' + str(np.int(180*phiOffset/np.pi)) + '.png' , 255*rendererGT.r[:,:,[2,1,0]])

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
changeRenderer = False
printStats = False
beginTraining = False
createGroundTruth = False
beginTesting = False
exploreSurface = False
newTeapotAsGT = False

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
