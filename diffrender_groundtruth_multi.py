__author__ = 'pol'

import matplotlib
# matplotlib.use('Qt4Agg')
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
import pdb
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
prefix = 'cian_data_dir_whitebg_1_repeat'
# prefix = 'train4_occlusion_shapemodel_newscenes_eccvworkshop'
previousGTPrefix = 'cian_data_dir_whitebg'

#Main script options:

renderFromPreviousGT = False
useShapeModel = True
useOpenDR = False
useBlender = False
renderBlender = False
captureEnvMapFromBlender = False
parseSceneInstantiations = False
loadBlenderSceneFile = True
useCycles = True
unpackModelsFromBlender = False
unpackSceneFromBlender = False
loadSavedSH = False
generateTriplets = False

replaceNewGroundtruth = True
renderOcclusions = False
occlusionMin = 0.0
occlusionMax = 0.9
renderTeapots =  True
renderMugs = False
showMug = False
centeredObject = True
fixedCamDistance = True

useDirectionalLight = True
renderBackground = False

teapotSceneIndex = 0
mugSceneIndex = 0
if renderTeapots and renderMugs:
    mugSceneIndex = 1

glModes = ['glfw','mesa']
glMode = glModes[1]

width, height = (64, 64)
win = -1

coords = np.meshgrid(np.arange(width) - width / 2, np.arange(height) - height / 2)
coordsMugX = np.array([0])
coordsMugY = np.array([0])
coordsTeapotX = np.array([0])
coordsTeapotY = np.array([0])

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
clip_start = 0.05
clip_end = 10
frustum = {'near': clip_start, 'far': clip_end, 'width': width, 'height': height}
camDistance = 0.4

teapotsFile = 'teapots.txt'
teapots = [line.strip() for line in open(teapotsFile)]
renderTeapotsList = np.arange(len(teapots))[0:1]
mugs = [line.strip() for line in open('mugs.txt')]
renderMugsList = np.arange(len(mugs))[0:1]

sceneIdx = 0
# replaceableScenesFile = '../databaseFull/fields/scene_replaceables_backup.txt'
replaceableScenesFile = '../databaseFull/fields/scene_replaceables_backup_new.txt'
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
if useBlender:
    selection = [ teapots[i] for i in renderTeapotsList]
    scene_io_utils.loadTargetsBlendData()
    for teapotIdx, teapotName in enumerate(selection):
        teapot = bpy.data.scenes[teapotName[0:63]].objects['teapotInstance' + str(renderTeapotsList[teapotIdx])]
        teapot.layers[1] = True
        teapot.layers[2] = True
        targetModels = targetModels + [teapot]
        blender_teapots = blender_teapots + [teapot]


    if renderMugs:
        selectionMugs = [ mugs[i] for i in renderMugsList]
        scene_io_utils.loadMugsBlendData()
        for mugIdx, mugName in enumerate(selectionMugs):
            mug = bpy.data.scenes[mugName[0:63]].objects['mugInstance' + str(renderMugsList[mugIdx])]
            mug.layers[1] = True
            mug.layers[2] = True
            mugModels = mugModels + [mug]
            blender_mugs = blender_mugs + [mug]


v_teapots, f_list_teapots, vc_teapots, vn_teapots, uv_teapots, haveTextures_list_teapots, textures_list_teapots, vflat, varray, center_teapots = scene_io_utils.loadTeapotsOpenDRData(renderTeapotsList, useBlender, unpackModelsFromBlender, targetModels)

if renderMugs:
    v_mugs, f_list_mugs, vc_mugs, vn_mugs, uv_mugs, haveTextures_list_mugs, textures_list_mugs, vflat_mugs, varray_mugs, center_mugs = scene_io_utils.loadMugsOpenDRData(renderMugsList, useBlender, unpackModelsFromBlender, mugModels)

    v_mug = v_mugs[0][0]
    f_list_mug = f_list_mugs[0][0]
    chVColorsMug = ch.Ch([1,0,0])
    vc_mug = [chVColorsMug * np.ones(v_mug[0].shape)]
    vn_mug = vn_mugs[0][0]
    uv_mug = uv_mugs[0][0]
    haveTextures_list_mug = haveTextures_list_mugs[0][0]
    textures_list_mug = textures_list_mugs[0][0]

chObjAz = ch.Ch([0])

chDist = ch.Ch([camDistance])

chObjAzGT = ch.Ch([0])
chAzGT = ch.Ch([0])
chElGT = ch.Ch([0])
chAzRelGT = chAzGT - chObjAzGT
chDistGT = ch.Ch([camDistance])
chComponentGT = ch.Ch(np.array([2, 0.25, 0.25, 0.12,-0.17,0.36,0.1,0.,0.]))

chObjDistGT = ch.Ch([0])
chObjRotationGT = ch.Ch([0])

chObjAzMug = ch.Ch([0])
chObjDistMug = ch.Ch([0])
chObjRotationMug = ch.Ch([0])

light_colorGT = ch.ones(3)

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
chLightRadGT = ch.Ch([0.1])
chLightDistGT = ch.Ch([0.5])
chLightIntensityGT = ch.Ch([1])
chLightAzGT = ch.Ch([0])
chLightElGT = ch.Ch([0])
chGlobalConstantGT = ch.Ch([0.5])
angleGT = ch.arcsin(chLightRadGT/chLightDistGT)
zGT = chZonalHarmonics(angleGT)

shDirLightGT = chZonalToSphericalHarmonics(zGT, np.pi/2 - chLightElGT, chLightAzGT - np.pi/2)

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

chDisplacementGT = ch.Ch([0.0,0.0,0.0])
chScaleGT = ch.Ch([1, 1.,1.])

currentTeapotModel = 0
currentMugModel = 0

if renderTeapots:
    addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list,  v_teapots[currentTeapotModel][0], f_list_teapots[currentTeapotModel][0], vc_teapots[currentTeapotModel][0], vn_teapots[currentTeapotModel][0], uv_teapots[currentTeapotModel][0], haveTextures_list_teapots[currentTeapotModel][0], textures_list_teapots[currentTeapotModel][0])

if renderMugs:
    addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list,  v_mugs[currentMugModel][0], f_list_mugs[currentMugModel][0], vc_mugs[currentMugModel][0], vn_mugs[currentMugModel][0], uv_mugs[currentMugModel][0], haveTextures_list_mugs[currentMugModel][0], textures_list_mugs[currentMugModel][0])

center = center_teapots[currentTeapotModel]

if useOpenDR:

    rendererGT = createRendererGT(glMode, chAzGT, chElGT, chDistGT, center, v, vc, f_list, vn, light_colorGT, chComponentGT, chVColorsGT, targetPosition[:].copy(), chDisplacementGT, width,height, uv, haveTextures_list, textures_list, frustum, None )
    rendererGT.overdraw = False
    rendererGT.nsamples = 8
    rendererGT.msaa = False
    rendererGT.initGL()
    rendererGT.initGLTexture()

    vis_gt = np.array(rendererGT.indices_image!=1).copy().astype(np.bool)
    vis_mask = np.array(rendererGT.indices_image==1).copy().astype(np.bool)

    shapeIm = vis_gt.shape

numPixels = height * width

## For blender parallell rendering.
import multiprocessing
numTileAxis = np.ceil(np.sqrt(multiprocessing.cpu_count())/2)
numTileAxis = 3


smCenterGT = ch.array([0,0,0.1])
shapeVerticesScaling = 0.09
teapotFilePath = 'data/teapotModel.pkl'
if useShapeModel:
    teapot_i = -1

    import shape_model
    #%% Load data

    teapotModel = shape_model.loadObject(teapotFilePath)
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

    chVerticesGT = chVerticesGT*shapeVerticesScaling
    smCenterGT = ch.array([0,0,0.1])

    smVerticesGT = [chVerticesGT]
    chNormalsGT = shape_model.chGetNormals(chVerticesGT, faces)
    smNormalsGT = [chNormalsGT]

else:
    latentDim = 1
    chShapeParamsGT = ch.array([0])


if useShapeModel:
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



### Renderer (only teapot)

teapotToRender = -1

# renderTeapotsList = np.arange(len(teapots))
renderTeapotsList = np.arange(len(teapots))[0:1]

targetModels = []

chAz = ch.Ch([0])
chObjAz = ch.Ch([0])
chAzRel = chAz - chObjAz

chEl =  ch.Ch([0.0])
chDist = ch.Ch([camDistance])

#Initialize to a random set of SH coefficients
chLightSHCoeffs = ch.Ch(np.array([2, 0.25, 0.25, 0.12,-0.17,0.36,0.1,0.,0.]))

# if multiObjects:
chObjDist = ch.Ch([0])
chObjRotation = ch.Ch([0])

chObjAzMug = ch.Ch([0])
chObjDistMug = ch.Ch([0])
chObjRotationMug = ch.Ch([0])

chVColorsMug = ch.Ch([1,0,0])

chComponent = chLightSHCoeffs * clampedCosCoeffs

light_color = ch.ones(3)

chVColors = ch.Ch([0.4,0.4,0.4])

chDisplacement = ch.Ch([0.0, 0.0,0.0])
chScale = ch.Ch([1.0,1.0,1.0])


if useShapeModel:
    import shape_model
    #%% Load data

    shapeParams = np.zeros(latentDim)
    chShapeParams = ch.Ch(shapeParams.copy())


    chVertices = shape_model.VerticesModel(chShapeParams=chShapeParams, meshLinearTransform=meshLinearTransform,W = W,b=b)
    chVertices.init()

    chVertices = ch.dot(geometry.RotateZ(-np.pi/2)[0:3,0:3],chVertices.T).T

    smFaces = [[faces]]
    smVColors = [chVColors*np.ones(chVertices.shape)]
    smUVs = ch.Ch(np.zeros([chVertices.shape[0],2]))
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

    if useShapeModel:
        center = smCenter
        UVs = smUVs
        v = smVertices
        vn = smNormals
        Faces = smFaces
        VColors = smVColors
        UVs = smUVs
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

    if renderMugs:
        verticesMug, normalsMug, mugPosOffset = transformObject(v_mug, vn_mug, chScale, chObjAzMug + np.pi / 2, chObjDistMug, chObjRotationMug, np.array([0,0,0]))

        VerticesB = [v] + [verticesMug]
        NormalsB = [vn] + [normalsMug]
        FacesB = [Faces] + [f_list_mug]
        VColorsB = [VColors] + [vc_mug]
        UVsB = [UVs] + [uv_mug]
        HaveTexturesB = [HaveTextures] + [haveTextures_list_mug]
        TexturesListB = [TexturesList] + [textures_list_mug]

        renderer = createRendererTarget(glMode, chAz, chEl, chDist, center, VerticesB, VColorsB, FacesB, NormalsB, light_color,chComponent, chVColors, np.array([0,0,0]), chDisplacement, width, height, UVsB, HaveTexturesB, TexturesListB, frustum, None)
    else:

        renderer = createRendererTarget(glMode, chAz, chEl, chDist, smCenter, [v], [smVColors], [smFaces], [vn], light_color, chComponent, chVColors, 0, chDisplacement, width,height, [smUVs], [smHaveTextures], [smTexturesList], frustum, None )
    renderer.msaa = True
    renderer.overdraw = True

    # chShapeParams[:] = np.zeros([latentDim])
    chVerticesMean = chVertices.r.copy()

# else:
#     renderer = renderer_teapots[teapotToRender]

########################################################
#######          Initialization ends here
########################################################

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

#Multi
trainObjDistGT = np.array([])
trainObjRotationGT = np.array([])
trainObjDistMug = np.array([])
trainObjRotationMug = np.array([])
trainObjAzMug = np.array([])
trainVColorsMug = np.array([])
trainTeapotElRel = np.array([])
trainMugElRel = np.array([])
trainMugPosOffset = np.array([])
trainTeapotPosOffset = np.array([])

trainLightCoefficientsGT = np.array([]).reshape([0,9])
trainLightCoefficientsGTRel = np.array([]).reshape([0,9])
trainAmbientIntensityGT = np.array([])
trainEnvMapPhiOffsets = np.array([])
trainShapeModelCoeffsGT = np.array([]).reshape([0,latentDim])

trainBBMug = np.array([],dtype=np.int8).reshape([0,4])
trainBBTeapot = np.array([],dtype=np.int8).reshape([0,4])

trainTeapotPresent = np.array([], dtype=np.bool)
trainMugPresent = np.array([], dtype=np.bool)

gtDir = 'groundtruth/' + prefix + '/'
if not os.path.exists(gtDir + 'images/'):
    os.makedirs(gtDir + 'images/')

if not os.path.exists(gtDir + 'sphericalharmonics/'):
    os.makedirs(gtDir + 'sphericalharmonics/')

if not os.path.exists(gtDir + 'images_opendr/'):
    os.makedirs(gtDir + 'images_opendr/')

if not os.path.exists(gtDir + 'masks_occlusion/'):
    os.makedirs(gtDir + 'masks_occlusion/')

if generateTriplets:
    if not os.path.exists(gtDir + 'triplets1/'):
        os.makedirs(gtDir + 'triplets1/')

    if not os.path.exists(gtDir + 'triplets2/'):
        os.makedirs(gtDir + 'triplets2/')

print("Generating renders")

sceneLines = [line.strip() for line in open(replaceableScenesFile)]
scenesToRender = range(len(sceneLines))[:]
scenesToRender = range(len(sceneLines))[0:1]

trainSize = 100000

renderTeapotsList = np.arange(len(teapots))[0:1]

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

gtDtype = [('trainIds', trainIds.dtype.name), ('trainAzsGT', trainAzsGT.dtype.name),('trainObjAzsGT', trainObjAzsGT.dtype.name),('trainElevsGT', trainElevsGT.dtype.name),
           ('trainLightAzsGT', trainLightAzsGT.dtype.name),('trainLightElevsGT', trainLightElevsGT.dtype.name),('trainLightIntensitiesGT', trainLightIntensitiesGT.dtype.name),
           ('trainVColorGT', trainVColorGT.dtype.name, (3,) ),('trainScenes', trainScenes.dtype.name),('trainTeapotIds', trainTeapotIds.dtype.name),
           ('trainEnvMaps', trainEnvMaps.dtype.name),('trainOcclusions', trainOcclusions.dtype.name),('trainTargetIndices', trainTargetIndices.dtype.name),
           ('trainLightCoefficientsGT',trainLightCoefficientsGT.dtype, (9,)), ('trainLightCoefficientsGTRel', trainLightCoefficientsGTRel.dtype, (9,)),
           ('trainAmbientIntensityGT', trainAmbientIntensityGT.dtype), ('trainEnvMapPhiOffsets', trainEnvMapPhiOffsets.dtype),
           ('trainShapeModelCoeffsGT', trainShapeModelCoeffsGT.dtype, (latentDim,)),
           ('trainObjDistGT', trainObjDistGT.dtype),
           ('trainObjRotationGT', trainObjRotationGT.dtype),
           ('trainObjDistMug', trainObjDistMug.dtype),
           ('trainObjRotationMug', trainObjRotationMug.dtype),
           ('trainObjAzMug', trainObjAzMug.dtype),
           ('trainVColorsMug', trainVColorsMug.dtype, (3,)),
           ('trainTeapotElRel', trainTeapotElRel.dtype),
           ('trainMugElRel', trainMugElRel.dtype),
           ('trainMugPosOffset', trainMugPosOffset.dtype,(3,)),
           ('trainTeapotPosOffset', trainTeapotPosOffset.dtype,(3,)),
           ('trainBBMug', trainBBMug.dtype, (4,)),
           ('trainBBTeapot', trainBBTeapot.dtype, (4,)),
           ('trainTeapotPresent', trainTeapotPresent.dtype),
           ('trainMugPresent', trainMugPresent.dtype)]

groundTruth = np.array([], dtype = gtDtype)
groundTruthFilename = gtDir + 'groundTruth.h5'

# gtDirToRender = 'groundtruth/' + previousGTPrefix + '/'
# gtDataFileToRender = h5py.File(gtDirToRender + 'groundTruth.h5', 'w')
# gtDatasetToRender = gtDataFileToRender.create_dataset(previousGTPrefix, data=groundTruth, maxshape=(None,))

nextId = 0
if not replaceNewGroundtruth:
    gtDataFile = h5py.File(groundTruthFilename, 'a')
    try:
        gtDataset = gtDataFile[prefix]
        if gtDataset.size > 0:
            nextId = gtDataset['trainIds'][-1] + 1
    except:

        gtDataset = gtDataFile.create_dataset(prefix, data=groundTruth, maxshape=(None,))
else:
    gtDataFile = h5py.File(groundTruthFilename, 'w')
    gtDataset = gtDataFile.create_dataset(prefix, data=groundTruth, maxshape=(None,))

train_i = nextId

#Re-producible groundtruth generation.

if train_i == 0:
    np.random.seed(1)

# np.random.seed(2)

unlinkedObj = None

scenesToRenderOcclusions = []
scenes = []
lenScenes = 0

#Compute how many different locations can the teapot be instantiated across all scenes.
for sceneIdx in scenesToRender:

    sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndices, targetPositions = scene_io_utils.getSceneInformation(sceneIdx, replaceableScenesFile)
    sceneDicFile = 'data/scene' + str(sceneNumber) + '.pickle'

    # if renderOcclusions:
    #     targetIndicesNew = []
    #     occlusionSceneFile = 'data/occlusions/occlusionScene' + str(sceneNumber) + '.pickle'
    #     with open(occlusionSceneFile, 'rb') as pfile:
    #         occlusions = pickle.load(pfile)
    #
    #     for targetidx, targetIndex in enumerate(targetIndices):
    #         if not occlusions[targetIndex][1]:
    #             print("Scene idx " + str(sceneIdx) + " at index " + str(targetIndex) + " has no proper occlusion.")
    #         else:
    #             targetIndicesNew = targetIndicesNew + [targetIndex]
    #     targetIndices = targetIndicesNew
    #
    # collisionSceneFile = 'data/collisions/collisionScene' + str(sceneNumber) + '.pickle'
    # scenes = scenes + [targetIndices]
    # with open(collisionSceneFile, 'rb') as pfile:
    #     collisions = pickle.load(pfile)
    #
    # for targetidx, targetIndex in enumerate(targetIndices):
    #     if not collisions[targetIndex][1]:
    #         print("Scene idx " + str(sceneIdx) + " at index " + str(targetIndex) + " collides everywhere.")

    scenes = scenes + [targetIndices]
    lenScenes += len(targetIndices)

groundTruthInfo = {'prefix':prefix, 'previousGTPrefix':previousGTPrefix, 'renderFromPreviousGT':renderFromPreviousGT, 'useShapeModel':useShapeModel, 'renderOcclusions':renderOcclusions, 'useOpenDR':useOpenDR, 'useBlender':useBlender, 'renderBlender':renderBlender, 'captureEnvMapFromBlender':captureEnvMapFromBlender, 'parseSceneInstantiations':parseSceneInstantiations, 'loadBlenderSceneFile':loadBlenderSceneFile, 'useCycles':useCycles, 'unpackModelsFromBlender':unpackModelsFromBlender, 'unpackSceneFromBlender':unpackSceneFromBlender, 'loadSavedSH':loadSavedSH, 'renderTeapots':renderTeapots, 'renderMugs':renderMugs, 'width':width, 'height':height, 'angle':angle, 'clip_start':clip_start, 'clip_end':clip_end, 'camDistance':camDistance, 'renderTeapotsList':renderTeapotsList, 'renderMugsList':renderMugsList, 'replaceableScenesFile':replaceableScenesFile, 'teapotsFile':teapotsFile, 'SHFilename':SHFilename, 'light_colorGT':light_colorGT, 'chDisplacement':chDisplacement, 'chDisplacementGT':chDisplacementGT, 'chScale':chScale, 'chScaleGT':chScaleGT}
with open(gtDir + 'gtInfo.pickle', 'wb') as pfile:
    pickle.dump(groundTruthInfo, pfile)

#Generate GT labels before rendering them.
if not renderFromPreviousGT:
    for scene_i, sceneIdx in enumerate(scenesToRender):

        sceneNumber, sceneFileName, instances, roomName, roomInstanceNum, targetIndicesScene, targetPositions = scene_io_utils.getSceneInformation(sceneIdx, replaceableScenesFile)
        print("Generating groundtruth for scene: " + str(sceneNumber))

        # targetIndices = scenes[scene_i]
        # if not targetIndices:
        #     continue
        targetIndices = targetIndicesScene

        sceneDicFile = 'data/scene' + str(sceneNumber) + '.pickle'

        collisionSceneFile = 'data/collisions/collisionScene' + str(sceneNumber) + '.pickle'
        with open(collisionSceneFile, 'rb') as pfile:
            collisions = pickle.load(pfile)

        # if renderOcclusions:
        #     occlusionSceneFile = 'data/occlusions/occlusionScene' + str(sceneNumber) + '.pickle'
        #     with open(occlusionSceneFile, 'rb') as pfile:
        #         occlusions = pickle.load(pfile)

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

            if renderTeapots:
                targetModels = []
                blender_teapots = []
                teapots = [line.strip() for line in open('teapots.txt')]
                selection = [teapots[i] for i in renderTeapotsList]
                scene_io_utils.loadTargetsBlendData()
                for teapotIdx, teapotName in enumerate(selection):
                    teapot = bpy.data.scenes[teapotName[0:63]].objects['teapotInstance' + str(renderTeapotsList[teapotIdx])]
                    teapot.layers[1] = True
                    teapot.layers[2] = True
                    targetModels = targetModels + [teapotIdx]
                    blender_teapots = blender_teapots + [teapot]

            if renderMugs:
                blender_mugs = []
                selectionMugs = [mugs[i] for i in renderMugsList]
                scene_io_utils.loadMugsBlendData()
                for mugIdx, mugName in enumerate(selectionMugs):
                    mug = bpy.data.scenes[mugName[0:63]].objects['mugInstance' + str(renderMugsList[mugIdx])]
                    mug.layers[1] = True
                    mug.layers[2] = True
                    mugModels = mugModels + [mug]
                    blender_mugs = blender_mugs + [mug]

            setupSceneGroundtruth(scene, width, height, clip_start, 2000, 'CUDA', 'CUDA_MULTI_2')

            treeNodes = scene.world.node_tree

            links = treeNodes.links

            cubeScene = createCubeScene(scene)
            setEnviornmentMapStrength(20, cubeScene)
            # cubeScene.render.engine = 'BLENDER_RENDER'

        unlinkedObj = None
        unlinkedCubeObj = None
        envMapFilename = None

        for targetidx, targetIndex in enumerate(targetIndices):
            targetPosition = targetPositions[np.where(targetIndex==np.array(targetIndicesScene))[0][0]]

            # if sceneIdx != currentScene or targetIndex != currentTargetIndex:
        #     targetPosition = targetPositions[np.where(targetIndex == np.array(targetIndicesScene))[0]]

            v, f_list, vc, vn, uv, haveTextures_list, textures_list = copy.deepcopy(v2), copy.deepcopy(f_list2), copy.deepcopy(vc2), copy.deepcopy(vn2)\
                , copy.deepcopy(uv2), copy.deepcopy(haveTextures_list2), copy.deepcopy(textures_list2)

            removeObjectData(len(v) - 1 - targetIndex, v, f_list, vc, vn, uv, haveTextures_list, textures_list)

            # if sceneIdx != currentScene or targetIndex != currentTargetIndex:
            parentIdx = instances[targetIndex][1]

            if useBlender:
                if unlinkedObj != None:
                    scene.objects.link(unlinkedObj)
                    cubeScene.objects.link(unlinkedCubeObj)
                unlinkedObj = scene.objects[str(targetIndex)]
                unlinkedCubeObj = cubeScene.objects['cube' + str(targetIndex)]
                scene.objects.unlink(unlinkedObj)
                cubeScene.objects.unlink(unlinkedCubeObj)


                parentSupportObj = scene.objects[str(parentIdx)]

                supportWidthMax, supportWidthMin = modelWidth(parentSupportObj.dupli_group.objects,
                                                              parentSupportObj.matrix_world)
                supportDepthMax, supportDepthMin = modelDepth(parentSupportObj.dupli_group.objects,
                                                              parentSupportObj.matrix_world)
                # supportRad = min(0.5 * np.sqrt((supportDepthMax - supportDepthMin) ** 2),0.5 * np.min((supportWidthMax - supportWidthMin) ** 2))
                supportRad = np.sqrt(0.5 * (supportDepthMax - supportDepthMin) ** 2 + 0.5 * (supportWidthMax - supportWidthMin) ** 2)

                distRange = min(supportRad, 0.3)

            sceneCollisions = {}
            collisionsFile = 'data/collisions/discreteCollisions_scene' + str(sceneNumber) + '_targetIdx' + str(targetIndex) + '.pickle'

            distInterval = 0.05
            rotationRange = 2 * np.pi
            rotationInterval = 10 * np.pi / 180

            if not parseSceneInstantiations and os.path.exists(collisionsFile):

                with open(collisionsFile, 'rb') as pfile:
                    sceneCollisions = pickle.load(pfile)

                supportRad = sceneCollisions['supportRad']
                instantiationBinsTeapot = sceneCollisions['instantiationBinsTeapot']
                distRange = sceneCollisions['distRange']
                rotationRange = sceneCollisions['rotationRange']
                rotationInterval = sceneCollisions['rotationInterval']
                distInterval = sceneCollisions['distInterval']
                instantiationBinsMug = sceneCollisions['instantiationBinsMug']
                totalBinsTeapot = sceneCollisions['totalBinsTeapot']
                totalBinsMug = sceneCollisions['totalBinsMug']
            else:
                assert(useBlender)

                print("Parsing collisions for scene " + str(sceneNumber))
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

                objDisplacementMat = computeHemisphereTransformation(chObjRotationGT, 0, chObjDistGT, np.array([0, 0, 0]))

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
                sceneCollisions = {'totalBinsTeapot': totalBinsTeapot, 'totalBinsMug': totalBinsMug,
                                   'supportRad': supportRad, 'instantiationBinsTeapot': instantiationBinsTeapot,
                                   'distRange': distRange, 'rotationRange': rotationRange,
                                   'distInterval': distInterval, 'rotationInterval': rotationInterval,
                                   'instantiationBinsMug': instantiationBinsMug}
                with open(collisionsFile, 'wb') as pfile:
                    pickle.dump(sceneCollisions, pfile)

                #Go to next target Index or scene if there are no plausible place instantiations for teapot or mug.
                if instantiationBinsTeapot.sum() < 1 or instantiationBinsMug.sum() < 1:
                    print("This scene position has not place to instantiate the objects.")
                    continue

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

                    if renderTeapots:
                        currentTeapotModel = teapot_i
                        center = center_teapots[teapot_i]

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

                    if renderMugs:
                        verticesMug, normalsMug, mugPosOffset = transformObject(v_mug, vn_mug, chScale, chObjAzMug + np.pi / 2, chObjDistMug, chObjRotationMug, targetPosition)

                        VerticesMug = [verticesMug]
                        NormalsMug = [normalsMug]
                        FacesMug = [f_list_mug]
                        VColorsMug = [vc_mug]
                        UVsMug = [uv_mug]
                        HaveTexturesMug = [haveTextures_list_mug]
                        TexturesListMug = [textures_list_mug]
                    else:
                        VerticesMug = []
                        NormalsMug = []
                        FacesMug = []
                        VColorsMug = []
                        UVsMug = []
                        HaveTexturesMug = []
                        TexturesListMug = []

                    if renderTeapots:
                        VerticesTeapot = [vGT]
                        NormalsTeapot = [vnGT]
                        FacesTeapot = [Faces]
                        VColorsTeapot = [VColors]
                        UVsTeapot = [UVs]
                        HaveTexturesTeapot = [HaveTextures]
                        TexturesListTeapot = [TexturesList]
                    else:
                        VerticesTeapot = []
                        NormalsTeapot = []
                        FacesTeapot = []
                        VColorsTeapot = []
                        UVsTeapot = []
                        HaveTexturesTeapot = []
                        TexturesListTeapot = []

                    # addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list, verticesMug, f_list_mug, vc_mug,  normalsMug, uv_mug, haveTextures_list_mug, textures_list_mug)
                    # addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list, vGT, Faces, VColors, vnGT, UVs, HaveTextures, TexturesList)

                    VColorsTeapot = [[np.ones_like(VColorsTeapot[0][0]) * chVColorsGT.reshape([1, 3])]]
                    if renderBackground:
                        vc_scene = VColorsTeapot + VColorsMug +  vc
                        vn_scene = NormalsTeapot + NormalsMug +  vn
                    else:
                        vc_scene = VColorsTeapot + VColorsMug
                        vn_scene = NormalsTeapot + NormalsMug


                    vc_scene_illum = []
                    for mesh in range(len(vc_scene)):
                        if useDirectionalLight:
                            vc_scene_illum += [computeGlobalAndDirectionalLighting(vn_scene[mesh], vc_scene[mesh], chLightAzGT, chLightElGT, chLightIntensityGT, chGlobalConstantGT)]
                        else:
                            vc_scene_illum += [computeSphericalHarmonics(vn_scene[mesh], vc_scene[mesh], light_colorGT, chComponentGT)]

                    if renderBackground:
                        v_scene = VerticesTeapot + VerticesMug +  v
                        f_scene = FacesTeapot + FacesMug +  f_list
                        UVsTeapot + UVsMug + uv
                        HaveTexturesTeapot + HaveTexturesMug + haveTextures_list
                        TexturesListTeapot + TexturesListMug + textures_list
                    else:
                        v_scene = VerticesTeapot + VerticesMug
                        f_scene = FacesTeapot + FacesMug
                        uvs_scene = UVsTeapot + UVsMug
                        haveTextures_scene = HaveTexturesTeapot + HaveTexturesMug
                        textures_list_scene = TexturesListTeapot + TexturesListMug
                    rendererGT = createRendererGT(glMode, chAzGT, chElGT, chDistGT, center, v_scene, vc_scene_illum, f_scene, vn_scene, light_colorGT, chComponentGT, chVColorsGT, targetPosition.copy(), chDisplacementGT, width, height, uvs_scene, haveTextures_scene, textures_list_scene, frustum, None)
                    rendererGT.overdraw = False
                    rendererGT.nsamples = 8
                    rendererGT.msaa = True
                    rendererGT.initGL()
                    rendererGT.initGLTexture()


                ## Blender: Unlink and link new teapot.
                if useBlender:

                    if renderTeapots:
                        # if currentScene != -1 and currentTargetIndex != -1 and currentTeapot != -1 and teapot != None:
                        if teapot.name in scene.objects:
                            scene.objects.unlink(teapot)

                            if useShapeModel:
                                deleteInstance(teapot)

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

                        # center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)
                        placeNewTarget(scene, teapot, targetPosition[:].copy())
                        teapot.layers[1] = True
                        teapot.layers[0] = True
                        original_matrix_world = teapot.matrix_world.copy()


                    if renderMugs:
                        if mug.name in scene.objects:
                            scene.objects.unlink(mug)
                        #     deleteInstance(mug)

                        mug = blender_mugs[currentTeapotModel]
                        placeNewTarget(scene, mug, targetPosition[:].copy())

                        mug.layers[1] = True
                        mug.layers[0] = True
                        original_matrix_world_mug = mug.matrix_world.copy()

                # LIGHT RANDOMIZATION
                hdrchoice = np.random.choice(len(hdrstorender))
                hdrchoice = 0

                hdrValues = hdrstorender[hdrchoice][1]
                hdrFile = hdrstorender[hdrchoice][0]

                hdridx = hdrValues[0]
                envMapCoeffs = hdrValues[1]

                envMapFilename = hdrFile

                if not useBlender:
                    envMapTexture = np.array(imageio.imread(envMapFilename))[:,:,0:3]
                numTeapotTrain = 0
                numAttempts = 0
                maxAttempts = max(int(trainSize/(lenScenes*len(renderTeapotsList))),1)
                exitInstantiationLoop = False

                while numTeapotTrain < maxAttempts and not exitInstantiationLoop:

                    numAttempts = numAttempts + 1
                    if numAttempts > 50 and numAttempts/(numTeapotTrain + 1) > 20:
                        exitInstantiationLoop = True

                    ignore = False

                    meanValIntensityOffset = np.random.uniform(-0.5,0.5)
                    meanValIntensityOffset = 0
                    chAmbientIntensityGTVals = (0.8 + meanValIntensityOffset)/(0.3*envMapCoeffs[0,0] + 0.59*envMapCoeffs[0,1]+ 0.11*envMapCoeffs[0,2])
                    chAmbientIntensityGTVals = chGlobalConstantGT.r


                    #LIGHT RANDOMIZATION
                    phiOffsetVals = np.random.uniform(0,2*np.pi, 1)
                    # phiOffsetVals = 0

                    # phiOffset[:] = 0
                    from numpy.random import choice

                    chAzGTVals = np.mod(np.random.uniform(0,np.pi, 1) - np.pi/2, 2*np.pi)

                    chElGTVals = np.random.uniform(0.05,np.pi/2, 1)

                    # chLightAzGTVals = np.random.uniform(0,2*np.pi, 1)
                    chLightAzGTVals = chLightAzGT.r
                    # chLightElGTVals = np.random.uniform(0,np.pi/2, 1)
                    chLightElGTVals = chLightElGT.r

                    chLightIntensityGTVals = chLightIntensityGT.r
                    # chLightIntensityGTVals = np.random.uniform(0,1)
                    # chGlobalConstantGTVals = np.random.uniform(0.1,0.9)
                    # chAmbientIntensityGTVals = chGlobalConstantGTVals

                    chVColorsGTVals =  np.random.uniform(0.0,1.0, [1, 3])

                    envMapCoeffsRotatedVals = np.dot(light_probes.chSphericalHarmonicsZRotation(totalOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]
                    envMapCoeffsRotatedRelVals = np.dot(light_probes.chSphericalHarmonicsZRotation(phiOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]

                    # SHAPE RANDOMIZATION
                    shapeParams = np.random.randn(latentDim)
                    shapeParams = np.zeros(latentDim)
                    # shapeParams[0:4] = np.random.randn(4)
                    chShapeParamsGTVals = shapeParams

                    ## Update renderer scene latent variables.

                    ignore = False

                    # chAmbientIntensityGT[:] = chGlobalConstantGTVals
                    chAmbientIntensityGT[:] = chGlobalConstantGT.r

                    phiOffset[:] = phiOffsetVals

                    chAzGT[:] = chAzGTVals
                    chElGT[:] = chElGTVals

                    chLightAzGT[:] = chLightAzGTVals
                    chLightElGT[:] = chLightElGTVals

                    teapotCamElGT = 0
                    teapotPosOffsetVals = 0

                    if useOpenDR:
                        cameraEye = np.linalg.inv(np.r_[rendererGT.camera.view_mtx, np.array([[0, 0, 0, 1]])])[0:3, 3]

                        vecToCenter = targetPosition - cameraEye
                        vecToCenter = vecToCenter / np.linalg.norm(vecToCenter)
                        rightCamVec = np.cross(vecToCenter, np.array([0, 0, 1]))

                    chObjAzGTVals = 0
                    chObjDistGTVals = 0
                    chObjRotationGTVals = 0
                    if renderTeapots:
                        chObjAzGTVals = np.random.uniform(0, np.pi * 2)
                        chObjDistGTVals = np.random.uniform(0, np.minimum(supportRad, 0.3))
                        chObjAzGT[:] = chObjAzGTVals

                        chVColorsGT[:] = chVColorsGTVals
                        try:
                            chShapeParamsGT[:] = shapeParams
                        except:
                            chShapeParamsGT[:] = np.random.randn(latentDim)

                        #Instantiate such that we always see the handle if only a bit!

                        chObjRotationGTVals = np.random.uniform(0,np.pi*2)

                        if not centeredObject:
                            teapotPlacement = np.random.choice(instantiationBinsTeapot.sum())

                            chObjDistGTVals = totalBinsTeapot[0].ravel()[instantiationBinsTeapot][teapotPlacement]
                            chObjRotationGTVals = totalBinsTeapot[1].ravel()[instantiationBinsTeapot][teapotPlacement]

                            chObjDistGTVals = np.random.uniform(chObjDistGTVals - distInterval/2, chObjDistGTVals + distInterval/2)
                            chObjRotationGTVals = np.mod(np.random.uniform(chObjRotationGTVals - rotationInterval/2, chObjRotationGTVals + rotationInterval/2), 2*np.pi)
                        else:
                            chObjDistGTVals = 0
                            chObjRotationGTVals = 0

                        chObjDistGT[:] = chObjDistGTVals
                        chObjRotationGT[:] = chObjRotationGTVals
                        if not centeredObject:
                            vecToTeapot = targetPosition + teapotPosOffset.r - cameraEye
                            vecToTeapot = vecToTeapot / np.linalg.norm(vecToTeapot)
                            teapotPosRight = np.sign(rightCamVec.dot(vecToTeapot))
                            angleToTeapot = np.arccos(vecToTeapot.dot(vecToCenter))

                            if np.isnan(angleToTeapot):
                                angleToMug = 0
                                ignore = True
                            chObjAzGTRel = chAzGT.r - teapotPosRight * angleToTeapot

                            teapotPosOffsetVals = teapotPosOffset.r

                    chObjDistMugVals = 0
                    chObjRotationMugVals = 0
                    chObjAzMugVals = 0
                    mugCamElGT = 0
                    mugPosOffsetVals = 0
                    chVColorsMugVals = np.random.uniform(0.0, 1, [1, 3])
                    if renderMugs:
                        chObjDistMugVals = np.random.uniform(0,np.minimum(supportRad, 0.4))
                        chObjRotationMugVals = np.random.uniform(0,np.pi*2)

                        instantiationBinsMugUpdated = instantiationBinsMug
                        if renderTeapots:

                            if not centeredObject:

                                ys = -np.cos(totalBinsMug[1]) * totalBinsMug[0]
                                xs = np.sin(totalBinsMug[1]) * totalBinsMug[0]

                                y = -np.cos(chObjRotationGTVals) * chObjDistGTVals
                                x = np.sin(chObjRotationGTVals) * chObjDistGTVals

                                instantiationBinsMugUpdated = instantiationBinsMug.copy()
                                # instantiationBinsMugUpdated[(totalBinsMug[0] > chObjDistGTVals - 0.15) & (totalBinsMug[0] < chObjDistGTVals + 0.15) & (totalBinsMug[1] > chObjRotationGTVals - 15*np.pi/180) & (totalBinsMug[1] < chObjRotationGTVals + 15*np.pi/180)] = False

                                instantiationBinsMugUpdated[np.sqrt((ys-y)**2 + (xs-x)**2) < 0.15] = False

                                if instantiationBinsMugUpdated.sum() == 0:
                                    ignore = True

                                mugPlacement = np.random.choice(instantiationBinsMugUpdated.sum())
                                chObjDistMugVals = totalBinsMug[0].ravel()[instantiationBinsMugUpdated][mugPlacement]
                                chObjRotationMugVals = totalBinsMug[1].ravel()[instantiationBinsMugUpdated][mugPlacement]
                                chObjDistMugVals = np.random.uniform(chObjDistMugVals - distInterval/2, chObjDistMugVals + distInterval/2)
                                chObjRotationMugVals = np.mod(np.random.uniform(chObjRotationMugVals - rotationInterval/2, chObjRotationMugVals + rotationInterval/2), 2*np.pi)

                            else:
                                chObjDistGTVals = 0
                                chObjRotationGTVals = 0

                        chObjDistMug[:] = chObjDistMugVals
                        chObjRotationMug[:] = chObjRotationMugVals

                        vecToMug = targetPosition + mugPosOffset.r - cameraEye
                        vecToMug = vecToMug / np.linalg.norm(vecToMug)
                        mugPosRight = np.sign(rightCamVec.dot(vecToMug))
                        angleToMug = np.arccos(vecToMug.dot(vecToCenter))

                        if np.isnan(angleToMug):
                            angleToMug = 0
                            ignore = True

                        chObjAzMugVals = np.random.uniform(chAzGT.r - mugPosRight*angleToMug - 2 * np.pi / 3, chAzGT.r - mugPosRight * angleToMug + 2 * np.pi / 3)
                        chObjAzMugRel = chAzGT.r - mugPosRight*angleToMug
                        chObjAzMug[:] = chObjAzMugVals

                        chVColorsMug[:] = chVColorsMugVals

                        vecMugToCamGT = cameraEye - (mugPosOffset + center)
                        mugCamElGT = 2 * ch.arctan(ch.norm(ch.array([0, -1, 0]) * ch.norm(vecMugToCamGT) - vecMugToCamGT * ch.norm(ch.array([0, -1, 0]))) / ch.norm(ch.array([0, -1, 0]) * ch.norm(vecMugToCamGT) + ch.norm(ch.array([0, -1, 0])) * vecMugToCamGT))

                        vecTeapotToCamGT = cameraEye - (teapotPosOffset + center)
                        teapotCamElGT = 2 * ch.arctan(ch.norm(ch.array([0, -1, 0]) * ch.norm(vecTeapotToCamGT) - vecTeapotToCamGT * ch.norm(ch.array([0, -1, 0]))) / ch.norm(ch.array([0, -1, 0]) * ch.norm(vecTeapotToCamGT) + ch.norm(
                                ch.array([0, -1, 0])) * vecTeapotToCamGT))

                        mugPosOffsetVals = mugPosOffset.r

                    if useBlender and not ignore:

                        placeCamera(scene.camera, -chAzGT.r[:].copy() * 180 / np.pi,
                                    chElGT.r[:].copy() * 180 / np.pi, chDistGT.r[0].copy(),
                                    center[:].copy() + targetPosition[:].copy())

                        if renderTeapots:
                            azimuthRot = mathutils.Matrix.Rotation(chObjAzGT.r[:].copy(), 4, 'Z')
                            teapot.matrix_world = mathutils.Matrix.Translation(original_matrix_world.to_translation() + mathutils.Vector(teapotPosOffset.r)) * azimuthRot * (mathutils.Matrix.Translation(-original_matrix_world.to_translation())) * original_matrix_world
                            setObjectDiffuseColor(teapot, chVColorsGT.r.copy())

                            if useShapeModel:
                                mesh = teapot.dupli_group.objects[0]
                                for vertex_i, vertex in enumerate(mesh.data.vertices):
                                    vertex.co = mathutils.Vector(chVerticesGT.r[vertex_i])



                        if renderMugs:
                            setObjectDiffuseColor(mug, chVColorsMug.r.copy())
                            azimuthRotMug = mathutils.Matrix.Rotation(chObjAzMug.r[:].copy() - np.pi / 2, 4, 'Z')
                            mug.matrix_world = mathutils.Matrix.Translation(original_matrix_world_mug.to_translation() + mathutils.Vector(mugPosOffset.r)) * azimuthRotMug * (mathutils.Matrix.Translation(-original_matrix_world_mug.to_translation())) * original_matrix_world_mug

                        scene.update()

                    ## Some validation checks:

                    if useOpenDR and not ignore:
                        if renderTeapots:
                            occlusion = getOcclusionFraction(rendererGT, id=teapotSceneIndex)
                            vis_occluded = np.array(rendererGT.indices_image == teapotSceneIndex+1).copy().astype(np.bool)
                            vis_occluded = np.ones_like(vis_occluded)
                            vis_im = np.array(rendererGT.image_mesh_bool([teapotSceneIndex])).copy().astype(np.bool)
                            vis_im = np.ones_like(vis_im)


                        if renderMugs:
                            occlusionMug = getOcclusionFraction(rendererGT, id=mugSceneIndex)
                            vis_occluded_mug = np.array(rendererGT.indices_image == mugSceneIndex+1).copy().astype(np.bool)
                            vis_im_mug = np.array(rendererGT.image_mesh_bool([mugSceneIndex])).copy().astype(np.bool)

                        if renderTeapots:

                            if occlusion < occlusionMin or occlusion > occlusionMax:
                                ignore = True

                            # if occlusion > 0.9 or vis_occluded.sum() < 10 or np.isnan(occlusion):
                            #Cian change:
                            if occlusion > 0.01 or vis_occluded.sum() < 10 or np.isnan(occlusion):
                                ignore = True

                            if np.sum(vis_im[:,0]) > 1 or np.sum(vis_im[0,:]) > 1 or np.sum(vis_im[:,-1]) > 1 or np.sum(vis_im[-1,:]) > 1:
                                ignore = True

                        if renderMugs and not showMug:
                            if  occlusionMug > 0.9 or vis_occluded_mug.sum() < 10 or np.isnan(occlusionMug):
                                ignore = True

                            #Check that objects are not only partly in the viewing plane of the camera:
                            if np.sum(vis_im_mug[:,0]) > 1 or np.sum(vis_im_mug[0,:]) > 1 or np.sum(vis_im_mug[:,-1]) > 1 or np.sum(vis_im_mug[-1,:]) > 1:
                                if showMug:
                                    ignore = True
                    #Don't take into account "ignore" variable for now...
                    ignore = False
                    # if not ignore:
                    #     # Ignore if camera collides with occluding object as there are inconsistencies with OpenDR and Blender.
                    #     cameraEye = np.linalg.inv(np.r_[rendererGT.camera.view_mtx, np.array([[0, 0, 0, 1]])])[0:3,3]
                    #
                    #     vDists = rendererGT.v.r[rendererGT.f[rendererGT.visibility_image[
                    #         rendererGT.visibility_image != 4294967295].ravel()].ravel()] - cameraEye
                    #
                    #     minDistToObjects = 0.2
                    #     maxDistToObjects = 0.6
                    #
                    #     #Ignore when teapot or mug is up to 10 cm to the camera eye, or too far (more than 1 meter).
                    #
                    #     if np.min(np.linalg.norm(vDists, axis=1)) <= clip_start:
                    #         ignore = True
                    #
                    #     if renderTeapots:
                    #         vDistsTeapot = rendererGT.v.r[rendererGT.f[rendererGT.visibility_image[vis_occluded].ravel()].ravel()] - cameraEye
                    #         if  np.min(np.linalg.norm(vDistsTeapot, axis=1)) <= minDistToObjects or np.min(np.linalg.norm(vDistsTeapot, axis=1)) > maxDistToObjects:
                    #             ignore = True
                    #
                    #     if renderMugs:
                    #         vDistsMug = rendererGT.v.r[rendererGT.f[rendererGT.visibility_image[vis_occluded_mug].ravel()].ravel()] - cameraEye
                    #         if np.min(np.linalg.norm(vDistsMug, axis=1)) <= minDistToObjects or np.min(np.linalg.norm(vDistsMug, axis=1)) > maxDistToObjects:
                    #             ignore = True
                    #
                    #     if useBlender:
                    #         if renderTeapots:
                    #             cubeTeapot = getCubeObj(teapot)
                    #             cubeScene.objects.link(cubeTeapot)
                    #
                    #         if renderMugs:
                    #             cubeMug = getCubeObj(mug)
                    #             cubeScene.objects.link(cubeMug)
                    #
                    #         cubeParentSupportObj = cubeScene.objects['cube'+str(parentIdx)]
                    #         cubeRoomObj = cubeScene.objects['cube' + str(roomInstanceNum)]
                    #         cubeScene.update()
                    #
                    #         if renderTeapots:
                    #             if collision.targetCubeSceneCollision(cubeTeapot, cubeScene, 'cube'+str(roomInstanceNum), cubeParentSupportObj):
                    #                 print("Teapot intersects with an object.")
                    #                 ignore = True
                    #
                    #             if not ignore and collision.instancesIntersect(mathutils.Matrix.Translation(mathutils.Vector((0, 0, +0.02))), [cubeTeapot], mathutils.Matrix.Identity(4), [cubeParentSupportObj]):
                    #                 print("Teapot interesects supporting object.")
                    #                 ignore = True
                    #
                    #             # ipdb.set_trace()
                    #             if not ignore and collision.instancesIntersect(mathutils.Matrix.Identity(4), [cubeTeapot], mathutils.Matrix.Identity(4), [cubeRoomObj]):
                    #                 print("Teapot intersects room")
                    #                 ignore = True
                    #
                    #             if not ignore and not ignore and not collision.instancesIntersect(mathutils.Matrix.Translation(mathutils.Vector((0, 0, -0.02))), [cubeTeapot], mathutils.Matrix.Identity(4), [cubeParentSupportObj]):
                    #                 print("Teapot not on table.")
                    #                 ignore = True
                    #
                    #         if renderMugs:
                    #             if not ignore and collision.targetCubeSceneCollision(cubeMug, cubeScene, 'cube' + str(roomInstanceNum), cubeParentSupportObj):
                    #                 print("Mug intersects with an object.")
                    #                 ignore = True
                    #
                    #             if not ignore and collision.instancesIntersect(mathutils.Matrix.Translation(mathutils.Vector((0, 0, +0.02))), [cubeMug], mathutils.Matrix.Identity(4), [cubeParentSupportObj]):
                    #                 print("Mug intersects supporting object")
                    #                 ignore = True
                    #
                    #             if not ignore and collision.instancesIntersect(mathutils.Matrix.Identity(4), [cubeMug], mathutils.Matrix.Identity(4), [cubeRoomObj]):
                    #                 print("Mug intersects room")
                    #                 ignore = True
                    #
                    #             if not ignore and not collision.instancesIntersect(mathutils.Matrix.Translation(mathutils.Vector((0, 0, -0.02))), [cubeMug], mathutils.Matrix.Identity(4), [cubeParentSupportObj]):
                    #                 print("Mug not on table.")
                    #                 ignore = True
                    #
                    #             if not ignore:
                    #                 print("No collision issues")
                    #
                    #         if renderTeapots:
                    #             cubeScene.objects.unlink(cubeTeapot)
                    #             deleteObject(cubeTeapot)
                    #
                    #         if renderMugs:
                    #             cubeScene.objects.unlink(cubeMug)
                    #             deleteObject(cubeMug)

                    ## Environment map update if using Cycles.

                    if not ignore and useBlender:
                        if captureEnvMapFromBlender:
                            envMapCoeffs = captureSceneEnvMap(scene, envMapTexture, roomInstanceNum,
                                                              totalOffset.r.copy(), links, treeNodes, teapot, center,
                                                              targetPosition, width, height, 2000, gtDir, train_i)

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

                    envMapCoeffsRotatedVals = envMapCoeffsRotated.r
                    envMapCoeffsRotatedRelVals = envMapCoeffsRotatedRel.r

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

                            rendererGTGray = 0.3 * rendererGT[:, :, 0].r[:] + 0.59 * rendererGT[:, :, 1].r[:] + 0.11 * rendererGT[:, :, 2].r[:]

                            meanIntensityScale = np.mean(blenderRenderGray[vis_occluded]) / np.mean(rendererGTGray[vis_occluded]).copy()
                            chAmbientIntensityGT[:] = chAmbientIntensityGT.r[:].copy() * meanIntensityScale

                            rendererGTGray = 0.3 * rendererGT[:, :, 0].r[:] + 0.59 * rendererGT[:, :, 1].r[:] + 0.11 * rendererGT[:, :, 2].r[:]
                            meanIntensityScale2 = np.mean(blenderRenderGray[vis_occluded]) / np.mean(rendererGTGray[vis_occluded]).copy()
                            chAmbientIntensityGT[:] = chAmbientIntensityGT.r[:].copy() * meanIntensityScale2

                            rendererGTGray = 0.3 * rendererGT[:, :, 0].r[:] + 0.59 * rendererGT[:, :, 1].r[:] + 0.11 * rendererGT[:, :, 2].r[:]
                            meanIntensityScale3 = np.mean(blenderRenderGray[vis_occluded]) / np.mean(rendererGTGray[vis_occluded]).copy()

                            chAmbientIntensityGT[:] = chAmbientIntensityGT.r[:].copy() * meanIntensityScale3

                            rendererGTGray = 0.3 * rendererGT[:, :, 0].r[:] + 0.59 * rendererGT[:, :, 1].r[:] + 0.11 * rendererGT[:, :, 2].r[:]
                            meanIntensityScale4 = np.mean(blenderRenderGray[vis_occluded]) / np.mean(rendererGTGray[vis_occluded]).copy()
                            chAmbientIntensityGT[:] = chAmbientIntensityGT.r[:].copy() * meanIntensityScale4

                            rendererGTGray = 0.3 * rendererGT[:, :, 0].r[:] + 0.59 * rendererGT[:, :, 1].r[:] + 0.11 * rendererGT[:, :, 2].r[:]
                            meanIntensityScale5 = np.mean(blenderRenderGray[vis_occluded]) / np.mean(rendererGTGray[vis_occluded]).copy()

                            chAmbientIntensityGT[:] = chAmbientIntensityGT.r[:].copy() * meanIntensityScale5

                            if np.mean(rendererGTGray, axis=(0, 1)) < 0.01:
                                ignore = True

                    if useOpenDR:
                        image = rendererGT.r[:].copy()
                        lin2srgb(image)

                    if not ignore:

                        if useOpenDR:
                            cv2.imwrite(gtDir + 'images_opendr/im' + str(train_i) + '.jpeg', 255 * image[:, :, [2, 1, 0]], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                            if renderTeapots:
                                np.save(gtDir + 'masks_occlusion/mask' + str(train_i) + '.npy_occluded', vis_occluded)
                                np.save(gtDir + 'masks_occlusion/mask' + str(train_i) + '.npy', vis_im)
                                coordsTeapotX = coords[1][vis_im]
                                coordsTeapotY = coords[0][vis_im]

                            if renderMugs:
                                np.save(gtDir + 'masks_occlusion/mask' + str(train_i) + '_mug_occluded.npy', vis_occluded_mug)
                                np.save(gtDir + 'masks_occlusion/mask' + str(train_i) + '_mug.npy', vis_im_mug)

                                coordsMugX = coords[1][vis_im_mug]
                                coordsMugY = coords[0][vis_im_mug]

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

                        trainObjDistGT = chObjDistGTVals
                        trainObjRotationGT = chObjRotationGTVals
                        trainObjDistMug = chObjDistMugVals
                        trainObjRotationMug = chObjRotationMugVals
                        trainObjAzMug = chObjAzMugVals
                        trainVColorsMug = chVColorsMugVals

                        trainMugElRel = mugCamElGT
                        trainTeapotElRel = teapotCamElGT

                        trainMugPosOffset = mugPosOffsetVals
                        trainTeapotPosOffset = teapotPosOffsetVals

                        trainBBMug = np.array([coordsMugX.min(),coordsMugX.max(),coordsMugY.min(),coordsMugY.max(),])
                        trainBBTeapot = np.array([coordsTeapotX.min(),coordsTeapotX.max(),coordsTeapotY.min(),coordsTeapotY.max(),])

                        trainTeapotPresent = renderTeapots
                        trainMugPresent = renderMugs

                        gtDataset.resize(gtDataset.shape[0] + 1, axis=0)

                        gtDataset[-1] = np.array([(trainIds, trainAzsGT, trainObjAzsGT, trainElevsGT,
                                                           trainLightAzsGT, trainLightElevsGT,
                                                           trainLightIntensitiesGT, trainVColorGT, trainScenes,
                                                           trainTeapotIds, trainEnvMaps, trainOcclusions,
                                                           trainTargetIndices, trainLightCoefficientsGT,
                                                           trainLightCoefficientsGTRel, trainAmbientIntensityGT,
                                                           phiOffsetVals, trainShapeModelCoeffsGT,
                                                           trainObjDistGT,
                                                           trainObjRotationGT,
                                                           trainObjDistMug,
                                                           trainObjRotationMug,
                                                           trainObjAzMug,
                                                           trainVColorsMug,
                                                           trainTeapotElRel,
                                                           trainMugElRel,
                                                           trainMugPosOffset,
                                                           trainTeapotPosOffset,
                                                           trainBBMug,
                                                          trainBBTeapot,
                                                          trainTeapotPresent,
                                                          trainMugPresent
                                                         )], dtype=gtDtype)

                        gtDataFile.flush()
                        train_i = train_i + 1
                        numTeapotTrain = numTeapotTrain + 1

                        if np.mod(train_i, 100) == 0:
                            print("Generated " + str(train_i) + " GT instances.")
                            print("Generating groundtruth. Iteration of " + str(range(int(trainSize/(lenScenes*len(hdrstorender)*len(renderTeapotsList))))) + " teapots")


if renderFromPreviousGT:

    groundTruthFilename = 'groundtruth/' + previousGTPrefix + '/groundTruth.h5'
    gtDataFileToRender = h5py.File(groundTruthFilename, 'r')
    groundTruthToRender = gtDataFileToRender[previousGTPrefix]

else:
    exit()

pdb.set_trace()

currentScene = -1
currentTeapot = -1
currentTargetIndex = -1

teapot = None

if renderFromPreviousGT:
    rangeGT = np.arange(len(groundTruthToRender))
else:
    rangeGT = np.arange(len(groundTruthToRender))

teapot_i = 0

# experimentPrefix = 'train4_occlusion_shapemodel_10k'
# experimentDir = 'experiments/' + experimentPrefix + '/'
# subsetToRender = np.load(experimentDir + 'test.npy')[np.arange(0,100)]


subsetToRender = np.arange(len(rangeGT))

# subsetToRender = np.arange(len(rangeGT))

if useShapeModel:
    teapot_i = -1
    # addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list,  v_teapots[currentTeapotModel][0], f_list_teapots[currentTeapotModel][0], vc_teapots[currentTeapotModel][0], vn_teapots[currentTeapotModel][0], uv_teapots[currentTeapotModel][0], haveTextures_list_teapots[currentTeapotModel][0], textures_list_teapots[currentTeapotModel][0])

idxToRender = rangeGT[:][subsetToRender]
sceneIdxsToRender = groundTruthToRender['trainScenes'][idxToRender]
sortedSceneIndices = np.argsort(sceneIdxsToRender)

sortedSceneAndTargetIdxs = np.arange(len(idxToRender))
for sceneIdx in np.unique(sceneIdxsToRender[sortedSceneIndices]):
    sortedTargetIdxs = np.argsort(groundTruthToRender['trainTargetIndices'][idxToRender][np.where(sceneIdxsToRender[sortedSceneIndices]==sceneIdx)])
    sortedSceneAndTargetIdx = idxToRender[sortedSceneIndices][np.where(sceneIdxsToRender[sortedSceneIndices]==sceneIdx)][sortedTargetIdxs]
    sortedSceneAndTargetIdxs[np.where(sceneIdxsToRender[sortedSceneIndices]==sceneIdx)] = sortedSceneAndTargetIdx

if renderFromPreviousGT:
    for gtIdx in sortedSceneAndTargetIdxs:

        if not replaceNewGroundtruth:
            if gtIdx in gtDataset['trainIds']:
                continue

        sceneNumber = groundTruthToRender['trainScenes'][gtIdx]

        # train_i = np.where(idxToRender==gtIdx)[0][0]
        train_i = gtIdx

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

            # Configure scene
            if useBlender:

                if renderTeapots:
                    targetModels = []
                    blender_teapots = []
                    teapots = [line.strip() for line in open('teapots.txt')]
                    selection = [teapots[i] for i in renderTeapotsList]
                    scene_io_utils.loadTargetsBlendData()
                    for teapotIdx, teapotName in enumerate(selection):
                        teapot = bpy.data.scenes[teapotName[0:63]].objects['teapotInstance' + str(renderTeapotsList[teapotIdx])]
                        teapot.layers[1] = True
                        teapot.layers[2] = True
                        targetModels = targetModels + [teapotIdx]
                        blender_teapots = blender_teapots + [teapot]

                if renderMugs:
                    blender_mugs = []
                    selectionMugs = [mugs[i] for i in renderMugsList]
                    scene_io_utils.loadMugsBlendData()
                    for mugIdx, mugName in enumerate(selectionMugs):
                        mug = bpy.data.scenes[mugName[0:63]].objects['mugInstance' + str(renderMugsList[mugIdx])]
                        mug.layers[1] = True
                        mug.layers[2] = True
                        mugModels = mugModels + [mug]
                        blender_mugs = blender_mugs + [mug]

                setupSceneGroundtruth(scene, width, height, clip_start, 5000, 'CUDA', 'CUDA_MULTI_2')

                treeNodes = scene.world.node_tree

                links = treeNodes.links


        unlinkedObj = None
        envMapFilename = None

        targetIndex = groundTruthToRender['trainTargetIndices'][gtIdx]

        if sceneIdx != currentScene and not renderMugs and sceneIdx == 44 and sceneNumber == 114:
            if len(v) > 1:
                removeObjectData(1, v, f_list, vc, vn, uv, haveTextures_list, textures_list)

        if sceneIdx != currentScene or targetIndex != currentTargetIndex:
            targetPosition = targetPositions[np.where(targetIndex==np.array(targetIndicesScene))[0][0]]
            import copy
            v, f_list, vc, vn, uv, haveTextures_list, textures_list = copy.deepcopy(v2), copy.deepcopy(f_list2), copy.deepcopy(vc2), copy.deepcopy(vn2), copy.deepcopy(uv2), copy.deepcopy(haveTextures_list2),  copy.deepcopy(textures_list2)

            removeObjectData(len(v) -1 - targetIndex, v, f_list, vc, vn, uv, haveTextures_list, textures_list)


        if sceneIdx != currentScene or targetIndex != currentTargetIndex:
            if useBlender:
                if unlinkedObj != None:
                    scene.objects.link(unlinkedObj)

                try:
                    unlinkedObj = scene.objects[str(targetIndex)]
                    scene.objects.unlink(unlinkedObj)
                except:
                    pass

                if sceneIdx == 44 and sceneNumber == 114:
                    try:
                        scene.objects.unlink(scene.objects['fc035f7d732166b97b6fd5468f603b31_cleaned'])
                    except:
                        pass

        teapot_i = groundTruthToRender['trainTeapotIds'][gtIdx]

        if useShapeModel:
            teapot_i = -1

        if sceneIdx != currentScene or targetIndex != currentTargetIndex or teapot_i != currentTeapot:

            ##Destroy and create renderer
            if useOpenDR:
                rendererGT.makeCurrentContext()
                rendererGT.clear()
                contextdata.cleanupContext(contextdata.getContext())
                if glMode == 'glfw':
                    glfw.destroy_window(rendererGT.win)
                del rendererGT

                if generateTriplets:
                    renderer.makeCurrentContext()
                    renderer.clear()
                    contextdata.cleanupContext(contextdata.getContext())
                    if glMode == 'glfw':
                        glfw.destroy_window(renderer.win)

                    del renderer

                if renderTeapots:
                    currentTeapotModel = teapot_i
                    center = center_teapots[teapot_i]

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

                if renderMugs:
                    verticesMug, normalsMug, mugPosOffset = transformObject(v_mug, vn_mug, chScale, chObjAzMug + np.pi / 2, chObjDistMug,
                                                                            chObjRotationMug, targetPosition)

                    VerticesMug = [verticesMug]
                    NormalsMug = [normalsMug]
                    FacesMug = [f_list_mug]
                    VColorsMug = [vc_mug]
                    UVsMug = [uv_mug]
                    HaveTexturesMug = [haveTextures_list_mug]
                    TexturesListMug = [textures_list_mug]
                else:
                    VerticesMug = []
                    NormalsMug = []
                    FacesMug = []
                    VColorsMug = []
                    UVsMug = []
                    HaveTexturesMug = []
                    TexturesListMug = []

                if renderTeapots:
                    VerticesTeapot = [vGT]
                    NormalsTeapot = [vnGT]
                    FacesTeapot = [Faces]
                    VColorsTeapot = [VColors]
                    UVsTeapot = [UVs]
                    HaveTexturesTeapot = [HaveTextures]
                    TexturesListTeapot = [TexturesList]
                else:
                    VerticesTeapot = []
                    NormalsTeapot = []
                    FacesTeapot = []
                    VColorsTeapot = []
                    UVsTeapot = []
                    HaveTexturesTeapot = []
                    TexturesListTeapot = []

                # addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list, verticesMug, f_list_mug, vc_mug,  normalsMug, uv_mug, haveTextures_list_mug, textures_list_mug)
                # addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list, vGT, Faces, VColors, vnGT, UVs, HaveTextures, TexturesList)

                rendererGT = createRendererGT(glMode, chAzGT, chElGT, chDistGT, center, VerticesTeapot + VerticesMug + v,
                                              VColorsTeapot + VColorsMug + vc, FacesTeapot + FacesMug + f_list, NormalsTeapot + NormalsMug + vn,
                                              light_colorGT, chComponentGT, chVColorsGT, targetPosition.copy(), chDisplacementGT, width, height,
                                              UVsTeapot + UVsMug + uv, HaveTexturesTeapot + HaveTexturesMug + haveTextures_list,
                                              TexturesListTeapot + TexturesListMug + textures_list, frustum, None)
                rendererGT.overdraw = False
                rendererGT.nsamples = 8
                rendererGT.msaa = True
                rendererGT.initGL()
                rendererGT.initGLTexture()

                rendererGT.makeCurrentContext()

                if generateTriplets:
                    if useShapeModel:
                        center = smCenter
                        UVs = smUVs
                        v = smVertices
                        vn = smNormals
                        Faces = smFaces
                        VColors = smVColors
                        UVs = smUVs
                        HaveTextures = smHaveTextures
                        TexturesList = smTexturesList
                    else:
                        v, vn = v_teapots[currentTeapotModel][0], vn_teapots[currentTeapotModel][0]
                        Faces = f_list_teapots[currentTeapotModel][0]
                        VColors = vc_teapots[currentTeapotModel][0]
                        UVs = uv_teapots[currentTeapotModel][0]
                        HaveTextures = haveTextures_list_teapots[currentTeapotModel][0]
                        TexturesList = textures_list_teapots[currentTeapotModel][0]

                    v, vn, teapotPosOffset = transformObject(v, vn, chScale, chObjAz, chObjDist, chObjRotation, np.array([0, 0, 0]))

                    if renderMugs:
                        verticesMug, normalsMug, mugPosOffset = transformObject(v_mug, vn_mug, chScale, chObjAzMug + np.pi / 2, chObjDistMug,
                                                                                chObjRotationMug, np.array([0, 0, 0]))

                        VerticesB = [v] + [verticesMug]
                        NormalsB = [vn] + [normalsMug]
                        FacesB = [Faces] + [f_list_mug]
                        VColorsB = [VColors] + [vc_mug]
                        UVsB = [UVs] + [uv_mug]
                        HaveTexturesB = [HaveTextures] + [haveTextures_list_mug]
                        TexturesListB = [TexturesList] + [textures_list_mug]

                        renderer = createRendererTarget(glMode, chAz, chEl, chDist, center, VerticesB, VColorsB, FacesB, NormalsB, light_color,
                                                        chComponent, chVColors, np.array([0, 0, 0]), chDisplacement, width, height, UVsB,
                                                        HaveTexturesB, TexturesListB, frustum, None)
                    else:

                        renderer = createRendererTarget(glMode, chAz, chEl, chDist, smCenter, [v], [smVColors], [smFaces], [vn], light_color,
                                                        chComponent, chVColors, 0, chDisplacement, width, height, [smUVs], [smHaveTextures],
                                                        [smTexturesList], frustum, None)
                    rendererGT.makeCurrentContext()


            ## Blender: Unlink and link new teapot.
            if useBlender:
                if renderTeapots:
                    # if currentScene != -1 and currentTargetIndex != -1 and currentTeapot != -1 and teapot != None:
                    if teapot.name in scene.objects:
                        scene.objects.unlink(teapot)

                        if useShapeModel:
                            deleteInstance(teapot)

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

                    # center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)
                    placeNewTarget(scene, teapot, targetPosition[:].copy())
                    teapot.layers[1] = True
                    teapot.layers[0] = True
                    original_matrix_world = teapot.matrix_world.copy()

                if renderMugs:
                    if mug.name in scene.objects:
                        scene.objects.unlink(mug)
                    # deleteInstance(mug)

                    mug = blender_mugs[currentTeapotModel]

                    placeNewTarget(scene, mug, targetPosition[:].copy())

                    mug.layers[1] = True
                    mug.layers[0] = True
                    original_matrix_world_mug = mug.matrix_world.copy()


        hdridx = groundTruthToRender['trainEnvMaps'][gtIdx]

        envMapFilename = ""
        for hdrFile, hdrValues in hdritems:
            if hdridx == hdrValues[0]:

                envMapCoeffs[:] = hdrValues[1]
                envMapFilename = hdrFile

                # updateEnviornmentMap(envMapFilename, scene)
                envMapTexture = np.array(imageio.imread(envMapFilename))[:,:,0:3]
                break

        assert(envMapFilename != "")
        # if envMapFilename == "":
        #     ipdb.set_trace()

        print("Render " + str(gtIdx) + "of " + str(len(groundTruthToRender)))
        ignore = False
        # chAmbientIntensityGT[:] = groundTruthToRender['trainAmbientIntensityGT'][gtIdx]
        chAmbientIntensityGT[:] = groundTruthToRender['trainAmbientIntensityGT'][gtIdx]

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
        try:
            chObjDistGT[:] = groundTruthToRender['trainObjDistGT'][gtIdx]
        except:
            chObjDistGT[:] = 0
        try:
            chObjRotationGT[:] = groundTruthToRender['trainObjRotationGT'][gtIdx]
        except:
            chObjRotationGT[:] = 0

        if renderMugs:
            chObjDistMug[:] = groundTruthToRender['trainObjDistMug'][gtIdx]
            chObjRotationMug[:] = groundTruthToRender['trainObjRotationMug'][gtIdx]
            chObjAzMug[:] = groundTruthToRender['trainObjAzMug'][gtIdx]
            chVColorsMug[:] = groundTruthToRender['trainVColorsMug'][gtIdx]

        if captureEnvMapFromBlender and not ignore and useBlender:

            envMapCoeffs = captureSceneEnvMap(scene, envMapTexture, roomInstanceNum, totalOffset.r.copy(), links, treeNodes, teapot, center, targetPosition, width, height, 2000, gtDir, train_i)

        if useBlender and captureEnvMapFromBlender:
            envMapCoeffsRotated[:] = np.dot(light_probes.chSphericalHarmonicsZRotation(0), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]
            envMapCoeffsRotatedRel[:] = np.dot(light_probes.chSphericalHarmonicsZRotation(-chObjAzGT.r), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]
        else:
            envMapCoeffsRotated[:] = np.dot(light_probes.chSphericalHarmonicsZRotation(totalOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]
            envMapCoeffsRotatedRel[:] = np.dot(light_probes.chSphericalHarmonicsZRotation(phiOffset), envMapCoeffs[[0,3,2,1,4,5,6,7,8]])[[0,3,2,1,4,5,6,7,8]]

        envMapCoeffsRotatedVals = envMapCoeffsRotated.r
        envMapCoeffsRotatedRelVals = envMapCoeffsRotatedRel.r

        if useOpenDR and not ignore:
            if renderTeapots:
                occlusion = getOcclusionFraction(rendererGT, id=teapotSceneIndex)
                vis_occluded = np.array(rendererGT.indices_image == teapotSceneIndex + 1).copy().astype(np.bool)
                vis_im = np.array(rendererGT.image_mesh_bool([teapotSceneIndex])).copy().astype(np.bool)

        if renderBlender and useBlender and not ignore:

            placeCamera(scene.camera, -chAzGT.r[:].copy() * 180 / np.pi,
                        chElGT.r[:].copy() * 180 / np.pi, chDistGT.r[0].copy(),
                        center[:].copy() + targetPosition[:].copy())

            if renderTeapots:
                azimuthRot = mathutils.Matrix.Rotation(chObjAzGT.r[:].copy(), 4, 'Z')
                teapot.matrix_world = mathutils.Matrix.Translation(
                    original_matrix_world.to_translation() + mathutils.Vector(teapotPosOffset.r)) * azimuthRot * (
                                      mathutils.Matrix.Translation(-original_matrix_world.to_translation())) * original_matrix_world
                setObjectDiffuseColor(teapot, chVColorsGT.r.copy())

                if useShapeModel:
                    mesh = teapot.dupli_group.objects[0]
                    for vertex_i, vertex in enumerate(mesh.data.vertices):
                        vertex.co = mathutils.Vector(chVerticesGT.r[vertex_i])

            if renderMugs:
                setObjectDiffuseColor(mug, chVColorsMug.r.copy())
                azimuthRotMug = mathutils.Matrix.Rotation(chObjAzMug.r[:].copy() - np.pi / 2, 4, 'Z')
                mug.matrix_world = mathutils.Matrix.Translation(
                    original_matrix_world_mug.to_translation() + mathutils.Vector(mugPosOffset.r)) * azimuthRotMug * (
                                   mathutils.Matrix.Translation(-original_matrix_world_mug.to_translation())) * original_matrix_world_mug

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

        if not ignore:

            if generateTriplets:
                lightCoeffsRel = envMapCoeffsRotatedRel.r[None, :].copy().squeeze()
                lightCoeffsRel = 0.3 * lightCoeffsRel[:, 0] + 0.59 * lightCoeffsRel[:, 1] + 0.11 * lightCoeffsRel[:, 2]
                chLightSHCoeffs[:] = lightCoeffsRel * chAmbientIntensityGT.r
                chObjAz[:] = 0
                chAz[:] = chAzRelGT.r + np.random.choice([-1,1]) * np.random.uniform(0,10) * np.pi / 180
                chEl[:] = chElGT.r
                chVColors[:] = chVColorsGT.r
                chShapeParams[:] =  chShapeParamsGT.r

                renderer.makeCurrentContext()
                if useOpenDR:
                    cv2.imwrite(gtDir + 'triplets1/im' + str(train_i) + '.jpeg' , 255*lin2srgb(renderer.r.copy())[:,:,[2,1,0]], [int(cv2.IMWRITE_JPEG_QUALITY), 100])


                chAz[:] = chAzRelGT.r + np.random.choice([-1,1]) * np.random.uniform(10,40) * np.pi / 180

                if useOpenDR:
                    cv2.imwrite(gtDir + 'triplets2/im' + str(train_i) + '.jpeg', 255 * lin2srgb(renderer.r.copy())[:, :, [2, 1, 0]], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            rendererGT.makeCurrentContext()
            if useBlender and renderBlender:
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

            trainObjDistGT = chObjDistGT.r
            trainObjRotationGT = chObjRotationGT.r
            trainObjDistMug = chObjDistMug.r
            trainObjRotationMug = chObjRotationMug.r
            trainObjAzMug = chObjAzMug.r
            trainVColorsMug = chVColorsMug.r

            try:
                trainMugElRel = groundTruthToRender['trainMugElRel'][gtIdx]
            except:
                trainMugElRel = -1

            try:
                trainTeapotElRel = groundTruthToRender['trainTeapotElRel'][gtIdx]
            except:
                trainTeapotElRel = -1

            try:
                trainMugPosOffset = groundTruthToRender['trainMugPosOffset'][gtIdx]
            except:
                trainMugPosOffset = np.array([0,0,0])

            try:
                trainTeapotPosOffset = groundTruthToRender['trainTeapotPosOffset'][gtIdx]
            except:
                trainTeapotPosOffset = np.array([0,0,0])

            try:
                trainBBMug = groundTruthToRender['trainBBMug'][gtIdx]
            except:
                trainBBMug = np.array([0,0,0,0])

            try:
                trainBBTeapot = groundTruthToRender['trainBBTeapot'][gtIdx]
            except:
                trainBBTeapot = np.array([0,0,0,0])

            trainTeapotPresent = renderTeapots

            trainMugPresent = renderMugs

            gtDataset.resize(gtDataset.shape[0] + 1, axis=0)

            gtDataset[-1] = np.array([(trainIds, trainAzsGT, trainObjAzsGT, trainElevsGT,
                                               trainLightAzsGT, trainLightElevsGT,
                                               trainLightIntensitiesGT, trainVColorGT, trainScenes,
                                               trainTeapotIds, trainEnvMaps, trainOcclusions,
                                               trainTargetIndices, trainLightCoefficientsGT,
                                               trainLightCoefficientsGTRel, trainAmbientIntensityGT,
                                               trainEnvMapPhiOffsets, trainShapeModelCoeffsGT,
                                               trainObjDistGT,
                                               trainObjRotationGT,
                                               trainObjDistMug,
                                               trainObjRotationMug,
                                               trainObjAzMug,
                                               trainVColorsMug,
                                               trainTeapotElRel,
                                               trainMugElRel,
                                               trainMugPosOffset,
                                               trainTeapotPosOffset,
                                               trainBBMug,
                                               trainBBTeapot,
                                               trainTeapotPresent,
                                               trainMugPresent
                                               )], dtype=gtDtype)

        train_i = train_i + 1

        currentScene  = sceneIdx
        currentTargetIndex = targetIndex
        currentTeapot = teapot_i
        gtDataFile.flush()

    # np.savetxt(gtDir + 'data.txt',np.array(np.hstack([trainIds[:,None], trainAzsGT[:,None], trainObjAzsGT[:,None], trainElevsGT[:,None], phiOffsets[:,None], trainOcclusions[:,None]])), fmt="%g")
    gtDataFile.close()

    gtDataFileToRender.close()