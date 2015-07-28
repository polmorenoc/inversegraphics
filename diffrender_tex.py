__author__ = 'pol'

import matplotlib
matplotlib.use('TkAgg')

import bpy
import sceneimport
import mathutils
from math import radians
import timeit
import time
import opendr
import chumpy as ch
from opendr.renderer import ColoredRenderer
from opendr.renderer import TexturedRenderer
from opendr.lighting import LambertianPointLight
from opendr.lighting import SphericalHarmonics
import geometry
from opendr.camera import ProjectPoints
import numpy as np
import cv2
from sklearn.preprocessing import normalize
from utils import *
import timeit

import matplotlib.pyplot as plt

plt.ion()

# pylab.pause(0.0001)
#
# pylab.plot([1,2,3])
# pylab.figure(1)
# pylab.show(block=False)

rn = TexturedRenderer()

rnmod = TexturedRenderer()

renderTeapotsList = [2]

[targetScenes, targetModels, transformations] = sceneimport.loadTargetModels(renderTeapotsList)
teapot = targetModels[0]
teapot.layers[1] = True
teapot.layers[2] = True

width, height = (200, 200)

angle = 60 * 180 / numpy.pi
clip_start = 0.05
clip_end = 10
camDistance = 0.4
azimuth = 95
elevation = 65

cam = bpy.data.cameras.new("MainCamera")
camera = bpy.data.objects.new("MainCamera", cam)
world = bpy.data.worlds.new("MainWorld")

replaceableScenesFile = '../databaseFull/fields/scene_replaceables.txt'
sceneLines = [line.strip() for line in open(replaceableScenesFile)]
sceneLineNums = numpy.arange(len(sceneLines))
sceneNum =  sceneLineNums[0]
sceneLine = sceneLines[sceneNum]
sceneParts = sceneLine.split(' ')
sceneFile = sceneParts[0]
sceneNumber = int(re.search('.+?scene([0-9]+)\.txt', sceneFile, re.IGNORECASE).groups()[0])
sceneFileName = re.search('.+?(scene[0-9]+\.txt)', sceneFile, re.IGNORECASE).groups()[0]
targetIndex = int(sceneParts[1])
instances = sceneimport.loadScene('../databaseFull/scenes/' + sceneFileName)
targetParentPosition = instances[targetIndex][2]
targetParentIndex = instances[targetIndex][1]
teapot.layers[1] = True
teapot.layers[2] = True
teapot.matrix_world = mathutils.Matrix.Translation(targetParentPosition)
center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)
original_matrix_world = teapot.matrix_world.copy()

sceneDicFile = 'sceneDic.pickle'
sceneDic = {}
if False:

    [blenderScenes, modelInstances] = sceneimport.importBlenderScenes(instances, True, targetIndex)

    targetParentInstance = modelInstances[targetParentIndex]
    targetParentInstance.layers[2] = True

    roomName = ''
    for model in modelInstances:
        reg = re.compile('(room[0-9]+)')
        res = reg.match(model.name)
        if res:
            roomName = res.groups()[0]

    scene = sceneimport.composeScene(modelInstances, targetIndex)

    roomInstance = scene.objects[roomName]
    roomInstance.layers[2] = True
    targetParentInstance.layers[2] = True

    setupScene(scene, targetIndex,roomName, world, camDistance, camera, width, height, 16, False, False)
    scene.objects.link(teapot)

    azimuthRot = mathutils.Matrix.Rotation(radians(-azimuth), 4, 'Z')
    elevationRot = mathutils.Matrix.Rotation(radians(-elevation), 4, 'X')
    originalLoc = mathutils.Vector((0,-camDistance, 0))
    location = center + azimuthRot * elevationRot * originalLoc
    camera = scene.camera
    camera.location = location
    scene.update()
    look_at(camera, center)
    scene.update()
    scene.render.filepath = 'opendr_blender.png'
    bpy.ops.render.render( write_still=True )

    # ipdb.set_trace()
    # v,f_list, vc, vn, uv, haveTextures_list, textures_list = sceneimport.unpackObjects(teapot)
    v = []
    f_list = []
    vc  = []
    vn  = []
    uv  = []
    haveTextures_list  = []
    textures_list  = []
    print("Unpacking blender data for OpenDR.")
    for modelInstance in scene.objects:
        if modelInstance.dupli_group != None:
            vmod,f_listmod, vcmod, vnmod, uvmod, haveTextures_listmod, textures_listmod = sceneimport.unpackObjects(modelInstance)
            # gray = np.dot(np.array([0.3, 0.59, 0.11]), vcmod[0].T).T
            # sat = 0.5
            #
            # vcmod[0][:,0] = vcmod[0][:,0] * sat + (1-sat) * gray
            # vcmod[0][:,1] = vcmod[0][:,1] * sat + (1-sat) * gray
            # vcmod[0][:,2] = vcmod[0][:,2] * sat + (1-sat) * gray
            v = v + vmod
            f_list = f_list + f_listmod
            vc = vc + vcmod
            vn = vn + vnmod
            uv = uv + uvmod
            haveTextures_list  = haveTextures_list + haveTextures_listmod
            textures_list  = textures_list + textures_listmod

    #Serialize
    sceneDic = {'v':v,'f_list':f_list,'vc':vc,'uv':uv,'haveTextures_list':haveTextures_list,'vn':vn,'textures_list': textures_list}
    with open(sceneDicFile, 'wb') as pfile:
        pickle.dump(sceneDic, pfile)

    print("Serialized scene!")
else:
    with open(sceneDicFile, 'rb') as pfile:
        sceneDic = pickle.load(pfile)
        v = sceneDic['v']
        f_list = sceneDic['f_list']
        vc = sceneDic['vc']
        uv = sceneDic['uv']
        haveTextures_list = sceneDic['haveTextures_list']
        vn = sceneDic['vn']
        textures_list = sceneDic['textures_list']

    print("Loaded serialized scene!")

vmod,fmod_list, vcmod, vnmod, uvmod, haveTexturesmod_list, texturesmod_list = sceneimport.unpackObjects(teapot)

rangeMeshes = range(len(v))
vch = [ch.array(v[mesh]) for mesh in rangeMeshes]
vnch = [ch.array(vn[mesh]) for mesh in rangeMeshes]
vcch = [ch.array(vc[mesh]) for mesh in rangeMeshes]

mesh = 0
vchmod = [ch.array(vmod[mesh])]
vnchmod = [ch.array(vnmod[mesh])]
vcchmod = [ch.array(vcmod[mesh])]

azScale = radians(30)
elScale = radians(15)
componentScale = 4
distScale = 0.2

chAzScaled = ch.Ch([radians(azimuth/azScale)])
chElScaled = ch.Ch([radians(elevation/elScale)])
chDistScaled = ch.Ch([camDistance/distScale])
chComponentScaled = ch.Ch([1, 0.25, 0.25, 0.,0.,0.,0.,0.,0.])


chAz = chAzScaled*azScale
chEl = chElScaled*elScale
chDist = chDistScaled*distScale
chComponent = chComponentScaled*componentScale

chDistMat = geometry.Translate(x=ch.Ch(0), y=-chDist, z=ch.Ch(0))
chToObjectTranslate = geometry.Translate(x=center.x, y=center.y, z=center.z)

chRotAzMat = geometry.RotateZ(a=-chAz)
chRotElMat = geometry.RotateX(a=-chEl)
chCamModelWorld = ch.dot(chToObjectTranslate, ch.dot(chRotAzMat, ch.dot(chRotElMat,chDistMat)))

chInvCam = ch.inv(ch.dot(chCamModelWorld, np.array(mathutils.Matrix.Rotation(radians(270), 4, 'X'))))

chRod = opendr.geometry.Rodrigues(rt=chInvCam[0:3,0:3]).reshape(3)
chTranslation = chInvCam[0:3,3]


translation, rotation = (chTranslation, chRod)


light_color=ch.ones(3)


A_list = [SphericalHarmonics(vn=vnch[mesh],
                       components=chComponent,
                       light_color=light_color) for mesh in rangeMeshes]
vc_list = [A_list[mesh]*vcch[mesh] for mesh in rangeMeshes]


mesh =0
A_mod = [SphericalHarmonics(vn=vnchmod[mesh],
                       components=chComponent,
                       light_color=light_color)]
vcmod_list = [A_mod[mesh]*vcchmod[mesh]]


v = ch.vstack(vch)
f = []
for mesh in f_list:
    lenMeshes = len(f)
    for polygons in mesh:
        f = f + [polygons + lenMeshes]
fstack = np.vstack(f)
vn = ch.vstack(vnch)
vc = ch.vstack(vc_list)
ft = np.vstack(uv)

texturesch = []
for texture_list in textures_list:
    for texture in texture_list:
        if texture != None:
            texturesch = texturesch + [ch.array(texture)]
texture_stack = ch.concatenate([tex.ravel() for tex in texturesch])

vmod = ch.vstack(vchmod)
fmod = []
for mesh in fmod_list:
    lenMeshes = len(fmod)
    for polygons in mesh:
        fmod = fmod + [polygons + lenMeshes]
fstackmod = np.vstack(fmod)
vnmod = ch.vstack(vnchmod)
vcmod = ch.vstack(vcmod_list)
ftmod = np.vstack(uvmod)

textureschmod = []
for texturemod_list in texturesmod_list:
    for texture in texturemod_list:
        if texture != None:
            textureschmod = textureschmod + [ch.array(texture)]
if textureschmod != []:
    texturemod_stack = ch.concatenate([tex.ravel() for tex in textureschmod])
else:
    texturemod_stack = ch.Ch([])



from opendr.camera import ProjectPoints
rn.camera = ProjectPoints(v=v, rt=rotation, t=translation, f= 1.12*ch.array([width,width]), c=ch.array([width,height])/2.0, k=ch.zeros(5))
rn.camera.openglMat = np.array(mathutils.Matrix.Rotation(radians(180), 4, 'X'))
rn.frustum = {'near': clip_start, 'far': clip_end, 'width': width, 'height': height}

#Ugly initializations, think of a better way that still works well with Chumpy!
rn.set(v=v, f=fstack, vn=vn, vc=vc, ft=ft, texture_stack=texture_stack, v_list=vch, f_list=f_list, vc_list=vc_list, ft_list=uv, textures_list=textures_list, haveUVs_list=haveTextures_list, bgcolor=ch.ones(3), overdraw=True)

rnmod.camera = ProjectPoints(v=vmod, rt=rotation, t=translation, f= 1.12*ch.array([width,width]), c=ch.array([width,height])/2.0, k=ch.zeros(5))
rnmod.camera.openglMat = np.array(mathutils.Matrix.Rotation(radians(180), 4, 'X'))
rnmod.frustum = {'near': clip_start, 'far': clip_end, 'width': width, 'height': height}
rnmod.set(v=vmod, f=fstackmod, vn=vnmod, vc=vcmod, ft=ftmod, texture_stack=texturemod_stack, v_list=vchmod, f_list=fmod_list, vc_list=vcmod_list, ft_list=uvmod, textures_list=texturesmod_list, haveUVs_list=haveTexturesmod_list, bgcolor=ch.ones(3), overdraw=True)

chAzOld = chAz[0].r
chElOld = chEl[0].r

# Show it
print("Beginning render.")
t = time.time()
rn.r
elapsed_time = time.time() - t
print("Ended render in  " + str(elapsed_time))

figuregt = plt.figure(1)
plt.imshow(rn.r)
ipdb.set_trace()

plt.imsave('opendr_opengl_gt.png', rn.r)
image = np.copy(np.array(rn.r)).astype(np.float64)

chAzScaled[0] = chAzScaled.r - radians(5)/azScale
chElScaled[0] = chElScaled.r - radians(5)/elScale
# chComponentScaled[0] = chComponentScaled[0].r - 1/componentScale
# A.components[0] = A.components[0].r - 0.3
# A.components[1] = A.components[1].r + 0.2
# A.components[2] = A.components[2].r + 0.15
# chEl[0] = chEl.r + radians(10)
# chDist[0] = chDist.r - 0.1
# Show it
print("Beginning render.")
t = time.time()
rn.r
elapsed_time = time.time() - t
print("Ended render in  " + str(elapsed_time))
plt.imsave('opendr_opengl_first.png', rn.r)
chImage = ch.array(image)
E_raw = chImage - rnmod
SqE_raw = 2*ch.SumOfSquares(rnmod - chImage)/chImage.size
iterat = 0

figurerender = plt.figure(2)
render = plt.imshow(rnmod.r)
# global render
# plt.show()
# render.draw()

figureerror = plt.figure(3)
plt.imshow(E_raw.r)


elapsed_time = time.time() - t
def cb2(_):
    global t
    elapsed_time = time.time() - t
    print("Ended interation in  " + str(elapsed_time))

    global E_raw
    global iterat
    iterat = iterat + 1
    print("Callback! " + str(iterat))

    # global render
    plt.figure(2)
    # render.set_data(rn.r)
    plt.imshow(rnmod.r)

    plt.figure(3)
    # render.set_data(rn.r)
    plt.imshow(E_raw.r)

    t = time.time()

chAzScaled
chElScaled
chDistScaled
chComponentScaled
free_variables = [chAzScaled, chElScaled]

mintime = time.time()
boundEl = (0, radians(90))
boundAz = (0, radians(360))
boundscomponents = (0,None)
bounds = [boundEl, boundAz ]
methods=['dogleg', 'minimize', 'BFGS', 'L-BFGS-B', 'Nelder-Mead']


ch.minimize({'raw': SqE_raw}, bounds=bounds, method=methods[0], x0=free_variables, callback=cb2, options={'disp':True})
elapsed_time = time.time() - mintime
print("Minimization time:  " + str(elapsed_time))

#
ipdb.set_trace()
plt.imsave('opendr_opengl_final.png', rn.r)