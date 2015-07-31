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
from opendr.renderer import ColoredRenderer
from opendr.renderer import TexturedRenderer
from opendr.lighting import LambertianPointLight
from opendr.lighting import SphericalHarmonics
from opendr.filters import gaussian_pyramid
import geometry
from opendr.camera import ProjectPoints
import numpy as np
import cv2
from sklearn.preprocessing import normalize
from utils import *
import timeit

import matplotlib.pyplot as plt

plt.ion()

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
azimuth = 315
elevation = 25

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

chAz = ch.Ch([radians(azimuth)])
chEl = ch.Ch([radians(elevation)])

chDist = chDistScaled*distScale
chDist = ch.Ch([camDistance])
chComponent = chComponentScaled*componentScale
chComponent = ch.Ch(np.array([2, 0.25, 0.25, 0.12,-0.17,0.36,0.1,0.,0.]))

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
# rn.camera.openglMat = np.array(mathutils.Matrix.Rotation(radians(180), 4, 'X'))
rn.camera.openglMat = np.array(mathutils.Matrix.Rotation(radians(180), 4, 'X'))
rn.frustum = {'near': clip_start, 'far': clip_end, 'width': width, 'height': height}

#Ugly initializations, think of a better way that still works well with Chumpy!
rn.set(v=v, f=fstack, vn=vn, vc=vc, ft=ft, texture_stack=texture_stack, v_list=vch, f_list=f_list, vc_list=vc_list, ft_list=uv, textures_list=textures_list, haveUVs_list=haveTextures_list, bgcolor=ch.ones(3), overdraw=True)

rnmod.camera = ProjectPoints(v=vmod, rt=rotation, t=translation, f= 1.12*ch.array([width,width]), c=ch.array([width,height])/2.0, k=ch.zeros(5))
rnmod.camera.openglMat = np.array(mathutils.Matrix.Rotation(radians(180), 4, 'X'))
rnmod.frustum = {'near': clip_start, 'far': clip_end, 'width': width, 'height': height}
rnmod.set(v=vmod, f=fstackmod, vn=vnmod, vc=vcmod, ft=ftmod, texture_stack=texturemod_stack, v_list=vchmod, f_list=fmod_list, vc_list=vcmod_list, ft_list=uvmod, textures_list=texturesmod_list, haveUVs_list=haveTexturesmod_list, bgcolor=ch.ones(3), overdraw=True)

# Show it
print("Beginning render.")
t = time.time()
rn.r
elapsed_time = time.time() - t
print("Ended render in  " + str(elapsed_time))


f, ((ax1, ax2), (ax3, ax4), (ax5,ax6)) = plt.subplots(3, 2, subplot_kw={'aspect':'equal'})
pos1 = ax1.get_position()
pos5 = ax5.get_position()
pos5.x0 = pos1.x0
ax5.set_position(pos5)


f.tight_layout()

# figuregt = plt.figure(1)
ax1.set_title("Ground Truth")
ax1.imshow(rn.r)
# ipdb.set_trace()
plt.imsave('opendr_opengl_gt.png', rn.r)
plt.draw()
imagegt = np.copy(np.array(rn.r)).astype(np.float64)

vis_im = np.array(rnmod.visibility_image != 4294967295).copy()

oldChAz = chAz[0].r
oldChEl = chEl[0].r

chAz[0] = chAz[0].r + radians(40)
chEl[0] = chEl[0].r - radians(20)
chComponent[0] = chComponent[0].r

# Show it
shapeIm = vis_im.shape

print("Beginning render.")
t = time.time()
rn.r
elapsed_time = time.time() - t
print("Ended render in  " + str(elapsed_time))
plt.imsave('opendr_opengl_first.png', rn.r)
chImage = ch.array(imagegt)

E_raw_simple = rnmod - chImage

negVisIm = ~vis_im
chImageWhite = imagegt.copy()
chImageWhite[np.tile(negVisIm.reshape([shapeIm[0],shapeIm[1],1]),[1,1,3]).astype(np.bool)] = 1
E_raw = rnmod - chImageWhite
SE_raw = E_raw*E_raw

E_pyr = gaussian_pyramid(E_raw, n_levels=4, normalization='SSE')
E_pyr_simple = gaussian_pyramid(E_raw_simple, n_levels=4, normalization='SSE')

SSqE_raw = ch.SumOfSquares(E_raw)/np.sum(vis_im)
SSqE_raw_simple = ch.SumOfSquares(E_raw_simple)/np.sum(vis_im)
SSqE_pyr = ch.SumOfSquares(E_pyr)/np.sum(vis_im)
SSqE_pyr_simple = ch.SumOfSquares(E_pyr_simple)/np.sum(vis_im)

iterat = 0

# figurerender = plt.figure(2)
ax2.set_title("Backprojection")
pim2 = ax2.imshow(rnmod.r)
# global render
# plt.show()
# render.draw()
plt.draw()
# ax1.figure(1)
edges = rnmod.boundarybool_image
gtoverlay = imagegt.copy()
gtoverlay[np.tile(edges.reshape([shapeIm[0],shapeIm[1],1]),[1,1,3]).astype(np.bool)] = 1
pim1 = ax1.imshow(gtoverlay)
# figureerror = plt.figure(3)

ax3.set_title("Error (Abs of residuals)")
pim3 = ax3.imshow(np.abs(E_raw.r))

ax5.set_title("Dr wrt. Azimuth")
drazsum = np.sum(SE_raw.dr_wrt(chAz).reshape(shapeIm[0],shapeIm[1],3), axis=2).reshape(shapeIm[0],shapeIm[1],1)
# drazsumneg = drazsum.copy()
# drazsumneg[drazsumneg < 0] = 0
# drazsumpos = drazsum.copy()
# drazsumpos[drazsumpos >= 0] = 0
# drazsumfinal = np.concatenate([drazsumpos, drazsumneg, np.zeros([shapeIm[0],shapeIm[1],1])],axis=2)
img = ax5.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
plt.colorbar(img, ax=ax5,use_gridspec=True)

ax6.set_title("Dr wrt. Elevation")
drazsum = np.sum(SE_raw.dr_wrt(chEl).reshape(shapeIm[0],shapeIm[1],3), axis=2).reshape(shapeIm[0],shapeIm[1],1)
# drazsumneg = drazsum.copy()
# drazsumneg[drazsumneg < 0] = 0
# drazsumpos = drazsum.copy()
# drazsumpos[drazsumpos >= 0] = 0
# drazsumfinal = np.concatenate([drazsumpos, drazsumneg, np.zeros([shapeIm[0],shapeIm[1],1])],axis=2)
img6 = ax6.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
plt.colorbar(img6, ax=ax6,use_gridspec=True)
# plt.draw()
pos1 = ax1.get_position()
pos5 = ax5.get_position()
pos5.x0 = pos1.x0
ax5.set_position(pos5)

plt.show()
pos1 = ax1.get_position()
pos5 = ax5.get_position()
pos5.x0 = pos1.x0
ax5.set_position(pos5)

ipdb.set_trace()
plt.pause(0.1)

elapsed_time = time.time() - t
def cb2(_):
    global t
    elapsed_time = time.time() - t
    print("Ended interation in  " + str(elapsed_time))

    global E_raw
    global iterat
    iterat = iterat + 1
    print("Callback! " + str(iterat))
    print("Sq Error: " + str(SSqE_raw.r))
    global imagegt
    global rnmod

    # if iterat % 5 == 0:
    edges = rnmod.boundarybool_image
    gtoverlay = imagegt.copy()
    gtoverlay[np.tile(edges.reshape([shapeIm[0],shapeIm[1],1]),[1,1,3]).astype(np.bool)] = 1
    pim1.set_data(gtoverlay)

    # plt.figure(2)
    # render.set_data(rn.r)
    pim2.set_data(rnmod.r)

    # plt.figure(3)
    # render.set_data(rn.r)
    pim3.set_data(np.abs(E_raw.r))


    drazsum = np.sum(SE_raw.dr_wrt(chAz).reshape(shapeIm[0],shapeIm[1],3), axis=2).reshape(shapeIm[0],shapeIm[1],1)
    # drazsumneg = drazsum.copy()
    # drazsumneg[drazsumneg < 0] = 0
    # drazsumpos = drazsum.copy()
    # drazsumpos[drazsumpos >= 0] = 0
    # drazsumfinal = np.concatenate([drazsumpos, drazsumneg, np.zeros([shapeIm[0],shapeIm[1],1])],axis=2)
    img = ax5.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)


    drazsum = np.sum(SE_raw.dr_wrt(chEl).reshape(shapeIm[0],shapeIm[1],3), axis=2).reshape(shapeIm[0],shapeIm[1],1)
    # drazsumneg = drazsum.copy()
    # drazsumneg[drazsumneg < 0] = 0
    # drazsumpos = drazsum.copy()
    # drazsumpos[drazsumpos >= 0] = 0
    # drazsumfinal = np.concatenate([drazsumpos, drazsumneg, np.zeros([shapeIm[0],shapeIm[1],1])],axis=2)
    # img6.set_data(drazsum.squeeze())
    img = ax6.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
    # plt.draw()

    f.canvas.draw()
    plt.pause(0.1)

    t = time.time()

# , chComponent[0]
free_variables = [chAz, chEl]

mintime = time.time()
boundEl = (0, np.pi/2.0)
boundAz = (0, None)
boundscomponents = (0,None)
bounds = [boundAz,boundEl]
methods=['dogleg', 'minimize', 'BFGS', 'L-BFGS-B', 'Nelder-Mead']

ipdb.set_trace()

# Sq err,delta,grad simple, grad check, grad pyramid

# Grad
# ch.optimization.gradCheck(SSqE_raw, [chAz], [0.01])
#
# # Grad
# ch.optimization.scipyGradCheck({'raw': SSqE_raw}, [chAz])

ch.minimize({'raw': SSqE_pyr}, bounds=bounds, method=methods[1], x0=free_variables, callback=cb2, options={'disp':True})

# ch.minimize({'raw': SSqE_pyr}, bounds=bounds, method=methods[3], x0=free_variables, callback=cb2, options={'disp':True})

elapsed_time = time.time() - mintime
print("Minimization time:  " + str(elapsed_time))

# plt.figure(1)
edges = rnmod.boundarybool_image
gtoverlay = imagegt.copy()
gtoverlay[np.tile(edges.reshape([shapeIm[0],shapeIm[1],1]),[1,1,3]).astype(np.bool)] = 1
pim1.set_data(gtoverlay)

# global render
# plt.figure(2)
# render.set_data(rn.r)
pim2.set_data(rnmod.r)
# plt.figure(3)
# render.set_data(rn.r)
pim3.set_data(np.abs(E_raw.r))

# plt.figure(4)
ax4.set_title("Full backprojection")
ax4.imshow(rn.r)

ax5.set_title("Dr wrt. Azimuth")
drazsum = np.sum(SE_raw.dr_wrt(chAz).reshape(shapeIm[0],shapeIm[1],3), axis=2).reshape(shapeIm[0],shapeIm[1],1)
# drazsumneg = drazsum.copy()
# drazsumneg[drazsumneg < 0] = 0
# drazsumpos = drazsum.copy()
# drazsumpos[drazsumpos >= 0] = 0
# drazsumfinal = np.concatenate([drazsumpos, drazsumneg, np.zeros([shapeIm[0],shapeIm[1],1])],axis=2)
img = ax5.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)

ax6.set_title("Dr wrt. Elevation")
drazsum = np.sum(SE_raw.dr_wrt(chEl).reshape(shapeIm[0],shapeIm[1],3), axis=2).reshape(shapeIm[0],shapeIm[1],1)
# drazsumneg = drazsum.copy()
# drazsumneg[drazsumneg < 0] = 0
# drazsumpos = drazsum.copy()
# drazsumpos[drazsumpos >= 0] = 0
# drazsumfinal = np.concatenate([drazsumpos, drazsumneg, np.zeros([shapeIm[0],shapeIm[1],1])],axis=2)
img = ax6.imshow(drazsum.squeeze(),cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)

f.canvas.draw()
plt.pause(0.1)

plt.imsave('opendr_opengl_final.png', rn.r)