__author__ = 'pol'

import matplotlib
matplotlib.use('TkAgg')

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
from sklearn.preprocessing import normalize
import timeit
import pickle 
import re
import matplotlib.pyplot as plt

plt.ion()

def loadScene(sceneFile):
    sceneLines = [line.strip() for line in open(sceneFile)]

    numModels = sceneLines[2].split()[1]
    instances = []
    for line in sceneLines:
        parts = line.split()
        if parts[0] == 'newModel':
            modelId = parts[2]
        if parts[0] == 'parentContactPosition':
            parentContactPosition = mathutils.Vector([float(parts[1])*inchToMeter, float(parts[2])*inchToMeter, float(parts[3])*inchToMeter])            
        if parts[0] == 'parentIndex':
            parentIndex = int(parts[1])
        if parts[0] == 'transform':
            transform = mathutils.Matrix([[float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])], [float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])], [ float(parts[9]), float(parts[10]), float(parts[11]), float(parts[12])], [float(parts[13]), float(parts[14]), float(parts[15]), float(parts[16])]]).transposed()
            # ipdb.set_trace()
            transform[0][3] = transform[0][3]*inchToMeter
            transform[1][3] = transform[1][3]*inchToMeter
            transform[2][3] = transform[2][3]*inchToMeter
            # ipdb.set_trace()
            
            instances.append([modelId, parentIndex, parentContactPosition, transform])

    return instances


# pylab.pause(0.0001)
#
# pylab.plot([1,2,3])
# pylab.figure(1)
# pylab.show(block=False)

rn = TexturedRenderer()

rnmod = TexturedRenderer()

renderTeapotsList = [2]

width, height = (200, 200)

angle = 60 * 180 / np.pi
clip_start = 0.05
clip_end = 10
camDistance = 0.4
azimuth = 95
elevation = 65


replaceableScenesFile = '../databaseFull/fields/scene_replaceables.txt'
sceneLines = [line.strip() for line in open(replaceableScenesFile)]
sceneLineNums = np.arange(len(sceneLines))
sceneNum =  sceneLineNums[0]
sceneLine = sceneLines[sceneNum]
sceneParts = sceneLine.split(' ')
sceneFile = sceneParts[0]
sceneNumber = int(re.search('.+?scene([0-9]+)\.txt', sceneFile, re.IGNORECASE).groups()[0])
sceneFileName = re.search('.+?(scene[0-9]+\.txt)', sceneFile, re.IGNORECASE).groups()[0]
targetIndex = int(sceneParts[1])
instances = loadScene('../databaseFull/scenes/' + sceneFileName)
targetParentPosition = instances[targetIndex][2]
targetParentIndex = instances[targetIndex][1]

center = targetParentPosition

sceneDicFile = 'sceneDic.pickle'
sceneDic = {}

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

# vmod,fmod_list, vcmod, vnmod, uvmod, haveTexturesmod_list, texturesmod_list = sceneimport.unpackObjects(teapot)


rangeMeshes = range(len(v))
vch = [ch.array(v[mesh]) for mesh in rangeMeshes]
vnch = [ch.array(vn[mesh]) for mesh in rangeMeshes]
vcch = [ch.array(vc[mesh]) for mesh in rangeMeshes]

light_color=ch.ones(3)

componentScale = 4
chComponentScaled = ch.Ch([1, 0.25, 0.25, 0.,0.,0.,0.,0.,0.])
chComponent = chComponentScaled*componentScale

A_list = [SphericalHarmonics(vn=vnch[mesh],
                       components=chComponent,
                       light_color=light_color) for mesh in rangeMeshes]
vc_list = [A_list[mesh]*vcch[mesh] for mesh in rangeMeshes]

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


# rangeMeshes = range(len(v))
# vchmod = [ch.array(v[-1])]
# vnchmod = [ch.array(vn[-1])]
# vcchmod = [ch.array(vc[-1])]
# uvmod = [ch.array(uv[-1])]
# haveTexturesmod_list = [haveTextures_list[-1]] 

# A_mod = [SphericalHarmonics(vn=vnchmod[0],
#                        components=chComponent,
#                        light_color=light_color)]
# vcmod_list = [A_mod[0]*vcchmod[0]]

# vmod = ch.vstack(vchmod)
# fmod_list = [f_list[-1]]
# fmod = []
# for mesh in fmod_list:
#     lenMeshes = len(fmod)
#     for polygons in mesh:
#         fmod = fmod + [polygons + lenMeshes]
# fstackmod = np.vstack(fmod)
# vnmod = ch.vstack(vnchmod)
# vcmod = ch.vstack(vcmod_list)
# ftmod = np.vstack(uvmod)

# texturesmod_list = []
# texturesmod_list.append([textures_list[-1]])
# textureschmod = []
# for texturemod_list in textures_list[-1]:
#     for texture in texturemod_list:
#         if texture != None:
#             textureschmod = textureschmod + [ch.array(texture)]
# if textureschmod != []:
#     texturemod_stack = ch.concatenate([tex.ravel() for tex in textureschmod])
# else:
#     texturemod_stack = ch.Ch([])


azScale = radians(30)
elScale = radians(15)

distScale = 0.2

chAzScaled = ch.Ch([radians(azimuth/azScale)])
chElScaled = ch.Ch([radians(elevation/elScale)])
chDistScaled = ch.Ch([camDistance/distScale])



chAz = chAzScaled*azScale
chEl = chElScaled*elScale
chDist = chDistScaled*distScale


chDistMat = geometry.Translate(x=ch.Ch(0), y=-chDist, z=ch.Ch(0))
chToObjectTranslate = geometry.Translate(x=center.x, y=center.y, z=center.z)

chRotAzMat = geometry.RotateZ(a=-chAz)
chRotElMat = geometry.RotateX(a=-chEl)
chCamModelWorld = ch.dot(chToObjectTranslate, ch.dot(chRotAzMat, ch.dot(chRotElMat,chDistMat)))

chInvCam = ch.inv(ch.dot(chCamModelWorld, np.array(mathutils.Matrix.Rotation(radians(270), 4, 'X'))))

chRod = opendr.geometry.Rodrigues(rt=chInvCam[0:3,0:3]).reshape(3)
chTranslation = chInvCam[0:3,3]


translation, rotation = (chTranslation, chRod)



from opendr.camera import ProjectPoints
rn.camera = ProjectPoints(v=v, rt=rotation, t=translation, f= 1.12*ch.array([width,width]), c=ch.array([width,height])/2.0, k=ch.zeros(5))
rn.camera.openglMat = np.array(mathutils.Matrix.Rotation(radians(180), 4, 'X'))
rn.frustum = {'near': clip_start, 'far': clip_end, 'width': width, 'height': height}

#Ugly initializations, think of a better way that still works well with Chumpy!
rn.set(v=v, f=fstack, vn=vn, vc=vc, ft=ft, texture_stack=texture_stack, v_list=vch, f_list=f_list, vc_list=vc_list, ft_list=uv, textures_list=textures_list, haveUVs_list=haveTextures_list, bgcolor=ch.ones(3), overdraw=True)

ipdb.set_trace()

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