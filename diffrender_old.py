__author__ = 'pol'

import bpy
import sceneimport
import mathutils
from math import radians
import timeit
import time
import opendr as opendr
import chumpy as ch
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.lighting import SphericalHarmonics

from opendr.camera import ProjectPoints
import numpy as np
import cv2
# from chumpy.utils import row, col
# from opendr.simple import *
# from opendr.util_tests import get_earthmesh
from sklearn.preprocessing import normalize
from utils import *
import timeit
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
plt.ion()

rn = ColoredRenderer()

renderTeapotsList = [2]

[targetScenes, targetModels, transformations] = sceneimport.loadTargetModels(renderTeapotsList)
teapot = targetModels[0]

width, height = (400, 400)

angle = 60 * 180 / numpy.pi
clip_start = 0.2
clip_end = 1
camDistance = 0.45
scene = setupBlender(teapot, width, height, angle, clip_start, clip_end, camDistance)
scene.render.filepath = 'opendr_blender.png'
# bpy.ops.render.render( write_still=True )

center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)
azimuth = 48
elevation = 80
azimuthRot = mathutils.Matrix.Rotation(radians(-azimuth), 4, 'Z')
elevationRot = mathutils.Matrix.Rotation(radians(-elevation), 4, 'X')
originalLoc = mathutils.Vector((0,-camDistance, 0))
location = center + azimuthRot * elevationRot * originalLoc
camera = scene.camera
camera.location = location
scene.update()
look_at(camera, center)
scene.update()
bpy.ops.render.render( write_still=True )

center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)
azimuth = 40
elevation = 80
azimuthRot = mathutils.Matrix.Rotation(radians(-azimuth), 4, 'Z')
elevationRot = mathutils.Matrix.Rotation(radians(-elevation), 4, 'X')
originalLoc = mathutils.Vector((0,-camDistance, 0))
location = center + azimuthRot * elevationRot * originalLoc
camera = scene.camera
camera.location = location
scene.update()
look_at(camera, center)
scene.update()


image = cv2.imread(scene.render.filepath)
image = image/255.0
image = cv2.cvtColor(numpy.float32(image*255), cv2.COLOR_BGR2RGB)/255.0

f = []
v = []
vc = []
vn = []
uv = []
vertexMeshIndex = 0
for mesh in teapot.dupli_group.objects:
    if mesh.type == 'MESH':
        # mesh.data.validate(verbose=True, clean_customdata=True)
        fmesh, vmesh, vcmesh,  nmesh, uvmesh = buildData(mesh.data)
        f = f + [fmesh + vertexMeshIndex]
        vc = vc + [vcmesh]
        transf = np.array(np.dot(teapot.matrix_world, mesh.matrix_world))
        vmesh = np.hstack([vmesh, np.ones([vmesh.shape[0],1])])
        vmesh = ( np.dot(transf , vmesh.T)).T[:,0:3]
        v = v + [vmesh]
        transInvMat = np.linalg.inv(transf).T
        nmesh = np.hstack([nmesh, np.ones([nmesh.shape[0],1])])
        nmesh = (np.dot(transInvMat , nmesh.T)).T[:,0:3]
        vn = vn + [normalize(nmesh, axis=1)]
        uv = uv + [uvmesh]

        vertexMeshIndex = vertexMeshIndex + len(vmesh)

f = np.vstack(f).astype(dtype=np.uint32)

v = np.vstack(v).astype(np.float32)
vc = np.vstack(vc).astype(np.float32)
#Desaturate a bit.
gray = np.dot(np.array([0.3, 0.59, 0.11]), vc.T).T
sat = 0.5

vc[:,0] = vc[:,0] * sat + (1-sat) * gray
vc[:,1] = vc[:,1] * sat + (1-sat) * gray
vc[:,2] = vc[:,2] * sat + (1-sat) * gray

vn = np.vstack(vn).astype(np.float32)
uv = np.vstack(uv).astype(np.float32)


gtCamRot = cv2.Rodrigues(np.array((camera.matrix_world ).inverted().to_3x3()))[0].squeeze() #[0,0,0]
camRot = cv2.Rodrigues(np.array((camera.matrix_world ).inverted().to_3x3()))[0].squeeze() #[0,0,0]
camRot = cv2.Rodrigues(np.array((camera.matrix_world  * mathutils.Matrix.Rotation(radians(0), 4, 'Z') * mathutils.Matrix.Rotation(radians(180), 4, 'X')).inverted().to_3x3()))[0].squeeze() #[0,0,0]


camLoc = (camera.matrix_world * mathutils.Matrix.Rotation(radians(180), 4, 'X')).inverted().to_translation()

gtrotation = ch.array(camRot)

translation, rotation = ch.array(camLoc), ch.array(camRot)
vch = ch.array(v)
vnch = ch.array(vn)
vcch = ch.array(vc)
fch = f.astype('uint32')

from opendr.camera import ProjectPoints
rn.camera = ProjectPoints(v=vch, rt=rotation, t=translation, f=ch.array([width,width]), c=ch.array([width,height])/2.0, k=ch.zeros(5))
rn.frustum = {'near': clip_start, 'far': clip_end, 'width': width, 'height': height}
rn.set(v=vch, f=fch, bgcolor=ch.zeros(3))

# Construct point light source
l1 = LambertianPointLight(
    f=f,
    v=vch,
    vn=vnch,
    num_verts=len(vch),
    light_pos=ch.array([-0,-0,0.5]),
    vc=vcch,
    light_color=ch.array([1., 1., 1.])*1.5)


# Construct point light source
l2 = LambertianPointLight(
    f=f,
    v=vch,
    vn=vnch,
    num_verts=len(vch),
    light_pos=ch.array([-0,-0,-0.5]),
    vc=vcch,
    light_color=ch.array([1., 1., 1.]))

rn.vc = l1 + l2 + vcch*0.25

# Show it
print("Beginning render.")
t = time.process_time()
rn.r
elapsed_time = time.process_time() - t
print("Ended render in  " + str(elapsed_time))

plt.imsave('opendr_opengl_final.png', rn.r)

chImage = ch.array(image)
E_raw = rn - chImage
SqE_raw = ch.SumOfSquares(rn - chImage)
iterat = 0
global t
t = time.process_time()
def cb(_):
    global t
    elapsed_time = time.process_time() - t
    print("Ended interation in  " + str(elapsed_time))

    global E_raw
    global iterat
    iterat = iterat + 1
    # res = np.copy(np.array(SqE_raw.r))
    resimg = np.copy(np.array(rn.r))
    print("Callback! " + str(iterat))
    imres = np.copy(np.sqrt(np.array(E_raw.r * E_raw.r)))
    plt.imsave('iter_Err' + str(iterat) + '.png',imres)
    plt.imsave('iter_dr' + str(iterat) + '.png',resimg)
    # plt.imshow('Sq error', res)
    # cv2.waitKey(1)
    t = time.process_time()
# ipdb.set_trace()
#
free_variables = [rotation]

# ch.minimize({'raw': E_raw}, x0=free_variables, callback=cb)