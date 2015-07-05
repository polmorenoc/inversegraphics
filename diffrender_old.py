__author__ = 'pol'

import bpy
import sceneimport
import mathutils
from math import radians
import timeit
import time
import opendrold as opendr
import chumpy as ch
from opendrold.renderer import ColoredRenderer
from opendrold.lighting import LambertianPointLight
from opendrold.lighting import SphericalHarmonics

from opendrold.camera import ProjectPoints
import numpy as np
import cv2
# from chumpy.utils import row, col
# from opendrold.simple import *
# from opendrold.util_tests import get_earthmesh
from sklearn.preprocessing import normalize
from utils import *

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
plt.ion()


# mesh = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))

# fname = '../databaseFull/models/teapots/fa1fa0818738e932924ed4f13e49b59d/Teapot N300912_cleaned.obj'
# m = load_mesh(fname)
# Create renderer
rn = ColoredRenderer()

renderTeapotsList = [1]

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
azimuth = 261
elevation = 75
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
# ftmp = f.copy()
# f[:,0] = f[:,2]
# f[:,2] = ftmp[:,0]
v = np.vstack(v).astype(np.float32)
vc = np.vstack(vc).astype(np.float32)
vn = np.vstack(vn).astype(np.float32)
uv = np.vstack(uv).astype(np.float32)


# * mathutils.Matrix.Rotation(radians(180), 4, 'Y'))
gtCamRot = cv2.Rodrigues(np.array((camera.matrix_world ).inverted().to_3x3()))[0].squeeze() #[0,0,0]
camRot = cv2.Rodrigues(np.array((camera.matrix_world ).inverted().to_3x3()))[0].squeeze() #[0,0,0]
# camRot = camera.matrix_world.inverted().to_euler()

camLoc = (camera.matrix_world ).inverted().to_translation()
gtrotation = ch.array(camRot)

translation, rotation = ch.array(camLoc), ch.array(camRot)
vch = ch.array(v)
vnch = ch.array(vn)
vcch = ch.array(vc)
fch = f.astype('uint32')

from opendrold.camera import ProjectPoints
rn.camera = ProjectPoints(v=vch, rt=rotation, t=translation, f=ch.array([width,width]), c=ch.array([width,height])/2.0, k=ch.zeros(5))
rn.frustum = {'near': clip_start, 'far': clip_end, 'width': width, 'height': height}
rn.set(v=vch, f=fch, bgcolor=ch.zeros(3))

# Construct point light source
l1 = LambertianPointLight(
    f=f,
    v=vch,
    vn=vnch,
    num_verts=len(vch),
    light_pos=ch.array([-0,-0,-0.5]),
    vc=vcch,
    light_color=ch.array([1., 1., 1.]))


# Construct point light source
l2 = LambertianPointLight(
    f=f,
    v=vch,
    vn=vnch,
    num_verts=len(vch),
    light_pos=ch.array([-0,-0,0.5]),
    vc=vcch,
    light_color=ch.array([1., 1., 1.]))

# vcl1 = l1.r + l2.r


rn.vc = l1 + l2

# A = SphericalHarmonics(vn=vnch, vc=vcch,
#                    components=[3.,2.,0.,0.,0.,0.,0.,0.,0.],
#                    light_color=ch.ones(3))
# rn.vc = A

# Show it
print("Beginning render.")
t = time.process_time()
rn.r
elapsed_time = time.process_time() - t
print("Ended render in  " + str(elapsed_time))
# plt.imshow(iplm)
# plt.show()
# imr = cv2.resize(im, (0,0), fx=2.0/upscale, fy=2.0/upscale)
# imc = imr[imr.shape[0]/2 - width/2: imr.shape[0]/2 + width/2, imr.shape[1]/2 - height/2: imr.shape[1]/2 + height/2]
# plt.imshow(imc)
# plt.show()

plt.imsave('opendr_opengl_final.png', rn.r)

chImage = ch.array(image)
E_raw = (rn - chImage)*(rn - chImage)
iterat = 0
# plt.imsave('groundtruth' + '.png',image)
# plt.imsave('initialmodel' + '.png',im)
def cb(_):
    global E_raw
    global iterat
    iterat = iterat + 1
    res = np.copy(np.array(E_raw.r))
    resimg = np.copy(np.array(rn.r))
    print("Callback! " + str(iterat))
    plt.imsave('iter_' + str(iterat) + '.png',res)
    plt.imsave('iter_dr' + str(iterat) + '.png',resimg)
    # plt.imshow('Sq error', res)
    # cv2.waitKey(1)
#
free_variables = [rotation]
ch.minimize({'raw': E_raw}, x0=free_variables, callback=cb)