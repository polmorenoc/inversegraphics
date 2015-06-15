__author__ = 'pol'

import opendr
import chumpy as ch
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
import numpy as np
import cv2
from chumpy.utils import row, col
import matplotlib.pyplot as plt
from opendr.simple import *

rn = ColoredRenderer()

# m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))

fname = '../databaseFull/models/teapots/fa1fa0818738e932924ed4f13e49b59d/Teapot N300912_cleaned.obj'
mesh = load_mesh(fname)

mesh.v = np.asarray(mesh.v, order='C')
# mesh.vc = mesh.v*0 + 1
mesh.v -= row(np.mean(mesh.v, axis=0))
mesh.v /= np.max(mesh.v)
mesh.v *= 2.0

# mesh.v = mesh.v.dot(cv2.Rodrigues(np.asarray(np.array(rotation), np.float64))[0])
# mesh.v = mesh.v + row(np.asarray(trans))


w, h = (320, 240)

rn.camera = ProjectPoints(v=mesh.v, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,w])/2., c=ch.array([w,h])/2., k=ch.zeros(5))
rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn.set(v=mesh.v, f=mesh.f, bgcolor=ch.zeros(3))

# Construct point light source
rn.vc = LambertianPointLight(
    f=mesh.f,
    v=rn.v,
    num_verts=len(mesh.v),
    light_pos=ch.array([-1000,-1000,-1000]),
    vc=mesh.vc,
    light_color=ch.array([1., 1., 1.]))

# Show it

plt.ion()
plt.imshow(rn.r)
plt.show()

dr = rn.dr_wrt(rn.v) # or rn.vc, or rn.camera.rt, rn.camera.t, rn.camera.f, rn.camera.c, etc

