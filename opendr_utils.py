__author__ = 'pol'

import opendr
import chumpy as ch
import geometry
import mathutils
import numpy as np
from math import radians
from opendr.camera import ProjectPoints
from opendr.renderer import TexturedRenderer
import ipdb

def setupCamera(v, chAz, chEl, chDist, objCenter, width, height):

    chDistMat = geometry.Translate(x=ch.Ch(0), y=-chDist, z=ch.Ch(0))
    chToObjectTranslate = geometry.Translate(x=objCenter.x, y=objCenter.y, z=objCenter.z)

    chRotAzMat = geometry.RotateZ(a=-chAz)
    chRotElMat = geometry.RotateX(a=-chEl)
    chCamModelWorld = ch.dot(chToObjectTranslate, ch.dot(chRotAzMat, ch.dot(chRotElMat,chDistMat)))

    chMVMat = ch.dot(chCamModelWorld, np.array(mathutils.Matrix.Rotation(radians(270), 4, 'X')))

    chInvCam = ch.inv(chMVMat)

    modelRotation = chInvCam[0:3,0:3]

    chRod = opendr.geometry.Rodrigues(rt=modelRotation).reshape(3)
    chTranslation = chInvCam[0:3,3]

    translation, rotation = (chTranslation, chRod)
    camera = ProjectPoints(v=v, rt=rotation, t=translation, f= 1.12*ch.array([width,width]), c=ch.array([width,height])/2.0, k=ch.zeros(5))
    camera.openglMat = np.array(mathutils.Matrix.Rotation(radians(180), 4, 'X'))
    return camera, modelRotation

def setupTexturedRenderer(vstack, vch,f_list, vc_list, vnch, uv, haveTextures_list, textures_list, camera, frustum):

    renderer = TexturedRenderer()

    f = []
    for mesh in f_list:
        lenMeshes = len(f)
        for polygons in mesh:
            f = f + [polygons + lenMeshes]
    fstack = np.vstack(f)
    if len(vnch)==1:
        vnstack = vnch[0]
    else:
        vnstack = ch.vstack(vnch)
    if len(vc_list)==1:
        vcstack = vc_list[0]
    else:
        vcstack = ch.vstack(vc_list)

    ftstack = np.vstack(uv)

    texturesch = []
    for texture_list in textures_list:
        for texture in texture_list:
            if texture != None:
                texturesch = texturesch + [ch.array(texture)]

    if len(texturesch) == 0:
        texture_stack = ch.Ch([])
    elif len(texturesch) == 1:
        texture_stack = texturesch[0].ravel()
    else:
        texture_stack = ch.concatenate([tex.ravel() for tex in texturesch])

    renderer.camera = camera

    renderer.frustum = frustum
    renderer.set(v=vstack, f=fstack, vn=vnstack, vc=vcstack, ft=ftstack, texture_stack=texture_stack, v_list=vch, f_list=f_list, vc_list=vc_list, ft_list=uv, textures_list=textures_list, haveUVs_list=haveTextures_list, bgcolor=ch.ones(3), overdraw=True)

    return renderer