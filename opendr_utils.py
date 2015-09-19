__author__ = 'pol'

import opendr
import chumpy as ch
import geometry
import mathutils
import numpy as np
from math import radians
from opendr.camera import ProjectPoints
from opendr.renderer import TexturedRenderer
from opendr.lighting import SphericalHarmonics
from opendr.lighting import LambertianPointLight
import ipdb
import light_probes


def getOcclusionFraction(renderer):

    vis_occluded = np.array(renderer.indices_image!=0).copy().astype(np.bool)
    vis_im = np.array(renderer.image_mesh_bool([0])).copy().astype(np.bool)

    return 1. - np.sum(vis_occluded)/np.sum(vis_im)

#From http://www.ppsloan.org/publications/StupidSH36.pdf
def chZonalHarmonics(a):
    zl0 = -ch.sqrt(ch.pi)*(-1.0 + ch.cos(a))
    zl1 = 0.5*ch.sqrt(3.0*ch.pi)*ch.sin(a)**2
    zl2 = -0.5*ch.sqrt(5.0*ch.pi)*ch.cos(a)*(-1.0 + ch.cos(a))*(ch.cos(a)+1.0)
    z = [zl0, zl1, zl2]
    return ch.concatenate(z)

# http://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf
chSpherical_harmonics = {
    (0, 0): lambda theta, phi: ch.Ch([0.282095]),

    (1, -1): lambda theta, phi: 0.488603 * ch.sin(theta) * ch.sin(phi),
    (1, 0): lambda theta, phi: 0.488603 * ch.cos(theta),
    (1, 1): lambda theta, phi: 0.488603 * ch.sin(theta) * ch.cos(phi),

    (2, -2): lambda theta, phi: 1.092548 * ch.sin(theta) * ch.cos(phi) * ch.sin(theta) * ch.sin(phi),
    (2, -1): lambda theta, phi: 1.092548 * ch.sin(theta) * ch.sin(phi) * ch.cos(theta),
    (2, 0): lambda theta, phi: 0.315392 * (3 * ch.cos(theta)**2 - 1),
    (2, 1): lambda theta, phi: 1.092548 * ch.sin(theta) * ch.cos(phi) * ch.cos(theta),
    (2, 2): lambda theta, phi: 0.546274 * (((ch.sin(theta) * ch.cos(phi)) ** 2) - ((ch.sin(theta) * ch.sin(phi)) ** 2))
}

#From http://www.ppsloan.org/publications/StupidSH36.pdf
def chZonalToSphericalHarmonics(z, theta, phi):
    sphCoeffs = []
    for l in np.arange(len(z)):
        for m in np.arange(np.int(-(l*2+1)/2),np.int((l*2+1)/2) + 1):
            ylm_d = chSpherical_harmonics[(l,m)](theta,phi)
            sh = np.sqrt(4*np.pi/(2*l + 1))*z[l]*ylm_d
            sphCoeffs = sphCoeffs + [sh]

    #Correct order in band l=1.
    sphCoeffs[1],sphCoeffs[3] = sphCoeffs[3],sphCoeffs[1]
    chSphCoeffs = ch.concatenate(sphCoeffs)
    return chSphCoeffs

#From http://www.ppsloan.org/publications/StupidSH36.pdf
def clampedCosineCoefficients():

    constants = []
    for l in np.arange(3):
        for m in np.arange(np.int(-(l*2+1)/2),np.int((l*2+1)/2) + 1):
            normConstant = np.pi
            if l > 1 and l % 1 == 0:
                normConstant = 0
            if l == 1:
                normConstant = 2*np.pi/3
            if l > 1 and l % 2 == 0:
                normConstant = 2*np.pi*(((-1)**(l/2.-1.))/((l+2)*(l-1)))*(np.math.factorial(l)/((2**(l))*(np.math.factorial(l/2)**2)))
                # normConstant = 0.785398

            constants = constants + [normConstant]

    return np.array(constants)

def setupCamera(v, chAz, chEl, chDist, objCenter, width, height):

    chCamModelWorld = computeHemisphereTransformation(chAz, chEl, chDist, objCenter)

    chMVMat = ch.dot(chCamModelWorld, np.array(mathutils.Matrix.Rotation(radians(270), 4, 'X')))

    chInvCam = ch.inv(chMVMat)

    modelRotation = chInvCam[0:3,0:3]

    chRod = opendr.geometry.Rodrigues(rt=modelRotation).reshape(3)
    chTranslation = chInvCam[0:3,3]

    translation, rotation = (chTranslation, chRod)
    camera = ProjectPoints(v=v, rt=rotation, t=translation, f= 1.12*ch.array([width,width]), c=ch.array([width,height])/2.0, k=ch.zeros(5))
    camera.openglMat = np.array(mathutils.Matrix.Rotation(radians(180), 4, 'X'))
    return camera, modelRotation

def computeHemisphereTransformation(chAz, chEl, chDist, objCenter):

    chDistMat = geometry.Translate(x=ch.Ch(0), y=-chDist, z=ch.Ch(0))
    chToObjectTranslate = geometry.Translate(x=objCenter[0], y=objCenter[1], z=objCenter[2])

    chRotAzMat = geometry.RotateZ(a=chAz)
    chRotElMat = geometry.RotateX(a=-chEl)
    chCamModelWorld = ch.dot(chToObjectTranslate, ch.dot(chRotAzMat, ch.dot(chRotElMat,chDistMat)))

    return chCamModelWorld

def computeSphericalHarmonics(vn, vc, light_color, components):

    # vnflat = [item for sublist in vn for item in sublist]
    # vcflat = [item for sublist in vc for item in sublist]
    rangeMeshes = range(len(vn))
    A_list = [SphericalHarmonics(vn=vn[mesh],
                       components=components,
                       light_color=light_color) for mesh in rangeMeshes]

    vc_list = [A_list[mesh]*vc[mesh] for mesh in rangeMeshes]
    return vc_list

def computeGlobalAndPointLighting(v, vn, vc, light_pos, globalConstant, light_color):
    # Construct point light source
    rangeMeshes = range(len(vn))
    vc_list = []
    for mesh in rangeMeshes:
        l1 = LambertianPointLight(
            v=v[mesh],
            vn=vn[mesh],
            num_verts=len(v[mesh]),
            light_pos=light_pos,
            vc=vc[mesh],
            light_color=light_color)

        vcmesh = vc[mesh]*(l1 + globalConstant)
        vc_list = vc_list + [vcmesh]
    return vc_list

def setupTexturedRenderer(renderer, vstack, vch, f_list, vc_list, vnch, uv, haveTextures_list, textures_list, camera, frustum, sharedWin=None):

    f = []
    f_listflat = [item for sublist in f_list for item in sublist]

    for mesh in f_listflat:
        lenMeshes = len(f)
        for polygons in mesh:
            f = f + [polygons + lenMeshes]
    fstack = np.vstack(f)

    # vnflat = [item for sublist in vnch for item in sublist]
    if len(vnch)==1:
        vnstack = vnch[0]
    else:
        vnstack = ch.vstack(vnch)

    # vc_listflat = [item for sublist in vc_list for item in sublist]
    if len(vc_list)==1:
        vcstack = vc_list[0]
    else:
        vcstack = ch.vstack(vc_list)

    uvflat = [item for sublist in uv for item in sublist]
    ftstack = np.vstack(uvflat)

    texturesch = []
    textures_listflat = [item for sublist in textures_list for item in sublist]
    for texture_list in textures_listflat:
        if texture_list != None:
            for texture in texture_list:
                if texture != None:
                    texturesch = texturesch + [ch.array(texture)]

    if len(texturesch) == 0:
        texture_stack = ch.Ch([])
    elif len(texturesch) == 1:
        texture_stack = texturesch[0].ravel()
    else:
        texture_stack = ch.concatenate([tex.ravel() for tex in texturesch])

    haveTextures_listflat = [item for sublist in haveTextures_list for item in sublist]

    renderer.set(camera=camera, frustum=frustum, v=vstack, f=fstack, vn=vnstack, vc=vcstack, ft=ftstack, texture_stack=texture_stack, v_list=vch, f_list=f_listflat, vc_list=vc_list, ft_list=uvflat, textures_list=textures_listflat, haveUVs_list=haveTextures_listflat, bgcolor=ch.ones(3), overdraw=True)
    renderer.sharedWin = sharedWin
    # renderer.clear()
    renderer.initGL()
    renderer.initGLTexture()

def addObjectData(v, f_list, vc, vn, uv, haveTextures_list, textures_list, vmod, fmod_list, vcmod, vnmod, uvmod, haveTexturesmod_list, texturesmod_list):
    v.insert(0,vmod)
    f_list.insert(0,fmod_list)
    vc.insert(0,vcmod)
    vn.insert(0,vnmod)
    uv.insert(0,uvmod)
    haveTextures_list.insert(0,haveTexturesmod_list)
    textures_list.insert(0,texturesmod_list)

def removeObjectData(objIdx, v, f_list, vc, vn, uv, haveTextures_list, textures_list):

    del v[objIdx]
    del f_list[objIdx]
    del vc[objIdx]
    del vn[objIdx]
    del uv[objIdx]
    del haveTextures_list[objIdx]
    del textures_list[objIdx]

