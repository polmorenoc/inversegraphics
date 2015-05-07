#!/usr/bin/env python3.4m
 
import bpy
import numpy
from PIL import Image
import mathutils
from math import radians
import h5py
import scipy.io
from score_image import *
import cv2
import sys
import io
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
import pickle
import ipdb
from tabulate import tabulate

def loadData():
    #data
    # fdata = h5py.File('../data/data-all-flipped-cropped-512.mat','r')
    # data = fdata["data"]

    data = scipy.io.loadmat('../data/data-all-flipped-cropped-512-scipy.mat')['data']

    images = h5py.File('../data/images-all-flipped-cropped-512-color.mat','r')
    # images = f["images"]
    # imgs = numpy.array(images)
    # N = imgs.shape[0]
    # imgs = imgs.transpose(0,2,3,1)

    # f = h5py.File('../data/all-flipped-cropped-512-crossval6div2_py-experiment.mat')
    # experiments = f["experiments_data"]

    # f = h5py.File('../data/all-flipped-cropped-512-crossval6div2_py-experiment.mat')
    experiments = scipy.io.loadmat('../data/all-flipped-cropped-512-crossval6all2-experiment.mat')

    return data, images, experiments['experiments_data']

def modifySpecular(scene, delta):
    for model in scene.objects:
        if model.type == 'MESH':
            for mat in model.data.materials:
                mat.specular_shader = 'PHONG'
                mat.specular_intensity = mat.specular_intensity + delta
                mat.specular_hardness = mat.specular_hardness / 4.0


def makeMaterial(name, diffuse, specular, alpha):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = diffuse
    mat.diffuse_shader = 'LAMBERT' 
    mat.diffuse_intensity = 1.0 
    mat.specular_color = specular
    mat.specular_shader = 'COOKTORR'
    mat.specular_intensity = 0.5
    mat.alpha = alpha
    mat.ambient = 1
    return mat
 

def setMaterial(ob, mat):
    me = ob.data
    me.materials.append(mat)

def look_at(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()

    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()


def modelHeight(scene):
    maxZ = -999999;
    minZ = 99999;
    for model in scene.objects:
        if model.type == 'MESH':
            for v in model.data.vertices:
                if (model.matrix_world * v.co).z > maxZ:
                    maxZ = (model.matrix_world * v.co).z
                if (model.matrix_world * v.co).z < minZ:
                    minZ = (model.matrix_world * v.co).z


    return minZ, maxZ

def modelWidth(scene):
    maxY = -999999;
    minY = 99999;
    for model in scene.objects:
        if model.type == 'MESH':
            for v in model.data.vertices:
                if (model.matrix_world * v.co).y > maxY:
                    maxY = (model.matrix_world * v.co).y
                if (model.matrix_world * v.co).y < minY:
                    minY = (model.matrix_world * v.co).y


    return minY, maxY


def centerOfGeometry(scene):
    center = mathutils.Vector((0.0,0.0,0.0))
    numVertices = 0.0
    for model in scene.objects:
        if model.type == 'MESH':
            numVertices = numVertices + len(model.data.vertices)
            for v in model.data.vertices:
                center = center + (model.matrix_world * v.co)


    return center/numVertices

def setEulerRotation(scene, eulerVectorRotation):
    for model in scene.objects:
        if model.type == 'MESH':
            model.rotation_euler = eulerVectorRotation

    scene.update()

def rotateMatrixWorld(scene, rotationMat):
    for model in scene.objects:
        if model.type == 'MESH':
            model.matrix_world = rotationMat * model.matrix_world

    scene.update()  