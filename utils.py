#!/usr/bin/env python3.4m
 
import bpy
import numpy
import mathutils
from math import radians
import h5py
import scipy.io

import sys
import io
import os
import pickle
import ipdb


inchToMeter = 0.0254

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

def loadGroundTruth():
    lines = [line.strip() for line in open('output/groundtruth.txt')]
    groundTruthLines = []
    imageFiles = []
    groundTruth = numpy.zeros([len(lines), 4])
    for instance in lines:
        parts = instance.split(' ')
        framestr = '{0:04d}'.format(int(parts[4]))
        outfilename = "scene_obj" + str(int(parts[3])) + "_" + framestr
        imageFile = "output/images/" + outfilename + ".png"
        if os.path.isfile(imageFile):
            imageFiles = imageFiles + [imageFile]
            groundTruthLines = groundTruthLines + [[float(parts[0]), float(parts[1]), float(parts[2]),int(parts[3]), int(parts[4])]]

    groundTruth = numpy.array(groundTruthLines)

    groundTruth = numpy.hstack((groundTruth,numpy.zeros((groundTruth.shape[0],1))))

    lines = [line.strip() for line in open('output/occlusions.txt')]

    for instance in lines:
        parts = instance.split(' ')
        index = numpy.where((groundTruth[:, 3] == int(parts[0])) & (groundTruth[:, 4] == int(parts[1])))[0][0]
        groundTruth[index, 5] = float(parts[2])
 
    return groundTruth, imageFiles

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
    loc_camera = obj_camera.location

    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()


def modelHeight(objects, transform):
    maxZ = -999999;
    minZ = 99999;
    for model in objects:
        if model.type == 'MESH':
            for v in model.data.vertices:
                if (transform * model.matrix_world * v.co).z > maxZ:
                    maxZ = (transform * model.matrix_world * v.co).z
                if (transform * model.matrix_world * v.co).z < minZ:
                    minZ = (transform * model.matrix_world * v.co).z


    return minZ, maxZ

def modelDepth(objects, transform):
    maxY = -999999;
    minY = 99999;
    for model in objects:
        if model.type == 'MESH':
            for v in model.data.vertices:
                if (transform * model.matrix_world * v.co).y > maxY:
                    maxY = (transform * model.matrix_world * v.co).y
                if (transform * model.matrix_world * v.co).y < minY:
                    minY = (transform * model.matrix_world * v.co).y


    return minY, maxY


def modelWidth(objects, transform):
    maxX = -999999;
    minX = 99999;
    for model in objects:
        if model.type == 'MESH':
            for v in model.data.vertices:
                if (transform * model.matrix_world * v.co).x > maxX:
                    maxX = (transform * model.matrix_world * v.co).x
                if (transform * model.matrix_world * v.co).x < minX:
                    minX = (transform * model.matrix_world * v.co).x


    return minX, maxX


def centerOfGeometry(objects, transform):
    center = mathutils.Vector((0.0,0.0,0.0))
    numVertices = 0.0
    for model in objects:
        if model.type == 'MESH':
            numVertices = numVertices + len(model.data.vertices)
            for v in model.data.vertices:
                center = center + (transform * model.matrix_world * v.co)


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


def AutoNodeOff():
    mats = bpy.data.materials
    for cmat in mats:
        cmat.use_nodes=False

def AutoNode():
    mats = bpy.data.materials
    for cmat in mats:
        #print(cmat.name)
        cmat.use_nodes=True
        TreeNodes=cmat.node_tree
        links = TreeNodes.links
    
        shader=''
        for n in TreeNodes.nodes:
    
            if n.type == 'ShaderNodeTexImage' or n.type == 'RGBTOBW':
                TreeNodes.nodes.remove(n)

            if n.type == 'OUTPUT_MATERIAL':
                shout = n       
                        
            if n.type == 'BACKGROUND':
                shader=n              
            if n.type == 'BSDF_DIFFUSE':
                shader=n  
            if n.type == 'BSDF_GLOSSY':
                shader=n              
            if n.type == 'BSDF_GLASS':
                shader=n  
            if n.type == 'BSDF_TRANSLUCENT':
                shader=n     
            if n.type == 'BSDF_TRANSPARENT':
                shader=n   
            if n.type == 'BSDF_VELVET':
                shader=n     
            if n.type == 'EMISSION':
                shader=n 
            if n.type == 'HOLDOUT':
                shader=n   

        if cmat.raytrace_mirror.use and cmat.raytrace_mirror.reflect_factor>0.001:
            print("MIRROR")
            if shader:
                if not shader.type == 'BSDF_GLOSSY':
                    print("MAKE MIRROR SHADER NODE")
                    TreeNodes.nodes.remove(shader)
                    shader = TreeNodes.nodes.new('BSDF_GLOSSY')    # RGB node
                    shader.location = 0,450
                    #print(shader.glossy)
                    links.new(shader.outputs[0],shout.inputs[0]) 
                        
        if not shader:
            shader = TreeNodes.nodes.new('BSDF_DIFFUSE')    # RGB node
            shader.location = 0,450
             
            shout = TreeNodes.nodes.new('OUTPUT_MATERIAL')
            shout.location = 200,400          
            links.new(shader.outputs[0],shout.inputs[0])                
                   
                   
                   
        if shader:                         
            textures = cmat.texture_slots
            for tex in textures:
                                
                if tex:
                    if tex.texture.type=='IMAGE':
                         
                        img = tex.texture.image
                        #print(img.name)  
                        shtext = TreeNodes.nodes.new('ShaderNodeTexImage')
          
                        shtext.location = -200,400 
        
                        shtext.image=img
        
                        if tex.use_map_color_diffuse:
                            links.new(shtext.outputs[0],shader.inputs[0]) 
    
                        if tex.use_map_normal:
                            t = TreeNodes.nodes.new('RGBTOBW')
                            t.location = -0,300 
                            links.new(t.outputs[0],shout.inputs[2]) 
                            links.new(shtext.outputs[0],t.inputs[0]) 



def cameraLookingInsideRoom(cameraAzimuth):
    if cameraAzimuth > 270 and cameraAzimuth < 90:
        return True
    return False


def setupScene(scene, modelInstances, targetIndex, roomName, world, distance, camera, width, height, numSamples, useCycles):
    if useCycles:
        #Switch Engine to Cycles
        scene.render.engine = 'CYCLES'
        AutoNode()
        # bpy.context.scene.render.engine = 'BLENDER_RENDER'

        cycles = bpy.context.scene.cycles

        cycles.samples = 1024
        cycles.max_bounces = 128
        cycles.min_bounces = 3
        cycles.caustics_reflective = True
        cycles.caustics_refractive = True
        cycles.diffuse_bounces = 128
        cycles.glossy_bounces = 128
        cycles.transmission_bounces = 128
        cycles.volume_bounces = 128
        cycles.transparent_min_bounces = 8
        cycles.transparent_max_bounces = 128



    scene.render.resolution_x = width #perhaps set resolution in code
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100


    scene.camera = camera
    camera.up_axis = 'Y'
    camera.data.angle = 60 * 180 / numpy.pi
    camera.data.clip_start = 0.01
    camera.data.clip_end = 10
    distance = 0.4

    center = centerOfGeometry(modelInstances[targetIndex].dupli_group.objects, modelInstances[targetIndex].matrix_world)
    # center = mathutils.Vector((0,0,0))
    # center = instances[targetIndex][1]

    originalLoc = mathutils.Vector((0,-distance , 0))
    elevation = 45.0
    azimuth = 0

    scene.render.use_raytrace = False
    scene.render.use_shadows = False

    elevationRot = mathutils.Matrix.Rotation(radians(-elevation), 4, 'X')
    azimuthRot = mathutils.Matrix.Rotation(radians(-azimuth), 4, 'Z')
    location = center + azimuthRot * elevationRot * originalLoc
    camera.location = location

    # lamp_data2 = bpy.data.lamps.new(name="LampBotData", type='POINT')
    # lamp2 = bpy.data.objects.new(name="LampBot", object_data=lamp_data2)
    # lamp2.location = targetParentPosition + mathutils.Vector((0,0,1.5))
    # lamp2.data.energy = 0.00010
    # # lamp.data.size = 0.25
    # lamp2.data.use_diffuse = True
    # lamp2.data.use_specular = True
    # # scene.objects.link(lamp2)
        

        # # toggle lamps
        # if obj.type == 'LAMP':
        #     obj.cycles_visibility.camera = not obj.cycles_visibility.camera
    roomInstance = scene.objects[roomName]

    ceilMinX, ceilMaxX = modelWidth(roomInstance.dupli_group.objects, roomInstance.matrix_world)
    ceilWidth = (ceilMaxX - ceilMinX)
    ceilMinY, ceilMaxY = modelDepth(roomInstance.dupli_group.objects, roomInstance.matrix_world)
    ceilDepth = (ceilMaxY - ceilMinY) 
    ceilMinZ, ceilMaxZ = modelHeight(roomInstance.dupli_group.objects, roomInstance.matrix_world)
    ceilPos =  mathutils.Vector(((ceilMaxX + ceilMinX) / 2.0, (ceilMaxY + ceilMinY) / 2.0 , ceilMaxZ))

    numLights = int(numpy.floor((ceilWidth-0.2)/1.15))
    lightInterval = ceilWidth/numLights

    for light in range(numLights):
        lightXPos = light*lightInterval + lightInterval/2.0
        lamp_data = bpy.data.lamps.new(name="Rect", type='AREA')
        lamp = bpy.data.objects.new(name="Rect", object_data=lamp_data)
        lamp.data.size = 0.15
        lamp.data.size_y = ceilDepth - 0.2
        lamp.data.shape = 'RECTANGLE'
        lamp.location = mathutils.Vector((ceilPos.x - ceilWidth/2.0 + lightXPos, ceilPos.y, ceilMaxZ))
        lamp.data.energy = 0.0005

        if useCycles:
            lamp.data.cycles.use_multiple_importance_sampling = True
            lamp.data.use_nodes = True
            lamp.data.node_tree.nodes['Emission'].inputs[1].default_value = 20
        scene.objects.link(lamp)
        lamp.layers[1] = True

    world.light_settings.use_environment_light = False
    # world.light_settings.environment_energy = 1
    world.horizon_color = mathutils.Color((0.0,0.0,0.0))
    # world.light_settings.samples = 20
    world.light_settings.use_ambient_occlusion = True
    world.light_settings.ao_blend_type = 'MULTIPLY'
    world.light_settings.ao_factor = 1
    world.light_settings.use_indirect_light = True
    world.light_settings.gather_method = 'APPROXIMATE'
    scene.world = world

    scene.update()

    look_at(camera, center)

    bpy.ops.scene.render_layer_add()

    camera.layers[1] = True
    scene.render.layers[0].use_pass_object_index = True
    scene.render.layers[1].use_pass_object_index = True
    scene.render.layers[1].use_pass_combined = False

    scene.layers[1] = False
    scene.layers[0] = True
    scene.render.layers[0].use = True
    scene.render.layers[1].use = False