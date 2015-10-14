#!/usr/bin/env python3.4m
 
import bpy
import bpy_extras
import numpy
import numpy as np
import mathutils
from math import radians
import h5py
import scipy.io
import cv2
import sys
import io
import os
import pickle
import ipdb
import re
from collision import instancesIntersect

inchToMeter = 0.0254

def addLamp(scene, lightAz, lightEl, lightDist, center, lightIntensity):
        #Add directional light to match spherical harmonics
    lamp_data = bpy.data.lamps.new(name="point", type='POINT')
    lamp = bpy.data.objects.new(name="point", object_data=lamp_data)
    lamp.layers[1] = True
    lamp.layers[2] = True
    lampLoc = getRelativeLocation(lightAz, lightAz, lightEl, center)
    lamp.location = mathutils.Vector((lampLoc[0],lampLoc[1],lampLoc[2]))
    lamp.data.cycles.use_multiple_importance_sampling = True
    lamp.data.use_nodes = True
    lamp.data.node_tree.nodes['Emission'].inputs[1].default_value = lightIntensity
    scene.objects.link(lamp)

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

def loadGroundTruth(rendersDir):
    lines = [line.strip() for line in open(rendersDir + 'groundtruth.txt')]
    groundTruthLines = []
    imageFiles = []
    segmentFiles = []
    segmentSingleFiles = []
    unoccludedFiles = []
    prefixes = []
    for instance in lines:
        parts = instance.split(' ')
        framestr = '{0:04d}'.format(int(parts[4]))
        prefix = ''
        az = float(parts[0])
        objAz = float(parts[1])
        el = float(parts[2])
        objIndex = int(parts[3])
        frame = int(parts[4])
        sceneNum = int(parts[5])
        targetIndex = int(parts[6])

        spoutPosX = int(float(parts[7]))
        spoutPosY = int(float(parts[8]))
        handlePosX = int(float(parts[9]))
        handlePosY = int(float(parts[10]))
        tipPosX = int(float(parts[11]))
        tipPosY = int(float(parts[12]))
        spoutOccluded = int(float(parts[13]))
        handleOccluded = int(float(parts[14]))
        tipOccluded = int(float(parts[15]))

        if len(parts) == 17:
            prefix = parts[16]
        outfilename = "render" + prefix + "_obj" + str(objIndex) + "_scene" + str(sceneNum) + '_target' + str(targetIndex) + '_' + framestr
        outfilenamesingle = "render" + prefix + "_obj" + str(objIndex) + "_scene" + str(sceneNum) + '_target' + str(targetIndex) + '_single_' + framestr
        outfilenameunoccluded = "render" + prefix + "_obj" + str(objIndex) + "_scene" + str(sceneNum) + '_target' + str(targetIndex) + '_unoccluded' + framestr
        imageFile = rendersDir + "images/" +  outfilename + ".png"
        segmentFile =  rendersDir + "images/" +  outfilename + "_segment.png"
        segmentFileSingle =  rendersDir + "images/" +  outfilenamesingle + "_segment.png"
        unoccludedFile =  rendersDir + "images/" +  outfilenameunoccluded + ".png"
        if os.path.isfile(imageFile):
            imageFiles = imageFiles + [imageFile]
            segmentFiles = segmentFiles + [segmentFile]
            segmentSingleFiles = segmentSingleFiles + [segmentFileSingle]
            unoccludedFiles = unoccludedFiles + [unoccludedFile]
            prefixes = prefixes + [prefix]
            groundTruthLines = groundTruthLines + [[az, objAz, el, objIndex, frame, 0.0, sceneNum, targetIndex, spoutPosX, spoutPosY, handlePosX, handlePosY, tipPosX, tipPosY, spoutOccluded, handleOccluded, tipOccluded]]

    # groundTruth = numpy.zeros([len(groundTruthLines), 5])
    groundTruth = numpy.array(groundTruthLines)

    # groundTruth = numpy.hstack((groundTruth,numpy.zeros((groundTruth.shape[0],1))))

    lines = [line.strip() for line in open(rendersDir + 'occlusions.txt')]

    for instance in lines:
        parts = instance.split(' ')
        prefix = ''
        if len(parts) == 6:
            prefix = parts[5]
        eqPrefixes = [ x==y for (x,y) in zip(prefixes, [prefix]*len(prefixes))]
        try:
            index = numpy.where((groundTruth[:, 3] == int(parts[0])) & (groundTruth[:, 4] == int(parts[1])) & (groundTruth[:,6] == int(parts[2])) & (groundTruth[:,7] == int(parts[3])) & (eqPrefixes))[0][0]
            groundTruth[index, 5] = float(parts[4])
        except:
            print("Problem!")


 
    return groundTruth, imageFiles, segmentFiles, segmentSingleFiles, unoccludedFiles, prefixes

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
    vecPoint = mathutils.Vector((point[0],point[1],point[2]))
    direction = vecPoint - loc_camera
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

def cleanBPYScene(scene):

    for blenderScene in bpy.data.scenes:
        if blenderScene != scene:
            if len(blenderScene.name) > 7 and blenderScene.name[0:7] == 'teapots':
                bpy.data.scenes.remove(blenderScene)


def addEnvironmentMapWorld(envMapFilename, scene):
    scene.world.use_nodes = True
    treeNodes=scene.world.node_tree
    envTextureNode = treeNodes.nodes.new('ShaderNodeTexEnvironment')
    mappingNode = treeNodes.nodes.new('ShaderNodeMapping')
    links = treeNodes.links
    links.new(mappingNode.outputs[0],envTextureNode.inputs[0])


    texCoordNode = treeNodes.nodes.new('ShaderNodeTexCoord')
    links.new(texCoordNode.outputs[0],mappingNode.inputs[0])
    # mathNode = treeNodes.nodes.new('ShaderNodeMath')
    # links.new(envTextureNode.outputs[0],mathNode.inputs[0])
    rgbToBWNode = treeNodes.nodes.new('ShaderNodeRGBToBW')
    # links.new(envTextureNode.outputs[0],treeNodes.nodes['Background'].inputs[0])
    links.new(envTextureNode.outputs[0],rgbToBWNode.inputs[0])
    links.new(rgbToBWNode.outputs[0],treeNodes.nodes['Background'].inputs[0])

    # mathNode.inputs[1].default_value = 1
    image = bpy.data.images.load(envMapFilename)
    envTextureNode.image = image


def setEnviornmentMapStrength(strength, scene):
    backgroundNode = scene.world.node_tree.nodes['Background']
    backgroundNode.inputs[1].default_value = strength

def updateEnviornmentMap(envMapFilename, scene):
    envTextureNode = scene.world.node_tree.nodes['Environment Texture']
    envTextureNode.image.user_clear()
    bpy.data.images.remove(envTextureNode.image)
    image = bpy.data.images.load(envMapFilename)
    envTextureNode.image = image

def rotateEnviornmentMap(angle, scene):
    mappingNode = scene.world.node_tree.nodes['Mapping']
    mappingNode.rotation[2] = angle

def cameraLookingInsideRoom(cameraAzimuth):
    if cameraAzimuth > 270 and cameraAzimuth < 90:
        return True
    return False

def deleteInstance(instance):

    for mesh in instance.dupli_group.objects:
        mesh.user_clear()
        bpy.data.objects.remove(mesh)

    instance.dupli_group.user_clear()
    bpy.data.groups.remove(instance.dupli_group)
    instance.user_clear()
    bpy.data.objects.remove(instance)


def placeNewTarget(scene, target, targetPosition):
    target.layers[1] = True
    target.layers[2] = True
    scene.objects.link(target)
    target.matrix_world = mathutils.Matrix.Translation(targetPosition)
    center = centerOfGeometry(target.dupli_group.objects, target.matrix_world)
    original_matrix_world = target.matrix_world.copy()
    camera = scene.camera
    scene.update()
    look_at(camera, center)
    scene.update()

def placeCamera(camera, azimuth, elevation, camDistance, center):
    location = getRelativeLocation(azimuth, elevation, camDistance, center)
    camera.location = location
    look_at(camera, center)

def getRelativeLocation(azimuth, elevation, distance, center):
    azimuthRot = mathutils.Matrix.Rotation(radians(-azimuth), 4, 'Z')
    elevationRot = mathutils.Matrix.Rotation(radians(-elevation), 4, 'X')
    originalLoc = mathutils.Vector((0,-distance, 0))
    location = center + azimuthRot * elevationRot * originalLoc
    return location

def setupScene(scene, roomInstanceNum, world, camera, width, height, numSamples, useCycles, useGPU):

    if useCycles:
        #Switch Engine to Cycles
        scene.render.engine = 'CYCLES'
        if useGPU:
            bpy.context.scene.cycles.device = 'GPU'
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'
            bpy.context.user_preferences.system.compute_device = 'CUDA_MULTI_0'

        scene.use_nodes = True

        AutoNode()
        # bpy.context.scene.render.engine = 'BLENDER_RENDER'

        cycles = bpy.context.scene.cycles

        cycles.samples = 512
        cycles.max_bounces = 36
        cycles.min_bounces = 4
        cycles.caustics_reflective = False
        cycles.caustics_refractive = False
        cycles.diffuse_bounces = 36
        cycles.glossy_bounces = 12
        cycles.transmission_bounces = 2
        cycles.volume_bounces = 12
        cycles.transparent_min_bounces = 2
        cycles.transparent_max_bounces = 2

        world.cycles_visibility.camera = False
        world.use_nodes = True

        world.cycles.sample_as_light = True
        world.cycles.sample_map_resolution = 2048

    scene.render.threads = 4
    scene.render.tile_x = height/2
    scene.render.tile_y = width/2

    scene.render.image_settings.compression = 0
    scene.render.resolution_x = width #perhaps set resolution in code
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100

    scene.camera = camera
    # scene.objects.link(camera)

    camera.up_axis = 'Y'
    camera.data.angle = 60 * 180 / numpy.pi
    camera.data.clip_start = 0.01
    camera.data.clip_end = 10

    roomInstance = scene.objects[str(roomInstanceNum)]

    if useCycles:
        roomInstance.cycles_visibility.shadow = False

    scene.world = world
    scene.world.light_settings.distance = 0.1

    if not useCycles:
        scene.render.use_raytrace = False
        scene.render.use_shadows = False

        # scene.view_settings.exposure = 5
        # scene.view_settings.gamma = 0.5
        scene.world.light_settings.use_ambient_occlusion = True
        scene.world.light_settings.ao_blend_type = 'ADD'
        scene.world.light_settings.use_indirect_light = True
        scene.world.light_settings.indirect_bounces = 1
        scene.world.light_settings.use_cache = True

        scene.world.light_settings.ao_factor = 1
        scene.world.light_settings.indirect_factor = 1
        scene.world.light_settings.gather_method = 'APPROXIMATE'

    world.light_settings.use_environment_light = False
    world.light_settings.environment_energy = 0.0
    world.horizon_color = mathutils.Color((0.0,0.0,0.0))
    # world.light_settings.samples = 20

    # world.light_settings.use_ambient_occlusion = False
    #
    # world.light_settings.ao_factor = 1
    # world.exposure = 1.1
    # world.light_settings.use_indirect_light = True

    scene.sequencer_colorspace_settings.name = 'Raw'
    scene.update()

    bpy.ops.scene.render_layer_add()
    bpy.ops.scene.render_layer_add()

    camera.layers[1] = True
    scene.render.layers[0].use_pass_object_index = True
    scene.render.layers[1].use_pass_object_index = True
    scene.render.layers[1].use_pass_combined = True
    camera.layers[2] = True
    scene.layers[1] = False
    scene.layers[2] = False
    scene.layers[0] = True
    scene.render.layers[0].use = True
    scene.render.layers[1].use = False
    scene.render.layers[2].use = False
    scene.render.use_sequencer = False


def addAmbientLightingScene(scene, useCycles):

    roomName = ''
    for model in scene.objects:
        reg = re.compile('(room[0-9]+)')
        res = reg.match(model.name)
        if res:
            roomName = res.groups()[0]

    roomInstance = scene.objects[roomName]
    ceilMinX, ceilMaxX = modelWidth(roomInstance.dupli_group.objects, roomInstance.matrix_world)
    ceilWidth = (ceilMaxX - ceilMinX)
    ceilMinY, ceilMaxY = modelDepth(roomInstance.dupli_group.objects, roomInstance.matrix_world)
    ceilDepth = (ceilMaxY - ceilMinY)
    ceilMinZ, ceilMaxZ = modelHeight(roomInstance.dupli_group.objects, roomInstance.matrix_world)
    ceilPos =  mathutils.Vector(((ceilMaxX + ceilMinX) / 2.0, (ceilMaxY + ceilMinY) / 2.0 , ceilMaxZ))

    numLights = int(numpy.floor((ceilWidth-0.2)/1.2))
    lightInterval = ceilWidth/numLights

    for light in range(numLights):
        lightXPos = light*lightInterval + lightInterval/2.0
        lamp_data = bpy.data.lamps.new(name="Rect", type='AREA')
        lamp = bpy.data.objects.new(name="Rect", object_data=lamp_data)
        lamp.data.size = 0.2
        lamp.data.size_y = ceilDepth - 0.2
        lamp.data.shape = 'RECTANGLE'
        lamp.location = mathutils.Vector((ceilPos.x - ceilWidth/2.0 + lightXPos, ceilPos.y, ceilMaxZ))
        lamp.data.energy = 0.0025
        if useCycles:
            lamp.data.cycles.use_multiple_importance_sampling = True
            lamp.data.use_nodes = True
            lamp.data.node_tree.nodes['Emission'].inputs[1].default_value = 30

        scene.objects.link(lamp)
        lamp.layers[1] = True
        lamp.layers[2] = True


def targetSceneCollision(target, scene):

    for sceneInstance in scene.objects:
        if sceneInstance.type == 'EMPTY' and sceneInstance != target and sceneInstance.name != roomName and sceneInstance != targetParentInstance:
            if instancesIntersect(target, sceneInstance):
                return True

    return False


def view_plane(camd, winx, winy, xasp, yasp):
    #/* fields rendering */
    ycor = yasp / xasp
    use_fields = False
    if (use_fields):
      ycor *= 2

    def BKE_camera_sensor_size(p_sensor_fit, sensor_x, sensor_y):
        #/* sensor size used to fit to. for auto, sensor_x is both x and y. */
        if (p_sensor_fit == 'VERTICAL'):
            return sensor_y;

        return sensor_x;

    if (camd.type == 'ORTHO'):
      #/* orthographic camera */
      #/* scale == 1.0 means exact 1 to 1 mapping */
      pixsize = camd.ortho_scale
    else:
      #/* perspective camera */
      sensor_size = BKE_camera_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
      pixsize = (sensor_size * camd.clip_start) / camd.lens

    #/* determine sensor fit */
    def BKE_camera_sensor_fit(p_sensor_fit, sizex, sizey):
        if (p_sensor_fit == 'AUTO'):
            if (sizex >= sizey):
                return 'HORIZONTAL'
            else:
                return 'VERTICAL'

        return p_sensor_fit

    sensor_fit = BKE_camera_sensor_fit(camd.sensor_fit, xasp * winx, yasp * winy)

    if (sensor_fit == 'HORIZONTAL'):
      viewfac = winx
    else:
      viewfac = ycor * winy

    pixsize /= viewfac

    #/* extra zoom factor */
    pixsize *= 1 #params->zoom

    #/* compute view plane:
    # * fully centered, zbuffer fills in jittered between -.5 and +.5 */
    xmin = -0.5 * winx
    ymin = -0.5 * ycor * winy
    xmax =  0.5 * winx
    ymax =  0.5 * ycor * winy

    #/* lens shift and offset */
    dx = camd.shift_x * viewfac # + winx * params->offsetx
    dy = camd.shift_y * viewfac # + winy * params->offsety

    xmin += dx
    ymin += dy
    xmax += dx
    ymax += dy

    #/* fields offset */
    #if (params->field_second):
    #    if (params->field_odd):
    #        ymin -= 0.5 * ycor
    #        ymax -= 0.5 * ycor
    #    else:
    #        ymin += 0.5 * ycor
    #        ymax += 0.5 * ycor

    #/* the window matrix is used for clipping, and not changed during OSA steps */
    #/* using an offset of +0.5 here would give clip errors on edges */
    xmin *= pixsize
    xmax *= pixsize
    ymin *= pixsize
    ymax *= pixsize

    return xmin, xmax, ymin, ymax


def projection_matrix(camd, scene):
    r = scene.render
    left, right, bottom, top = view_plane(camd, r.resolution_x, r.resolution_y, 1, 1)

    farClip, nearClip = camd.clip_end, camd.clip_start

    Xdelta = right - left
    Ydelta = top - bottom
    Zdelta = farClip - nearClip

    mat = [[0]*4 for i in range(4)]

    mat[0][0] = nearClip * 2 / Xdelta
    mat[1][1] = nearClip * 2 / Ydelta
    mat[2][0] = (right + left) / Xdelta #/* note: negate Z  */
    mat[2][1] = (top + bottom) / Ydelta
    mat[2][2] = -(farClip + nearClip) / Zdelta
    mat[2][3] = -1
    mat[3][2] = (-2 * nearClip * farClip) / Zdelta
    # ipdb.set_trace()
    # return sum([c for c in mat], [])
    projMat = mathutils.Matrix(mat)
    return projMat.transposed()

def image_projection(scene, point):
    p4d = mathutils.Vector.Fill(4, 1)
    p4d.x = point.x
    p4d.y = point.y
    p4d.z = point.z
    projectionMat = projection_matrix(scene.camera.data, scene)

    proj = projectionMat * scene.camera.matrix_world.inverted() * p4d
    return [scene.render.resolution_x*(proj.x/proj.w + 1)/2, scene.render.resolution_y*(proj.y/proj.w + 1)/2]

def image_project(scene, camera, point):
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, point)

    # print("2D Coords:", co_2d)

    # If you want pixel coords
    render_scale = scene.render.resolution_percentage / 100
    render_size = ( int(scene.render.resolution_x * render_scale), int(scene.render.resolution_y * render_scale))
    return (round(co_2d.x * render_size[0]), round(co_2d.y * render_size[1]))

#Need to verify!
def closestCameraIntersection(scene, point):
    for instance in scene.objects:

        if instance.type == 'EMPTY' and instance.dupli_type == 'GROUP':
            instanceLoc = numpy.array(instance.location)
            camLoc = numpy.array(scene.camera.location)
            pointLoc = numpy.array(point)
            invInstanceTransf = instance.matrix_world.inverted()
            localCamTmp = invInstanceTransf * scene.camera.location
            if numpy.linalg.norm(instanceLoc - camLoc) < numpy.linalg.norm(pointLoc - camLoc) and (instanceLoc - camLoc).dot(pointLoc - camLoc) > 0:
                for mesh in instance.dupli_group.objects:
                    if mesh.type == 'MESH':
                        invMeshTransf = mesh.matrix_world.inverted()
                        localCam = invMeshTransf * localCamTmp
                        localPoint = invMeshTransf * invInstanceTransf * point

                        location, normal, index = mesh.ray_cast(localCam, localPoint)
                        if index != -1:
                            #Success.
                            return True

    return False

def sceneIntersection(scene, point):
    result, object, matrix, location, normal = scene.ray_cast(scene.camera.location, point)

    return result

# def flattenMesh(mesh, transform):
#     return
# def flattenInstance(instance, transform):


def setupBlender(teapot, width, height, angle, clip_start, clip_end, camDistance):
    cam = bpy.data.cameras.new("MainCamera")
    camera = bpy.data.objects.new("MainCamera", cam)
    world = bpy.data.worlds.new("MainWorld")
    bpy.ops.scene.new()
    bpy.context.scene.name = 'Main Scene'
    scene = bpy.context.scene
    scene.objects.link(teapot)
    scene.camera = camera
    camera.up_axis = 'Y'
    camera.data.angle = angle
    camera.data.clip_start = clip_start
    camera.data.clip_end = clip_end
    scene.world = world
    world.light_settings.use_environment_light = False
    world.light_settings.environment_energy = 0.0
    world.horizon_color = mathutils.Color((0.0,0.0,0.0))
    scene.render.resolution_x = width #perhaps set resolution in code
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.update()
    lamp_data2 = bpy.data.lamps.new(name="LampBotData", type='POINT')
    lamp2 = bpy.data.objects.new(name="LampBot", object_data=lamp_data2)
    lamp2.location = mathutils.Vector((0,0,1.5))
    lamp2.data.energy = 1
    scene.objects.link(lamp2)
    lamp_data2 = bpy.data.lamps.new(name="LampBotData2", type='POINT')
    lamp2 = bpy.data.objects.new(name="LampBot2", object_data=lamp_data2)
    lamp2.location = mathutils.Vector((0,0,-1.5))
    lamp2.data.energy = 0.5
    scene.objects.link(lamp2)
    bpy.context.screen.scene = scene
    # teapot.matrix_world = mathutils.Matrix.Translation(mathutils.Vector((0,0,0)))
    center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)
    azimuth = 90
    elevation = 25
    azimuthRot = mathutils.Matrix.Rotation(radians(-azimuth), 4, 'Z')
    elevationRot = mathutils.Matrix.Rotation(radians(-elevation), 4, 'X')
    originalLoc = mathutils.Vector((0,-camDistance, 0))
    location = center + azimuthRot * elevationRot * originalLoc
    camera.location = location
    scene.update()
    look_at(camera, center)
    scene.update()

    return scene


###########################################################
#
#   Round values of the 3D vector
#
###########################################################

def r3d(v):
    return round(v[0],6), round(v[1],6), round(v[2],6)

###########################################################
#
#   Round values of the 2D vector
#
###########################################################

def r2d(v):
    return round(v[0],6), round(v[1],6)


###########################################################
#
#   Convert object name to be suitable for C definition
#
###########################################################

def clearName(name):
    tmp=name.upper()
    ret=""
    for i in tmp:
        if (i in " ./\-+#$%^!@"):
            ret=ret+"_"
        else:
            ret=ret+i
    return ret


###########################################################
#
#   Build data for each object (MESH)
#
###########################################################

def getDrWrtAzimuth(SqE_raw, rotation):
    rot, rot_dr = cv2.Rodrigues(np.array(rotation))
    a = mathutils.Matrix(rot).to_euler()[2]
    a2 = np.arctan(rot[1,0]/rot[0,0])
    b = mathutils.Matrix(rot).to_euler()[1]
    b2 = np.arctan(-rot[2,0]/np.sqrt(rot[2,1]**2 + rot[2,2]**2))
    g = mathutils.Matrix(rot).to_euler()[0]
    g2 = np.arctan(rot[2,1]/rot[2,2])

    dra11 = -np.sin(a)*np.cos(b)
    dra12 = -np.sin(a)*np.sin(b)*np.sin(g) - np.sin(a)*np.cos(g)
    dra13 = -np.sin(a)*np.sin(b)*np.cos(g) + np.cos(a)*np.sin(g)
    dra21 = np.cos(a)*np.cos(b)
    dra22 = np.cos(a)*np.sin(b)*np.sin(g) - np.sin(a)*np.cos(g)
    dra23 = np.cos(a)*np.sin(b)*np.cos(g) + np.sin(a)*np.cos(g)
    dra31 = 0
    dra32 = 0
    dra33 = 0

    rotwrtaz = np.array([[dra11,dra12,dra13], [dra21,dra22,dra23], [dra31,dra32,dra33]])
    # ipdb.set_trace()
    drazimuth = np.dot(SqE_raw.dr_wrt(rotation), np.dot(rot_dr , rotwrtaz.ravel()))/(2*400*400)

    return drazimuth, np.array([a,g,b])

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
