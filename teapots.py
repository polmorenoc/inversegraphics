#!/usr/bin/env python3.4m
 
import bpy
import numpy
from PIL import Image
import mathutils
from math import radians

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



baseDir = '../databaseFull/models/'

lines = [line.strip() for line in open('teapots.txt')]

for object in bpy.data.scenes['Scene'].objects: print(object.name)

lamp = bpy.data.scenes['Scene'].objects[1]
lamp.location = (0,0.0,2)

camera = bpy.data.scenes['Scene'].objects[2]

camera.data.angle = 60 * 180 / numpy.pi
location = (0,-0.5,0.0)
location = mathutils.Matrix.Rotation(radians(-45.0), 4, 'X') * mathutils.Vector(location)
camera.location = location

look_at(camera, mathutils.Vector((0,0,0)))


# red = makeMaterial('Red', (1,0,0), (1,1,1), 1)

# for item in bpy.data.objects:
#     if item.type == 'MESH':
#         for mat in item.data.materials:
#             mat = red

world = bpy.context.scene.world

# World settings
# world.use_sky_blend = True

# Environment lighting
world.light_settings.use_environment_light = True
world.light_settings.environment_energy = 0.10
world.horizon_color = mathutils.Color((0.0,0.0,0.0))
# wset.use_environment_light = True
# wset.use_ambient_occlusion = False
# wset.ao_blend_type = 'MULTIPLY'
# wset.ao_factor = 0.8
# wset.gather_method = 'APPROXIMATE'

# cycles = bpy.context.scene.cycles

# cycles.max_bounces = 128
# cycles.min_bounces = 3
# cycles.caustics_reflective = True
# cycles.caustics_refractive = True
# cycles.diffuse_bounces = 128
# cycles.glossy_bounces = 128
# cycles.transmission_bounces = 128
# cycles.volume_bounces = 128
# cycles.transparent_min_bounces = 8
# cycles.transparent_max_bounces = 128



for teapot in lines:
    

    fullTeapot = baseDir + teapot

    print("Reading " + fullTeapot + '.dae')

    bpy.ops.scene.new()
    bpy.context.scene.name = teapot
    scene = bpy.context.scene

    scene.objects.link(lamp)
    # scene.objects.link(camera)

    scene.camera = camera

    scene.render.use_raytrace = True
    # scene.render.use_full_sample = True
    scene.render.antialiasing_samples = '16'


    # scene.render.engine = 'CYCLES'

    # cycles = scene.cycles
    # cycles.samples = 128
    # cycles.max_bounces = 128
    # cycles.min_bounces = 3
    # cycles.caustics_reflective = True
    # cycles.caustics_refractive = True
    # cycles.diffuse_bounces = 128
    # cycles.glossy_bounces = 128
    # cycles.transmission_bounces = 128
    # cycles.volume_bounces = 128
    # cycles.transparent_min_bounces = 8
    # cycles.transparent_max_bounces = 128

    scene.render.filepath = teapot + '.png'

    bpy.utils.collada_import(fullTeapot + '.dae')
    # bpy.ops.import_scene.autodesk_3ds(filepath=fullTeapot + '.3DS')

    minZ, maxZ = modelHeight(scene)

    scale = 0.15/(maxZ-minZ)

    for mesh in scene.objects:
        if mesh.type == 'MESH':
            scaleMat = mathutils.Matrix.Scale(scale, 4)
            mesh.matrix_world =  scaleMat * mesh.matrix_world
                 
            
    # scene.update()
    minZ, maxZ = modelHeight(scene)


    for mesh in scene.objects:
        if mesh.type == 'MESH':
                # mesh.location = mesh.location - mathutils.Vector((0,0,-minZ))
            mesh.matrix_world = mathutils.Matrix.Translation((0,0,-minZ)) * mesh.matrix_world
    
    # scene.update()
    # scene.world = world

    for object in bpy.context.scene.objects: print(object.name)


    width = 300
    height = 300

    scene.render.resolution_x = width #perhaps set resolution in code
    scene.render.resolution_y = height



    camera.data.angle = 60 * 180 / numpy.pi
    
    camera.location = location

    look_at(camera, mathutils.Vector((0,0,0.1)))


    # red = makeMaterial('Red', (1,0,0), (1,1,1), 1)

    # for item in bpy.data.objects:
    #     if item.type == 'MESH':
    #         for mat in item.data.materials:
    #             mat = red

    scene.world = world

    # Environment lighting
    # scene.world.light_settings.use_environment_light = True

    bpy.ops.render.render( write_still=False )

    blendImage = bpy.data.images['Render Result']

    image = numpy.flipud(numpy.array(blendImage.extract_render(scene=scene)).reshape([height/2,width/2,4]))

    # Truncate intensities larger than 1.
    image[numpy.where(image > 1)] = 1

    im = Image.fromarray(numpy.uint8(image*255))

    im.save(teapot + '_2.png')

