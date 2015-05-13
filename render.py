#!/usr/bin/env python3.4m
 
import sceneimport
from utils import * 

inchToMeter = 0.0254

sceneFile = '../databaseFull/scenes/scene00051.txt' 
targetIndex = 9
world = bpy.context.scene.world

instances = sceneimport.loadScene(sceneFile)

# targets = sceneimport.loadTargetModels()
[blenderScenes, modelInstances] = sceneimport.importBlenderScenes(instances, targetIndex)

scene = sceneimport.composeScene(modelInstances, targetIndex)

scene.update()
bpy.context.screen.scene = scene

#Switch Engine to Cycles
# bpy.context.scene.render.engine = 'CYCLES'
# AutoNode()
# bpy.context.scene.render.engine = 'BLENDER_RENDER'

# cycles = bpy.context.scene.cycles

# cycles.samples = 4
# cycles.max_bounces = 128
# cycles.min_bounces = 3
# cycles.caustics_reflective = True
# cycles.caustics_refractive = True
# cycles.diffuse_bounces = 128
# cycles.transmission_bounces = 128
# cycles.volume_bounces = 128
# cycles.transparent_min_bounces = 8
# cycles.transparent_max_bounces = 128

width = 600
height = 600

scene.render.resolution_x = width #perhaps set resolution in code
scene.render.resolution_y = height

camera = bpy.data.scenes['Scene'].objects[2]
scene.camera = camera
camera.up_axis = 'Z'
camera.data.angle = 60 * 180 / numpy.pi
distance = 2

center = centerOfGeometry(modelInstances[targetIndex].dupli_group.objects, modelInstances[targetIndex].matrix_world)
# center = mathutils.Vector((0,0,0))
# center = instances[targetIndex][1]

originalLoc = mathutils.Vector((0,-distance , 0))
elevation = 45.0
azimuth = 0

scene.render.use_raytrace = True


elevationRot = mathutils.Matrix.Rotation(radians(-elevation), 4, 'X')
azimuthRot = mathutils.Matrix.Rotation(radians(-azimuth), 4, 'Z')
location = center + azimuthRot * elevationRot * originalLoc
camera.location = location

lamp_data2 = bpy.data.lamps.new(name="LampBotData", type='POINT')
lamp2 = bpy.data.objects.new(name="LampBot", object_data=lamp_data2)
lamp2.location = location
lamp2.data.energy = 1
# lamp.data.size = 0.25
lamp2.data.use_diffuse = True
lamp2.data.use_specular = True
scene.objects.link(lamp2)
# world.light_settings.use_environment_light = False
# world.light_settings.environment_energy = 5
# world.horizon_color = mathutils.Color((0.0,0.0,0.0))
# world.light_settings.samples = 20
# scene.world = world

scene.update()

look_at(camera, center)

bpy.ops.scene.render_layer_add()

# for c in range(2,3):
#     scene.objects[c].layers[1] = True

# bpy.data.objects['room09'].layers[0] = False
bpy.data.objects['room09'].layers[1] = True
# scene.render.use_compositing = False

# for ob in modelInstances[0].dupli_group.objects:
#     ob.layers[1] = False

camera.layers[1] = True
lamp2.layers[1] = True
scene.render.layers[0].use_pass_object_index = True
scene.render.layers[1].use_pass_object_index = True
scene.render.layers[1].use_pass_combined = False


scene.layers[1] = False
scene.layers[0] = True
scene.render.layers[0].use = True
scene.render.layers[1].use = False

# azimuths = [0,5,10,20,30,40,50]
# for azimuths in positions:
#     bpy.context.scene.frame_set(frame_num)
#     ob.location = position
#     ob.keyframe_insert(data_path="location", index=-1)
#     frame_num += 10


scene.update()

minZ, maxZ = modelHeight(scene)


# scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = 'scene.png'
bpy.ops.render.render( write_still=True )


# for target in targets:


# modifySpecular(scene, 0.3)

# # ipdb.set_trace()

# minZ, maxZ = modelHeight(scene)

# minY, maxY = modelWidth(scene)

# scaleZ = 0.254/(maxZ-minZ)
# scaleY = 0.1778/(maxY-minY)

# scale = min(scaleZ, scaleY)

# for mesh in scene.objects:
#     if mesh.type == 'MESH':
#         scaleMat = mathutils.Matrix.Scale(scale, 4)
#         mesh.matrix_world =  scaleMat * mesh.matrix_world
             
# minZ, maxZ = modelHeight(scene)
# scene.objects.link(lamp2)

# scene.objects.link(lamp)

# # lamp2.location = (0,0, 2)


# center = centerOfGeometry(scene)
# for mesh in scene.objects:
#     if mesh.type == 'MESH':
#         mesh.matrix_world = mathutils.Matrix.Translation(-center) * mesh.matrix_world