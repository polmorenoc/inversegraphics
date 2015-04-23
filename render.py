#!/usr/bin/env python3.4m
 
import bpy
import numpy
from PIL import Image

# bpy.ops.scene.new(type='EMPTY')

# bpy.data.scenes['Scene'].objects.unlink(bpy.context.active_object)

# for item in bpy.data.objects:
#     if item.type == 'MESH':
#         item.user_clear()
#         bpy.data.objects.remove(item)


#Switch Engine to Cycles
bpy.context.scene.render.engine = 'CYCLES'
#bpy.context.scene.render.engine = 'BLENDER_RENDER'

cycles = bpy.context.scene.cycles

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

 
#tell blender to use CUDA / GPU devices
#bpy.context.user_preferences.system.compute_device_type = 'CUDA'

    # red = makeMaterial('Red', (1,0,0), (1,1,1), 1)

    # for item in bpy.data.objects:
    #     if item.type == 'MESH':
    #         for mat in item.data.materials:
    #             mat = red

bpy.data.scenes['Scene'].render.filepath = 'prova.png'

# bpy.utils.collada_import("/home/pol/Documents/3DScene/databaseFull/models/teapots/01af6b064c2d71b944715c5630d326dd/Teapot_fixed.dae")
#bpy.utils.collada_import("/home/pol/Documents/3DScene/databaseFull/models/teapots/e41b46d4249bf5cd81709c7a26423382/Teapot N240608_fixed.dae")


for object in bpy.data.scenes['Scene'].objects: print(object.name)

for item in bpy.data.objects:
    print(item.name, item.type)

width = 360
height = 240


bpy.context.scene.render.resolution_x = width #perhaps set resolution in code
bpy.context.scene.render.resolution_y = height

bpy.ops.render.render( write_still=True )
#bpy.ops.render.opengl(animation=False, view_context=False)
#bpy.ops.object.bake(type='COMBINED', filepath="baked.png", width=512, height=512)

blendImage = bpy.data.images['Render Result']

image = numpy.flipud(numpy.array(blendImage.extract_render(scene=bpy.data.scenes['Scene'])).reshape([height/2,width/2,4]))

im = Image.fromarray(numpy.uint8(image*255))

im.save("lele.png")