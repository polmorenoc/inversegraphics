#!/usr/bin/env python3.4m
 
import bpy
import numpy


bpy.ops.scene.new(type='EMPTY')

bpy.ops.render.render(write_still=True)

bpy.data.scenes['Scene'].render.filepath = '/home/user/Documents/image.jpg'
bpy.context.scene.render.resolution_x = w #perhaps set resolution in code
bpy.context.scene.render.resolution_y = h

bpy.ops.render.render( write_still=True )