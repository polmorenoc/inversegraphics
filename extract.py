import matplotlib
# matplotlib.use('Agg')
import bpy
import numpy
import matplotlib.pyplot as plt

width = 110

height = 110



scene = bpy.data.scenes[0]
scene.render.resolution_x = width #perhaps set resolution in code
scene.render.resolution_y = height
scene.render.resolution_percentage = 100

bpy.data.objects['Cube'].pass_index = 1

scene.render.layers[0].use_pass_object_index = True

bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree

rl = tree.nodes[1]
links = tree.links

v = tree.nodes.new('CompositorNodeViewer')
v.location = 750,210
v.use_alpha = False
links.new(rl.outputs['Image'], v.inputs['Image'])  # link Image output to Viewer input

outnode = tree.nodes.new('CompositorNodeOutputFile')
outnode.base_path = 'indexob.png'
links.new(rl.outputs['IndexOB'], outnode.inputs['Image'])

bpy.data.objects['Cube'].pass_index = 1

bpy.ops.render.render( write_still=False )

blendImage = bpy.data.images['Render Result']

image = numpy.flipud(numpy.array(blendImage.extract_render(scene=scene)).reshape([height,width,4]))[:,:,0:3]
#
# blendImage2 = bpy.data.images['Viewer Node']
#
# image2 = numpy.flipud(numpy.array(blendImage2.extract_render(scene=scene)).reshape([256,256,4]))[:,:,0:3]

print("DOne")