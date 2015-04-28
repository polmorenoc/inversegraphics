import bpy
for i in range(20):
    bpy.ops.render.render( write_still=False )

    blendImage = bpy.data.images['Render Result']

    scene = bpy.data.scenes[0]

    z = blendImage.extract_render(scene=scene)