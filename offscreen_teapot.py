__author__ = 'matt'
import OpenGL.GL as GL
import OpenGL.GL.shaders as shaders
import glfw
import cv2
import numpy as np
from OpenGL.arrays import vbo
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from PIL import Image
plt.ion()
import matplotlib.pyplot as plt
import ipdb
import platform
import ctypes
import bpy
import mathutils
import sceneimport
import time
from utils import *
from opendr.lighting import LambertianPointLight
import chumpy as ch
import sys
import io
import os
import pickle
from sklearn.preprocessing import normalize

renderTeapotsList = [2]

[targetScenes, targetModels, transformations] = sceneimport.loadTargetModels(renderTeapotsList)
teapot = targetModels[0]

width, height = (400, 400)

angle = 60 * 180 / numpy.pi
clip_start = 0.2
clip_end = 1
camDistance = 0.45
scene = setupBlender(teapot, width, height, angle, clip_start, clip_end, camDistance)
scene.render.filepath = 'opendr_blender.png'
# bpy.ops.render.render( write_still=True )

center = centerOfGeometry(teapot.dupli_group.objects, teapot.matrix_world)
azimuth = 261
elevation = 75
azimuthRot = mathutils.Matrix.Rotation(radians(-azimuth), 4, 'Z')
elevationRot = mathutils.Matrix.Rotation(radians(-elevation), 4, 'X')
originalLoc = mathutils.Vector((0,-camDistance, 0))
location = center + azimuthRot * elevationRot * originalLoc
camera = scene.camera
camera.location = location
scene.update()
look_at(camera, center)
scene.update()
bpy.ops.render.render( write_still=True )


image = cv2.imread(scene.render.filepath)
image = image/255.0
image = cv2.cvtColor(numpy.float32(image*255), cv2.COLOR_BGR2RGB)/255.0

# toOpenglAxisRot = mathutils.Matrix.Rotation(radians(90), 4, 'X')
# teapot.matrix_world = mathutils.Matrix.Translation(teapot.location) * toOpenglAxisRot * mathutils.Matrix.Translation(-teapot.location) * teapot.matrix_world
# teapot.matrix_world = mathutils.Matrix.Translation(-teapot.location)

# Assign attributes to renderer
# from opendr.util_tests import get_earthmesh
# m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))
vl = []
vnl = []
fl = []
vtl = []
vcl = []
fcl = []

vertexMeshIndex = 0
# for mesh in teapot.dupli_group.objects:
#     if mesh.type == 'MESH':
#         # ipdb.set_trace()
#         mesh.data.transform(teapot.matrix_world * mesh.matrix_world)
#         mesh.data.update()
#         mesh.data.calc_normals()
#         mesh.data.calc_tessface()
#         for vertex in mesh.data.vertices:
#             vl = vl + [vertex.co[:]]
#             vcl = vcl + [np.array([1,1,1])]
#             vnl = vnl + [vertex.normal[:]]
#
#         for face in mesh.data.tessfaces:
#             if len(face.vertices) != 3:
#                 print("Polygon not triangle!")
#             fl = fl + [[face.vertices[0] + vertexMeshIndex, face.vertices[1]+ vertexMeshIndex,face.vertices[2] + vertexMeshIndex]]
#
#             vcolor = numpy.ones(3)
#             try:
#                 vcolor = mesh.data.materials[face.material_index].diffuse_color[:]
#                 if vcolor == (0.0,0.0,0.0) and mesh.data.materials[face.material_index].specular_color[:] != (0.0,0.0,0.0):
#                     vcolor = mesh.data.materials[face.material_index].specular_color[:]
#                     # print("Using specular!")
#
#                 # if mesh.data.materials[face.material_index].name == 'Oro':
#                 fcl = fcl + [vcolor]
#                 fcl = fcl + [vcolor]
#                 fcl = fcl + [vcolor]
#                 vcl[face.vertices[0] + vertexMeshIndex] = vcolor
#                 vcl[face.vertices[1] + vertexMeshIndex] = vcolor
#                 vcl[face.vertices[2] + vertexMeshIndex] = vcolor
#             except:
#                 print("Problem with material index and vertex color!")
#                 ipdb.set_trace()
#
#         vertexMeshIndex = vertexMeshIndex + len(mesh.data.vertices)


f = []
v = []
vc = []
vn = []
uv = []
vertexMeshIndex = 0
for mesh in teapot.dupli_group.objects:
    if mesh.type == 'MESH':
        # mesh.data.validate(verbose=True, clean_customdata=True)
        fmesh, vmesh, vcmesh,  nmesh, uvmesh = buildData(mesh.data)
        f = f + [fmesh + vertexMeshIndex]
        vc = vc + [vcmesh]
        transf = np.array(np.dot(teapot.matrix_world, mesh.matrix_world))
        vmesh = np.hstack([vmesh, np.ones([vmesh.shape[0],1])])
        vmesh = ( np.dot(transf , vmesh.T)).T[:,0:3]
        v = v + [vmesh]
        transInvMat = np.linalg.inv(transf).T
        nmesh = np.hstack([nmesh, np.ones([nmesh.shape[0],1])])
        nmesh = (np.dot(transInvMat , nmesh.T)).T[:,0:3]
        vn = vn + [normalize(nmesh, axis=1)]
        uv = uv + [uvmesh]

        vertexMeshIndex = vertexMeshIndex + len(vmesh)

f = np.vstack(f).astype(dtype=np.uint32)
# ftmp = f.copy()
# f[:,0] = f[:,2]
# f[:,2] = ftmp[:,0]
v = np.vstack(v).astype(np.float32)
vc = np.vstack(vc).astype(np.float32)
vn = np.vstack(vn).astype(np.float32)
uv = np.vstack(uv).astype(np.float32)

glfw.init()
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
# glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.VISIBLE, GL.GL_FALSE)
# glfw.window_hint(glfw.DEPTH_BITS,32)
win = glfw.create_window(width, height, "test",  None, None)
glfw.make_context_current(win)
#
#
GL.USE_ACCELERATE = True

fbo = GL.glGenFramebuffers(1)
GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo )
#
render_buf = GL.glGenRenderbuffers(1)
GL.glBindRenderbuffer(GL.GL_RENDERBUFFER,render_buf)
GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGB, width, height)
GL.glFramebufferRenderbuffer(GL.GL_DRAW_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, render_buf)


z_buf = GL.glGenRenderbuffers(1)
GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, z_buf)
GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT, width, height)
GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, z_buf)

status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
print(str(status))

# GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0 )

GL.glViewport(0, 0, width, height)

GL.glEnable(GL.GL_DEPTH_TEST)
GL.glDepthMask(GL.GL_TRUE)
GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
# GL.glCullFace(GL.GL_FRONT)
# GL.Disable(GL_LIGHTING); # causes error
GL.glEnable(GL.GL_CULL_FACE)
# GL.glFrontFace( GL.GL_CW )


print ('Vendor: %s' % (GL.glGetString(GL.GL_VENDOR)))
print ('Opengl version: %s' % (GL.glGetString(GL.GL_VERSION)))
print ('GLSL Version: %s' % (GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION)))
print ('Renderer: %s' % (GL.glGetString(GL.GL_RENDERER)))

############################
# ENABLE SHADER

FRAGMENT_SHADER = shaders.compileShader("""#version 330 core
// Interpolated values from the vertex shaders
in vec3 theColor;
// Ouput data
out vec3 color;
void main(){
	color = theColor;
}""", GL.GL_FRAGMENT_SHADER)

# VERTEX_SHADER120 = shaders.compileShader("""#version 120
#         attribute vec3 position;
# attribute vec3 color;
# varying vec3 f_color;
# void main(void) {
#   gl_Position = vec4(position, 1.0);
#   f_color = color;
# }""", GL.GL_VERTEX_SHADER)
#
# FRAGMENT_SHADER120 = shaders.compileShader("""#version 120
#         varying vec3 f_color;
# void main(void) {
#   gl_FragColor = vec4(1, 1, 0, 1.0);
# }""", GL.GL_FRAGMENT_SHADER)
#
# print(GL.glGetShaderInfoLog(FRAGMENT_SHADER120))
# print(GL.glGetShaderSource(FRAGMENT_SHADER))

VERTEX_SHADER = shaders.compileShader("""#version 330 core
// Input vertex data, different for all executions of this shader.
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
uniform mat4 MVP;
out vec3 theColor;
// Values that stay constant for the whole mesh.
void main(){
	// Output position of the vertex, in clip space : MVP * position
	gl_Position =  MVP* vec4(position,1);
	theColor = color;
}""", GL.GL_VERTEX_SHADER)

# print(GL.glGetShaderInfoLog(VERTEX_SHADER120))
# print(GL.glGetShaderSource(VERTEX_SHADER))


shader = shaders.compileProgram(VERTEX_SHADER,FRAGMENT_SHADER)

print(GL.glGetProgramInfoLog(shader))

print("Beginning render.")
t = time.process_time()

shaders.glUseProgram(shader)

GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, fbo)
# GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)

position_location = GL.glGetAttribLocation(shader, 'position')
color_location = GL.glGetAttribLocation(shader, 'color')
MVP_location = GL.glGetUniformLocation(shader, 'MVP')

#
GL.glClear(GL.GL_COLOR_BUFFER_BIT)
GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

  #Create the Vertex Array Object
vertexArrayObject = GL.GLuint(0)
GL.glGenVertexArrays(1, vertexArrayObject)
GL.glBindVertexArray(vertexArrayObject)

# GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)

# ipdb.set_trace()

# with open("mesh.pickle", "rb") as input_file:
#      mesh = pickle.load(input_file)
#
# f = mesh['f']
# v = mesh['v']
# vn = mesh['vn']
# vc = mesh['vc']
# fc = mesh['fc']

# f = np.vstack(fl).astype(dtype=np.uint32)
#
# v = np.vstack(vl).astype(np.float32)
#
# fc = np.vstack(fcl).astype(np.float32)
#
# vc = np.vstack(vcl).astype(np.float32)
#
# vn = np.array(np.vstack(vnl), dtype=np.float32)


# mesh = {'v':v, 'vc':vc, 'vn':vn, 'fc':fc, 'f':f}
# with open('mesh.pickle', 'wb') as pfile:
#             pickle.dump(mesh, pfile)

frange= np.arange(len(f)*3, dtype=np.uint32).reshape(f.shape)

vfravel = v[f.ravel()]

vbo_verts = vbo.VBO(vfravel)

vbo_indices = vbo.VBO(frange, target=GL.GL_ELEMENT_ARRAY_BUFFER)

# vc = np.array([[1, 0, 0], [1, 0,0], [1, 0, 0]], dtype=np.float32)
#
#
# vt = v
# v[:,0] =vt[:,2]
#

# vnt = vn
# vn[:,0] =vnt[:,2]
#
# vn[:,2] =vnt[:,0]

vch = ch.array(v)
vnch = ch.array(vn)
vcch = ch.array(vc)

# Construct point light source
l1 = LambertianPointLight(
    f=f,
    v=vch,
    vn=vnch,
    num_verts=len(vch),
    light_pos=ch.array([-0,-0,-0.5]),
    vc=vcch,
    light_color=ch.array([1., 1., 1.]))


# Construct point light source
l2 = LambertianPointLight(
    f=f,
    v=vch,
    vn=vnch,
    num_verts=len(vch),
    light_pos=ch.array([-0,-0,0.5]),
    vc=vcch,
    light_color=ch.array([1., 1., 1.]))

vcl1 = l1 + l2
vcl1fravel = vcl1[f.ravel()]
vbo_colors =   vbo.VBO(np.array(vcl1fravel, dtype=np.float32))
# ipdb.set_trace()

vbo_indices.bind()
# indices_buffer = GL.glGenBuffers(1)
# GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, indices_buffer)
# GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices, GL.GL_STATIC_DRAW)

# ipdb.set_trace()
# vbo_indices.bind()
# vertex_buffer = GL.glGenBuffers(1)
# GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vertex_buffer)
vbo_verts.bind()
GL.glEnableVertexAttribArray(position_location) # from 'location = 0' in shader
# GL.glBufferData(GL.GL_ARRAY_BUFFER, verts, GL.GL_STATIC_DRAW)
GL.glVertexAttribPointer(position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
#
# vc_buffer = GL.glGenBuffers(1)
# GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vc_buffer)

vbo_colors.bind()
GL.glEnableVertexAttribArray(color_location) # from 'location = 0' in shader
# GL.glBufferData(GL.GL_ARRAY_BUFFER, vc, GL.GL_STATIC_DRAW)
GL.glVertexAttribPointer(color_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

# vbo_colors.unbind()
# vbo_verts.unbind()
# vbo_indices.unbind()

fx = width
fy = width
cx = width/2
cy = height/2
w = width
h = height
far = 10
near = 0.1

# GL.glEnableClientState(GL.GL_VERTEX_ARRAY);
# GL.glVertexPointerf( vertices )

GL.glBindVertexArray(0)

# vbo_verts.unbind()
GL.glBindVertexArray(vertexArrayObject)

# vbo_verts.data = verts
# assert(np.array_equal(vbo_verts.data,verts))

projectionMatrix = np.array([[fx/cx, 0,0,0],   [0, fy/cy, 0,0],    [0,0, -(near + far)/(far - near), -2*near*far/(far-near)],   [0,0, -1, 0]], dtype=np.float32)
# projectionMatrix = np.array([[height/width, 0,0,0], [0, 1, 0,0], [0,0, 1, 0], [0,0,-1,1]], dtype=np.float32)

viewMatrix = np.array(camera.matrix_world.inverted())

print(camera.matrix_world)

print(viewMatrix)

# rot = mathutils.Matrix.Rotation(np.pi/2, 4, 'Y')
# viewMatrix = np.array(rot, dtype=np.float32)
# # viewMatrix = np.eye(4, dtype=np.float32)
# viewMatrix[0,3] = 0.5

# viewMatrix = np.linalg.inv(viewMatrix)
#
# rot = mathutils.Matrix.Rotation(0, 4, 'Y')
# trans = mathutils.Matrix.Translation(mathutils.Vector((0,0,0)))
# viewMatrix = trans * rot

MVP = np.dot(projectionMatrix, viewMatrix)

MVP = np.array(MVP, dtype=np.float32)

print(MVP)

# MVP = np.eye(4, dtype=np.float32)

GL.glUniformMatrix4fv(MVP_location, 1, GL.GL_TRUE, MVP)

GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, fbo)
GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

# ipdb.set_trace()
# GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(vbo_verts))
GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)

GL.glDrawElements(GL.GL_TRIANGLES, len(vbo_indices)*3, GL.GL_UNSIGNED_INT, None)

GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)


GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

# pixels = GL.glReadPixels(0,0,width, height, GL.GL_RGBA, GL.GL_FLOAT)

pixels = GL.glReadPixels( 0,0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
im = Image.frombuffer("RGB", (width,height), pixels, "raw", "RGB", 0, 0)
pixels =  np.array(im.transpose(Image.FLIP_TOP_BOTTOM))

plt.imsave('offscreen-opengl.png', pixels)


# pixels =  np.array(im)[:,:,0:3]
print(pixels.shape)
elapsed_time = time.process_time() - t
print("Ended render in  " + str(elapsed_time))

plt.imshow(pixels)


ipdb.set_trace()

glfw.swap_buffers(win)

#
# im.show()

# while True:
#     print ("True")

# while True:
#     glfw.swap_buffers(win)

print('glValidateProgram: ' + str(GL.glValidateProgram(shader)))
print('glGetProgramInfoLog ' + str(GL.glGetProgramInfoLog(shader)))
print('GL_MAX_VERTEX_ATTRIBS: ' + str(GL.glGetInteger(GL.GL_MAX_VERTEX_ATTRIBS)))

# im = GL.getImage()
# cv2.imshow('a', im)

