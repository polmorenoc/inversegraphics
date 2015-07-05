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
import sceneimport
import mathutils
import time


width, height = 480, 480
glfw.init()
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
# glfw.window_hint(glfw.VISIBLE, GL.GL_FALSE)
win = glfw.create_window(width, height, "test",  None, None)
glfw.make_context_current(win)
#
#
GL.USE_ACCELERATE = True
#
# fbo = GL.glGenFramebuffers(1)
# GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo )
#
# render_buf = GL.glGenRenderbuffers(1)
# GL.glBindRenderbuffer(GL.GL_RENDERBUFFER,render_buf)
# GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGB, width, height)
# GL.glFramebufferRenderbuffer(GL.GL_DRAW_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, render_buf)

status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
print(str(status))

# GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0 )

GL.glViewport(0, 0, width, height)

GL.glEnable(GL.GL_DEPTH_TEST)
GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
# GL.Disable(GL_LIGHTING); # causes error
GL.glEnable(GL.GL_CULL_FACE)
# GL.glPixelStorei(GL.GL_PACK_ALIGNMENT,4)
# GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT,4)
GL.glDepthFunc(GL.GL_LESS)



print ('Vendor: %s' % (GL.glGetString(GL.GL_VENDOR)))
print ('Opengl version: %s' % (GL.glGetString(GL.GL_VERSION)))
print ('GLSL Version: %s' % (GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION)))
print ('Renderer: %s' % (GL.glGetString(GL.GL_RENDERER)))

############################
# ENABLE SHADER

FRAGMENT_SHADER = shaders.compileShader("""#version 330 core
// Interpolated values from the vertex shaders
in vec4 theColor;
// Ouput data
out vec4 color;
void main(){
	color = vec4(theColor);
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
out vec4 theColor;
// Values that stay constant for the whole mesh.
void main(){
	// Output position of the vertex, in clip space : MVP * position
	gl_Position =  MVP* vec4(position,1);
	theColor = vec4(color,1);
}""", GL.GL_VERTEX_SHADER)

# print(GL.glGetShaderInfoLog(VERTEX_SHADER120))
# print(GL.glGetShaderSource(VERTEX_SHADER))


shader = shaders.compileProgram(VERTEX_SHADER,FRAGMENT_SHADER)

print(GL.glGetProgramInfoLog(shader))

print("Beginning render.")
t = time.process_time()



shaders.glUseProgram(shader)

# GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, fbo)

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

verts = np.array(  [[1.0, -1.0,  1.0],[-1.0, -1.0,  -1.0],[1.0,  1.0,  1.0], [-1.0,  1.0,  1.0],[-1.0, -1.0, -1.0],[1.0, -1.0, -1.0], [1.0,  1.0, -1.0], [-1.0,  1.0, -1.0]], dtype=np.float32)
# ipdb.set_trace()

#Create the index buffer object
indices = np.array([[1,2,3]], dtype=np.uint16)
vbo_verts = vbo.VBO(verts)

#Can arrays be empty?

vbo_indices = vbo.VBO(indices, target=GL.GL_ELEMENT_ARRAY_BUFFER)
vc = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 1.0, 1.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 1.0, 1.0]], dtype=np.float32)

vbo_colors =  vbo.VBO(vc)
# ipdb.set_trace()

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

vbo_indices.bind()
# vbo_colors.unbind()
# vbo_verts.unbind()
# vbo_indices.unbind()

fx = width
fy = width
cx = width/2
cy = height/2
w = width
h = height
far = 5
near = 0.1

# GL.glEnableClientState(GL.GL_VERTEX_ARRAY);
# GL.glVertexPointerf( vertices )

GL.glBindVertexArray(0)

# vbo_verts.unbind()


GL.glBindVertexArray(vertexArrayObject)

GL.glEnableVertexAttribArray(position_location)
GL.glEnableVertexAttribArray(color_location)

# vbo_verts.data = verts
# assert(np.array_equal(vbo_verts.data,verts))

projectionMatrix = np.array([[near, 0,0,0],   [0, near, 0,0],    [0,0, -(near + far)/(far - near), -2*near*far/(far-near)],   [0,0, -1, 0]], dtype=np.float32)
# projectionMatrix = np.array([[near, 0,0,0],   [0, near, 0,0],    [0,0, -2/(far - near), -(near + far)/(far-near)],   [0,0, 0, 1]], dtype=np.float32)
# projectionMatrix = np.array([[1, 0,0,0], [0, 0.5, 0,0], [0,0, 0, 1], [0,0,-1,0]], dtype=np.float32)
# projectionMatrix =  np.eye(4, dtype=np.float32)
rot = mathutils.Matrix.Rotation(np.pi, 4, 'Z')
rot = np.eye(4, dtype=np.float32)
roty = np.array(mathutils.Matrix.Rotation(np.pi/4.0, 4, 'X'), np.float32)
roty = np.eye(4, dtype=np.float32)
viewMatrix = np.array(rot, dtype=np.float32)
# viewMatrix = np.eye(4, dtype=np.float32)
viewMatrix[2,3] = 1

print(vbo_verts.data)
print(vbo_indices.data)
print(vbo_colors.data)

#
# rot = mathutils.Matrix.Rotation(0, 4, 'Y')
# trans = mathutils.Matrix.Translation(mathutils.Vector((0,0,0)))
# viewMatrix = trans * rot
invViewMat = np.linalg.inv(viewMatrix)

MVP = np.dot(np.dot(projectionMatrix, invViewMat), roty)

# ipdb.set_trace()

GL.glUniformMatrix4fv(MVP_location, 1, GL.GL_FALSE, MVP.T)
#
# vertexPositions.unbind()
# indexPositions.unbind()

# GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, fbo)

# ipdb.set_trace()
# GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(indices) )
GL.glDrawElements(GL.GL_TRIANGLES, len(vbo_indices)*3, GL.GL_UNSIGNED_SHORT, None)

# GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)

# GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

# pixels = GL.glReadPixels(0,0,width, height, GL.GL_RGBA, GL.GL_FLOAT)

# screenshot = GL.glReadPixels( 0,0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
# im = Image.frombuffer("RGBA", (width,height), screenshot, "raw", "RGBA", 0, 0)
# pixels =  np.array(im)
# print(pixels.shape)
elapsed_time = time.process_time() - t
print("Ended render in  " + str(elapsed_time))

# plt.imshow(pixels)
# plt.imsave('offscreen-opengl.png', pixels)
# ipdb.set_trace()

glfw.swap_buffers(win)

#
# im.show()

# while True:
#     print ("True")

while True:
    glfw.swap_buffers(win)

print('glValidateProgram: ' + str(GL.glValidateProgram(shader)))
print('glGetProgramInfoLog ' + str(GL.glGetProgramInfoLog(shader)))
print('GL_MAX_VERTEX_ATTRIBS: ' + str(GL.glGetInteger(GL.GL_MAX_VERTEX_ATTRIBS)))

# im = GL.getImage()
# cv2.imshow('a', im)

