from math import sin, cos, ceil, floor, pi
import importlib
import bpy
import numpy as np
import mathutils
from contextlib import contextmanager
from uuid import uuid4
from bpy.utils import register_module, unregister_module
from bpy import props as p
import json
import ipdb
import matplotlib.pyplot as plt

# bl_info = {
#     "name": "Lightprobe",
#     "description": "Gives ability to add light probes to a cycles render. \
# Light probes sample incoming light at that location and generate 9 \
# coefficients that can be used to quickly simluate that lighting in a real-time \
# game engine.",
#     "category": "Object",
#     "author": "Andrew Moffat",
#     "version": (1, 0),
#     "blender": (2, 7, 1)
# }

JSON_FILE_NAME = "lightprobes.json"
FAILSAFE_OFFSET = 0.00001

import cv2

spherical_harmonics_coeffs = np.array([
    0.282095,
    0.488603 ,
    0.488603 ,
     0.488603,
     1.092548,
     1.092548,
    0.315392 ,
    1.092548 ,
    0.546274 ])

def processSphericalEnvironmentMap(envMapTexture):
    mask = np.ones([envMapTexture.shape[0],envMapTexture.shape[1]]).astype(np.uint8)
    mask[np.int(mask.shape[0]/2), np.int(mask.shape[1]/2)] = 0
    distMask = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    envMapTexture[distMask > mask.shape[0]/2,:] = 0
    envMapMean = envMapTexture[distMask <= mask.shape[0]/2].mean()
    envMapTexture[distMask <= mask.shape[0]/2, :] = envMapTexture[distMask <= mask.shape[0]/2]
    return envMapTexture, envMapMean

def is_lightprobe(ob):
    return ob.name.startswith("lightprobe-")

def all_active_lightprobes():
    for ob in bpy.context.scene.objects:
        if is_lightprobe(ob):
            yield ob

def get_all_lightprobe_data():
    all_data = []
    scale_by = bpy.context.scene.unit_settings.scale_length
    
    for probe in all_active_lightprobes():
        coeffs = get_coeff_prop(probe)
        if not coeffs:
            continue

        data = {}
        data["loc"] = [p*scale_by for p in list(probe.location)]
        data["name"] = probe.name or None
        data["coeffs"] = coeffs
        all_data.append(data)
        
    return all_data
    

def get_or_create_probe_file():
    if JSON_FILE_NAME not in bpy.data.texts:
        bpy.data.texts.new(JSON_FILE_NAME)
        
    f = bpy.data.texts[JSON_FILE_NAME]
    return f


def write_lightprobe_data(data):
    f = get_or_create_probe_file()
    f.clear()        
    
    data = json.dumps(data, indent=4, sort_keys=True)
    f.write(data)
    
    
# def fetch_integration_callback(name):
#     parts = name.split(".")
#     fn_name = parts[-1]
#     module_name = ".".join(parts[:-1])
#
#     try:
#         module = importlib.import_module(module_name)
#     except:
#         return None
#     else:
#         fn = getattr(module, fn_name, None)
#         return fn
#
#     name, theta_res, phi_res, samples =
# # def pre_bake_hook(name, context, probe):
#     fn = fetch_integration_callback(name)
#     ret = None
#     if fn:
#         ret = fn(context, probe)
#     return ret
#
# # def post_bake_hook(name, context, data, pre_bake_data):
#     fn = fetch_integration_callback(name)
#     if fn:
#         fn(context, data, pre_bake_data)


def hide_object(ob):
    """ hides an object from cycles rendering, and returns a function that,
    when called, will restore the visibility """ 
    v = ob.cycles_visibility

    names = ("shadow", "camera", "glossy", "diffuse", "transmission", "scatter")
    for name in names:
        setattr(v, name, False)
        
        
def set_coeff_prop(ob, coeffs):
    """ sets our SH coeffs onto an object datablock.  we can't use the python
    dictionary that we've generated, so we'll flatten it to something that
    can be stored in a datablock """
        
    ob["lightprobe_coeffs"] = json.dumps(coeffs)
    
    
def get_coeff_prop(ob):
    """ retrieve our SH coeffs from our object datablock.  this is essentially
    unserializing it to our original data """
    
    coeffs = ob.get("lightprobe_coeffs", None)
    if not coeffs:
        return None
        
    data = json.loads(coeffs)
    return data
    
def setup_material(ob):

    scene = bpy.context.scene
    
    mat = bpy.data.materials.new(ob.name)
    mat.use_nodes = True
    
    tree = mat.node_tree
    
    for node in list(tree.nodes):
        tree.nodes.remove(node)
        
    #diffuse = tree.nodes.new("ShaderNodeBsdfGlossy")
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (1, 1, 1, 1)
    diffuse.inputs["Roughness"].default_value = 1.0
    diffuse_out = diffuse.outputs["BSDF"]
    
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    # output.color_space = 'NONE'
    output_in = output.inputs["Surface"]
    
    tree.links.new(diffuse_out, output_in)
    
    bake_node = tree.nodes.new("ShaderNodeTexImage")
    bake_node.label = bake_node.name
    bake_node.color_space = 'NONE'
    bake_out = bake_node.outputs["Color"]
    
    texture = create_lightmap_image(ob, 128, 128)
    bake_node.image = texture
    bake_node.image.colorspace_settings.name = 'Raw'

    ob.data.uv_textures["lightmap"].active = True

    color_uvmap = tree.nodes.new("ShaderNodeUVMap")
    color_uvmap.uv_map = "lightmap"
    tree.links.new(color_uvmap.outputs["UV"], bake_node.inputs["Vector"])
    
    ob.data.materials.append(mat)
        
@contextmanager
def no_interfere_ctx():
    old_selected_objects = bpy.context.selected_objects
    active_object = bpy.context.active_object
    yield
    for obj in bpy.context.selected_objects:
        obj.select = False
    for obj in old_selected_objects:
        obj.select = True
    bpy.context.scene.objects.active = active_object
    
    
@contextmanager
def active(ob):
    with no_interfere_ctx():
        ob.select = True
        bpy.context.scene.objects.active = ob
        yield
    

def override_ctx(**kwargs):
    ctx = bpy.context.copy()
    ctx.update(kwargs)
    return ctx

def create_lightmap_image(ob, width, height):
    name = ob.name
    bpy.ops.image.new(override_ctx(object=ob), name=name, width=width,
            height=height, alpha=False, float=False)
    return bpy.data.images[name]

def get_lightmap(ob):
    return bpy.data.images[ob.name]

def add_lightprobe():
    with no_interfere_ctx():
        bpy.ops.mesh.primitive_cube_add()
        probe = bpy.context.object
        
        for _ in range(len(probe.data.uv_layers)):
            bpy.ops.mesh.uv_texture_remove()
        
        bpy.ops.mesh.uv_texture_add()
        probe.data.uv_layers[0].name = "lightmap"
        
        bpy.ops.uv.lightmap_pack(PREF_CONTEXT="ALL_FACES",
                PREF_PACK_IN_ONE=True, PREF_NEW_UVLAYER=False,
                PREF_APPLY_IMAGE=False, PREF_IMG_PX_SIZE=512, PREF_BOX_DIV=12,
                PREF_MARGIN_DIV=0.1)
        
        bpy.ops.object.modifier_add(type="SUBSURF")
        bpy.context.object.modifiers["Subsurf"].levels = 4
        
        bpy.ops.object.modifier_add(type="TRIANGULATE")
        bpy.ops.object.convert(target='MESH')

    
        probe.scale = mathutils.Vector((0.3, 0.3, 0.3))
        bpy.ops.object.shade_smooth()
        
        
    probe.name = "lightprobe-" + uuid4().hex
    return probe

def bake(ob):
    with active(ob):
        cycles = bpy.context.scene.cycles
        old_samples = cycles.samples

        name, theta_res, phi_res, samples = probeProperties()

        cycles.samples = samples
        bpy.ops.object.bake(type="COMBINED")
        cycles.samples = old_samples


def get_lightprobe_coefficients(probe, theta_res, phi_res):
    probe.data.calc_tessface()
    bake(probe)
    lightmap = get_lightmap(probe)
    return get_all_coefficients(probe, lightmap, theta_res, phi_res)


def sample_image(channels, width, height, pixel_data, loc):
    """ samples a blender location at a particular xy integer location """    
    x, y = loc[0], loc[1]
    
    pix_loc = int((y * width * channels) + x * channels)
    r = pixel_data[pix_loc + 0]
    g = pixel_data[pix_loc + 1]
    b = pixel_data[pix_loc + 2]
     
    return mathutils.Color((r, g, b))

def bilinear_interpolate(image, uv):
    """ performs bilinear interpolation of a blender image using texture-space
    uv coordinates.  the boundary conditions are to extend the edges """
    
    lightmap_size = mathutils.Vector(image.size)
    width, height = lightmap_size[0], lightmap_size[1]
    
    px_x, px_y = 1.0/width, 1.0/height
    half_px_x, half_px_y = 1.0/(2*width), 1.0/(2*height)
    
    left_coord = ceil(width * (uv[0] - half_px_x) - 1)
    right_coord = ceil(width * (uv[0] + half_px_x) - 1)
    bottom_coord = ceil(height * (uv[1] - half_px_y) - 1)
    top_coord = ceil(height * (uv[1] + half_px_y) - 1)
    
    
    # these are asking how much of 1-pixel (in uv space) has our uv coordinate
    # traversed, starting at the left/bottom pixel boundary
    lerp_x = (uv[0] - (left_coord + 0.5) / width) / px_x
    lerp_y = (uv[1] - (bottom_coord + 0.5) / height) / px_y
    
    
    # boundary conditions
    if right_coord + 1 > width:
        right_coord = left_coord
    
    if left_coord < 0:
        left_coord = right_coord
        
    if top_coord + 1 > height:
        top_coord = bottom_coord
        
    if bottom_coord < 0:
        bottom_coord = top_coord
        
    
    ll_uv = mathutils.Vector((left_coord, bottom_coord))
    lr_uv = mathutils.Vector((right_coord, bottom_coord))
    ur_uv = mathutils.Vector((right_coord, top_coord))
    ul_uv = mathutils.Vector((left_coord, top_coord))
    
    pixel_data = image.pixels[:]
    chan = image.channels
    width, height = image.size
    
    lower_left = mathutils.Vector(sample_image(chan, width, height, pixel_data, ll_uv))
    lower_right = mathutils.Vector(sample_image(chan, width, height, pixel_data, lr_uv))
    upper_right = mathutils.Vector(sample_image(chan, width, height, pixel_data, ur_uv))
    upper_left = mathutils.Vector(sample_image(chan, width, height, pixel_data, ul_uv))
    
    del pixel_data
    
    top = upper_left.lerp(upper_right, lerp_x)
    bottom = lower_left.lerp(lower_right, lerp_x)
    color = mathutils.Color(bottom.lerp(top, lerp_y))
    
    return color


# http://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
def triangle_intersection(v1, v2, v3, ray, origin):
    """ performs moller-trumbore ray-triangle intersection and returns
    barycentric coordinates if an intersection exists, None otherwise """
    
    epsilon = 0.000001
    edge1 = v2 - v1
    edge2 = v3 - v1
    
    P = ray.cross(edge2)
    det = edge1.dot(P)
    
    if det > -epsilon and det < epsilon:
        return None
    
    inv_det = 1.0 / det
    T = origin - v1
    
    u = T.dot(P) * inv_det
    
    if u < 0 or u > 1:
        return None
    
    Q = T.cross(edge1)
    v = ray.dot(Q) * inv_det
    
    if v < 0 or u + v > 1:
        return None
    
    t = edge2.dot(Q) * inv_det
    
    if t > epsilon:
        w = 1 - u - v
        return mathutils.Vector((u, v, w))
    
    return None


def get_glsl_coefficients(coeffs):
    """ a convenience function for testing SH coefficients in the shader
    provided by the opengl orange book, second edition """
    
    tmpl = "const vec3 L%d%s%d = vec3(%f, %f, %f);"
    lines = []
    
    for l, mdata in coeffs.items():
        for m, color in mdata.items():
            
            sign = ""
            if m < 0:
                sign = "m"
                
            line = tmpl % (l, sign, abs(m), color[0], color[1], color[2])
            lines.append(line)
        
    return "\n".join(lines)
    

def get_all_coefficients(ob, lightmap, theta_res, phi_res):
    """ returns all SH coefficients.  theta_res and phi_res are the
    sampling resolutions for theta (zenith) and phi (azimuth) respectively.
    theta ranges from 0-pi, while phi ranges from 0-2pi """
    mapping = {}
    for l, m in spherical_harmonics.keys():
        color = get_coefficients(ob, lightmap, l, m, theta_res, phi_res)
        mapping.setdefault(l, {})[m] = color
    return mapping


def get_coefficients(ob, lightmap, l, m, theta_res, phi_res):
    """ returns the RGB spherical harmonic coefficients for a given
    l and m """
    c = mathutils.Color((0, 0, 0))
    harmonic = spherical_harmonics[(l, m)]

    # plt.imsave("lightmap.png", lightmap)
    for theta in (pi * y / float(theta_res) for y in range(theta_res)):
        for phi in (pi * 2 * x / float(phi_res) for x in range(phi_res)):
            color = sample_icosphere_color(ob, lightmap, theta, phi)
            c += (color * harmonic(theta, phi) * sin(theta)
                / (theta_res * phi_res))
            
    return c.r, c.g, c.b
# http://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf
spherical_harmonics = {
    (0, 0): lambda theta, phi: 0.282095,

    (1, 1): lambda theta, phi: 0.488603 * sin(theta) * cos(phi),
    (1, 0): lambda theta, phi: 0.488603 * cos(theta),
    (1, -1): lambda theta, phi: 0.488603 * sin(theta) * sin(phi),


    (2, -2): lambda theta, phi: 1.092548 * sin(theta) * cos(phi) * sin(theta) * sin(phi),
    (2, -1): lambda theta, phi: 1.092548 * sin(theta) * sin(phi) * cos(theta),
    (2, 0): lambda theta, phi: 0.315392 * (3 * cos(theta)**2 - 1),
    (2, 1): lambda theta, phi: 1.092548 * sin(theta) * cos(phi) * cos(theta),
    (2, 2): lambda theta, phi: 0.546274 * (((sin(theta) * cos(phi)) ** 2) - ((sin(theta) * sin(phi)) ** 2))
}



def SHProjection(envMap, shCoefficients):
    #Backprojected.
    phis = 2*ch.pi*np.tile((np.roll(np.arange(envMap.shape[1])[::-1], np.int(envMap.shape[1]/2))/envMap.shape[1]).reshape([1,envMap.shape[1],1]), [envMap.shape[0],1,3])
    thetas = np.pi*np.tile((np.arange(envMap.shape[0])/envMap.shape[0]).reshape([envMap.shape[0],1,1]), [1,envMap.shape[1],3])

    phis = np.mod(phis, 2*np.pi)

    normalize = 1

    pEnvMap = np.zeros([envMap.shape[0],envMap.shape[1], 3, 9])
    pEnvMap[:,:,:,0] = shCoefficients[0].reshape([1,1,3]) *  spherical_harmonics_coeffs[0]  * normalize
    pEnvMap[:,:,:,1] = shCoefficients[1].reshape([1,1,3]) * spherical_harmonics_coeffs[1]*np.sin(thetas) * np.cos(phis) * normalize
    pEnvMap[:,:,:,2] = shCoefficients[2].reshape([1,1,3]) * spherical_harmonics_coeffs[2]*np.cos(thetas) * normalize
    pEnvMap[:,:,:,3] = shCoefficients[3].reshape([1,1,3]) * spherical_harmonics_coeffs[3]*np.sin(thetas)*np.sin(phis) * normalize
    pEnvMap[:,:,:,4] = shCoefficients[4].reshape([1,1,3]) * spherical_harmonics_coeffs[4]*np.sin(thetas) * np.cos(phis) * np.sin(phis)* np.sin(thetas)  * normalize
    pEnvMap[:,:,:,5] = shCoefficients[5].reshape([1,1,3]) * spherical_harmonics_coeffs[5]*np.sin(thetas) * np.sin(phis) * np.cos(thetas)  * normalize
    pEnvMap[:,:,:,6] = shCoefficients[6].reshape([1,1,3]) * spherical_harmonics_coeffs[6]*(3 * np.cos(thetas)**2 - 1) * normalize
    pEnvMap[:,:,:,7] = shCoefficients[7].reshape([1,1,3]) * spherical_harmonics_coeffs[7]*np.sin(thetas) * np.cos(phis) * np.cos(thetas) * normalize
    pEnvMap[:,:,:,8] = shCoefficients[8].reshape([1,1,3]) * spherical_harmonics_coeffs[8]*(((np.sin(thetas) * np.cos(phis)) ** 2) - ((np.sin(thetas) * np.sin(phis)) ** 2)) * normalize

    return pEnvMap

def sphericalHarmonicsZRotation(angle):
    return np.array([[1,0,0,0,0,0,0,0,0],[0, np.cos(angle), 0, np.sin(angle), 0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0, -np.sin(angle), 0, np.cos(angle), 0,0,0,0,0],[0,0,0,0,np.cos(2*angle),0,0,0,np.sin(2*angle)],[0,0,0,0,0,np.cos(angle), 0, np.sin(angle),0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0, -np.sin(angle),0, np.cos(angle),0],[0,0,0,0,-np.sin(2*angle),0,0,0,np.cos(2*angle)]])

import chumpy as ch
from chumpy import depends_on, Ch
class SphericalHarmonicsZRotation(Ch):
    dterms = 'angle'

    def compute_r(self):
        return np.array([[1,0,0,0,0,0,0,0,0],[0, np.cos(self.angle), 0, np.sin(self.angle), 0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0, -np.sin(self.angle), 0, np.cos(self.angle), 0,0,0,0,0],[0,0,0,0,np.cos(2*self.angle),0,0,0,np.sin(2*self.angle)],[0,0,0,0,0,np.cos(self.angle), 0, np.sin(self.angle),0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0, -np.sin(self.angle),0, np.cos(self.angle),0],[0,0,0,0,-np.sin(2*self.angle),0,0,0,np.cos(2*self.angle)]])

    #No need to make it differentiable for now.
    def compute_dr_wrt(self, wrt):
        return None

import chumpy as ch
def chSphericalHarmonicsZRotation(angle):
    return SphericalHarmonicsZRotation(angle=angle)



def getEnvironmentMapCoefficients(envMap, normalize, phiOffset, type):
    """ returns the RGB spherical harmonic coefficients for a given
    l and m """
    if type == 'equirectangular':
        phis = 2*np.pi*np.tile((np.roll(np.arange(envMap.shape[1])[::-1], np.int(envMap.shape[1]/2))/envMap.shape[1]).reshape([1,envMap.shape[1],1]), [envMap.shape[0],1,3])
        thetas = np.pi*np.tile((np.arange(envMap.shape[0])/envMap.shape[0]).reshape([envMap.shape[0],1,1]), [1,envMap.shape[1],3])
    elif type == 'spherical':
        vcoords = (-envMap.shape[0]/2 + np.tile(np.arange(envMap.shape[0]).reshape([envMap.shape[0], 1,1]), [1,envMap.shape[1],3]))/(envMap.shape[0]/2)
        ucoords = (-envMap.shape[1]/2 + np.tile(np.arange(envMap.shape[1]).reshape([1,envMap.shape[1],1]), [envMap.shape[0],1,3]))/(envMap.shape[1]/2)
        thetas=np.arctan2(vcoords,ucoords)
        phis=pi*np.sqrt(ucoords*ucoords + vcoords*vcoords)
    else:
        print("Environment map format not recognized")
        return

    phis = np.mod(phis + phiOffset, 2*np.pi)

    L = np.zeros([9,3])
    # ipdb.set_trace()
    L[0] = np.sum(envMap * spherical_harmonics_coeffs[0] *np.sin(thetas) , axis=(0,1))
    L[1] = np.sum(envMap *spherical_harmonics_coeffs[1]*np.sin(thetas)*np.sin(thetas) * np.cos(phis), axis=(0,1))
    L[2] = np.sum(envMap *spherical_harmonics_coeffs[2]*np.sin(thetas)*np.cos(thetas), axis=(0,1))
    L[3] = np.sum(envMap *spherical_harmonics_coeffs[3]*np.sin(thetas)*np.sin(thetas)*np.sin(phis), axis=(0,1))
    L[4] = np.sum(envMap *spherical_harmonics_coeffs[4]*np.sin(thetas)*np.sin(thetas) * np.cos(phis)  * np.sin(phis)* np.sin(thetas), axis=(0,1))
    L[5] = np.sum(envMap *spherical_harmonics_coeffs[5]*np.sin(thetas)*np.sin(thetas) * np.sin(phis) * np.cos(thetas) , axis=(0,1))
    L[6] = np.sum(envMap *spherical_harmonics_coeffs[6]*np.sin(thetas)*(3 * np.cos(thetas)**2 - 1), axis=(0,1))
    L[7] = np.sum(envMap *spherical_harmonics_coeffs[7]*np.sin(thetas) * np.cos(phis) * np.cos(thetas)* np.sin(thetas), axis=(0,1))
    L[8] = np.sum(envMap *spherical_harmonics_coeffs[8]*np.sin(thetas)*(((np.sin(thetas) * np.cos(phis)) ** 2) - ((np.sin(thetas) * np.sin(phis)) ** 2)), axis=(0,1))


    L = 2*np.pi*L*np.pi/(envMap.shape[0]*envMap.shape[1]*normalize)
    return L


def sample_icosphere_color(ob, lightmap, theta, phi):
    """ takes a theta and phi and casts a ray out from the center of an
    icosphere, bilinearly sampling the surface where the ray intersects """
    ray = angle_to_ray(theta, phi)
    
    # we extend the ray arbitrarily so it's guaranteed to intersect with the
    # icosphere, instead of falling short
    ray *= 100
    
    face, location = find_intersecting_face(ob, ray)
    
    # it is possible that we couldn't find an intersecting face if the ray
    # we shot aligns perfectly with a vertex.  in this case, we'll offset the
    # ray slightly and try again.  this should not fail a second time.
    if face is None:
        ray.x += FAILSAFE_OFFSET
        ray.y += FAILSAFE_OFFSET
        ray.z += FAILSAFE_OFFSET
        
        face, location = find_intersecting_face(ob, ray)
        assert(face is not None)
    
    color = sample_lightmap(ob, lightmap, face, location)
    return color

    
def angle_to_ray(theta, phi):
    """ converts a spherical coordinate to cartesian coordinate """
    x = sin(theta) * cos(phi)
    y = sin(theta) * sin(phi)
    z = cos(theta)
    ray = mathutils.Vector((x, y, z)).normalized()
    return ray


def find_intersecting_face(ob, ray):
    """ finds the face where a ray from the center of an icosphere
    intersects """
    mesh = ob.data
    origin = mathutils.Vector()
    
    # we'll use scale to ensure that our transform applies to our vertices.
    # assume scale is uniform and just use x scale
    scale = ob.scale[0]
    
    for face in mesh.tessfaces:
        v = face.vertices
        v1 = mesh.vertices[v[0]].co * scale
        v2 = mesh.vertices[v[1]].co * scale
        v3 = mesh.vertices[v[2]].co * scale
        
        intersection = triangle_intersection(v1, v2, v3, ray, origin)
        if intersection:
            return face, intersection
        
    # we should never get here, but we may in the case of a ray aligning
    # perfectly with a vertex.  in this case, we'll catch this error up at
    # the caller
    return None, None
    
        
def sample_lightmap(ob, lightmap, face, loc):
    """ """
    mesh = ob.data
    uvs = mesh.tessface_uv_textures[0].data[face.index]
    location_uv = loc[0] * uvs.uv1 + loc[1] * uvs.uv2 + loc[2] * uvs.uv3
    
    return bilinear_interpolate(lightmap, location_uv)


        
def bakeOp(context):
    probe = context.object

    name, theta_res, phi_res, samples = probeProperties()

    # ret = pre_bake_hook(scene_settings.pre_bake_hook, context, probe)

    coeffs = get_lightprobe_coefficients(probe, theta_res,
            phi_res)
    set_coeff_prop(probe, coeffs)
    lp_data = get_all_lightprobe_data()
    write_lightprobe_data(lp_data)

    # post_bake_hook(scene_settings.post_bake_hook, context, lp_data, ret)
    return lp_data

    
# class BakeAllOp(context):
#     override = context.copy()
#     for probe in all_active_lightprobes():
#         override["object"] = probe
#         bpy.ops.object.bake_lightprobe(override)
#
#
    
def resizeAll(context):

    size = p.FloatProperty(name="Units", default=1)

    dimensions = mathutils.Vector((self.size, self.size, self.size))

    for probe in all_active_lightprobes():
        probe.dimensions = dimensions


def lightProbeOp(context):

    probe = add_lightprobe()
    hide_object(probe)
    probe.show_x_ray = True

    with active(probe):
        setup_material(probe)


def probeProperties():
    name = ""
    theta_res = 20
    phi_res = 40
    samples = 1000
    return name, theta_res, phi_res, samples
