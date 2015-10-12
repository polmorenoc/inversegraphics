__author__ = 'pol'
#From http://blender.stackexchange.com/a/9080

import bpy
import bmesh
from utils import *

def bmesh_copy_from_object(obj, objTransf, transform=True, triangulate=True, apply_modifiers=False):
    """
    Returns a transformed, triangulated copy of the mesh
    """

    assert(obj.type == 'MESH')

    if apply_modifiers and obj.modifiers:
        me = obj.to_mesh(bpy.context.scene, True, 'PREVIEW', calc_tessface=False)
        bm = bmesh.new()
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
    else:
        me = obj.data
        if obj.mode == 'EDIT':
            bm_orig = bmesh.from_edit_mesh(me)
            bm = bm_orig.copy()
        else:
            bm = bmesh.new()
            bm.from_mesh(me)

    # Remove custom data layers to save memory
    for elem in (bm.faces, bm.edges, bm.verts, bm.loops):
        for layers_name in dir(elem.layers):
            if not layers_name.startswith("_"):
                layers = getattr(elem.layers, layers_name)
                for layer_name, layer in layers.items():
                    layers.remove(layer)

    if transform:
        bm.transform(objTransf * obj.matrix_world)

    if triangulate:
        bmesh.ops.triangulate(bm, faces=bm.faces)

    return bm

def aabb_intersect(instance1, instance2):
    minX1, maxX1 = modelWidth(instance1.dupli_group.objects, instance1.matrix_world)
    minY1, maxY1 = modelDepth(instance1.dupli_group.objects, instance1.matrix_world)
    minZ1, maxZ1 = modelHeighaja00
    t(instance1.dupli_group.objects, instance1.matrix_world)

    minX2, maxX2 = modelWidth(instance2.dupli_group.objects, instance2.matrix_world)
    minY2, maxY2 = modelDepth(instance2.dupli_group.objects, instance2.matrix_world)
    minZ2, maxZ2 = modelHeight(instance2.dupli_group.objects, instance2.matrix_world)

    return ((maxX1 > minX2) and (minX1 < maxX2) and (maxY1 > minY2) and (minY1 < maxY2) and (maxZ1 > minZ2) and (minZ1 < maxZ2))


def bmesh_check_intersect_objects(obj, objTransf,  obj2, obj2Transf):
    """
    Check if any faces intersect with the other object

    returns a boolean
    """
    assert(obj != obj2)

    # Triangulate
    bm = bmesh_copy_from_object(obj, objTransf, transform=True, triangulate=True)
    bm2 = bmesh_copy_from_object(obj2, obj2Transf, transform=True, triangulate=True)

    # If bm has more edges, use bm2 instead for looping over its edges
    # (so we cast less rays from the simpler object to the more complex object)
    if len(bm.edges) > len(bm2.edges):
        bm2, bm = bm, bm2

    # Create a real mesh (lame!)
    scene = bpy.context.scene
    me_tmp = bpy.data.meshes.new(name="~temp~")
    bm2.to_mesh(me_tmp)
    bm2.free()
    obj_tmp = bpy.data.objects.new(name=me_tmp.name, object_data=me_tmp)
    scene.objects.link(obj_tmp)
    scene.update()
    ray_cast = obj_tmp.ray_cast

    intersect = False

    EPS_NORMAL = 0.000001
    EPS_CENTER = 0.01  # should always be bigger

    #for ed in me_tmp.edges:
    for ed in bm.edges:
        v1, v2 = ed.verts

        # setup the edge with an offset
        co_1 = v1.co.copy()
        co_2 = v2.co.copy()
        co_mid = (co_1 + co_2) * 0.5
        no_mid = (v1.normal + v2.normal).normalized() * EPS_NORMAL
        co_1 = co_1.lerp(co_mid, EPS_CENTER) + no_mid
        co_2 = co_2.lerp(co_mid, EPS_CENTER) + no_mid

        co, no, index = ray_cast(co_1, co_2)
        if index != -1:
            intersect = True
            break

    scene.objects.unlink(obj_tmp)
    bpy.data.objects.remove(obj_tmp)
    bpy.data.meshes.remove(me_tmp)

    scene.update()

    return intersect

def instancesIntersect(instance1, instance2):
    if aabb_intersect(instance1, instance2):
        print ("There's an AABB intersection!")
        for mesh2 in instance2.dupli_group.objects:
            if bmesh_check_intersect_objects(instance1, instance1.matrix_world,  mesh2, instance2.matrix_world):
                return True
    else:
        print ("There's NO AABB intersection!")

    return False


