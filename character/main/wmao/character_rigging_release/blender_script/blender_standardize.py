# Blender
import argparse
import os
import sys
import bpy
import math
import mathutils
from mathutils import Vector, Matrix
import numpy as np
from pdb import set_trace as st

dirname, _ = os.path.split(os.path.abspath(__file__))
import sys
sys.path.append(os.path.abspath(os.getcwd()))

def clear_scene(remain_objects=[]):
    # Ensure we are in Object mode
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    
    for obj in bpy.context.scene.objects:
        if not obj.visible_get():
            bpy.data.objects.remove(obj, do_unlink=True)
    for obj in remain_objects:
        obj.select_set(False)
    bpy.ops.object.delete()

def join_objs(objs=[]):
    if len(objs) == 0:
        return None
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.join()
    return bpy.context.active_object


# Function to find the armature in the scene
def find_obj(type="ARMATURE", inverse=False, is_join=False, is_select=True):
    bpy.ops.object.select_all(action='DESELECT')
    objs = []
    for obj in bpy.context.scene.objects:
        if not inverse and obj.type == type:
            objs.append(obj)
            obj.select_set(is_select)
        elif inverse and obj.type != type:
            objs.append(obj)
            obj.select_set(is_select)
    if len(objs) > 0 and is_join and type=='MESH' and inverse==False:
        objs = [join_objs(objs)]
    return objs

def angle_axis_to_rotation_matrix(axis, angle):
    # Ensure the axis is a unit vector
    axis = axis / np.linalg.norm(axis)
    
    # Extract components of the axis vector
    x, y, z = axis
    
    # Compute the skew-symmetric matrix K
    K = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])
    
    # Compute the rotation matrix using Rodrigues' formula
    I = np.eye(3)  # Identity matrix
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    return R

def get_vertices(mesh_objs):
    verts = []
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in mesh_objs:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = obj
        for vert in obj.data.vertices:
            verts.append(np.array(vert.co))
    verts = np.vstack(verts)
    return verts

def set_vertices(mesh_objs, scale, center):
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in mesh_objs:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = obj
        for vert in obj.data.vertices:
            vert.co = (vert.co - center) * scale + Vector((0,1,0))
        obj.data.update()

def load_file(file_path, is_join=True):
    """
    load mesh file, and remove non-mesh objects
    """
    bpy.ops.object.select_all(action='DESELECT')
    if file_path.lower().endswith('.obj'):
        bpy.ops.wm.obj_import(filepath=file_path)
    elif file_path.lower().endswith('.fbx'):
        bpy.ops.import_scene.fbx(filepath=file_path)
    elif file_path.lower().endswith('.glb'):
        bpy.ops.import_scene.gltf(filepath=file_path)
    else:
        raise('unsupport file type')
        
    # remove invisible obejcts
    for obj in bpy.context.scene.objects:
        if not obj.visible_get():
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Get all selected objects
    selected_objects = bpy.context.selected_objects

    # Print the names of the selected objects
    objs = []
    other_objs = []
    for obj in selected_objects:
        print(obj.name)
        if obj.type == "MESH":
            bpy.context.view_layer.objects.active = obj
            for modifier in obj.modifiers:
                bpy.ops.object.modifier_apply(modifier=modifier.name)
            objs.append(obj)
        else:
            other_objs.append(obj)
    for obj in other_objs:
        bpy.data.objects.remove(obj, do_unlink=True)
    if is_join:        
        objs = [join_objs(objs)]
    for obj in objs:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(rotation=True) # apply existing rotation
        bpy.context.object.rotation_mode = 'XYZ'
        bpy.context.object.rotation_euler[0] = -1.5708
        bpy.ops.object.transform_apply(rotation=True) # remove default rotation by blender
    return objs



if __name__ == "__main__":

    # /usr/blender-4.0.1-linux-x64/blender --background --python ./blender_standardize.py -- --target_file /aigc_cfs_gdp/weimao/character_rigging/test_data/quad_mesh_3.glb --out_dir /aigc_cfs_gdp/weimao/character_rigging/output/
    argv = sys.argv
    arg_idx = argv.index("--")
    path = argv[arg_idx + 1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_file', type=str, default="/aigc_cfs_gdp/weimao/character_rigging/test_data/quad_mesh_3.glb")
    parser.add_argument('--out_dir', type=str, default='/aigc_cfs_gdp/weimao/character_rigging/output/')

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)
#    args = parser.parse_args()
  
    mesh_folder = os.path.split(args.target_file)[0]
    mesh_filename = os.path.split(args.target_file)[1]
    mesh_basename = os.path.splitext(mesh_filename)[0]
    mesh_extension = os.path.splitext(mesh_filename)[1]
    os.makedirs(os.path.dirname(args.out_dir),exist_ok=True)
    clear_scene()
    
    """
    >>>>>> standardize the input mesh so that it is within a unit sphere and the origin is at the bounding box center of the mesh
    """
    objs = load_file(args.target_file, is_join=False)
    verts = get_vertices(objs)
    # scale * (v - center)
    vmax = verts.max(axis=0)
    vmin = verts.min(axis=0)
    center = (vmax + vmin)/2
    # center = verts.mean(axis=0)
    # scale = 2 / np.linalg.norm(vmax - vmin)
    scale =  1 / np.max(np.linalg.norm(verts - center[None], axis=1))
    set_vertices(objs, scale, Vector(center))
    npz_path = os.path.join(args.out_dir, 'standardize_data.npz')
    np.savez_compressed(npz_path, center=center, scale=scale, des="vn = scale*(v-center)")
    # export to various data type
    bpy.ops.object.select_all(action='DESELECT')
    # add default rotation by blender
    for obj in objs:
        obj.rotation_euler[0] = 1.5708
        obj.select_set(True)

    # 导出为GLTF
    gltf_path = os.path.join(args.out_dir, mesh_basename + "_standardized.gltf")
    bpy.ops.export_scene.gltf(filepath=gltf_path, use_selection=True,
                              export_format='GLTF_SEPARATE', export_animations=False)
    # 删除导入的对象
    for obj in objs:
        bpy.data.objects.remove(obj, do_unlink=True)

    # 导入GLTF模型
    bpy.ops.import_scene.gltf(filepath=gltf_path)
    # 获取导入的对象
    imported_objects = bpy.context.selected_objects
    obj_path = os.path.join(args.out_dir, mesh_basename + "_standardized.glb")
    bpy.ops.export_scene.gltf(filepath=obj_path, use_selection=True,
                              export_format='GLB', export_animations=False)
    # 删除导入的对象
    for obj in imported_objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # 导入GLTF模型
    bpy.ops.import_scene.gltf(filepath=gltf_path)
    # 获取导入的对象
    imported_objects = bpy.context.selected_objects
    # 导出为FBX
    obj_path = os.path.join(args.out_dir, mesh_basename + "_standardized.fbx")
    bpy.ops.export_scene.fbx(
        filepath=obj_path, use_selection=True, bake_anim=False)