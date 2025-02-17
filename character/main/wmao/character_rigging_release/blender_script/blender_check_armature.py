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

    # return objs



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
    objs = load_file(args.target_file, is_join=False)
    armatures = find_obj('ARMATURE')
    npy_path = os.path.join(args.out_dir, 'has_armature.npy')
    np.save(npy_path, int(len(armatures)>0))
    