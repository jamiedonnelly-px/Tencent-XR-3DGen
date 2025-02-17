# Blender
import argparse
import os
import sys
import bpy
import math
# import json
# import torch
import mathutils
from mathutils import Vector
import numpy as np
# from math import radians
# from pathlib import Path
from pdb import set_trace as st


dirname, _ = os.path.split(os.path.abspath(__file__))
# root_dir = os.path.join(dirname, "../../")
# sys.path.append(os.path.join(root_dir, "delta"))
import sys
sys.path.append(os.path.abspath(os.getcwd()))

def clear_scene():
    # Ensure we are in Object mode
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


# Function to find the armature in the scene
def find_obj(type="ARMATURE", inverse=False):
    objs = []
    for obj in bpy.context.scene.objects:
        if not inverse and obj.type == type:
            objs.append(obj)
        elif inverse and obj.type != type:
            objs.append(obj)
    return objs



if __name__ == "__main__":

    # /root/blender-4.0.1-linux-x64/blender --background --python ./blender/rigging.py -- --template_fbx_dir "/aigc_cfs_2/weimao/non-smalfit/dataset/SK_Dolphin/template.fbx" --rigging_info_dir "/aigc_cfs_2/weimao/non-smalfit/output/dolphin/rigging/rigging.npz" --out_dir "/aigc_cfs_2/weimao/non-smalfit/output/dolphin/rigging/"
    argv = sys.argv
    arg_idx = argv.index("--")
    path = argv[arg_idx + 1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--template_fbx_dir', type=str, default="/aigc_cfs_2/weimao/non-smalfit/dataset/SK_Elephant/SK_Elephant.FBX")
    parser.add_argument('--rigging_info_dir', type=str, default="/aigc_cfs_2/weimao/non-smalfit/output/elephant4/rigging/rigging.npz")
    # parser.add_argument('--origin_mesh', type=str, default="/aigc_cfs_2/weimao/non-smalfit/output/elephant4/rigging/rigging.npz")
    parser.add_argument('--out_dir', type=str, default='/aigc_cfs_2/weimao/non-smalfit/output/elephant4/rigging/')

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    clear_scene()
    os.makedirs(args.out_dir,exist_ok=True)
    
    try:
        bpy.ops.import_scene.fbx(filepath=args.template_fbx_dir)
    except Exception as e:
        print(f'error loading fbx {e}')
        
    bpy.ops.object.select_all(action='DESELECT')

    # load template armature
    armature_template = find_obj("ARMATURE")[0]
    
    # delete other objects
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    armature_template.select_set(False)
    bpy.ops.object.delete()
#    assert False

    rigging_data = np.load(args.rigging_info_dir)
    verts = rigging_data["verts_cano"]
    faces = rigging_data["faces"]
    joint_name = rigging_data["joint_name"]
    joint_cano = rigging_data["joint_cano"]
    weights = rigging_data["weights"]
    parents = rigging_data["parents"]
    
    vertices_blender = []
    faces_blender = []
    for v in verts:
        vertices_blender.append((v[0], v[1], v[2]))
    for f in faces:
        faces_blender.append((f[0], f[1], f[2]))
    joints_blender = []
    for joint_index in range(joint_cano.shape[0]):
        j = joint_cano[joint_index]
        # somehow the armature is rotated 90 degree along x axis
        # joints_blender.append((j[0], j[2], -j[1]))
        joints_blender.append((j[0], j[1], j[2]))
    name = "animal"
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices_blender, [], faces_blender)
    obj = bpy.data.objects.new(name, mesh)
    scene = bpy.context.scene
    scene.collection.objects.link(obj)
    obj.select_set(False)
    
    # edit armature location before binding
    scale = armature_template.scale[0]
#    armature_template.scale = Vector((1.0,1.0,1.0))
    armature_template.select_set(True)
    bpy.context.view_layer.objects.active = armature_template
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature_template.data.edit_bones
    for ji, jn in enumerate(joint_name):
        if jn in edit_bones:
            bone = edit_bones[jn]
            jpos = Vector(joints_blender[ji]) / scale
            dt = jpos - bone.head
            bone.head = bone.head + dt
            bone.tail = bone.tail + dt
        print(jn)
    bpy.ops.object.mode_set(mode='OBJECT')
    # Bind mesh to armature (skinning)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_template
    bpy.ops.object.parent_set(type="ARMATURE_NAME") # Create empty vertex groups
    
    # Set skin weights
    bpy.context.view_layer.objects.active = obj
    lbs_weights = weights
    for index, vertex_weights in enumerate(lbs_weights):
        for joint_index, joint_weight in enumerate(vertex_weights):
            if joint_weight > 0.0:
                vg = obj.vertex_groups[joint_name[joint_index]]
                vg.add([index], joint_weight, "REPLACE")

    # Use smooth normals
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    
    use_selection_flag = False
#    if obj_names:
#        bpy.ops.object.select_all(action='DESELECT')
#        select(obj_names)
#        use_selection_flag = True

     # Set export options
    export_settings = {
        'mesh_smooth_type': 'OFF',  # Options: 'OFF', 'FACE', 'EDGE', 'CUSTOM'
        'use_selection': use_selection_flag,
        'use_mesh_modifiers': True,
        'use_armature_deform_only': True,
        'bake_anim_use_all_actions': True,
        'bake_anim_simplify_factor': 1.0,
        'path_mode': 'COPY',  # Options: 'AUTO', 'ABSOLUTE', 'RELATIVE', 'COPY'
        'embed_textures': True,  # Embed textures in FBX
        'add_leaf_bones': False,
        'primary_bone_axis': 'Y',
        'secondary_bone_axis': 'X',
        'axis_forward': 'Y',
        'axis_up': 'Z',
        'global_scale': 1.0,
        'apply_unit_scale': True,
        'apply_scale_options': 'FBX_SCALE_NONE',
        'bake_space_transform': False,
        'object_types': {'MESH', 'ARMATURE'},
    }
    
    # bpy.ops.export_scene.fbx(filepath=file_path, use_selection=use_selection_flag, embed_textures=True, path_mode='COPY')   
    bpy.ops.export_scene.fbx(filepath=args.out_dir+'/mesh_rigged.fbx', check_existing=False, **export_settings)  
#    assert False
    
