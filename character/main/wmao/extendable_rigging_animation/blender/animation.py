import bpy
import numpy as np
import mathutils
import math
import copy
import argparse
import os
import sys


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

def get_keyframes(action):
    keyframes = set()
    for fcu in action.fcurves:
        for keyframe in fcu.keyframe_points:
            keyframes.add(keyframe.co[0])
    return sorted(keyframes)

def select_object_hierarchical(obj):
    obj.select_set(True)
    for child in obj.children:
        select_object_hierarchical(child)

if __name__ == "__main__":
    # /root/blender-3.5.0-linux-x64/blender -b --python ./blender/animation.py -- --src_animation /aigc_cfs_2/weimao/non-smalfit/dataset/SK_Dolphin/animation/Jump.fbx --tgt_fbx /aigc_cfs_2/weimao/non-smalfit/output/dolphin/rigging/mesh_rigged.fbx --out_fbx /aigc_cfs_2/weimao/non-smalfit/output/dolphin/animation/Jump.fbx
    argv = sys.argv
    arg_idx = argv.index("--")
    path = argv[arg_idx + 1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_animation', type=str, default="/Users/weimao/Documents/test_lbs/SK_Elephant/Elephant@Run_RM.FBX")
    parser.add_argument('--tgt_fbx', type=str, default="/Users/weimao/Documents/test_lbs/output/elephant4/rigging/rigged_mesh.fbx")
    parser.add_argument('--out_fbx', type=str, default='/Users/weimao/Documents/test_lbs/output/elephant4/animation/Run_RM.fbx')
    # args = parser.parse_args()
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    clear_scene()
    os.makedirs(os.path.dirname(args.out_fbx),exist_ok=True)
    
    try:
        bpy.ops.import_scene.fbx(filepath=args.src_animation)
    except Exception as e:
        print(f'error loading fbx {e}')
    src_ani = bpy.context.object
    bpy.ops.object.select_all(action='DESELECT')
    
    action = src_ani.animation_data.action
    kf = get_keyframes(action)
    nf = len(kf)
    bpy.context.scene.frame_end = nf
    try:
        bpy.ops.import_scene.fbx(filepath=args.tgt_fbx)
    except Exception as e:
        print(f'error loading fbx {e}')
    tgt_obj = bpy.context.object
    
    for i in range(1, nf+1):
        bpy.context.scene.frame_set(i)
        for i, bone in enumerate(tgt_obj.pose.bones):
            bone.rotation_mode = 'QUATERNION'
            bname = bone.name
            src_bone = src_ani.pose.bones.get(bname)
            src_bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = src_bone.rotation_quaternion
#            if i == 0:
            bone.location = src_bone.location
            bone.keyframe_insert(data_path='location', index=-1)        
            # Insert keyframes for rotation and location of the pose bone
            bone.keyframe_insert(data_path='rotation_quaternion', index=-1)
    
    bpy.ops.object.select_all(action='DESELECT')
    select_object_hierarchical(src_ani)
    bpy.ops.object.delete()
#    bpy.data.objects.remove(src_ani, do_unlink=True)
#    assert False
    
    export_settings = {
        'mesh_smooth_type': 'OFF',  # Options: 'OFF', 'FACE', 'EDGE', 'CUSTOM'
        'use_selection': False,
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
    
    bpy.ops.export_scene.fbx(filepath=args.out_fbx, check_existing=False, **export_settings)  
