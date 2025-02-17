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
    parser.add_argument('--src_animation', type=str, default="/Users/weimao/Documents/character_rigging/debug_data/Boxing.fbx")
    parser.add_argument('--tgt_fbx', type=str, default="/Users/weimao/Documents/character_rigging/debug_data/character_rigging/mesh_rigged.fbx")
    parser.add_argument('--out_fbx', type=str, default='/Users/weimao/Documents/character_rigging/debug_data/character_rigging/mesh_boxing_test.fbx')
#    args = parser.parse_args()
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    clear_scene()
    os.makedirs(os.path.dirname(args.out_fbx),exist_ok=True)
    
    try:
        bpy.ops.import_scene.fbx(filepath=args.src_animation)
    except Exception as e:
        print(f'error loading fbx {e}')
    src_ani = bpy.context.object
    # clear the rotation
    bpy.ops.object.rotation_clear(clear_delta=False)
    bpy.ops.object.select_all(action='DESELECT')
    armature_ani = find_obj(type="ARMATURE", inverse=False)[0]
    # assume the scale of animation is different from that of the targeting armature and targeting armature's scale is 1.0
    scale_ani = armature_ani.scale[0]
    print('over all scale is', scale_ani)
    
    action = src_ani.animation_data.action
    kf = get_keyframes(action)
    nf = len(kf)
    bpy.context.scene.frame_end = nf
    try:
        bpy.ops.import_scene.fbx(filepath=args.tgt_fbx)
    except Exception as e:
        print(f'error loading fbx {e}')
    tgt_obj = bpy.context.object
#    assert False

    joint_mapper_openxr2mixamo={
        "Hips":                "mixamorig:Hips",
        "LeftUpLeg":           "mixamorig:LeftUpLeg",   
        "LeftLeg":             "mixamorig:LeftLeg",     
        "LeftFoot":            "mixamorig:LeftFoot",    
        "LeftToeEnd":          "mixamorig:LeftToeBase", 
        "RightUpLeg":          "mixamorig:RightUpLeg",  
        "RightLeg":            "mixamorig:RightLeg",    
        "RightFoot":           "mixamorig:RightFoot",   
        "RightToeEnd":         "mixamorig:RightToeBase",
        "Spine":               "mixamorig:Spine",       
        "Spine1":              "mixamorig:Spine1",      
        "Spine2":              "mixamorig:Spine2",      
        "Neck":                "mixamorig:Neck",        
        "Head":                "mixamorig:Head",        
        "jaw":                 "",
        "left_eye_smplhf":     "",
        "right_eye_smplhf":    "",
        "LeftShoulder":        "mixamorig:LeftShoulder",
        "LeftArm":             "mixamorig:LeftArm",     
        "LeftForeArm":         "mixamorig:LeftForeArm", 
        "LeftHand":            "mixamorig:LeftHand",    
        "LeftHandIndex1":      "mixamorig:LeftHandIndex1",  
        "LeftHandIndex2":      "mixamorig:LeftHandIndex2",  
        "LeftHandIndex3":      "mixamorig:LeftHandIndex3",
        "LeftHandMiddle1":     "mixamorig:LeftHandMiddle1", 
        "LeftHandMiddle2":     "mixamorig:LeftHandMiddle2", 
        "LeftHandMiddle3":     "mixamorig:LeftHandMiddle3", 
        "LeftHandPinky1":      "mixamorig:LeftHandRing1",   
        "LeftHandPinky2":      "mixamorig:LeftHandRing2",   
        "LeftHandPinky3":      "mixamorig:LeftHandRing3",   
        "LeftHandRing1":       "mixamorig:LeftHandPinky1",  
        "LeftHandRing2":       "mixamorig:LeftHandPinky2",  
        "LeftHandRing3":       "mixamorig:LeftHandPinky3", 
        "LeftHandThumb1":      "mixamorig:LeftHandThumb1",  
        "LeftHandThumb2":      "mixamorig:LeftHandThumb2",  
        "LeftHandThumb3":      "mixamorig:LeftHandThumb3",   
        "RightShoulder":       "mixamorig:RightShoulder",   
        "RightArm":            "mixamorig:RightArm",        
        "RightForeArm":        "mixamorig:RightForeArm",    
        "RightHand":           "mixamorig:RightHand",       
        "RightHandIndex1":     "mixamorig:RightHandIndex1", 
        "RightHandIndex2":     "mixamorig:RightHandIndex2", 
        "RightHandIndex3":     "mixamorig:RightHandIndex3", 
        "RightHandMiddle1":    "mixamorig:RightHandMiddle1",
        "RightHandMiddle2":    "mixamorig:RightHandMiddle2",
        "RightHandMiddle3":    "mixamorig:RightHandMiddle3",
        "RightHandPinky1":     "mixamorig:RightHandRing1",  
        "RightHandPinky2":     "mixamorig:RightHandRing2",  
        "RightHandPinky3":     "mixamorig:RightHandRing3",  
        "RightHandRing1":      "mixamorig:RightHandPinky1", 
        "RightHandRing2":      "mixamorig:RightHandPinky2", 
        "RightHandRing3":      "mixamorig:RightHandPinky3",
        "RightHandThumb1":     "mixamorig:RightHandThumb1", 
        "RightHandThumb2":     "mixamorig:RightHandThumb2", 
        "RightHandThumb3":     "mixamorig:RightHandThumb3"
    }
    joint_mapper_mixamo2openxr ={
        "mixamorig:Hips":               "Hips",           
        "mixamorig:LeftUpLeg":          "LeftUpLeg",         
        "mixamorig:LeftLeg":            "LeftLeg",   
        "mixamorig:LeftFoot":           "LeftFoot",          
        "mixamorig:LeftToeBase":        "LeftToeEnd",        
        "mixamorig:RightUpLeg":         "RightUpLeg",        
        "mixamorig:RightLeg":           "RightLeg",          
        "mixamorig:RightFoot":          "RightFoot",         
        "mixamorig:RightToeBase":       "RightToeEnd",       
        "mixamorig:Spine":              "Spine",             
        "mixamorig:Spine1":             "Spine1",            
        "mixamorig:Spine2":             "Spine2",            
        "mixamorig:Neck":               "Neck",              
        "mixamorig:Head":               "Head",         
        "mixamorig:LeftShoulder":       "LeftShoulder",      
        "mixamorig:LeftArm":            "LeftArm",           
        "mixamorig:LeftForeArm":        "LeftForeArm",       
        "mixamorig:LeftHand":           "LeftHand",          
        "mixamorig:LeftHandIndex1":     "LeftHandIndex1",    
        "mixamorig:LeftHandIndex2":     "LeftHandIndex2",    
        "mixamorig:LeftHandIndex3":     "LeftHandIndex3",    
        "mixamorig:LeftHandMiddle1":    "LeftHandMiddle1",   
        "mixamorig:LeftHandMiddle2":    "LeftHandMiddle2",   
        "mixamorig:LeftHandMiddle3":    "LeftHandMiddle3",   
        "mixamorig:LeftHandRing1":      "LeftHandPinky1",   
        "mixamorig:LeftHandRing2":      "LeftHandPinky2",    
        "mixamorig:LeftHandRing3":      "LeftHandPinky3",    
        "mixamorig:LeftHandPinky1":     "LeftHandRing1",     
        "mixamorig:LeftHandPinky2":     "LeftHandRing2",     
        "mixamorig:LeftHandPinky3":     "LeftHandRing3",     
        "mixamorig:LeftHandThumb1":     "LeftHandThumb1",    
        "mixamorig:LeftHandThumb2":     "LeftHandThumb2",    
        "mixamorig:LeftHandThumb3":     "LeftHandThumb3",    
        "mixamorig:RightShoulder":      "RightShoulder",    
        "mixamorig:RightArm":           "RightArm",          
        "mixamorig:RightForeArm":       "RightForeArm",      
        "mixamorig:RightHand":          "RightHand",         
        "mixamorig:RightHandIndex1":    "RightHandIndex1",   
        "mixamorig:RightHandIndex2":    "RightHandIndex2",   
        "mixamorig:RightHandIndex3":    "RightHandIndex3",   
        "mixamorig:RightHandMiddle1":   "RightHandMiddle1",  
        "mixamorig:RightHandMiddle2":   "RightHandMiddle2",  
        "mixamorig:RightHandMiddle3":   "RightHandMiddle3",  
        "mixamorig:RightHandRing1":     "RightHandPinky1",   
        "mixamorig:RightHandRing2":     "RightHandPinky2",   
        "mixamorig:RightHandRing3":     "RightHandPinky3",   
        "mixamorig:RightHandPinky1":    "RightHandRing1",    
        "mixamorig:RightHandPinky2":    "RightHandRing2",    
        "mixamorig:RightHandPinky3":    "RightHandRing3",    
        "mixamorig:RightHandThumb1":    "RightHandThumb1",   
        "mixamorig:RightHandThumb2":    "RightHandThumb2",   
        "mixamorig:RightHandThumb3":    "RightHandThumb3"
    }
    for fi in range(1, nf+1):
        bpy.context.scene.frame_set(fi)
        for i, bone in enumerate(tgt_obj.pose.bones):
            bone.rotation_mode = 'QUATERNION'
            bname = bone.name
            if 'mixamorig' not in bname:
                bname = joint_mapper_openxr2mixamo[bname]
            src_bone = src_ani.pose.bones.get(bname)
            print(bname)
            src_bone.rotation_mode = 'QUATERNION'
            
            rot_quat = src_bone.rotation_quaternion
            bone.rotation_quaternion = rot_quat
            if i == 0:
                bone.location[1] = src_bone.location[1] * scale_ani
                bone.keyframe_insert(data_path='location', index=-1)        
            # Insert keyframes for rotation and location of the pose bone

            bone.keyframe_insert(data_path='rotation_quaternion', index=-1)
#        assert False
    bpy.ops.object.select_all(action='DESELECT')
    select_object_hierarchical(src_ani)
    bpy.ops.object.delete()
#    bpy.data.objects.remove(src_ani, do_unlink=True)
    
    obj = find_obj('MESH')[0]
    obj.data.use_auto_smooth = False
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj

    try:
        bpy.context.object.active_material.blend_method = 'CLIP'
    except:
        print('error on changing blend method')
    obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=args.out_fbx.replace('.fbx','.glb'), export_format='GLB',  export_yup=False, export_normals=True) 
    
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
    
    