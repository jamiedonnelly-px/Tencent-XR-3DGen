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

def threejs_euler2rot(jsx, jsy, jsz):
    jsx, jsy, jsz = np.deg2rad(jsx),np.deg2rad(jsy),np.deg2rad(jsz)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(jsx), -np.sin(jsx)],
        [0, np.sin(jsx), np.cos(jsx)]
    ])
    
    Ry = angle_axis_to_rotation_matrix(Rx[1], jsy)
    
    R = Ry @ Rx

    Rz = angle_axis_to_rotation_matrix(R[2], jsz)
    
    R = Rz @ Ry @ Rx
    return R

def get_blending_weights(mesh_obj, armature, joint_names):
    # get blending weights
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    weights = []
    joint_names_update = []
    for jn in joint_names:
        if jn in armature.pose.bones:
            joint_names_update.append(jn)
    joint_names = np.array(joint_names_update)
    nb = len(joint_names)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_mesh = mesh_obj.evaluated_get(depsgraph)
    mesh = eval_mesh.data
    
    # Access vertex groups (bones) attached to this mesh
    vertex_groups = mesh_obj.vertex_groups
    n_verts = len(mesh.vertices)
    w = np.zeros([n_verts, nb])
    vs = np.zeros([n_verts, 3]) 
    for vgroup in vertex_groups:
        print('vgroup name', vgroup.name)
        ji = np.where(vgroup.name==joint_names)[0].item()
        for vi, v in enumerate(mesh.vertices):
            # Iterate through each vertex group and get the weight for this vertex
            vs[vi] = np.array(list(obj.matrix_world @ v.co))
            for gp in v.groups:
                if gp.group == vgroup.index:
                    w[vi,ji] = gp.weight
                
    # Clean up: free the evaluated mesh to avoid memory leaks
    eval_mesh.to_mesh_clear()  
    return vs, w, joint_names

def change_bone_orientation(armature, bone_orient):
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.armature.select_all(action='DESELECT')
    edit_bones = armature_template.data.edit_bones
    rot_resi = {}
    for bone in edit_bones:
        matrix_orig = Matrix(bone_orient[bone.name].tolist()).to_3x3()
        matrix_now = bone.matrix.to_3x3()
        rot_resi[bone.name] = matrix_now.inverted() @ matrix_orig
    
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.parent_clear(type='CLEAR')
       
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='POSE')
    for bone in armature_template.pose.bones:

        rot = rot_resi[bone.name].to_quaternion()
        bone.rotation_mode = 'QUATERNION'
        bone.rotation_quaternion = rot
        print(bone.name)
        print(np.array(bone.matrix))
        print(bone_orient[bone.name])
    bpy.ops.pose.armature_apply()

def connect_armature(armature, parents, joint_names):
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.armature.select_all(action='DESELECT')
    for jn, pi in zip(joint_names, parents):
        if jn not in armature.data.edit_bones:
            continue
        if pi < 0:
            continue
        pn = joint_names[pi]
        print(f'parent {pn}, joint {jn}')
        armature.data.edit_bones[jn].select = True
        armature.data.edit_bones[pn].select = True
        armature.data.edit_bones.active = armature.data.edit_bones[pn]
        if 'Shoulder' in jn or 'UpLeg' in jn or 'Pinky1' in jn or 'Thumb1' in jn or 'Index1' in jn or 'Middle1' in jn or 'Ring1':
            bpy.ops.armature.parent_set(type='OFFSET')
        else:
            bpy.ops.armature.parent_set(type='CONNECTED')
        bpy.ops.armature.select_all(action='DESELECT')
        
def add_empty(rot):
    rot = Matrix(rot.tolist())
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.empty_add(type='ARROWS', align='WORLD', 
        location=(0, 0, 0), scale=(0.1, 0.1, 0.1))
    emp = bpy.context.object
    emp.rotation_mode = 'XYZ'
    emp.rotation_euler = rot.to_euler()
    emp.location = rot.to_translation() * 0.01
    
def rotate_bone(bone, pt1, pt2, is_right=True):
    diff = np.array(pt2 - pt1)
    diff = diff / (np.linalg.norm(diff) + 1e-10)
    if is_right:
        xax = np.array([-1,0,0.])
    else:
        xax = np.array([1,0,0.])
        
    ang = np.arccos((diff * xax).sum())
    print(f'direction:{diff};ang {ang}')
    bone.bone.select = True
    if is_right:
        bpy.ops.transform.rotate(value=-ang, orient_axis='Z', orient_type='GLOBAL', 
                                orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), 
                                mirror=False, snap=False, snap_elements={'INCREMENT'}, 
                                use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, 
                                use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False, 
                                release_confirm=True)
    else:
        bpy.ops.transform.rotate(value=ang, orient_axis='Z', orient_type='GLOBAL', 
                                orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), 
                                mirror=False, snap=False, snap_elements={'INCREMENT'}, 
                                use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, 
                                use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False, 
                                release_confirm=True)
        
    bone.bone.select = False

def rotate_bone_general(armature_template, bone_name, target_dir, rot_ord='ZY'):
    for ro in rot_ord:
        if ro == "Z":
            i = 2
        elif ro == 'Y':
            i = 1
        elif ro == 'X':
            i = 0
        bone = armature_template.pose.bones[bone_name]
        bone.bone.select = True
        tail = bone.tail * armature_template.scale[0]
        head = bone.head * armature_template.scale[0]
        diff = np.array(tail - head)
        # get z rotation
        mask = np.array([1, 1, 1.])
        mask[i] = 0
        diff = (diff*mask) / max(np.linalg.norm(diff*mask), 1e-10)
        theta = np.arccos((target_dir * diff * mask).sum())
        ax = np.cross(diff*mask, target_dir*mask)
        #rot = angle_axis_to_rotation_matrix(ax,theta)
        if ax[i] != 0:
            theta = ax[i]/np.abs(ax[i]) * theta
            bpy.ops.transform.rotate(value=theta, orient_axis=ro, orient_type='GLOBAL', 
                                    orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                    orient_matrix_type='GLOBAL', constraint_axis=(ro=='X', ro=='Y', ro=='Z'), 
                                    mirror=False, snap=False, snap_elements={'INCREMENT'}, 
                                    use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, 
                                    use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False, 
                                    release_confirm=True)
        
        bone.bone.select = False

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

def rotate_meshes(objs, euler_angles, mode='ZYX', center=(0,0,0)):
    # euler_angles = [np.deg2rad(float(args.threejs_x)), 
    #                 np.deg2rad(float(args.threejs_y)), 
    #                 np.deg2rad(float(args.threejs_z))]
    for obj in objs:
        bpy.context.view_layer.objects.active = obj
        obj.location[0] = -center[0]
        obj.location[1] = -center[1]
        obj.location[2] = -center[2]
        bpy.ops.object.transform_apply(location=True)
        obj.rotation_mode = mode
        obj.rotation_euler = euler_angles    
        bpy.ops.object.transform_apply(rotation=True)
        obj.location[0] = center[0]
        obj.location[1] = center[1]
        obj.location[2] = center[2]
        bpy.ops.object.transform_apply(location=True)

def get_joint_location(armature, bone_name):
    # Access the evaluated object through the dependency graph
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_armature = armature.evaluated_get(depsgraph)
    # Get the pose bone's world-space location
    pose_bone_name = bone_name  # Replace with your bone's name
    pose_bone = evaluated_armature.pose.bones[pose_bone_name]
    # Get the world-space location
    bone_world_matrix = evaluated_armature.matrix_world @ pose_bone.matrix
    bone_head_world = bone_world_matrix.translation
    return bone_head_world

if __name__ == "__main__":

#    # /root/blender-4.0.1-linux-x64/blender --background --python ./blender/rigging.py -- --template_fbx_dir "/aigc_cfs_2/weimao/non-smalfit/dataset/SK_Dolphin/template.fbx" --rigging_info_dir "/aigc_cfs_2/weimao/non-smalfit/output/dolphin/rigging/rigging.npz" --out_dir "/aigc_cfs_2/weimao/non-smalfit/output/dolphin/rigging/"
    argv = sys.argv
    arg_idx = argv.index("--")
    path = argv[arg_idx + 1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--template_fbx_dir', type=str, default="/home/wei/Downloads/akai_e_espiritu.fbx")
    parser.add_argument('--joint_dir', type=str, default="/home/wei/Downloads/character_rigging/joint3d_v2.npz")
    parser.add_argument('--target_dir', type=str, default="/home/wei/Downloads/character_rigging/123230_cfebf1825eef3cd30bfbea3a8fe4947b_standardized.glb")
    parser.add_argument('--out_file', type=str, default='/home/wei/Downloads/character_rigging/mesh_rigged1.fbx')
    parser.add_argument('--threejs_x', type=str, default='90')
    parser.add_argument('--threejs_y', type=str, default='0')
    parser.add_argument('--threejs_z', type=str, default='0')

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)
    # args = parser.parse_args()
  
    clear_scene()
    os.makedirs(os.path.dirname(args.out_file),exist_ok=True)
    
    """
    >>>>>> step 1, Load template armature
    """
    
    try:
        bpy.ops.import_scene.fbx(filepath=args.template_fbx_dir)
    except Exception as e:
        print(f'error loading fbx {e}')
        
    bpy.ops.object.select_all(action='DESELECT')

    # load template armature
    armature_template = find_obj("ARMATURE")[0]
    bpy.context.object.show_in_front = True
    bpy.context.object.data.show_axes = True
    
    clear_scene(remain_objects=[armature_template])
    armature_template.select_set(True)
    bpy.context.view_layer.objects.active = armature_template
    armature_template.rotation_mode = 'XYZ'
    armature_template.rotation_euler = Vector((0,0,0))
    # Loop through all actions in the current armature's action list
    for action in bpy.data.actions:
        if action.users > 0:
            # Loop through the FCurves of the action
            for fcurve in action.fcurves:
                if fcurve.data_path.startswith('pose.bones'):  # Check if it's a bone keyframe
                    # Clear all keyframe points
                    fcurve.keyframe_points.clear()

    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.loc_clear()
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    scale = armature_template.scale[0]
    print('over all scale is', scale)
    rigging_data = np.load(args.joint_dir)
    joint_names = rigging_data["joint_names"]
    parents = rigging_data["parents"]
    print('joint_names', joint_names)
    joints = {}
    joints_to_remove = []
    for jn in joint_names:
        if len(rigging_data[jn]) == 0:
            joints_to_remove.append(jn)
            joints[jn] = None
        else:
            joints[jn] = rigging_data[jn] / scale
    is_mixamo = True
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
    if 'mixamorig' not in joint_names[0]:
        is_mixamo = False

    # mixamo_joint_names = np.array([
    #               "mixamorig:Hips",
    #               "mixamorig:Spine",
    #               "mixamorig:Spine1",
    #               "mixamorig:Spine2",
    #               "mixamorig:Neck",
    #               "mixamorig:Head",
    #               "mixamorig:RightShoulder",
    #               "mixamorig:RightArm",
    #               "mixamorig:RightForeArm",
    #               "mixamorig:LeftShoulder",
    #               "mixamorig:LeftArm",
    #               "mixamorig:LeftForeArm",
    #               "mixamorig:RightUpLeg",
    #               "mixamorig:RightLeg",
    #               "mixamorig:LeftUpLeg",
    #               "mixamorig:LeftLeg"])
    
    """
    >>>>>> step 2, Load target mesh, manifold it and skinning: 
        apply all existing modifiers and remove objects that are not mesh.
    """
    obj = load_file(args.target_dir, is_join=True)[0]
    # update 01-11-24: the rotation centre is changed from (0,0,0) to (0,1,0)
    # bpy.context.object.location[1] = -1
    # bpy.ops.object.transform_apply(location=True)
    # euler_angles = [np.deg2rad(float(args.threejs_x)), 
    #                 np.deg2rad(float(args.threejs_y)), 
    #                 np.deg2rad(float(args.threejs_z))]
    # obj.rotation_mode = 'ZYX'
    # obj.rotation_euler = euler_angles    
    # bpy.ops.object.transform_apply(rotation=True)
    # bpy.context.object.location[1] = 1
    # bpy.ops.object.transform_apply(location=True)
    euler_angles = [np.deg2rad(float(args.threejs_x)), 
                    np.deg2rad(float(args.threejs_y)), 
                    np.deg2rad(float(args.threejs_z))]
    rotate_meshes([obj], euler_angles, mode='ZYX', center=(0,1,0))

    # # use voxel remesh may cause artifacts for fingles 
    # bpy.ops.object.mode_set(mode='OBJECT')
    # bpy.context.object.data.remesh_voxel_size = 0.01
    # bpy.ops.object.voxel_remesh()
    # # simplify the mesh to speed up
    # decimate_mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
    # decimate_mod.ratio = 5000.0/len(obj.data.vertices)
    # bpy.ops.object.modifier_apply(modifier="Decimate")

    # combine duplicated vertices
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.remove_doubles(threshold=0.001)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    bpy.context.view_layer.objects.active = armature_template
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.armature.select_all(action='DESELECT')
    edit_bones = armature_template.data.edit_bones
    bone_orient_dict = {}
    bone_names = []
    for bone in edit_bones:
        if not is_mixamo:
            try:
                bone.name = joint_mapper_mixamo2openxr[bone.name]
            except:
                print(f'bone {bone.name} did not find match in openxr')
        bone_orient_dict[bone.name] = np.array(bone.matrix)
        bone_names.append(bone.name)
    for bn in bone_names:
        if bn not in joint_names or bn in joints_to_remove:
            print('remove bone', bn)
            edit_bones.remove(edit_bones[bn])
    # for ji, jn in enumerate(joint_names):
    #     if joints[jn] is None:
    #         continue
    for bone in edit_bones:
        jn = bone.name
        print(f'{bone.name}:{bone.head}_{bone.tail}')
        p = joints[jn]
        if len(p.shape) == 2 and p.shape[0] == 2:
            p1 = p[0]
            p2 = p[1]
        else:
            p1 = p[0]
            p2 = p1 + 0.4 * (p1 - np.array(bone.parent.head))
        bone.head = p1
        bone.tail = p2 
        print(f'after transform {bone.head}_{bone.tail}')
    
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_template
    bpy.ops.object.parent_set(type="ARMATURE_AUTO") # Create empty vertex groups    
    
    # get the blending weights and vertex locations
    verts_combined, weights, joint_names_update = get_blending_weights(obj, armature_template, joint_names)
#    # nverts x nb
#    vids = np.where(weights.sum(axis=-1)<0.01)[0]
#    vids_remain = np.setdiff1d(np.arange(verts_combined.shape[0]),vids)
#    for vid in vids:
#        dist = np.linalg.norm(verts_combined[vids_remain] - verts_combined[vid], axis=1)
#        idcand = np.argmin(dist)
#        weights[vid] = weights[vids_remain[idcand]]
    vids_zerow = np.where(weights.sum(axis=-1)<0.01)[0] 
    if len(vids_zerow) > 0:
        # get hand verts
        hand_joint_id = []
        for ji, joint_name in enumerate(joint_names_update):
            if 'Hand' in joint_name:
                hand_joint_id.append(ji)
        hand_vid = np.where(weights[:,hand_joint_id].sum(axis=-1) > 0.5)[0]
        hand_verts = verts_combined[hand_vid]
        hand_weights = weights[hand_vid]
        
        bpy.ops.object.mode_set(mode='OBJECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        # use voxel remesh may cause artifacts for fingles 
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.object.data.remesh_voxel_size = 0.01
        bpy.ops.object.voxel_remesh()
        obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_template
        bpy.ops.object.parent_set(type="ARMATURE_AUTO") # Create empty vertex groups 
        verts_combined, weights, joint_names_update = get_blending_weights(obj, armature_template, joint_names_update) 
        
        # get hand verts
        body_joint_id = []
        for ji, joint_name in enumerate(joint_names_update):
            if 'Hand' not in joint_name:
                body_joint_id.append(ji)
        body_vid = np.where(weights[:,body_joint_id].sum(axis=-1) > 0.5)[0]
        body_verts = verts_combined[body_vid]
        body_weights = weights[body_vid]
        
        verts_combined = np.concatenate([hand_verts,body_verts],axis=0)
        weights = np.concatenate([hand_weights,body_weights],axis=0)
     # 
     
     
#    assert False, len(np.where(weights.sum(axis=-1)<0.01)[0])
    
    
    # move the weights for shoulder to last spine
    if is_mixamo:
        rshidx = np.where(joint_names_update=="mixamorig:RightShoulder")
        lshidx = np.where(joint_names_update=="mixamorig:LeftShoulder")
        sp2idx = np.where(joint_names_update=="mixamorig:Spine2")
    else:
        rshidx = np.where(joint_names_update=="RightShoulder")
        lshidx = np.where(joint_names_update=="LeftShoulder")
        sp2idx = np.where(joint_names_update=="Spine2") 
    weights[:, sp2idx] = weights[:,sp2idx] + weights[:,rshidx] + weights[:, lshidx]
    weights[:, lshidx] = 0
    weights[:, rshidx] = 0
    
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_template
    bpy.ops.object.parent_clear(type='CLEAR')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.delete()
    

    """
    >>>>>> step 3, Load target mesh and copy the weights from manifoled mesh to this mesh: 
        apply all existing modifiers and remove objects that are not mesh.
    """
    # copy the blending weights to the initial mesh
    objs_orig = load_file(args.target_dir, is_join=False)
    for obj_orig in objs_orig:
        # bpy.context.view_layer.objects.active = obj_orig
        # euler_angles = [np.deg2rad(float(args.threejs_x)), 
        #                 np.deg2rad(float(args.threejs_y)), 
        #                 np.deg2rad(float(args.threejs_z))]
        # obj_orig.rotation_mode = 'ZYX'
        # obj_orig.rotation_euler = euler_angles    
        # bpy.ops.object.transform_apply(rotation=True)
        rotate_meshes([obj_orig], euler_angles, mode='ZYX', center=(0,1,0))


        obj_orig.select_set(True)
        bpy.context.view_layer.objects.active = armature_template
        bpy.ops.object.parent_set(type='ARMATURE_NAME') # Create empty vertex groups 
        
        # set blending weights
        bpy.context.view_layer.objects.active = obj_orig
        for index, v in enumerate(obj_orig.data.vertices):
            vert = np.array(obj_orig.matrix_world @ v.co)
            dist = np.linalg.norm(vert[None] - verts_combined,axis=-1)
            idx = np.argmin(dist)
            vertex_weights = weights[idx]
            for joint_index, joint_weight in enumerate(vertex_weights):
                if joint_weight > 0.0:
                    vg = obj_orig.vertex_groups[joint_names_update[joint_index]]
                    vg.add([index], joint_weight, "REPLACE")
    
    """
    >>>>>> step 4, Transform the mesh to T pose.
    """               
    bpy.context.view_layer.objects.active = armature_template
    bpy.ops.object.mode_set(mode='POSE')
    for bone in armature_template.pose.bones:
        bone.bone.select = False
        
    # transform to T Pose assume the charscter is Y up and facing Z
    if is_mixamo:
        rotate_bone_general(armature_template, 'mixamorig:RightShoulder', np.array([-1,0,0.]), rot_ord='ZY')
        rotate_bone_general(armature_template, 'mixamorig:RightArm', np.array([-1,0,0.]), rot_ord='ZY')
        rotate_bone_general(armature_template, 'mixamorig:RightForeArm', np.array([-1,0,0.]), rot_ord='ZY')
        
        # correct hand orientation
        if "mixamorig:RightHandThumb1" in armature_template.pose.bones and 'mixamorig:RightHandThumb3' in armature_template.pose.bones:
            thumb1 = get_joint_location(armature, 'mixamorig:RightHandThumb1')
            thumb3 = get_joint_location(armature, 'mixamorig:RightHandThumb3')
            thumb_dir = (thumb3 - thumb1)/(np.linalg.norm(thumb3-thumb1) + 1e-5)
            hand_dir = np.array([1.,0,0])
            palm_dir = np.cross(hand_dir, thumb_dir)
            cos = palm_dir[2]    
            if cos > 0.5:
                theta = np.arccos(cos)
                delt_theta = np.pi/2 - theta
                bone = armature_template.pose.bones["mixamorig:RightArm"]
                bone.bone.select = True
                bpy.ops.transform.rotate(value=delt_theta/3, orient_axis='X', 
                                         orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                         orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), 
                                         mirror=False, snap=False, snap_elements={'INCREMENT'}, 
                                         use_snap_project=False, snap_target='CLOSEST', 
                                         use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, 
                                         use_snap_selectable=False, release_confirm=True)
                bone = armature_template.pose.bones["mixamorig:RightForeArm"]
                bone.bone.select = True
                bpy.ops.transform.rotate(value=delt_theta/3, orient_axis='X', 
                                         orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                         orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), 
                                         mirror=False, snap=False, snap_elements={'INCREMENT'}, 
                                         use_snap_project=False, snap_target='CLOSEST', 
                                         use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, 
                                         use_snap_selectable=False, release_confirm=True)
                bone = armature_template.pose.bones["mixamorig:RightHand"]
                bone.bone.select = True
                bpy.ops.transform.rotate(value=delt_theta/3, orient_axis='X', 
                                         orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                         orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), 
                                         mirror=False, snap=False, snap_elements={'INCREMENT'}, 
                                         use_snap_project=False, snap_target='CLOSEST', 
                                         use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, 
                                         use_snap_selectable=False, release_confirm=True)

                
        
        rotate_bone_general(armature_template, 'mixamorig:LeftShoulder', np.array([1,0,0.]), rot_ord='ZY')
        rotate_bone_general(armature_template, 'mixamorig:LeftArm', np.array([1,0,0.]), rot_ord='ZY')
        rotate_bone_general(armature_template, 'mixamorig:LeftForeArm', np.array([1,0,0.]), rot_ord='ZY')

        rotate_bone_general(armature_template, 'mixamorig:RightUpLeg', np.array([0,-1,0.]), rot_ord='ZX')
        rotate_bone_general(armature_template, 'mixamorig:RightLeg', np.array([0,-1,0.]), rot_ord='ZX')
        
        rotate_bone_general(armature_template, 'mixamorig:LeftUpLeg', np.array([0,-1,0.]), rot_ord='ZX')
        rotate_bone_general(armature_template, 'mixamorig:LeftLeg', np.array([0,-1,0.]), rot_ord='ZX')
    else:
        rotate_bone_general(armature_template, 'RightShoulder', np.array([-1,0,0.]), rot_ord='ZY')
        rotate_bone_general(armature_template, 'RightArm', np.array([-1,0,0.]), rot_ord='ZY')
        rotate_bone_general(armature_template, 'RightForeArm', np.array([-1,0,0.]), rot_ord='ZY')
        # correct hand orientation
        if "RightHandThumb1" in armature_template.pose.bones and 'RightHandThumb3' in armature_template.pose.bones:
            thumb1 = get_joint_location(armature_template, 'RightHandThumb1')
            thumb3 = get_joint_location(armature_template, 'RightHandThumb3')
            thumb_dir = (thumb3 - thumb1)/(np.linalg.norm(thumb3-thumb1) + 1e-5)
            hand_dir = np.array([1.,0,0])
            palm_dir = np.cross(hand_dir, thumb_dir)
            cos = palm_dir[2]
            if cos > 0.5:
                theta = np.arccos(cos)
                delt_theta = 2*np.pi/3 - theta
                bone = armature_template.pose.bones["RightArm"]
                bone.bone.select = True
                bpy.ops.transform.rotate(value=delt_theta/3, orient_axis='X', 
                                         orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                         orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), 
                                         mirror=False, snap=False, snap_elements={'INCREMENT'}, 
                                         use_snap_project=False, snap_target='CLOSEST', 
                                         use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, 
                                         use_snap_selectable=False, release_confirm=True)
                bone.bone.select = False
                bone = armature_template.pose.bones["RightForeArm"]
                bone.bone.select = True
                bpy.ops.transform.rotate(value=delt_theta/3, orient_axis='X', 
                                         orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                         orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), 
                                         mirror=False, snap=False, snap_elements={'INCREMENT'}, 
                                         use_snap_project=False, snap_target='CLOSEST', 
                                         use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, 
                                         use_snap_selectable=False, release_confirm=True)
                bone.bone.select = False
                bone = armature_template.pose.bones["RightHand"]
                bone.bone.select = True
                bpy.ops.transform.rotate(value=delt_theta/3, orient_axis='X', 
                                         orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                         orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), 
                                         mirror=False, snap=False, snap_elements={'INCREMENT'}, 
                                         use_snap_project=False, snap_target='CLOSEST', 
                                         use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, 
                                         use_snap_selectable=False, release_confirm=True)
                bone.bone.select = False
        rotate_bone_general(armature_template, 'LeftShoulder', np.array([1,0,0.]), rot_ord='ZY')
        rotate_bone_general(armature_template, 'LeftArm', np.array([1,0,0.]), rot_ord='ZY')
        rotate_bone_general(armature_template, 'LeftForeArm', np.array([1,0,0.]), rot_ord='ZY')
        # correct hand orientation
        if "LeftHandThumb1" in armature_template.pose.bones and 'LeftHandThumb3' in armature_template.pose.bones:
            thumb1 = get_joint_location(armature_template, 'LeftHandThumb1')
            thumb3 = get_joint_location(armature_template, 'LeftHandThumb3')
            thumb_dir = (thumb3 - thumb1)/(np.linalg.norm(thumb3-thumb1) + 1e-5)
            hand_dir = np.array([1.,0,0])
            palm_dir = np.cross(hand_dir, thumb_dir)
            cos = palm_dir[2]
            if cos > 0.5:
                theta = np.arccos(cos)
                delt_theta = 2*np.pi/3 - theta
                bone = armature_template.pose.bones["LeftArm"]
                bone.bone.select = True
                bpy.ops.transform.rotate(value=delt_theta/3, orient_axis='X', 
                                         orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                         orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), 
                                         mirror=False, snap=False, snap_elements={'INCREMENT'}, 
                                         use_snap_project=False, snap_target='CLOSEST', 
                                         use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, 
                                         use_snap_selectable=False, release_confirm=True)
                bone.bone.select = False
                bone = armature_template.pose.bones["LeftForeArm"]
                bone.bone.select = True
                bpy.ops.transform.rotate(value=delt_theta/3, orient_axis='X', 
                                         orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                         orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), 
                                         mirror=False, snap=False, snap_elements={'INCREMENT'}, 
                                         use_snap_project=False, snap_target='CLOSEST', 
                                         use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, 
                                         use_snap_selectable=False, release_confirm=True)
                bone.bone.select = False
                bone = armature_template.pose.bones["LeftHand"]
                bone.bone.select = True
                bpy.ops.transform.rotate(value=delt_theta/3, orient_axis='X', 
                                         orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                         orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), 
                                         mirror=False, snap=False, snap_elements={'INCREMENT'}, 
                                         use_snap_project=False, snap_target='CLOSEST', 
                                         use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, 
                                         use_snap_selectable=False, release_confirm=True)
                bone.bone.select = False
                
        rotate_bone_general(armature_template, 'RightUpLeg', np.array([0,-1,0.]), rot_ord='ZX')
        rotate_bone_general(armature_template, 'RightLeg', np.array([0,-1,0.]), rot_ord='ZX')
        # correct the orientation of foot
        foot_head = armature_template.pose.bones["RightFoot"].head
        foot_tail = armature_template.pose.bones["RightFoot"].tail
        foot_dir = (foot_tail-foot_head)
        foot_dir[1] = 0
        foot_dir = foot_dir / (np.linalg.norm(foot_dir) + 1e-5)
        cos = foot_dir[2]
        if cos > 0:
            theta = np.arccos(cos) - np.pi/8
            bone = armature_template.pose.bones["RightUpLeg"]
            bone.bone.select = True
            bpy.ops.transform.rotate(value=theta/2, orient_axis='Y', 
                                     orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                     orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), 
                                     mirror=False, snap=False, snap_elements={'INCREMENT'}, 
                                     use_snap_project=False, snap_target='CLOSEST', 
                                     use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, 
                                     use_snap_selectable=False, release_confirm=True)
            bone.bone.select = False
            bone = armature_template.pose.bones["RightLeg"]
            bone.bone.select = True
            bpy.ops.transform.rotate(value=theta/2, orient_axis='X', 
                                     orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                     orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), 
                                     mirror=False, snap=False, snap_elements={'INCREMENT'}, 
                                     use_snap_project=False, snap_target='CLOSEST', 
                                     use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, 
                                     use_snap_selectable=False, release_confirm=True)
            bone.bone.select = False
            
        rotate_bone_general(armature_template, 'LeftUpLeg', np.array([0,-1,0.]), rot_ord='ZX')
        rotate_bone_general(armature_template, 'LeftLeg', np.array([0,-1,0.]), rot_ord='ZX')
        # correct the orientation of foot
        foot_head = armature_template.pose.bones["LeftFoot"].head
        foot_tail = armature_template.pose.bones["LeftFoot"].tail
        foot_dir = (foot_tail-foot_head)
        foot_dir[1] = 0
        foot_dir = foot_dir / (np.linalg.norm(foot_dir) + 1e-5)
        cos = foot_dir[2]
        if cos > 0:
            theta = -np.arccos(cos) + np.pi/8
            bone = armature_template.pose.bones["LeftUpLeg"]
            bone.bone.select = True
            bpy.ops.transform.rotate(value=theta/2, orient_axis='Y', 
                                     orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                     orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), 
                                     mirror=False, snap=False, snap_elements={'INCREMENT'}, 
                                     use_snap_project=False, snap_target='CLOSEST', 
                                     use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, 
                                     use_snap_selectable=False, release_confirm=True)
            bone.bone.select = False
            bone = armature_template.pose.bones["LeftLeg"]
            bone.bone.select = True
            bpy.ops.transform.rotate(value=theta/2, orient_axis='X', 
                                     orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                     orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), 
                                     mirror=False, snap=False, snap_elements={'INCREMENT'}, 
                                     use_snap_project=False, snap_target='CLOSEST', 
                                     use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, 
                                     use_snap_selectable=False, release_confirm=True)
            bone.bone.select = False

    for obj_orig in objs_orig:
        # Select the mesh and apply the armature modifier
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.objects.active = obj_orig
        
        for modifier in obj_orig.modifiers:
            if modifier.type == 'ARMATURE':
                bpy.ops.object.modifier_apply(modifier=modifier.name)
    
    # Set the armature to rest Pose
    bpy.context.view_layer.objects.active = armature_template
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.armature_apply()
    
#    for k, v in bone_orient_dict.items():
#        add_empty(v)
    change_bone_orientation(armature_template, bone_orient_dict)
    connect_armature(armature_template, parents, joint_names)
    
    # Return to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')
    for obj_orig in objs_orig:
        # Re-add the Armature modifier
        bpy.context.view_layer.objects.active = obj_orig
        bpy.ops.object.modifier_add(type='ARMATURE')
        obj_orig.modifiers['Armature'].object = armature_template
    try:
        bpy.context.object.active_material.blend_method = 'CLIP'
    except:
        print('error on changing blend method')
    
    armature_template.rotation_mode = 'XYZ'
    armature_template.rotation_euler[0] = 1.5708
    
    obj_orig.select_set(True)
    bpy.ops.export_scene.gltf(filepath=args.out_file.replace('.fbx', '.glb'), 
                              export_format='GLB',  export_yup=True, 
                              export_normals=True) 
    
    armature_template.rotation_mode = 'XYZ'
    armature_template.rotation_euler[0] = 1.5708
    use_selection_flag = False
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
        'axis_forward': '-Z',
        'axis_up': 'Y',
        'global_scale': 1.0,
        'apply_unit_scale': True,
        'apply_scale_options': 'FBX_SCALE_NONE',
        'bake_space_transform': False,
        'object_types': {'MESH', 'ARMATURE'},
    }
    
    bpy.ops.export_scene.fbx(filepath=args.out_file, check_existing=False, **export_settings)

    
# /usr/blender-4.0.1-linux-x64/blender -b --python ./blender_rigging_new.py -- --joint_dir /aigc_cfs_gdp/jiawei/data/general_generate/3ce0b65c-de88-43b9-a01b-dbdfa6d8303c/character_rigging/joint3d_v2.npz --target_dir /aigc_cfs_gdp/jiawei/data/general_generate/3ce0b65c-de88-43b9-a01b-dbdfa6d8303c/character_rigging/rotated.glb --out_file /aigc_cfs_gdp/jiawei/data/general_generate/3ce0b65c-de88-43b9-a01b-dbdfa6d8303c/character_rigging/mesh_rigged.fbx --threejs_x 90 --threejs_y 0 --threejs_z 0

