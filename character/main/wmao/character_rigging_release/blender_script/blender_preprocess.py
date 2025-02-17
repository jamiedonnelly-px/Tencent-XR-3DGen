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
    return vs, w

def change_bone_orientation(armature, bone_orient):
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.armature.select_all(action='DESELECT')
    edit_bones = armature.data.edit_bones
    rot_resi = {}
    for bone in edit_bones:
        matrix_orig = Matrix(bone_orient[bone.name].tolist()).to_3x3()
        matrix_now = bone.matrix.to_3x3()
        rot_resi[bone.name] = matrix_now.inverted() @ matrix_orig
    
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.parent_clear(type='CLEAR')
       
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='POSE')
    for bone in armature.pose.bones:

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
    for jn, pi in zip(joint_names,parents):
        if pi < 0:
            continue
        pn = joint_names[pi]
        print(f'parent {pn}, joint {jn}')
        armature.data.edit_bones[jn].select = True
        armature.data.edit_bones[pn].select = True
        armature.data.edit_bones.active = armature.data.edit_bones[pn]
        if 'Shoulder' in jn or 'UpLeg' in jn:
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
if __name__ == "__main__":

    # /root/blender-4.0.1-linux-x64/blender --background --python ./blender/rigging.py -- --template_fbx_dir "/aigc_cfs_2/weimao/non-smalfit/dataset/SK_Dolphin/template.fbx" --rigging_info_dir "/aigc_cfs_2/weimao/non-smalfit/output/dolphin/rigging/rigging.npz" --out_dir "/aigc_cfs_2/weimao/non-smalfit/output/dolphin/rigging/"
    argv = sys.argv
    arg_idx = argv.index("--")
    path = argv[arg_idx + 1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, default="/home/wei/Documents/character_rigging/debug_data/quad_remesh/quad_mesh.glb")
    parser.add_argument('--out_file', type=str, default='/home/wei/Documents/character_rigging/debug_data/mesh_rigged.fbx')
    parser.add_argument('--threejs_x', type=str, default='0')
    parser.add_argument('--threejs_y', type=str, default='0')
    parser.add_argument('--threejs_z', type=str, default='0')

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)
#    args = parser.parse_args()
  
    clear_scene()
    os.makedirs(os.path.dirname(args.out_file),exist_ok=True)
    

    """
    >>>>>> step 2, Load target mesh, manifold it and skinning: 
        apply all existing modifiers and remove objects that are not mesh.
    """
    obj = load_file(args.target_dir, is_join=True)[0]

    # update 07-11-24 rotation center change from (0,0,0) to (0,1,0)
    euler_angles = [np.deg2rad(float(args.threejs_x)), 
                    np.deg2rad(float(args.threejs_y)), 
                    np.deg2rad(float(args.threejs_z))]
    rotate_meshes([obj], euler_angles, mode='ZYX', center=(0,1,0))

    bpy.ops.object.mode_set(mode='OBJECT')
    obj.rotation_mode = 'XYZ'
    obj.rotation_euler[0] = 1.5708
    out_file = os.path.join(os.path.dirname(args.out_file),'mesh_rotated.obj')
    bpy.ops.wm.obj_export(filepath=out_file, 
                          export_uv=False, 
                          export_normals=False,
                          export_materials=False
                        )
    # obj.rotation_mode = 'ZYX'
    # obj.rotation_euler = euler_angles    
    # bpy.ops.object.transform_apply(rotation=True)
    
    # use voxel remesh may cause artifacts for fingles 
    bpy.context.object.data.remesh_voxel_size = 0.01
    bpy.ops.object.voxel_remesh()
    obj.rotation_mode = 'XYZ'
    obj.rotation_euler[0] = 1.5708
    bpy.ops.wm.obj_export(filepath=args.out_file)
    