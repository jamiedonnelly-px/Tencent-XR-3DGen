import bpy
import numpy as np
import os
from mathutils import Matrix, Vector, Quaternion, Euler
import random

def file_import(character_path):
    data_formate = character_path[-3:]
    if "fbx" == data_formate:
        bpy.ops.import_scene.fbx(filepath=character_path, use_anim=True)  
    elif "glb" == data_formate:
        bpy.ops.import_scene.gltf(filepath=character_path)
    else:
        print("data formate not support")
        return -1
    return 0

def file_export(character_path):
    data_formate = character_path[-3:]
    if "fbx" == data_formate:
        bpy.ops.export_scene.fbx(filepath=character_path,bake_anim=True) 
    elif "glb" == data_formate:
        bpy.ops.export_scene.gltf(filepath=character_path, export_format='GLB', export_materials='EXPORT', export_colors=True, use_selection=True)
    else:
        print("data formate not support")
        return -1
    return 0




def smpl_drive_mdm_extend(character_path,output_filepath,rotations_all_file,root_translation_file):

    if len(bpy.context.scene.objects)!=0:
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    bpy.ops.import_scene.fbx(filepath=character_path, use_anim=True)

    replace_names = {"left_hip":"thigh_l","right_hip":"thigh_r",
    "left_knee":"calf_l","right_knee":"calf_r","left_ankle":"foot_l","right_ankle":"foot_r",
    "left_foot":"toes_l","right_foot":"toes_r","left_collar":"clavicle_l","right_collar":"clavicle_r","left_shoulder":"upperarm_l","right_shoulder":"upperarm_r",
    "left_elbow":"lowerarm_l","right_elbow":"lowerarm_r","left_wrist":"hand_l","right_wrist":"hand_r"}
    SMPL_JOINT_NAMES = ["root","pelvis","left_hip","right_hip","spine1","left_knee","right_knee","spine2","left_ankle","right_ankle","spine3","left_foot","right_foot","neck","left_collar","right_collar","head","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist"]#"left_hand","right_hand",
 
    rotations_all = np.load(rotations_all_file)
    root_translation = np.load(root_translation_file)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')

    max_vertex_count = 0
    armature_with_most_vertices_name = ""

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            vertex_count = len(obj.data.vertices)

            for modifier in obj.modifiers:
                if modifier.type == 'ARMATURE' and modifier.object is not None:
                    if vertex_count > max_vertex_count:
                        max_vertex_count = vertex_count
                        armature_with_most_vertices_name = modifier.object.name
    print(armature_with_most_vertices_name)
    armature_object = bpy.data.objects[armature_with_most_vertices_name] 
    bone_names = []
    for name in SMPL_JOINT_NAMES[1:]:
        if name in replace_names.keys():
            bone_names.append(replace_names[name])
        else:
            bone_names.append(name)
    print(bone_names)

    bones_num  = len(bone_names)
    bones=[]
    for j in np.arange(0,bones_num,1):
        bone_name = bone_names[j]
        bone = armature_object.pose.bones[bone_name]
        bones.append(bone)

    bpy.ops.object.select_all(action='DESELECT')
    armature_object.select_set(True)
    bpy.context.view_layer.objects.active = armature_object
    bpy.ops.object.mode_set(mode='POSE')

    frames_num = rotations_all.shape[0]

    assert(bones_num==22)

    print(root_translation.shape)
    print(rotations_all.shape)

    step = 1
    for i in np.arange(0,frames_num,step):
        bpy.context.scene.frame_set(i//step+1)
        for j in np.arange(0,bones_num,1):
            # x = root_translation[0,i]
            # y = root_translation[1,i]
            # z = root_translation[2,i]

            # armature_object.location = Vector((x,-z,y))
            armature_object.location = Vector((0,0,0))
            bone_name = bone_names[j]
            rotation = rotations_all[i,j,:]
            bone = bones[j]
            bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = rotation
            
            bone.keyframe_insert(data_path='rotation_quaternion', frame=i//step+1)
            armature_object.keyframe_insert(data_path='location', frame=i//step+1)

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.export_scene.fbx(filepath=output_filepath,bake_anim=True,use_selection=False, embed_textures=True, path_mode='COPY')

if __name__ == '__main__':
    import sys
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] 
    fbx_path = argv[0]
    fbx_out_path = argv[1]
    rotations_all_file = argv[2]
    root_translation_file = argv[3]
    print(fbx_path)
    if not os.path.exists(fbx_path) or (not fbx_path.endswith('.fbx') and not fbx_path.endswith('.glb')):
        print("input fbx file not exist or not fbx glb file")
    else:
        print("file_path:"+fbx_path)
        output_filepath = fbx_out_path

        if not os.path.exists(rotations_all_file) or not os.path.exists(root_translation_file):
            print(f"{rotations_all_file} not exist or {root_translation_file} not exist")
        else:
            smpl_drive_mdm_extend(fbx_path,output_filepath,rotations_all_file,root_translation_file)
