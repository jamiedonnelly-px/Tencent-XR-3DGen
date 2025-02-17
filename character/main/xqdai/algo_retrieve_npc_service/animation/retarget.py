import bpy
import numpy as np
import os
from mathutils import Matrix, Vector, Quaternion, Euler
import math
from math import radians
import time
import sys
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

def clean():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    for material in bpy.data.materials:
        bpy.data.materials.remove(material)

    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture)

    for image in bpy.data.images:
        bpy.data.images.remove(image)

def file_import(character_path):
    data_formate = character_path[-3:]
    print(f"data_formate:{data_formate}")
    if "fbx" == data_formate:
        bpy.ops.import_scene.fbx(filepath=character_path, use_anim=True)  
    elif "glb" == data_formate:
        bpy.ops.import_scene.gltf(filepath=character_path)
    else:
        print("data formate not support")
        return -1
    print("file import success")
    return 0

def file_export(character_path):
    data_formate = character_path[-3:]
    print(f"data_formate:{data_formate}")
    if "fbx" == data_formate:
        # bpy.ops.export_scene.fbx(filepath=character_path,bake_anim=True) 
        bpy.ops.export_scene.fbx(filepath=character_path,use_selection=False, embed_textures=True, path_mode='COPY')
    elif "glb" == data_formate:
        export_settings = {
            "export_format": "GLB",
            "export_animations": True,
            "export_animation_mode": "SCENE",
            "export_frame_range": True,
            "export_frame_step":1,
            "export_force_sampling":True,
            "export_rest_position_armature":False,
            "export_optimize_animation_size":True,
            "export_optimize_animation_keep_anim_armature":True,
            "export_optimize_animation_keep_anim_object":True
        }
        bpy.ops.export_scene.gltf(filepath=character_path, **export_settings)
        # bpy.ops.export_scene.gltf(filepath=character_path, export_format='GLB', export_materials='EXPORT', export_colors=True, use_selection=True,export_animations=True,
        # export_frame_range=True)
    else:
        print("data formate not support")
        return -1
    print("file export success")
    return 0
    
def find_another_armature(name):
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            if obj.name != name:
                return obj.name
    return None

def retarget(character_path_src,character_path_dst):
    
    clean()

    file_import(character_path_src)
    
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
    
    file_import(character_path_dst)
    
    print(f"armature_with_most_vertices_name:{armature_with_most_vertices_name}")
    dst_obj_name = find_another_armature(armature_with_most_vertices_name)
    print(f"dst_obj_name:{dst_obj_name}")
    if dst_obj_name==None:
        return -1

    bpy.context.scene.rsl_retargeting_armature_source = bpy.data.objects[armature_with_most_vertices_name]
    bpy.context.scene.rsl_retargeting_armature_target = bpy.data.objects[dst_obj_name]
    
    bpy.ops.rsl.build_bone_list()
    # bpy.ops.object.select_all(action='DESELECT')
    if 'HumanML3D' in character_path_src:
        print("HumanML3D in character_path_src")

        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE' and obj.name == armature_with_most_vertices_name:
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
        bpy.context.object.rotation_mode = 'XYZ'
        bpy.context.object.rotation_euler[0] = 1.5708
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

        bones_mapping = {'pelvis':'Hips','spine1':'Spine','spine2':'Spine1','spine3':'Spine2','right_foot':'RightToeEnd','left_foot':'LeftToeEnd','right_hip':'RightUpLeg','left_hip':'LeftUpLeg','right_collar':'RightShoulder','left_collar':'LeftShoulder','neck':'Neck','right_ankle':'RightFoot','left_ankle':'LeftFoot','right_shoulder':'RightArm','left_shoulder':'LeftArm','right_elbow':'RightForeArm','left_elbow':'LeftForeArm','right_wrist':'RightHand','left_wrist':'LeftHand','right_knee':'RightLeg','left_knee':'LeftLeg'}
        for item in bpy.context.scene.rsl_retargeting_bone_list:
            if item.bone_name_source in bones_mapping.keys():
                # print(item.bone_name_source,bones_mapping[item.bone_name_source])
                item.bone_name_target = bones_mapping[item.bone_name_source]
            
    else:
        print("HumanML3D not in character_path_src")

        bones_mapping ={'mixamorig:Spine':'spine','mixamorig:Spine1':'spine1','mixamorig:Spine2':'spine2','mixamorig:RightArm':'RightArm','mixamorig:LeftArm':'LeftArm',
        'mixamorig:RightForeArm':'RightForeArm','mixamorig:LeftForeArm':'LeftForeArm','mixamorig:RightShoulder':'RightShoulder','mixamorig:LeftShoulder':'LeftShoulder',
        'mixamorig:LeftUpLeg':'LeftUpLeg','mixamorig:RightUpLeg':'RightUpLeg','mixamorig:LeftLeg':'LeftLeg','mixamorig:RightLeg':'RightLeg','mixamorig:LeftFoot':'LeftFoot',
        'mixamorig:RightFoot':'RightFoot','mixamorig:LeftToeBase':'LeftToeEnd','mixamorig:RightToeBase':'RightToeEnd'}
        for item in bpy.context.scene.rsl_retargeting_bone_list:
            if item.bone_name_source in bones_mapping.keys():
                item.bone_name_target = bones_mapping[item.bone_name_source]
        

    bpy.ops.rsl.retarget_animation()


    bpy.ops.object.select_all(action='DESELECT')
    
    bpy.data.objects[armature_with_most_vertices_name].select_set(True)
    for child in bpy.data.objects[armature_with_most_vertices_name].children:
        if child.type == 'MESH':
            child.select_set(True)
    
    bpy.context.view_layer.objects.active = bpy.data.objects[armature_with_most_vertices_name]
    
    bpy.ops.object.delete()
    
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and obj.name == dst_obj_name:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
    if 'HumanML3D' in character_path_src:
        bpy.context.object.rotation_mode = 'XYZ'
        bpy.context.object.rotation_euler[0] = -1.5708
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        armature_obj = bpy.context.scene.objects.get(dst_obj_name)

        bpy.ops.object.select_all(action='DESELECT')
        armature_obj.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        bpy.ops.transform.translate(value=(0, 0, bpy.context.object.dimensions.z/2+0.1))

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    output_filepath = character_path_dst.replace('.fbx','_animation.fbx').replace('.glb','_animation.glb')
    file_export(output_filepath)

    return 0




if __name__ == '__main__':
    # src_path="/mnt/aigc_cfs_cq/xiaqiangdai/project/algo_retrieve_npc_service/animation/mixamo/Idle1.fbx"
    # dst_path= "/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/ff72b5ee-7d8b-5f74-8a39-6bbf18c9435a/mesh/mesh.glb"
   
    # start_time = time.time()
    # print(retarget(src_path,dst_path))
    # end_time = time.time()
    # print("time: {:.2f} s".format(end_time - start_time))

    argv = sys.argv
    argv = argv[argv.index("--") + 1:] 
    character_path_src = argv[0]
    character_path_dst = argv[1]
    retarget(character_path_src,character_path_dst)