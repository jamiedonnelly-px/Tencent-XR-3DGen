import bpy
import numpy as np
import os
from mathutils import Matrix, Vector, Quaternion, Euler
import math
from math import radians
import time
import sys
import bmesh
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

def read_mesh_to_ndarray( mesh, mode = "object"):
    ''' read the vert coordinate of a deformed mesh
    :param mesh: mesh object
    :return: numpy array of the mesh
    '''
    assert mode in [ "edit", "object"]
    faces = []
    if mode is "object" :
        bm = bmesh.new()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bm.from_object(mesh, depsgraph)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        mverts_co = [(v.co) for v in bm.verts]
        mverts_co = np.asarray( mverts_co, dtype=np.float32)
        # faces = [[ v.index for v in face.verts ] for face in bm.faces]
        # faces=np.asarray(faces,dtype=np.int32)
        bm.free()
    elif mode is "edit" :
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.editmode_toggle()
        bm = bmesh.from_edit_mesh(mesh.data)
        mverts_co = [(v.co) for v in bm.verts]
        mverts_co = np.asarray( mverts_co, dtype=np.float32)
        # faces = [[ v.index for v in face.verts ] for face in bm.faces]
        # faces=np.asarray(faces,dtype=np.int32)
        bm.free()
        bpy.ops.object.editmode_toggle()

    return mverts_co, faces

def compute_mesh_size( meshes ):

    #计算角色中心点,身体长度
    verts = []
    for ind, mesh in enumerate(meshes):
        vert, _ = read_mesh_to_ndarray( mesh, mode="object")
        mat = np.asarray(mesh.matrix_world)
        R,t = mat[:3,:3], mat[:3,3:] #Apply World Scale
        if vert.shape[0] == 0:
            continue
        # print(R.shape,t.shape,vert.shape)
        verts.append( ( R@vert.T + t ).T )
    if len(verts)==0:
        return [0,0,0],1
    verts=np.concatenate(verts, axis=0)
    min_ = verts.min(axis=0)
    max_ = verts.max(axis=0)

    obj_center = ( min_ + max_ ) / 2

    length = (max_ - min_)[2]

    print( "object mode ", obj_center, length)

    return obj_center, length

def get_keyframes(action):
    keyframes = set()
    for fcu in action.fcurves:
        for keyframe in fcu.keyframe_points:
            keyframes.add(keyframe.co[0])
    return sorted(keyframes)

def retarget(character_path_src,character_path_dst,output_filepath):
    
    try:
        clean()
    except Exception as e:
        print(e)
        pass
    
    start_time = time.time()
    try:
        file_import(character_path_src)
    except:
        raise ValueError(f"retarget character_path_src import  error")
        return -1
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print(f"retarget file_import Time: {elapsed_time:.2f} seconds")
    
    start_time = time.time()
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
    
    src_ani = bpy.data.objects[armature_with_most_vertices_name]
    action = src_ani.animation_data.action
    kf = get_keyframes(action)
    nf = len(kf)
    bpy.context.scene.frame_end = nf

    try:
        file_import(character_path_dst)
    except:
        raise ValueError(f"retarget character_path_dst import  error")
        return -1
    
    print(f"armature_with_most_vertices_name:{armature_with_most_vertices_name}")
    dst_obj_name = find_another_armature(armature_with_most_vertices_name)
    print(f"dst_obj_name:{dst_obj_name}")
    if dst_obj_name==None:
        return -1
    armature = bpy.data.objects.get(dst_obj_name)
    print(armature)
    dst_amature = bpy.data.objects[dst_obj_name]
    if dst_amature is not None:
        if dst_amature.animation_data and dst_amature.animation_data.action:
            dst_amature.animation_data.action = None

    if dst_amature.animation_data and dst_amature.animation_data.nla_tracks:
        for track in dst_amature.animation_data.nla_tracks:
            dst_amature.animation_data.nla_tracks.remove(track)
    dst_amature.animation_data_clear()
    bpy.context.scene.rsl_retargeting_armature_source = bpy.data.objects[armature_with_most_vertices_name]
    bpy.context.scene.rsl_retargeting_armature_target = bpy.data.objects[dst_obj_name]
    
    bpy.ops.rsl.build_bone_list()

    bones_mapping ={'mixamorig:Hips':'Hips', 'mixamorig:Spine':'Spine','mixamorig:Spine1':'Spine1','mixamorig:Spine2':'Spine2','mixamorig:RightArm':'RightArm','mixamorig:LeftArm':'LeftArm',
    'mixamorig:RightForeArm':'RightForeArm','mixamorig:LeftForeArm':'LeftForeArm','mixamorig:RightShoulder':'RightShoulder','mixamorig:LeftShoulder':'LeftShoulder',
    'mixamorig:LeftUpLeg':'LeftUpLeg','mixamorig:RightUpLeg':'RightUpLeg','mixamorig:LeftLeg':'LeftLeg','mixamorig:RightLeg':'RightLeg','mixamorig:LeftFoot':'LeftFoot',
    'mixamorig:RightFoot':'RightFoot','mixamorig:LeftToeBase':'LeftToeEnd','mixamorig:RightToeBase':'RightToeEnd'}
    for item in bpy.context.scene.rsl_retargeting_bone_list:
        print(item.bone_name_source, item.bone_name_target)

        if item.bone_name_source in bones_mapping.keys() and ('mixamorig' not in item.bone_name_target):
            item.bone_name_target = bones_mapping[item.bone_name_source]
        elif 'mixamorig' in item.bone_name_target :#and item.bone_name_source in dst_armature.pose.bones:
            item.bone_name_target = item.bone_name_source

    # bones_mapping ={'Spine':'spine','Spine1':'spine1','Spine2':'spine2','RightArm':'RightArm','LeftArm':'LeftArm',
    # 'RightForeArm':'RightForeArm','LeftForeArm':'LeftForeArm','RightShoulder':'RightShoulder','LeftShoulder':'LeftShoulder',
    # 'LeftUpLeg':'LeftUpLeg','RightUpLeg':'RightUpLeg','LeftLeg':'LeftLeg','RightLeg':'RightLeg','LeftFoot':'LeftFoot',
    # 'RightFoot':'RightFoot','LeftToeBase':'LeftToeEnd','RightToeBase':'RightToeEnd','LeftToeEnd':'','RightToeEnd':''}
    # for item in bpy.context.scene.rsl_retargeting_bone_list:
    #     if item.bone_name_source in bones_mapping.keys():
    #         item.bone_name_target = bones_mapping[item.bone_name_source]
        

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
    
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')

    end_time = time.time()
    elapsed_time = end_time - start_time  
    print(f"retarget process Time: {elapsed_time:.2f} seconds")

    start_time = time.time()
    file_export(output_filepath)
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print(f"{output_filepath} retarget file_export Time: {elapsed_time:.2f} seconds")

    return 0


if __name__ == '__main__':
    # src_path="/aigc_cfs_gdp/xiaqiangdai/data/HumanML3D/models/011752.fbx"
    # dst_path="/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/retrieve_NPC/09a2c23a-0ce9-4d68-a3d4-289e606ed487/character_rigging/mesh_Walking.glb"
    # output_filepath = "/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/retrieve_NPC/09a2c23a-0ce9-4d68-a3d4-289e606ed487/character_rigging/mesh_Walking_animation.glb"
    # start_time = time.time()
    # print(retarget(src_path,dst_path,output_filepath))
    # end_time = time.time()
    # print("time: {:.2f} s".format(end_time - start_time))

    argv = sys.argv
    argv = argv[argv.index("--") + 1:] 
    character_path_src = argv[0]
    character_path_dst = argv[1]
    character_path_out = argv[2]
    try:
        retarget(character_path_src,character_path_dst,character_path_out)
    except:
        sys.exit(-1)

# /usr/blender-4.0.1-linux-x64/blender -b --addons rokoko-studio-live-blender-master --python ./blender_animation_rokoko.py -- /aigc_cfs_gdp/weimao/character_rigging/animations/Dancing.fbx /aigc_cfs_gdp/jiawei/data/general_generate/cd6e4527-b726-486c-a8ec-0d64b32a26d9/character_rigging/mesh_rigged.fbx /aigc_cfs_gdp/jiawei/data/general_generate/cd6e4527-b726-486c-a8ec-0d64b32a26d9/character_rigging/mesh_Dancing.fbx