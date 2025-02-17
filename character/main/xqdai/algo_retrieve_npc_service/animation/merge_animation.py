import bpy
import sys

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
        bpy.ops.export_scene.gltf(filepath=character_path,)
   
    else:
        print("data formate not support")
        return -1
    print("file export success")
    return 0


def merge_animations(model_in_path:str,model_idle_path:str,model_out_path:str):
    clean()
    
    file_import(model_in_path)
    first_model = bpy.context.selected_objects[0]

    # 导入第二个GLB文件
    file_import(model_idle_path)
    second_model = bpy.context.selected_objects[0]

    file_import(model_out_path)
    third_model = bpy.context.selected_objects[0]

    print(first_model.name)
    print(second_model.name)
    print(third_model.name)
    names = [first_model.name,second_model.name,third_model.name]
    # 将第二个模型的动画复制到第一个模型
    print(bpy.data.actions)
  
    actions = list(bpy.data.actions)
    for action in list(actions):
        print(actions)
        print(action.name)
        if action.name.replace('Root_Root','Root') not in names:
            print(action.name,'pass')
            continue

        if second_model.name in action.name:
            print("second_model")
            new_action = action.copy()

            new_action.name = action.name.replace(second_model.name, first_model.name)
            
            # 创建一个新的NLA Track并添加动作
            nla_track = first_model.animation_data.nla_tracks.new()
            nla_track.name = new_action.name
            nla_strip = nla_track.strips.new(name=new_action.name, start=0, action=new_action)

            bpy.data.actions.remove(action)
            actions.remove(action)
            continue
        
        if third_model.name in action.name:
            print("third_model")
            new_action = action.copy()
            
            new_action.name = action.name.replace(third_model.name, first_model.name)
            
            # 创建一个新的NLA Track并添加动作
            nla_track = first_model.animation_data.nla_tracks.new()
            nla_track.name = new_action.name
            nla_strip = nla_track.strips.new(name=new_action.name, start=0, action=new_action)

            bpy.data.actions.remove(action)
            actions.remove(action)
      

    bpy.data.objects.remove(second_model, do_unlink=True)
    bpy.data.objects.remove(third_model, do_unlink=True)
    first_model.animation_data.nla_tracks[0].name = 'EnterCapsule'
    first_model.animation_data.nla_tracks[1].name = 'Idle'
    first_model.animation_data.nla_tracks[2].name = 'ExitCapsule'
    # 删除第二，三个模型
    armature_to_keep = bpy.data.objects.get(first_model.name)

    if armature_to_keep and armature_to_keep.type == 'ARMATURE':
        meshes_to_keep = [obj for obj in bpy.data.objects if obj.type == 'MESH' and obj.parent == armature_to_keep]
        
        for obj in bpy.data.objects:

            if (obj.type == 'ARMATURE' or obj.type == 'MESH') and obj != armature_to_keep and obj not in meshes_to_keep:
                bpy.data.objects.remove(obj, do_unlink=True)
            
    for track in first_model.animation_data.nla_tracks:
        print(track.name)
    output_filepath = model_idle_path.replace('_animation_idle.fbx','_animation.fbx').replace('_animation_idle.glb','_animation.glb')
    file_export(output_filepath)

if __name__ == "__main__":
    # idle_path = "/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/f2620fd4-4276-53c6-b3aa-1c1303e6aca6/mesh/mesh_animation.glb"
    # out_path = "/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/f2620fd4-4276-53c6-b3aa-1c1303e6aca6/mesh/mesh_animation_out.glb"
    # in_path = "/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/f2620fd4-4276-53c6-b3aa-1c1303e6aca6/mesh/mesh_animation_in.glb"
    # merge_animations(in_path,idle_path,out_path)
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] 
    character_path_in = argv[0]
    character_path_idle = argv[1]
    character_path_out = argv[2]
    try:
        merge_animations(character_path_in,character_path_idle,character_path_out)
    except:
        sys.exit(-1)