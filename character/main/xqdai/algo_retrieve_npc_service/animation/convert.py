import os
import json
current_file_path = os.path.abspath(__file__)
current_file_directory = os.path.dirname(current_file_path)

json_path = "/mnt/aigc_cfs_cq/xiaqiangdai/project/algo_retrieve_npc_service/clothwrap2/combine/body_config/base_body_map.json"
f = open(json_path,'r')
json_data = json.load(f)
character_path_src = "/mnt/aigc_cfs_cq/xiaqiangdai/project/algo_retrieve_npc_service/animation/mixamo/Idle.fbx"
for key1 in json_data.keys():
    bodys = json_data[key1]
    for key2 in bodys.keys():
        glb_path = os.path.join(bodys[key2]['path'],'auto_rig/armature_mesh.glb')
        cmd = "/mnt/aigc_cfs_cq/xiaqiangdai/envs/blender-3.6.15-linux-x64/blender  --background --addons rokoko-studio-live-blender-master --python  "+os.path.join(current_file_directory,'retarget_test.py')+" -- "+character_path_src+" "+glb_path
        os.system(cmd)