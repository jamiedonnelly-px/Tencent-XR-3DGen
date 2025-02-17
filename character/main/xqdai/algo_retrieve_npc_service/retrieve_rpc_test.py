# from gradio_client import Client
import sys
import rpyc
import json
import threading
from queue import Queue
import time
import ujson
import uuid
import os
import numpy as np
import cv2
# from ipdb import set_trace as st

def cos_upload(connection,glb_local_path):
    """
        glb_local_path example:"/mesh/mesh.fbx"
    """
    result =connection.root.cos_upload(glb_local_path)
    result = ujson.loads(result)
    print(result)

def cos_upload_remote(connection,glb_local_path,remote_file_path):
    """
        glb_local_path example:"/mesh/mesh.fbx"
        remote_file_path example: "/pandorax/image/test.fbx"
    """
    result =connection.root.cos_upload_remote(glb_local_path,remote_file_path)
    result = ujson.loads(result)
    print(result)
    
def texReplace(connection,prompt,path,key):
    """
        prompt example:"red"
        path example: "/aigc_cfs_gdb/test"
        key example:"aaaaa"
    """
    result =connection.root.texReplace(prompt,path,key)
    result = ujson.loads(result)
    print(result)

def txt_retrive_npc(connection, entity,description):
    """
        npc text retrieve
        input:txt list([hair,top,trousers,shoe,outfit,others,gender])
        output info:[hair image list,top image list,trousers image list,shoe image list,outfit image list,others image list,
                    hair glb list,top glb list,trousers glb list,shoe glb list,outfit glb list,others glb list,
                    hair key list,top key list,trousers key list,shoe key list,outfit key list,others key list]
    """
    # entity=['hair','top','trousers','shoe','outfit','others',"action", "body","hair_color","male"]
    # description = 'a girl'
    result =connection.root.txt_retrive_npc(entity,description)
    result = ujson.loads(result)
    print(result)

def long_txt_retrive_npc(connection, long_prompt):
    """
        npc text retrieve
        input:txt example('The woman walked down the street with a confident stride, her black leather jacket hugging her curves in all the right places. She wore a simple white t-shirt underneath, tucked into a pair of high-waisted blue jeans that accentuated her long legs. Her black ankle boots clicked against the pavement as she made her way towards the cafe, and her oversized sunglasses shielded her eyes from the bright sun. A silver necklace with a small pendant hung around her neck, adding a touch of elegance to her otherwise casual outfit. She exuded a sense of effortless style and cool confidence that turned heads as she passed by.')
        output: similar like upper case
        output info:[hair image list,top image list,trousers image list,shoe image list,outfit image list,others image list,
                    hair glb list,top glb list,trousers glb list,shoe glb list,outfit glb list,others glb list,
                    hair key list,top key list,trousers key list,shoe key list,outfit key list,others key list]
    """
    result =connection.root.long_txt_retrive_npc(long_prompt)
    result = ujson.loads(result)
    print(result)

def long_retrive_npc_all_auto(connection,long_prompt):
    """
        npc text retrieve
        input:txt example('The woman walked down the street with a confident stride, her black leather jacket hugging her curves in all the right places. She wore a simple white t-shirt underneath, tucked into a pair of high-waisted blue jeans that accentuated her long legs. Her black ankle boots clicked against the pavement as she made her way towards the cafe, and her oversized sunglasses shielded her eyes from the bright sun. A silver necklace with a small pendant hung around her neck, adding a touch of elegance to her otherwise casual outfit. She exuded a sense of effortless style and cool confidence that turned heads as she passed by.')
        output:glb path,fbx path,glb folder path
    """
    result = connection.root.long_retrive_npc_all_auto(long_prompt)
    result = ujson.loads(result)
    print(result)

def retrive_npc_combine(connection,path_list,key_list,texture_replace=[False, False, False, False, False, False],
        gender="male",
        shape="fat",
        hair_color = "gold"):
    """
        input:txt example(['/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/Female hair/Female hair/F_HAIR_346/F_HAIR_346_fbx2020.glb',None, '/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/Bottoms/Bottoms/Bottoms01/BTM_93/BTM_93_fbx2020.glb', '/mnt/business_1/Data/DesignCenter/clock_fix_sample/20231215/fix_top_bottom/component/shoe/SK_Shoe_Sneaker03_F/SK_Shoe_Sneaker03_F.glb','/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/modify/Dresses/F_A/DR_673_F_A/DR_673_fbx2020.glb',  '/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/Glove Socks/Glove Socks/Socks/SOX_129/SOX_129_fbx2020.glb'],[F_HAIR_346_Asset,None,DSBA_BTM_4_Bottoms03,SH_241_SHOES01,DR_673_F_A_Dresses,SOX_204_SOX])
        if not be selected,that shoud be None
        output:glb folder path,glb path

    """
    # path_list=['hair_path','top_path','trousers_path','shoe_path','outfit_path','others_path']
    # key_list = ['hair_key','top_key','trousers_key','shoe_key','outfit_key','others_key']
    result = connection.root.retrive_npc_combine(
        path_list,
        key_list,
        texture_replace=texture_replace,
        gender=gender,
        shape=shape,
        hair_color = hair_color
    )
    result = ujson.loads(result)
    print(result)

def retrive_npc_all_auto(
        connection,
        path_list,
        key_list,
        texture_replace=[False, False, False, False, False, False],
        gender="male",
        shape="fat",
        hair_color=""
    ):
    """
        npc text retrieve
        input:txt example(['/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/Female hair/Female hair/F_HAIR_346/F_HAIR_346_fbx2020.glb',None, '/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/Bottoms/Bottoms/Bottoms01/BTM_93/BTM_93_fbx2020.glb', '/mnt/business_1/Data/DesignCenter/clock_fix_sample/20231215/fix_top_bottom/component/shoe/SK_Shoe_Sneaker03_F/SK_Shoe_Sneaker03_F.glb','/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/modify/Dresses/F_A/DR_673_F_A/DR_673_fbx2020.glb',  '/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/Glove Socks/Glove Socks/Socks/SOX_129/SOX_129_fbx2020.glb'],[F_HAIR_346_Asset,None,DSBA_BTM_4_Bottoms03,SH_241_SHOES01,DR_673_F_A_Dresses,SOX_204_SOX])
        output:glb path,fbx path,glb folder path
    """
    result = connection.root.retrive_npc_all_auto(
        path_list,
        key_list,
        texture_replace=[False, False, False, False, False, False],
        gender="male",
        shape="fat",
        hair_color=""
    )
    result = ujson.loads(result)
    print(result)

def retrive_npc_autoRig(connection,mesh_output_path):
    """
        retrive_npc_autoRig
        input:mesh folder path
        output:mesh folder path,glb path
    """
    result = connection.root.retrive_npc_autoRig(mesh_output_path)
    result = ujson.loads(result)
    print(result)


def retrive_npc_animation(connection,mesh_output_path):
    """
        retrive_npc_animation
        input:glb folder path
        output:glb path,gif path,glb folder path

    """
    result = connection.root.retrive_npc_animation(mesh_output_path)
    result = ujson.loads(result)
    print(result)

def retrive_npc_text_animation(connection,mesh_output_path,text_prompt):
    """
        retrive_npc_animation
        input:glb folder path,text_prompt
        output:glb path,gif path,glb folder path
    """
    result = connection.root.retrive_npc_text_animation(mesh_output_path,text_prompt)
    result = ujson.loads(result)
    print(result)

def render_gif_frontImage(connection, model_path):
    """
        render_gif_frontImage
        input:model_path
        output:360 video and front image

    """
    result = connection.root.render_gif_frontImage(model_path)
    result = ujson.loads(result)
    print(result)

def render_shader(connection, model_path):
    """
        render_shader
        input:model_path
        output:shader video and front image
    """
    result = connection.root.render_shader(model_path)
    result = ujson.loads(result)
    print(result)

def shape_text_retrieve(connection,gender, shape_promt):
    """
        shape_text_retrieve
        input:gender, shape_promt
        output:gender,shape,use_shoes,use_hair
    """
    result = connection.root.shape_text_retrieve(gender, shape_promt)
    result = ujson.loads(result)
    print(result)

def hair_is_change_ok(connection,mesh_path):
    """
        hair_is_change_ok
        input:mesh_path
        output:True or False
    """
    result = connection.root.hair_is_change_ok(mesh_path)
    result = ujson.loads(result)
    print(result)

def get_retrieve_json_path(connection):
    """
        hair_is_change_ok
        input:
        output:cfs and gdp json path
    """
    result = connection.root.get_retrieve_json_path()
    result = ujson.loads(result)
    print(result)

def retrive_npc_dislike(connection, mesh_output_path):
    """
        retrive_npc_dislike
        input:txt example('/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/data/0a58f6f2-c40b-5e46-8f93-302e79f3caf0')
        output:None

    """
    result = connection.root.retrive_npc_dislike(mesh_output_path)
    result = ujson.loads(result)
    print(result)

def retrive_npc_animation_dislike(connection, mesh_output_path):
    """
        retrive_npc_dislike
        input:txt example('/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/data/0a58f6f2-c40b-5e46-8f93-302e79f3caf0')
        output:None

    """
    result =connection.root.retrive_npc_animation_dislike( mesh_output_path)
    result = ujson.loads(result)
    print(result)

def retrive_npc_auto_binding_animation(connection, mesh_output_path):
    """
        retrive_npc_auto_binding_animation
        input:mesh_output_path
        output:glb path,gif path,glb folder path
    """
    result =connection.root.retrive_npc_auto_binding_animation(self, mesh_output_path)
    result = ujson.loads(result)
    print(result)

def retrive_npc_text_animation(connection, mesh_output_path,text_prompt):
    """
        retrive_npc_animation
        input:glb folder path,text_prompt
        output:glb path,gif path,glb folder path
    """
    result = connection.root.retrive_npc_text_animation(mesh_output_path,text_prompt)
    result = ujson.loads(result)
    print(result)

def exposed_render_shader(connection, model_path):
    """
        render_shader
        input:model_path
        output:shader video and front image
    """
    result = connection.root.render_shader(model_path)
    print(ujson.loads(result))

def get_retrieve_json_path(connection):
    """
        get_retrieve_json_pat
        input:
        output:cfs and gdp json path
    """
    result = connection.root.get_retrieve_json_path()
    print(ujson.loads(result))

if __name__ == "__main__":
    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    timestamp = int(time.time())
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
    rpyc_config["sync_request_timeout"] = None
    connection = rpyc.connect('', 0,config=rpyc_config)

    print(connection)
    image_path = "/mnt/aigc_cfs_cq/xiaqiangdai/project/image_retrieve/lib_generate/test_images/cam-0022_2.png"
    input_image = cv2.imread(image_path)
    input_image = ujson.dumps(input_image.tolist())
    result = connection.root.mask_predict(input_image)
    result= ujson.loads(result)
    print(np.array(result['mask']).shape)
    print(np.array(result['box']).shape)
    print(result['phrase'])



