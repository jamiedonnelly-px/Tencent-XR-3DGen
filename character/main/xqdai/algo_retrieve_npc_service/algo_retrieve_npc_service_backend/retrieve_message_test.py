import sys
import time
import uuid
import os

import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

from retrieve_npc_backend import  retrieve_npc_service

def pandorax_cos_upload(service:retrieve_npc_service,glb_local_path:str):
    """
        upload to cos remote
        glb_local_path example:"/mesh/mesh.fbx"
    """
    timestamp = int(time.time())
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
    unique_id_str = str(unique_id)
    
    message = {'channel':'cos_upload','data':[glb_local_path,unique_id_str],'id':unique_id_str}
   
    service.submit_task(message['channel'],message['id'],message['data'])
    result = service.user_get_result(message['channel'],message['id'])
    return result

def pandorax_cos_upload_remote(service:retrieve_npc_service,glb_local_path:str,remote_file_path:str):
    """
        upload to cos remote
        glb_local_path example:"/mesh/mesh.fbx"
        remote_file_path example: "/pandorax/image/test.fbx"
    """
    timestamp = int(time.time())
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
    unique_id_str = str(unique_id)
    
    message = {'channel':'cos_upload_remote','data':[glb_local_path,remote_file_path,unique_id_str],'id':unique_id_str}
   
    service.submit_task(message['channel'],message['id'],message['data'])
    result = service.user_get_result(message['channel'],message['id'])
    return result
    

def pandorax_txt_retrive_npc(service:retrieve_npc_service, entity:list,description:list):
    """
        npc text retrieve
        input:txt list([hair,top,trousers,shoe,outfit,others,gender])
        output info:[hair image list,top image list,trousers image list,shoe image list,outfit image list,others image list,
                    hair glb list,top glb list,trousers glb list,shoe glb list,outfit glb list,others glb list,
                    hair key list,top key list,trousers key list,shoe key list,outfit key list,others key list]
    """
    # entity=['hair','top','trousers','shoe','outfit','others',"action", "body shape prompt","hair_color","male"]
    # description = 'a girl'
    timestamp = int(time.time())
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
    unique_id_str = str(unique_id)
    
    message = {'channel':'txt_retrive_npc','data':[entity,description,unique_id_str],'id':unique_id_str}
   
    service.submit_task(message['channel'],message['id'],message['data'])
    result = service.user_get_result(message['channel'],message['id'])
    return result

def pandorax_long_txt_retrive_npc(service:retrieve_npc_service, long_prompt:str):
    """
        npc text retrieve
        input:txt example('The woman walked down the street with a confident stride, her black leather jacket hugging her curves in all the right places. She wore a simple white t-shirt underneath, tucked into a pair of high-waisted blue jeans that accentuated her long legs. Her black ankle boots clicked against the pavement as she made her way towards the cafe, and her oversized sunglasses shielded her eyes from the bright sun. A silver necklace with a small pendant hung around her neck, adding a touch of elegance to her otherwise casual outfit. She exuded a sense of effortless style and cool confidence that turned heads as she passed by.')
        output: similar like upper case
        output info:[hair image list,top image list,trousers image list,shoe image list,outfit image list,others image list,
                    hair glb list,top glb list,trousers glb list,shoe glb list,outfit glb list,others glb list,
                    hair key list,top key list,trousers key list,shoe key list,outfit key list,others key list]
    """
    timestamp = int(time.time())
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
    unique_id_str = str(unique_id)
    
    message = {'channel':'long_txt_retrive_npc','data':[long_prompt,unique_id_str],'id':unique_id_str}
   
    service.submit_task(message['channel'],message['id'],message['data'])
    result = service.user_get_result(message['channel'],message['id'])
    return result

def pandorax_long_retrive_npc_all_auto(service:retrieve_npc_service,long_prompt:str):
    """
        npc text retrieve
        input:txt example('The woman walked down the street with a confident stride, her black leather jacket hugging her curves in all the right places. She wore a simple white t-shirt underneath, tucked into a pair of high-waisted blue jeans that accentuated her long legs. Her black ankle boots clicked against the pavement as she made her way towards the cafe, and her oversized sunglasses shielded her eyes from the bright sun. A silver necklace with a small pendant hung around her neck, adding a touch of elegance to her otherwise casual outfit. She exuded a sense of effortless style and cool confidence that turned heads as she passed by.')
        output:glb path,fbx path,glb folder path
    """
    timestamp = int(time.time())
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
    unique_id_str = str(unique_id)
    
    message = {'channel':'long_retrive_npc_all_auto','data':[long_prompt,unique_id_str],'id':unique_id_str}
   
    service.submit_task(message['channel'],message['id'],message['data'])
    result = service.user_get_result(message['channel'],message['id'])
    return result

def pandorax_retrive_npc_combine(service:retrieve_npc_service,path_list:list,key_list:list,texture_replace=[False, False, False, False, False, False],
        gender="male",
        shape="fat",
        hair_color = "gold",
        scope = "webui",
        is_combined_first=True):
    """
        input:txt example(['/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/Female hair/Female hair/F_HAIR_346/F_HAIR_346_fbx2020.glb',None, '/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/Bottoms/Bottoms/Bottoms01/BTM_93/BTM_93_fbx2020.glb', '/mnt/business_1/Data/DesignCenter/clock_fix_sample/20231215/fix_top_bottom/component/shoe/SK_Shoe_Sneaker03_F/SK_Shoe_Sneaker03_F.glb','/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/modify/Dresses/F_A/DR_673_F_A/DR_673_fbx2020.glb',  '/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/Glove Socks/Glove Socks/Socks/SOX_129/SOX_129_fbx2020.glb'],[F_HAIR_346_Asset,None,DSBA_BTM_4_Bottoms03,SH_241_SHOES01,DR_673_F_A_Dresses,SOX_204_SOX])
        if not be selected,that shoud be None
        texture_replace:whether to enable texture replace in [hair,top,trousers,shoe,outfit,others]
        gender:male for female,
        shape:body shape prompt,
        hair_color:hair color prompt
        scope:xr or webui
        output:glb folder path,glb path

    """
    # path_list=['hair_path','top_path','trousers_path','shoe_path','outfit_path','others_path']
    # key_list = ['hair_key','top_key','trousers_key','shoe_key','outfit_key','others_key']
    timestamp = int(time.time())
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
    unique_id_str = str(unique_id)
    
    message = {'channel':'retrive_npc_combine','data':[path_list,key_list,unique_id_str,texture_replace,gender,shape,hair_color,scope,is_combined_first],'id':unique_id_str}
   
    service.submit_task(message['channel'],message['id'],message['data'])
    result = service.user_get_result(message['channel'],message['id'])
    return result

def pandorax_retrive_npc_all_auto(
        service:retrieve_npc_service,
        path_list:list,
        key_list:list,
        texture_replace=[False, False, False, False, False, False],
        gender="male",
        shape="fat",
        hair_color=""
    ):
    """
        npc text retrieve
        input:txt example(['/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/Female hair/Female hair/F_HAIR_346/F_HAIR_346_fbx2020.glb',None, '/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/Bottoms/Bottoms/Bottoms01/BTM_93/BTM_93_fbx2020.glb', '/mnt/business_1/Data/DesignCenter/clock_fix_sample/20231215/fix_top_bottom/component/shoe/SK_Shoe_Sneaker03_F/SK_Shoe_Sneaker03_F.glb','/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/modify/Dresses/F_A/DR_673_F_A/DR_673_fbx2020.glb',  '/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/Glove Socks/Glove Socks/Socks/SOX_129/SOX_129_fbx2020.glb'],[F_HAIR_346_Asset,None,DSBA_BTM_4_Bottoms03,SH_241_SHOES01,DR_673_F_A_Dresses,SOX_204_SOX])
        texture_replace:whether to enable texture replace in [hair,top,trousers,shoe,outfit,others]
        gender:male for female,
        shape:body shape prompt,
        hair_color:hair color prompt
        output:glb path,fbx path,glb folder path
    """
    timestamp = int(time.time())
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
    unique_id_str = str(unique_id)
    
    message = {'channel':'retrive_npc_all_auto','data':[path_list,key_list,unique_id_str,texture_replace,gender,shape,hair_color],'id':unique_id_str}
   
    service.submit_task(message['channel'],message['id'],message['data'])
    result = service.user_get_result(message['channel'],message['id'])
    return result

def pandorax_retrive_npc_autoRig(service:retrieve_npc_service,mesh_output_path:str):
    """
        retrive_npc_autoRig
        input:mesh folder path
        output:mesh folder path,glb path
    """
    timestamp = int(time.time())
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
    unique_id_str = str(unique_id)
    
    message = {'channel':'retrive_npc_autoRig','data':[mesh_output_path,unique_id_str],'id':unique_id_str}
   
    service.submit_task(message['channel'],message['id'],message['data'])
    result = service.user_get_result(message['channel'],message['id'])
    return result


def pandorax_retrive_npc_animation(service:retrieve_npc_service,model_path:str):
    """
        retrive_npc_animation
        input:glb folder path
        output:glb path,gif path,glb folder path

    """
    timestamp = int(time.time())
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
    unique_id_str = str(unique_id)
    
    message = {'channel':'retrive_npc_animation','data':[model_path,unique_id_str],'id':unique_id_str}
   
    service.submit_task(message['channel'],message['id'],message['data'])
    result = service.user_get_result(message['channel'],message['id'])
    return result

def pandorax_retrive_npc_text_animation(service:retrieve_npc_service,model_path:str,text_prompt:str):
    """
        retrive_npc_animation
        input:glb folder path,text_prompt
        output:glb path,gif path,glb folder path
    """
    timestamp = int(time.time())
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
    unique_id_str = str(unique_id)
    
    message = {'channel':'retrive_npc_text_animation','data':[model_path,text_prompt,unique_id_str],'id':unique_id_str}
    print(message)
    service.submit_task(message['channel'],message['id'],message['data'])
    result = service.user_get_result(message['channel'],message['id'])
    return result

def pandorax_npc_animation_text_retrieve(service:retrieve_npc_service,text_prompt:str):
    """
        npc_animation_text_retrieve
        input:text_prompt
        output:gif path list
    """
    timestamp = int(time.time())
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
    unique_id_str = str(unique_id)
    
    message = {'channel':'npc_animation_text_retrieve','data':[text_prompt,unique_id_str],'id':unique_id_str}
    print(message)
    service.submit_task(message['channel'],message['id'],message['data'])
    result = service.user_get_result(message['channel'],message['id'])
    return result

def pandorax_animation_gif_generate(service:retrieve_npc_service,model_path:str,gif_path:str):
    """
        animation_gif_generate
        input:glb folder path,gif_path
        output:glb path
    """
    timestamp = int(time.time())
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
    unique_id_str = str(unique_id)
    
    message = {'channel':'animation_gif_generate','data':[model_path,gif_path,unique_id_str],'id':unique_id_str}
    print(message)
    service.submit_task(message['channel'],message['id'],message['data'])
    result = service.user_get_result(message['channel'],message['id'])
    return result


def pandorax_render_shader(service:retrieve_npc_service, model_path:str):
    """
        render_shader
        input:model_path
        output:shader video and front image
    """
    timestamp = int(time.time())
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
    unique_id_str = str(unique_id)
    
    message = {'channel':'render_shader','data':[model_path,unique_id_str],'id':unique_id_str}
   
    service.submit_task(message['channel'],message['id'],message['data'])
    result = service.user_get_result(message['channel'],message['id'])
    return result


def pandorax_retrive_npc_auto_binding_animation(service:retrieve_npc_service,mesh_output_path):
    timestamp = int(time.time())
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
    unique_id_str = str(unique_id)
    
    message = {'channel':'auto_rig_manual_render','data':[mesh_output_path,unique_id_str],'id':unique_id_str}
   
    service.submit_task(message['channel'],message['id'],message['data'])
    result = service.user_get_result(message['channel'],message['id'])
    return result

if __name__ == "__main__":

    service = retrieve_npc_service(db_id=6)
    print(service)
    start_time = time.time()
    