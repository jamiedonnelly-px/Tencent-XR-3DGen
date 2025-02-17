import os

import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, parent_dir)


import argparse
import gradio as gr

import numpy as np

from PIL import Image

import uuid
import time
import json
import requests
import rpyc
import ujson
import threading
import cv2
import gc

import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.insert(0, parent_dir)
main_dir = os.path.dirname(parent_dir)

sys.path.append(os.path.join(main_dir,"segment_anything_webui"))
sys.path.append(os.path.join(main_dir,"text2image"))
sys.path.append(os.path.join(main_dir,"texture_generation"))
sys.path.append("/mnt/aigc_cfs_cq/xiaqiangdai/project/algo_retrieve_npc_service/algo_retrieve_npc_service_backend")
sys.path.append("/mnt/aigc_cfs_cq/xiaqiangdai/project/chat_generation_service_temp/Microservices/third_party/tdmq_everything/texture_generation/tdmq_interface")
from retrieve_npc_backend import retrieve_npc_service
from retrieve_message_test import pandorax_retrive_npc_combine

from main_call_texgen import TexgenInterface, init_job_id

from inference import run_inference
from client_t2i import T2iProducer,T2iBackendConsumer

retrieve_service = retrieve_npc_service()

def update_progress(progress, progress_path):
    with open(progress_path) as fi:
        prog = fi.read()
        # print(prog)
        try:
            title, ratio = prog.strip().split(' ')
            title = title.replace('_', ' ')
            ratio = float(ratio)
            progress(ratio, desc=title)
        except:
            print(f'can not parse progress: {prog}')

def keys_dict(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    folder_keys_dict = {}
    for key in json_data['data'].keys():
        for key_1 in json_data['data'][key].keys():
                folder_key = key_1
                folder_keys_dict[folder_key] = json_data['data'][key][key_1]
    return folder_keys_dict

names = ["hair","top","trousers","shoe","outfit"]
json_path ="/mnt/aigc_cfs_cq/xiaqiangdai/project/algo_retrieve_npc_service/npc_layer_retrieve/20241010_daz_decimate_add_ct.json"
global g_json_data
g_json_data = keys_dict(json_path)

def int_sd_model():
    text2image_producer = T2iProducer(os.path.join(main_dir,'text2image/configs/t2i_deploy_gdp_online.json'))
    text2image_backend_consumer = T2iBackendConsumer(os.path.join(main_dir,'text2image/configs/t2i_deploy_gdp_online.json'))
    consumer_thread = threading.Thread(target=text2image_backend_consumer.start_consumer)
    consumer_thread.start()
    return text2image_producer

text2image_producer = int_sd_model()
texture_service = TexgenInterface()

def sd_retrive_all(input_text:str):
    
    # 调用文生多图接口
    timestamp = int(time.time())
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
    unique_id_str = str(unique_id)
    text2image_producer.interface_query_text(prompt=input_text,job_id = unique_id_str,out_save_dir='/aigc_cfs_gdp/xiaqiangdai/t2i')
    path_output = os.path.join('/aigc_cfs_gdp/xiaqiangdai/t2i',unique_id_str,'t2i.png')
    time_all=0
    while not os.path.exists(path_output):
        time.sleep(1)
        time_all+=1
        if time_all>60:
            print(f"{path_output} not exists")
            return None
    output_img = np.array(Image.open(path_output))
    print(output_img.shape)
    return output_img



def create_uuid_folder(base_path="."):
    uuid_folder_name = str(uuid.uuid4())
    folder_path = os.path.join(base_path, uuid_folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path



def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

def deserialize(obj):
    if isinstance(obj, list):
        # 尝试将列表转换为 NumPy 数组
        try:
            return np.array(obj)
        except ValueError:
            # 如果转换失败，递归处理列表元素
            return [deserialize(i) for i in obj]
    elif isinstance(obj, dict):
        # 递归处理字典元素
        return {k: deserialize(v) for k, v in obj.items()}
    else:
        return obj


def image_retrive_all(image,gender):

    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config["sync_request_timeout"] = None
    connection = rpyc.connect('', 27690,config=rpyc_config)
    input_dict = {'image':image,'gender':gender}
    input_dict = convert_to_serializable(input_dict)
    input_image_json = ujson.dumps(input_dict)
    result = connection.root.retrive_image_all(input_image_json)
    output = ujson.loads(result)
    output = deserialize(output)

    threads_in =[]
    results_dict = {}
    paths_out = []
    
    for index,key in enumerate(names):
        if key in output.keys():
            if len(output[key]['key'].tolist())==0:
                for i in range(4):
                    paths_out.append(None)
                continue
            paths_out.append(output[key]['caption'])
            for index1,key1 in enumerate(output[key]['key'].tolist()):
                num = index*4+index1+1
                if key1!=None:
                    if index1!=0 or index==0:
                        paths_out.append(g_json_data[key1]["GLB_Mesh"])
                    else:
                        paths_out.append(g_json_data[key1]["GLB_Mesh"])
                        folder_temp = create_uuid_folder("/aigc_cfs_gdp/sz/server/tex_gen/pulsar_log/mcwy_debug_public")
                        texture_replace(threads_in,results_dict,num,output[key]['img_path'],'',key1,folder_temp)
                        print(output[key]['img_path'])
                else:
                    paths_out.append(None)
                
                
        else:
            for i in range(4):
                paths_out.append(None)
    for thread in threads_in:
        thread.join()
    print(results_dict.keys())
    for num in results_dict.keys():
        num_int = int(num)
        paths_out[num_int]=results_dict[num]
       

    mask1 = np.array(output['mask1']).astype(np.uint8)
    mask2 = np.array(output['mask2']).astype(np.uint8)
    print(f"mask1:{mask1.shape} {mask1.min()} {mask1.max()}")
    print(f"mask2:{mask2.shape} {mask2.min()} {mask2.max()}")
    print(paths_out[0])
    print(paths_out[4])
    print(paths_out[8])
    print(paths_out[12])
    print(paths_out[16])
   
    paths_out.append(mask1)
    paths_out.append(mask2)
    # print(paths_out)
    return paths_out



def image_retrive_parts(image,gender,part_info):

    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config["sync_request_timeout"] = None
    connection = rpyc.connect('', 0,config=rpyc_config)
    input_dict = {'image':image.tolist(),'part_info':part_info,'gender':gender}
    input_dict = convert_to_serializable(input_dict)
    input_image_json = ujson.dumps(input_dict)
    result = connection.root.retrieve_parts(input_image_json)
    output = ujson.loads(result)
    output = deserialize(output)

    print(output)

    threads_in =[]
    results_dict = {}
    paths_out = []

    key_list= [None]*6
    path_list = [None]*6
    enable_list = [False]*6
    image_list = [None]*6
    for index,key in enumerate(names):
        if key in output.keys():
            if len(output[key]['key'].tolist())==0:
                for i in range(4):
                    paths_out.append(None)
                continue    
            paths_out.append(output[key]['caption'])
            for index1,key1 in enumerate(output[key]['key'].tolist()):
                num = index*4+index1+1
                if key1!=None:
                    if index1!=0:
                        paths_out.append(g_json_data[key1]["GLB_Mesh"])
                    else:
                        paths_out.append(g_json_data[key1]["GLB_Mesh"])
                        # folder_temp = create_uuid_folder("/aigc_cfs_gdp/sz/server/tex_gen/pulsar_log/mcwy_debug_public")
                        # texture_replace(threads_in,results_dict,num,output[key]['img_path'],'',key1,folder_temp)
                        print(output[key]['img_path'])
                        # enable_list[index]=True
                        image_list[index] = output[key]['img_path']

                else:
                    paths_out.append(None)
                
                if index1==0:
                    key_list[index] = key1
                    path_list[index] = g_json_data[key1]["GLB_Mesh"]
                    
        else:
            for i in range(4):
                paths_out.append(None)

    print("==========",key_list)
    print("==========",path_list)
    print("==========",enable_list)
    print("==========",image_list)
    result = pandorax_retrive_npc_combine(retrieve_service,path_list, key_list,enable_list, 'male', 'mcwy1', '','xr')
    print("==========",result)

    combine_result = result[1]
    

    texgen_interface = TexgenInterface('/mnt/aigc_cfs_cq/xiaqiangdai/project/chat_generation_service_temp/Microservices/third_party/tdmq_everything/texture_generation/configs/client_texgen.json', 'uv_mcwy')

    job_id = init_job_id()
    tex_result = combine_result.replace('mesh.glb',f'replace_mesh_{job_id}.glb')
    mid_result_dir = os.path.join("/aigc_cfs_gdp/jiawei/data/texture_generation/", job_id)
    fast_mode_param = {
        "fast_mode": True,
        "raw_glb": combine_result,
        "out_glb": tex_result
    }
    # input_param = [
    #     {
    #         "in_mesh_key": "VRoid_4_4423006740960699034_Top",
    #         "in_mesh_path": None,
    #         "in_prompts": "red",
    #         "in_condi_img": None,
    #         "mode": "text"
    #     },
    #     {
    #         "in_mesh_key": "BTM_419",
    #         "in_mesh_path": None,
    #         "in_prompts": "",
    #         "in_condi_img": "/aigc_cfs_gdp/sz/batch_1106/in_imgs/0a1de9108cabcf6eea187536df3c627f.jpg",
    #         "mode": "image"
    #     },
    #     {
    #         "in_mesh_key": "SH_160",
    #         "in_mesh_path": None,
    #         "in_prompts": "red",
    #         "in_condi_img": "/aigc_cfs_gdp/sz/batch_1106/in_imgs/3c83e2b7599cc89a73a620662c878100.jpg",
    #         "mode": "mix"
    #     },
    # ]
    input_param=[]
    for i,key in enumerate(key_list):
        if i==0 or key==None:
            continue
        replace_key_dict = {
            "in_mesh_key": key,
            "in_mesh_path": None,
            "in_prompts": "",
            "in_condi_img": image_list[i],
            "mode": "image"
        }
        input_param.append(replace_key_dict)
    # batch fast mode
    success_flag, result_meshs = texture_service.blocking_call_batch_query(
        job_id,
        input_param,    # list of dict
        mid_result_dir=mid_result_dir,   # 存中间结果的目录，比如 根目录+job_id
        fast_mode_param=fast_mode_param,
    )
    paths_out.append(result_meshs[0])
    gc.collect()

    return paths_out

def image_retrive_parts1(image,gender,part_info):

    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config["sync_request_timeout"] = None
    connection = rpyc.connect('', 0,config=rpyc_config)
    input_dict = {'image':image.tolist(),'part_info':part_info,'gender':gender}
    input_dict = convert_to_serializable(input_dict)
    input_image_json = ujson.dumps(input_dict)
    result = connection.root.retrieve_parts(input_image_json)
    output = ujson.loads(result)
    output = deserialize(output)

    print(output)

    threads_in =[]
    results_dict = {}
    paths_out = []

    key_list= [None]*6
    path_list = [None]*6
    enable_list = [False]*6
    image_list = [None]*6
    for index,key in enumerate(names):
        if key in output.keys():
            if len(output[key]['key'].tolist())==0:
                continue    
            for index1,key1 in enumerate(output[key]['key'].tolist()):
                num = index*4+index1+1
                if key1!=None:
                    if index1!=0:
                        ii=0
                    else:
                        ii=0
                        # folder_temp = create_uuid_folder("/aigc_cfs_gdp/sz/server/tex_gen/pulsar_log/mcwy_debug_public")
                        # texture_replace(threads_in,results_dict,num,output[key]['img_path'],'',key1,folder_temp)
                        print(output[key]['img_path'])
                        # enable_list[index]=True
                        image_list[index] = output[key]['img_path']

                else:
                    ii=0
                
                if index1==0:
                    key_list[index] = key1
                    path_list[index] = g_json_data[key1]["GLB_Mesh"]
                    
        else:
            for i in range(4):
                ii=0

    print("==========",key_list)
    print("==========",path_list)
    print("==========",enable_list)
    print("==========",image_list)
    result = pandorax_retrive_npc_combine(retrieve_service,path_list, key_list,enable_list, 'male', 'mcwy1', '','xr')
    print("==========",result)

    combine_result = result[1]
    

    texgen_interface = TexgenInterface('/mnt/aigc_cfs_cq/xiaqiangdai/project/chat_generation_service_temp/Microservices/third_party/tdmq_everything/texture_generation/configs/client_texgen.json', 'uv_mcwy')

    job_id = init_job_id()
    tex_result = combine_result.replace('mesh.glb',f'replace_mesh_{job_id}.glb')
    mid_result_dir = os.path.join("/aigc_cfs_gdp/jiawei/data/texture_generation/", job_id)
    fast_mode_param = {
        "fast_mode": True,
        "raw_glb": combine_result,
        "out_glb": tex_result
    }
    # input_param = [
    #     {
    #         "in_mesh_key": "VRoid_4_4423006740960699034_Top",
    #         "in_mesh_path": None,
    #         "in_prompts": "red",
    #         "in_condi_img": None,
    #         "mode": "text"
    #     },
    #     {
    #         "in_mesh_key": "BTM_419",
    #         "in_mesh_path": None,
    #         "in_prompts": "",
    #         "in_condi_img": "/aigc_cfs_gdp/sz/batch_1106/in_imgs/0a1de9108cabcf6eea187536df3c627f.jpg",
    #         "mode": "image"
    #     },
    #     {
    #         "in_mesh_key": "SH_160",
    #         "in_mesh_path": None,
    #         "in_prompts": "red",
    #         "in_condi_img": "/aigc_cfs_gdp/sz/batch_1106/in_imgs/3c83e2b7599cc89a73a620662c878100.jpg",
    #         "mode": "mix"
    #     },
    # ]
    input_param=[]
    for i,key in enumerate(key_list):
        if i==0 or key==None:
            continue
        replace_key_dict = {
            "in_mesh_key": key,
            "in_mesh_path": None,
            "in_prompts": "",
            "in_condi_img": image_list[i],
            "mode": "image"
        }
        input_param.append(replace_key_dict)
    # batch fast mode
    success_flag, result_meshs = texture_service.blocking_call_batch_query(
        job_id,
        input_param,    # list of dict
        mid_result_dir=mid_result_dir,   # 存中间结果的目录，比如 根目录+job_id
        fast_mode_param=fast_mode_param,
    )

    return result_meshs[0]

mask_colors={'hair':[255,0,0],'top':[0,255,0],'trousers':[0,0,255],'shoe':[255,255,0],'outfit':[255,0,255],'other':[0,255,255]}

def draw_image(input_image,mask_state):
    mask_color_image = np.zeros(input_image.shape)
    print(f"mask_color_image:{mask_color_image.shape}")
    
    for key in mask_state.keys():
        if key=='outfit':
            continue
        for i in range(3):
            print(mask_state[key]['mask'].shape)
            mask_color_image[mask_state[key]['mask']!=0,i]=mask_colors[key][i]
    mask_color_image = mask_color_image.astype(np.uint8)
    print(mask_color_image.shape)

    mask_color_image_outfit = np.zeros(mask_color_image.shape)
    for key in mask_state.keys():
        if key!='outfit':
            continue
        for i in range(3):
            mask_color_image_outfit[mask_state[key]['mask']!=0,i]=mask_colors[key][i]
    mask_color_image_outfit = mask_color_image_outfit.astype(np.uint8)
    print(f"mask_color_image_outfit:{mask_color_image_outfit.shape}")
    return mask_color_image,mask_color_image_outfit

def ground_sam_image(input_image:np.ndarray,mask_state):
    mask_state.clear()
    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config["sync_request_timeout"] = None
    connection = rpyc.connect('', 0,config=rpyc_config)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    print(f"input_image:{input_image.shape}")
    input_image_json = ujson.dumps(input_image.tolist())
    result = connection.root.mask_predict(input_image_json)
    result = ujson.loads(result)
    masks = np.array(result['mask'])
    h  = masks.shape[-2]
    w  = masks.shape[-1]
    boxes = np.array(result['box'])
    names_map = {"hair":"hair","top cloth":"top","trousers":"trousers","shoe":"shoe","suit":"outfit"}
    
    image =  Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))

    print(result['phrase'])
    for i,phrase in enumerate(result['phrase']):
        cloth_name = phrase.split('(')[0]
        mask_state[names_map[cloth_name]] = {'mask':masks[i,0,:,:],'box':boxes[i]}
        print(f"phrase:{phrase} mask:{masks[i,0,:,:].shape}")
    
    mask_color_image = np.zeros((h,w,3))
    print(f"mask_color_image:{mask_color_image.shape}")
    
    for key in mask_state.keys():
        if key=='outfit':
            continue
        for i in range(3):
            mask_color_image[mask_state[key]['mask']!=0,i]=mask_colors[key][i]
    mask_color_image = mask_color_image.astype(np.uint8)
    print(mask_color_image.shape)

    mask_color_image_outfit = np.zeros(mask_color_image.shape)
    for key in mask_state.keys():
        if key!='outfit':
            continue
        for i in range(3):
            mask_color_image_outfit[mask_state[key]['mask']!=0,i]=mask_colors[key][i]
    mask_color_image_outfit = mask_color_image_outfit.astype(np.uint8)
    print(f"mask_color_image_outfit:{mask_color_image_outfit.shape}")

    return mask_color_image,mask_color_image_outfit

def clear_part(input_image:np.ndarray,mask_state,part_name):
    if part_name not in mask_state.keys():
        return
    del mask_state[part_name] 
    mask_color_image,mask_color_image_outfit = draw_image(input_image,mask_state)

    return mask_color_image,mask_color_image_outfit

def remove_small_components(mask, threshold_ratio):

    total_area = np.sum(mask == 1)
    
    min_size = total_area * threshold_ratio
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    new_mask = np.zeros_like(mask)
    
    for i in range(1, num_labels):  # 从1开始，跳过背景
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area >= min_size:
            new_mask[labels == i] = 1
    
    return new_mask

def get_combined_bounding_box(binary_mask):
    # 确保输入是二进制掩码
    
    binary_mask = binary_mask.astype(np.uint8)

    binary_mask = remove_small_components(binary_mask,0.1).astype(np.uint8)

    # 查找掩码中的轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None  # 如果没有找到任何轮廓，返回 None

    # 初始化最小和最大坐标
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')

    # 遍历所有轮廓
    for contour in contours:
        # 获取轮廓的最小外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # 计算综合的边界框
    combined_box = (x_min, y_min, x_max, y_max)

    return combined_box
    
def sam_seg(original_image,selected_points,body_part,body_mask_dict):
    output_image, output_mask = run_inference(32, 0.88, 0.95, 0, 1, 0.7, 0,0.7, 0.1, original_image, '', selected_points)

    target_color = np.array([1, 1, 1])
    binary_mask = np.all(output_mask != target_color, axis=-1)
    if body_part not in body_mask_dict.keys():
        body_mask_dict[body_part]={}
    box = get_combined_bounding_box(binary_mask)
    body_mask_dict[body_part]['mask'] = binary_mask
    body_mask_dict[body_part]['box'] = box
    print(binary_mask.shape)
    mask_color_image = np.zeros(output_mask.shape)
    for key in body_mask_dict.keys():
        if key=='outfit':
            continue
        for i in range(3):
            if key == body_part:
                mask_color_image[binary_mask == True,i]=mask_colors[key][i]
            else:
                mask_color_image[body_mask_dict[key]['mask']==True,i]=mask_colors[key][i]
    mask_color_image = mask_color_image.astype(np.uint8)
    print(mask_color_image.shape)
    selected_points.clear()

    mask_color_image_outfit = np.zeros(output_mask.shape)
    for key in body_mask_dict.keys():
        if key!='outfit':
            continue
        for i in range(3):
            if key == body_part:
                mask_color_image_outfit[binary_mask == True,i]=mask_colors[key][i]
            else:
                mask_color_image_outfit[body_mask_dict[key]['mask']==True,i]=mask_colors[key][i]
    mask_color_image_outfit = mask_color_image_outfit.astype(np.uint8)
    print(mask_color_image_outfit.shape)
    gc.collect()

    return mask_color_image,mask_color_image_outfit,original_image

colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]
def launch(
    port,
    listen=False,
):
    lock_path = 'runing'

    global listen_port
    listen_port = port
   
    css = """
    #config-accordion, #logs-accordion {color: black !important;}
    .dark #config-accordion, .dark #logs-accordion {color: white !important;}
    .stop {background: darkred !important;}
    """
    custom_css = """
    body {
        background-color: lightblue;
    }

    .label {
        color: red;
        font-size: 20px;
    }
    .image-container img {
    width: auto !important;
    height: auto !important;
}
    #model-out {
    height: 512px;
    }
    #model-in {
    height: 512px;
    }
    """

    with gr.Blocks(
        title="NPC Image Retrieve - Web Demo",
        css=custom_css,
    ) as demo:
        with gr.Row(equal_height=True):
            header = """
            # NPC Retrieve Demo
            """
            gr.Markdown(header)
        
        # with gr.Row(equal_height=True):
        #     with gr.Column():
        #         txt_input = gr.Textbox(
        #             value='',
        #             placeholder="please input prompt for image generation",
        #             show_label=False,
        #             interactive=True)
        #     with gr.Column():
        #         sd_btn = gr.Button(value="Image generation with text prompt", variant="primary")
        with gr.Row(equal_height=True):
            dropdown_options = ["male", "female"]
            gender_select = gr.Dropdown(choices=dropdown_options,value="male",  label="Select gender",interactive=True)

        # with gr.Tab(label='Auto'):
        #     with gr.Row(equal_height=True):
        #         with gr.Column():
        #             # input image
        #             original_image = gr.State(value=None)   # store original image without points, default None
        #             input_image = gr.Image(type="numpy")
                
        #         with gr.Column():
        #             mask_image1 = gr.Image(type="numpy",interactive=False)
        #         with gr.Column(): 
        #             mask_image2 = gr.Image(type="numpy",interactive=False)

        #     with gr.Row():
        #         with gr.Column(scale=2):
        #             auto_retrieve_btn = gr.Button(value="Auto segment and retrieve using Image", variant="primary")
        
        with gr.Tab(label='Mannul'):
            with gr.Row(equal_height=True):
                with gr.Column():
                    # input image
                    original_image = gr.State(value=None)   # store original image without points, default None
                    input_image = gr.Image(type="numpy")
                    mask_state = gr.State({})
                    # point prompt
                    with gr.Column():
                        selected_points = gr.State([])      # store points
                        with gr.Row():
                            gr.Markdown('You can click on the image to select points prompt. Default: foreground_point.')
                            undo_button = gr.Button('Undo point')
                        radio = gr.Radio(['foreground_point', 'background_point'], label='point labels')
                        with gr.Row(equal_height=True):
                            groundsam_button = gr.Button('auto segment')
                
                with gr.Column():
                    with gr.Row(equal_height=True):
                        mask_image1 = gr.Image(type="numpy")

                    with gr.Row(equal_height=True):
                        dropdown_options = ["hair", "top","trousers","shoe","outfit","other"]
                        body_part_select = gr.Dropdown(choices=dropdown_options,value="hair",  label="Select body part: hair(red),top(green),trousers(blue),shoe(yellow),outfit(purple)",interactive=True)

                    with gr.Row(equal_height=True):
                        sam_button = gr.Button('segment with mannul input')
                    
                    with gr.Row(equal_height=True):
                        clear_button = gr.Button('clear the corresponding part')

                with gr.Column(): 
                    mask_image2 = gr.Image(type="numpy")

            with gr.Row():
                with gr.Column(scale=2):
                    mannul_retrieve_btn = gr.Button(value="retrieve using Image", variant="primary")



        with gr.Tab(label='Output'):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    with gr.Row():
                        caption1 = gr.Textbox(label="caption1")
                    with gr.Row():
                        # save_mesh1_1 = gr.Image(value=None, label="Save Mesh1_1",interactive=False,show_download_button=False)
                        save_mesh1_1 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
                    with gr.Row():
                        save_mesh1_2 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
                    with gr.Row():
                        save_mesh1_3 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
                
                with gr.Column(scale=1):
                    with gr.Row():
                        caption2 = gr.Textbox(label="caption2")
                    with gr.Row():
                        save_mesh2_1 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
                    with gr.Row():
                        save_mesh2_2 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
                    with gr.Row():
                        save_mesh2_3 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)

                with gr.Column(scale=1):
                    with gr.Row():
                        caption3 = gr.Textbox(label="caption3")
                    with gr.Row():
                        save_mesh3_1 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
                    with gr.Row():
                        save_mesh3_2 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
                    with gr.Row():
                        save_mesh3_3 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)

                with gr.Column(scale=1):
                    with gr.Row():
                        caption4 = gr.Textbox(label="caption4")
                    with gr.Row():
                        save_mesh4_1 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
                    with gr.Row():
                        save_mesh4_2 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
                    with gr.Row():
                        save_mesh4_3 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
                
                with gr.Column(scale=1):
                    with gr.Row():
                        caption5 = gr.Textbox(label="caption5")
                    with gr.Row():
                        save_mesh5_1 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
                    with gr.Row():
                        save_mesh5_2 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
                    with gr.Row():
                        save_mesh5_3 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
        
        with gr.Tab(label='combine'):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    with gr.Row():
                        save_mesh_combine = gr.Model3D(value=None, label="Combined Mesh", interactive=False,elem_id="model-in")
                # with gr.Column(scale=1):
                #     with gr.Row():
                #         save_mesh_combine1 = gr.Model3D(value=None, label="Combined Mesh", interactive=False,elem_id="model-in")
                # with gr.Column(scale=1):
                #     with gr.Row():
                #         save_mesh_combine2 = gr.Model3D(value=None, label="Combined Mesh", interactive=False,elem_id="model-in")
            
        # part_keys = ['hair','top','bottom','shoe','outfit','others']

        def store_img(img):
            return img, []  # when new image is uploaded, `selected_points` should be empty
        input_image.upload(
            store_img,
            [input_image],
            [original_image, selected_points]
        )

        def get_point(img, sel_pix, point_type, evt: gr.SelectData):
            print("get_point")
            print(f"{evt.value} at {evt.index} from {evt.target}")
            if point_type == 'foreground_point':
                sel_pix.append((evt.index, 1))   # append the foreground_point
            elif point_type == 'background_point':
                sel_pix.append((evt.index, 0))    # append the background_point
            else:
                sel_pix.append((evt.index, 1))    # default foreground_point
            # draw points
            print(sel_pix)
            for point, label in sel_pix:
                cv2.drawMarker(img, point, colors[label], markerType=markers[label], markerSize=5, thickness=3)
            print(img.shape)
            # if img[..., 0][0, 0] == img[..., 2][0, 0]:  # BGR to RGB
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img if isinstance(img, np.ndarray) else np.array(img)
        input_image.select(
            get_point,
            [input_image, selected_points, radio],
            [input_image],
        )

        def undo_points(orig_img, sel_pix):
            if isinstance(orig_img, int):   # if orig_img is int, the image if select from examples
                temp = cv2.imread(image_examples[orig_img][0])
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            else:
                temp = orig_img.copy()
            # draw points
            if len(sel_pix) != 0:
                sel_pix.pop()
                for point, label in sel_pix:
                    cv2.drawMarker(temp, point, colors[label], markerType=markers[label], markerSize=5, thickness=5)
            # if temp[..., 0][0, 0] == temp[..., 2][0, 0]:  # BGR to RGB
            #     temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            return temp if isinstance(temp, np.ndarray) else np.array(temp)
        undo_button.click(
            undo_points,
            [original_image, selected_points],
            [input_image]
        )

        # auto_retrieve_btn.click(
        #     fn=image_retrive_all,
        #     inputs=[
        #         input_image,gender_select
        #     ],
        #     outputs=[
        #         caption1,save_mesh1_1,save_mesh1_2,save_mesh1_3,
        #         caption2,save_mesh2_1,save_mesh2_2,save_mesh2_3,
        #         caption3,save_mesh3_1,save_mesh3_2,save_mesh3_3,
        #         caption4,save_mesh4_1,save_mesh4_2,save_mesh4_3,
        #         caption5,save_mesh5_1,save_mesh5_2,save_mesh5_3,
        #         mask_image1,mask_image2
        #     ]
        # )

        # sd_btn.click(
        #     fn=sd_retrive_all,
        #     inputs=[
        #         txt_input
        #     ],
        #     outputs=[
        #         input_image
        #     ]
        # )

        sam_button.click(sam_seg, inputs=[original_image,selected_points,body_part_select,mask_state],
                    outputs=[mask_image1,mask_image2,input_image])
        
        groundsam_button.click(ground_sam_image, inputs=[original_image,mask_state],
                    outputs=[mask_image1,mask_image2])
        
        # mannul_retrieve_btn.click(image_retrive_parts, inputs=[original_image,gender_select,mask_state],
        #             outputs=[save_mesh_combine])

        mannul_retrieve_btn.click(image_retrive_parts, inputs=[original_image,gender_select,mask_state],
                    outputs=[caption1,save_mesh1_1,save_mesh1_2,save_mesh1_3,
                            caption2,save_mesh2_1,save_mesh2_2,save_mesh2_3,
                            caption3,save_mesh3_1,save_mesh3_2,save_mesh3_3,
                            caption4,save_mesh4_1,save_mesh4_2,save_mesh4_3,
                            caption5,save_mesh5_1,save_mesh5_2,save_mesh5_3,save_mesh_combine])
        
        clear_button.click(clear_part, inputs=[original_image,mask_state,body_part_select],
                    outputs=[mask_image1,mask_image2])

    launch_args = {"server_port": port}
    
    if listen:
        launch_args["server_name"] = "0.0.0.0"
        launch_args["inbrowser"] = True
    demo.queue().launch(**launch_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--port", type=int, default=8084)
    args = parser.parse_args()

    launch(
        args.port,
        listen=args.listen
    )


   
