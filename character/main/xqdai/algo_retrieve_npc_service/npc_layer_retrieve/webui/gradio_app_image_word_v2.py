import os

import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, parent_dir)

sys.path.append("/mnt/aigc_cfs_cq/xiaqiangdai/project/segment_anything_webui")
from inference import run_inference
import argparse
import gradio as gr
import torch

import cv2
import clip
import numpy as np
import rembg
import torch
from PIL import Image

import uuid
import time
import json
import requests

import sys
# sys.path.append('/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/bodyfit')
# from smpl_weights.run_auto_rig import auto_rig_layer

colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]

data_type = "muchangwuyu_extend" #muchangwuyu,readmeplay,muchangwuyu_extend

FAISS_GPU_IDX_PATH_ori = "/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/retrieve_libs/emb/NPC_faiss_gpu_index_normalised.npy"
KEYS_PATH_ori = '/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/retrieve_libs/emb/NPC_keys.pkl'
EMB_2_OBJ_ori = '/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/retrieve_libs/emb/NPC_emb_idx_2_obj_idx.pkl'

part_keys = ['hair','top','bottom','shoe','outfit','others']

FAISS_GPU_IDX_PATH = []
KEYS_PATH=[]
EMB_2_OBJ=[]
for key in part_keys:
    FAISS_GPU_IDX_PATH.append(FAISS_GPU_IDX_PATH_ori.replace('/emb','/emb_'+data_type+'/'+key))
    KEYS_PATH.append(KEYS_PATH_ori.replace('/emb','/emb_'+data_type+'/'+key))
    EMB_2_OBJ.append(EMB_2_OBJ_ori.replace('/emb','/emb_'+data_type+'/'+key))

if data_type=="readmeplay":
    Global_PATH = '/mnt/aigc_cfs_gz/layer_avatar_data/readplayerMe/man_sep'
else:
    Global_PATH=''


from CLIP_img_gen.retrieval import (
    load_faiss_index_gpu,
    load_emb_2_index,
    load_obj_keys,
    prepare_img,
    calc_img_emb,
    calc_txt_emb,
    retrive_topk_paths,
    keys_dict,
    keys_dict_image,
    retrive_topk_image_paths

)
json_path ="/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/mcwy_data_withObj_20240202.json"
global g_json_data
g_json_data = keys_dict_image(json_path)

from ipdb import set_trace as st
    
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



print(FAISS_GPU_IDX_PATH)
print(EMB_2_OBJ)
print(KEYS_PATH)
faiss_gpu_index=[]
emb2obj=[]
obj_keys=[]
for i,key in enumerate(part_keys):
    faiss_gpu_index.append(load_faiss_index_gpu(path = FAISS_GPU_IDX_PATH[i]))
    emb2obj.append(load_emb_2_index(path = EMB_2_OBJ[i]))
    obj_keys.append(load_obj_keys(path = KEYS_PATH[i]))

# sdxl_pipe = load_SDXL_pipeline()
rembg_session = rembg.new_session()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14')
model.to(device)


def wrap_cloth( mesh_output_path, paths_temp ):
    obj_lst = os.path.join(mesh_output_path, "object_lst.txt")

    json_object = json.dumps(paths_temp, indent=4)

    print("json_object", json_object)
    with open(obj_lst, "w") as f:
        f.write(json_object)

    # with open(obj_lst, "w") as f:
    #     for line in paths_temp:
    #         f.write(f"{line}\n")


    cmd = ["/root/miniconda3/envs/auto_rig/bin/python",
           "/mnt/aigc_cfs_cq/rabbityli/bodyfit/webui/cloth_warpper.py",
           "--lst_path",
           obj_lst]

    cmd = " ".join(cmd)
    os.system(cmd)
    return mesh_output_path


def retrive_single_txt(prompt,key_id):
    if ''==prompt or ' '==prompt:
        return [None,None,None],[None,None,None]
    query_txt_embed = calc_txt_emb(prompt, model, device)
    found_keys = retrive_topk_image_paths(
        faiss_gpu_index[key_id], query_txt_embed,
        emb2obj[key_id], obj_keys[key_id],
        Global_PATH,
        10,
        g_json_data,
        data_type)
    
    paths = []
    keys = []
    
    for key in found_keys.keys():
        paths.append(found_keys[key])
        keys.append(key)

    if len(found_keys.keys()) < 3:
        for i in range(3-len(found_keys.keys())):
            paths.append(None)
            keys.append(None)
    print(paths[:3])
    return keys[:3],paths[:3]

def retrive_txts(prompt,save_mesh1_index,save_mesh2_index,save_mesh3_index,save_mesh4_index,save_mesh5_index,save_mesh6_index):
    save_mesh1_index.clear()
    save_mesh1_index.append(0)
    save_mesh2_index.clear()
    save_mesh2_index.append(0)
    save_mesh3_index.clear()
    save_mesh3_index.append(0)
    save_mesh4_index.clear()
    save_mesh4_index.append(0)
    save_mesh5_index.clear()
    save_mesh5_index.append(0)
    save_mesh6_index.clear()
    save_mesh6_index.append(0)

    strs = prompt.split(',')
    print(f"befour:{strs} len:{len(strs)}")
    if len(strs)<6 and len(strs)>0:
        for i in range(6-len(strs)):
            strs.append(strs[len(strs)-1])
    elif len(strs)>6:
        strs = strs[:6]
    
    print(f"after:{strs} len:{len(strs)}")
    paths_out = []
    keys_out = []
    for key_id,s in enumerate(strs):
        keys,paths = retrive_single_txt(s,key_id)

        None_num  = paths.count(None)
        print(s,None_num)
        if None_num==3 and (''!=s and ' '!=s):
            if key_id==0:
                keys,paths = retrive_single_txt('hair',key_id)
            elif key_id==1:
                keys,paths = retrive_single_txt('coat',key_id)
            elif key_id==2:
                keys,paths = retrive_single_txt('trousers',key_id)
            elif key_id==3:
                keys,paths = retrive_single_txt('shoe',key_id)
            elif key_id==4:
                keys,paths = retrive_single_txt('outfit',key_id)
            else:
                keys,paths = retrive_single_txt('bracelet',key_id)

        for i in range(len(keys)):
            paths_out.append(paths[i])
            keys_out.append(keys[i])
    

    paths_out.append([paths_out[0],paths_out[1],paths_out[2],None])
    paths_out.append([paths_out[3],paths_out[4],paths_out[5],None])
    paths_out.append([paths_out[6],paths_out[7],paths_out[8],None])
    paths_out.append([paths_out[9],paths_out[10],paths_out[11],None])
    paths_out.append([paths_out[12],paths_out[13],paths_out[14],None])
    paths_out.append([paths_out[15],paths_out[16],paths_out[17],None])


    paths_out.append(paths_out[0])
    paths_out.append(paths_out[3])
    paths_out.append(paths_out[6])
    paths_out.append(paths_out[9])
    paths_out.append(paths_out[12])
    paths_out.append(paths_out[15])


    paths_out.append([keys_out[0],keys_out[1],keys_out[2],None])
    paths_out.append([keys_out[3],keys_out[4],keys_out[5],None])
    paths_out.append([keys_out[6],keys_out[7],keys_out[8],None])
    paths_out.append([keys_out[9],keys_out[10],keys_out[11],None])
    paths_out.append([keys_out[12],keys_out[13],keys_out[14],None])
    paths_out.append([keys_out[15],keys_out[16],keys_out[17],None])

    return paths_out

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
    """

    with gr.Blocks(
        title="NPC Retrieve - Web Demo",
        css=custom_css,
    ) as demo:
        with gr.Row(equal_height=True):
            header = """
            # NPC Retrieve Demo
            """
            gr.Markdown(header)
        
                
        with gr.Row().style(equal_height=True):
            with gr.Column(scale=4):
                with gr.Row().style(equal_height=True):
                    txt_input = gr.Textbox(
                        value='',
                        placeholder="please input six prompts with comma for hair,top,trousers,shoe,outfit,others,allow no result with ' '",
                        show_label=False,
                        interactive=True)
            with gr.Column(scale=2):
                txt_btn = gr.Button(value="Retrieve using Text", variant="primary")

        with gr.Row().style(equal_height=False):
            with gr.Column(scale=1):
                with gr.Row():
                    save_mesh1_1 = gr.Image(value=None, label="Save Mesh1_1", height = 200,width=200,interactive=False,show_download_button=False)
                with gr.Row():
                    save_mesh1_2 = gr.Image(value=None, label="Save Mesh1_2", height = 200,width=200,interactive=False,show_download_button=False)
                with gr.Row():
                    save_mesh1_3 = gr.Image(value=None, label="Save Mesh1_3", height = 200,width=200,interactive=False,show_download_button=False)
            
            with gr.Column(scale=1):
                with gr.Row():
                    save_mesh2_1 = gr.Image(value=None, label="Save Mesh2_1", height = 200,width=200,interactive=False,show_download_button=False)
                with gr.Row():
                    save_mesh2_2 = gr.Image(value=None, label="Save Mesh2_2", height = 200,width=200,interactive=False,show_download_button=False)
                with gr.Row():
                    save_mesh2_3 = gr.Image(value=None, label="Save Mesh2_3", height = 200,width=200,interactive=False,show_download_button=False)

            with gr.Column(scale=1):
                with gr.Row():
                    save_mesh3_1 = gr.Image(value=None, label="Save Mesh3_1", height = 200,width=200,interactive=False,show_download_button=False)
                with gr.Row():
                    save_mesh3_2 = gr.Image(value=None, label="Save Mesh3_2", height = 200,width=200,interactive=False,show_download_button=False)
                with gr.Row():
                    save_mesh3_3 = gr.Image(value=None, label="Save Mesh3_3", height = 200,width=200,interactive=False,show_download_button=False)

            with gr.Column(scale=1):
                with gr.Row():
                    save_mesh4_1 = gr.Image(value=None, label="Save Mesh4_1", height = 200,width=200,interactive=False,show_download_button=False)
                with gr.Row():
                    save_mesh4_2 = gr.Image(value=None, label="Save Mesh4_2", height = 200,width=200,interactive=False,show_download_button=False)
                with gr.Row():
                    save_mesh4_3 = gr.Image(value=None, label="Save Mesh4_3", height = 200,width=200,interactive=False,show_download_button=False)
            
            with gr.Column(scale=1):
                with gr.Row():
                    save_mesh5_1 = gr.Image(value=None, label="Save Mesh5_1", height = 200,width=200,interactive=False,show_download_button=False)
                with gr.Row():
                    save_mesh5_2 = gr.Image(value=None, label="Save Mesh5_2", height = 200,width=200,interactive=False,show_download_button=False)
                with gr.Row():
                    save_mesh5_3 = gr.Image(value=None, label="Save Mesh5_3", height = 200,width=200,interactive=False,show_download_button=False)
            
            with gr.Column(scale=1):
                with gr.Row():
                    save_mesh6_1 = gr.Image(value=None, label="Save Mesh6_1", height = 200,width=200,interactive=False,show_download_button=False)
                with gr.Row():
                    save_mesh6_2 = gr.Image(value=None, label="Save Mesh6_2", height = 200,width=200,interactive=False,show_download_button=False)
                with gr.Row():
                    save_mesh6_3 = gr.Image(value=None, label="Save Mesh6_3", height = 200,width=200,interactive=False,show_download_button=False)
        # part_keys = ['hair','top','bottom','shoe','outfit','others']
        with gr.Row().style(equal_height=False):
            with gr.Column(scale=1):
                output_mesh1_btn = gr.Button(value="switch_hair", variant="primary")
            with gr.Column(scale=1):
                output_mesh2_btn = gr.Button(value="switch_top", variant="primary")
            with gr.Column(scale=1):
                output_mesh3_btn = gr.Button(value="switch_trousers", variant="primary")
            with gr.Column(scale=1):
                output_mesh4_btn = gr.Button(value="switch_shoe", variant="primary")
            with gr.Column(scale=1):
                output_mesh5_btn = gr.Button(value="switch_outfit", variant="primary")
            with gr.Column(scale=1):
                output_mesh6_btn = gr.Button(value="switch_others", variant="primary")
        
        with gr.Row().style(equal_height=False):
            with gr.Column(scale=1):
                save_mesh1_list = gr.State([]) 
                save_key1_list = gr.State([]) 
                save_mesh1_index = gr.State([0]) 
                save_mesh1 = gr.Image(value=None, label="Save Mesh1", height = 200,width=200,interactive=False,show_download_button=False)
            with gr.Column(scale=1):
                save_mesh2_list = gr.State([]) 
                save_key2_list = gr.State([]) 
                save_mesh2_index = gr.State([0]) 
                save_mesh2 = gr.Image(value=None, label="Save Mesh2", height = 200,width=200,interactive=False,show_download_button=False)
            with gr.Column(scale=1):
                save_mesh3_list = gr.State([]) 
                save_key3_list = gr.State([]) 
                save_mesh3_index = gr.State([0]) 
                save_mesh3 = gr.Image(value=None, label="Save Mesh3", height = 200,width=200,interactive=False,show_download_button=False)
            with gr.Column(scale=1):
                save_mesh4_list = gr.State([]) 
                save_key4_list = gr.State([]) 
                save_mesh4_index = gr.State([0]) 
                save_mesh4 = gr.Image(value=None, label="Save Mesh4", height = 200,width=200,interactive=False,show_download_button=False)
            with gr.Column(scale=1):
                save_mesh5_list = gr.State([]) 
                save_key5_list = gr.State([]) 
                save_mesh5_index = gr.State([0]) 
                save_mesh5 = gr.Image(value=None, label="Save Mesh5", height = 200,width=200,interactive=False,show_download_button=False)
            with gr.Column(scale=1):
                save_mesh6_list = gr.State([]) 
                save_key6_list = gr.State([]) 
                save_mesh6_index = gr.State([0]) 
                save_mesh6 = gr.Image(value=None, label="Save Mesh6", height = 200,width=200,interactive=False,show_download_button=False)
        
        with gr.Row().style(equal_height=False):
            with gr.Column(scale=6):
                with gr.Row():
                    with gr.Column(scale=3):
                        model_generate_btn = gr.Button(value="model generation", variant="primary")
                    with gr.Column(scale=3):
                        dislike_folder = gr.State([]) 
                        dislike_btn = gr.Button(value="dislike", variant="primary")
                with gr.Row():
                    with gr.Column(scale=3):
                        all_generate_mesh = gr.Model3D(value=None, label="all generate Mesh", interactive=False)
                    with gr.Column(scale=3):
                        animation_gif = gr.Image(type="numpy")
        
        txt_btn.click(
            fn=retrive_txts,
            inputs=[
                txt_input,save_mesh1_index,save_mesh2_index,save_mesh3_index,save_mesh4_index,save_mesh5_index,save_mesh6_index
            ],
            outputs=[
                save_mesh1_1,save_mesh1_2,save_mesh1_3,
                save_mesh2_1,save_mesh2_2,save_mesh2_3,
                save_mesh3_1,save_mesh3_2,save_mesh3_3,
                save_mesh4_1,save_mesh4_2,save_mesh4_3,
                save_mesh5_1,save_mesh5_2,save_mesh5_3,
                save_mesh6_1,save_mesh6_2,save_mesh6_3,
                save_mesh1_list,save_mesh2_list,save_mesh3_list,save_mesh4_list,save_mesh5_list,save_mesh6_list,
                save_mesh1,save_mesh2,save_mesh3,save_mesh4,save_mesh5,save_mesh6,
                save_key1_list,save_key2_list,save_key3_list,save_key4_list,save_key5_list,save_key6_list
            ]
        )


        def output_mesh1_select(save_mesh1_list,index_list):
            print("Before update:", save_mesh1_list, index_list)
            index = index_list[0]
            if index<3:
                index+=1
            else:
                index = 0
            index_list.clear()
            index_list.append(index)
            print("After update:", save_mesh1_list, index_list)
            print("After update mesh:",save_mesh1_list[index])
            return save_mesh1_list[index]
        output_mesh1_btn.click(
            fn=output_mesh1_select,
            inputs = [save_mesh1_list,save_mesh1_index],
            outputs=[save_mesh1]
        )

        def output_mesh2_select(save_mesh2_list,index_list):
            print("Before update:", save_mesh2_list, index_list)
            index = index_list[0]
            if index<3:
                index+=1
            else:
                index = 0
            index_list.clear()
            index_list.append(index)
            print("After update:", save_mesh2_list, index_list)
            print("After update mesh:", save_mesh2_list[index])
            return save_mesh2_list[index]
        output_mesh2_btn.click(
            fn=output_mesh2_select,
            inputs = [save_mesh2_list,save_mesh2_index],
            outputs=[save_mesh2]
        )

        
        def output_mesh3_select(save_mesh3_list,index_list):
            print("Before update:", save_mesh3_list, index_list)
            index = index_list[0]
            if index<3:
                index+=1
            else:
                index = 0
            index_list.clear()
            index_list.append(index)
            print("After update:", save_mesh3_list, index_list)
            print("After update mesh:", save_mesh3_list[index])
            return save_mesh3_list[index]
        output_mesh3_btn.click(
            fn=output_mesh3_select,
            inputs = [save_mesh3_list,save_mesh3_index],
            outputs=[save_mesh3]
        )

        def output_mesh4_select(save_mesh4_list,index_list):
            print("Before update:", save_mesh4_list, index_list)
            index = index_list[0]
            if index<3:
                index+=1
            else:
                index = 0
            index_list.clear()
            index_list.append(index)
            print("After update:", save_mesh4_list, index_list)
            print("After update mesh:",save_mesh4_list[index])
            return save_mesh4_list[index]
        output_mesh4_btn.click(
            fn=output_mesh4_select,
            inputs = [save_mesh4_list,save_mesh4_index],
            outputs=[save_mesh4]
        )
        def output_mesh5_select(save_mesh5_list,index_list):
            print("Before update:", save_mesh5_list, index_list)
            index = index_list[0]
            if index<3:
                index+=1
            else:
                index = 0
            index_list.clear()
            index_list.append(index)
            print("After update:", save_mesh5_list, index_list)
            print("After update mesh:",save_mesh5_list[index])
            return save_mesh5_list[index]
        output_mesh5_btn.click(
            fn=output_mesh5_select,
            inputs = [save_mesh5_list,save_mesh5_index],
            outputs=[save_mesh5]
        )
        def output_mesh6_select(save_mesh6_list,index_list):
            print("Before update:", save_mesh6_list, index_list)
            index = index_list[0]
            if index<3:
                index+=1
            else:
                index = 0
            index_list.clear()
            index_list.append(index)
            print("After update:", save_mesh6_list, index_list)
            print("After update mesh:",save_mesh6_list[index])
            return save_mesh6_list[index]
        output_mesh6_btn.click(
            fn=output_mesh6_select,
            inputs = [save_mesh6_list,save_mesh6_index],
            outputs=[save_mesh6]
        )

        def dislike(dislike_folder_in):
            if len(dislike_folder_in)==0:
                return
            print(dislike_folder_in)
            f = open('data/dislike.txt','a+')
            f.write(dislike_folder_in[0])
            f.write('\n')
            f.close()
            
        dislike_btn.click(
            fn=dislike,
            inputs = [dislike_folder]
        )


        def model_generation(save_key1_list,save_mesh1_index,save_key2_list,save_mesh2_index,save_key3_list,save_mesh3_index, \
                save_key4_list,save_mesh4_index,save_key5_list,save_mesh5_index, save_key6_list,save_mesh6_index):
            if "muchangwuyu" not in data_type: 
                print("dataset not support")
                return None,None,[]

            input_list=[]
            if len(save_key1_list)!=4 or len(save_mesh1_index)!=1:
                print("length of save_key1_list is not four or index length error")
                return None,None,[]
            if len(save_key2_list)!=4 or len(save_mesh2_index)!=1:
                print("length of save_key2_list is not four or index length error")
                return None,None,[]
            if len(save_key3_list)!=4 or len(save_mesh3_index)!=1:
                print("length of save_key3_list is not four or index length error")
                return None,None,[]
            if len(save_key4_list)!=4 or len(save_mesh4_index)!=1:
                print("length of save_key4_list is not four or index length error")
                return None,None,[]
            if len(save_key5_list)!=4 or len(save_mesh5_index)!=1:
                print("length of save_key5_list is not four or index length error")
                return None,None,[]
            if len(save_key6_list)!=4 or len(save_mesh6_index)!=1:
                print("length of save_key6_list is not four or index length error")
                return None,None,[]

            if save_mesh1_index[0]<0 or save_mesh1_index[0]>3:
                print("save_mesh1_index error")
                return None,None,[]
            if save_mesh2_index[0]<0 or save_mesh2_index[0]>3:
                print("save_mesh1_index error")
                return None,None,[]
            if save_mesh3_index[0]<0 or save_mesh3_index[0]>3:
                print("save_mesh1_index error")
                return None,None,[]
            if save_mesh4_index[0]<0 or save_mesh4_index[0]>3:
                print("save_mesh1_index error")
                return None,None,[]
            if save_mesh5_index[0]<0 or save_mesh5_index[0]>3:
                print("save_mesh1_index error")
                return None,None,[]
            if save_mesh6_index[0]<0 or save_mesh6_index[0]>3:
                print("save_mesh1_index error")
                return None,None,[]

            timestamp = int(time.time())
            unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
            mesh_output_path = "/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/data/"+str(unique_id)
            if not os.path.exists(mesh_output_path):
                os.makedirs(mesh_output_path)

            global g_json_data

            paths_temp = {}
        
            if save_key1_list[save_mesh1_index[0]]!=None:
                key1 = save_key1_list[save_mesh1_index[0]]
                input_list.append(key1)
                paths_temp[g_json_data[key1][4]]='hair'
            if save_key2_list[save_mesh2_index[0]]!=None:
                key2= save_key2_list[save_mesh2_index[0]]
                input_list.append(key2)
                paths_temp[g_json_data[key2][4]]='top'
            if save_key3_list[save_mesh3_index[0]]!=None:
                key3 = save_key3_list[save_mesh3_index[0]]
                input_list.append(key3)
                paths_temp[g_json_data[key3][4]]='trousers'
            if save_key4_list[save_mesh4_index[0]]!=None:
                key4 = save_key4_list[save_mesh4_index[0]]
                input_list.append(key4)
                paths_temp[g_json_data[key4][4]]='shoe'
            if save_key5_list[save_mesh5_index[0]]!=None:
                key5 = save_key5_list[save_mesh5_index[0]]
                input_list.append(key5)
                paths_temp[g_json_data[key5][4]]='outfit'
            # if save_key6_list[save_mesh6_index[0]]!=None:
            #     key6 = save_key6_list[save_mesh6_index[0]]
            #     input_list.append(key6)
            #     paths_temp[g_json_data[key6][4]]='others'

                
            print(paths_temp)
            wrap_cloth( mesh_output_path, paths_temp)

            input = {"folder":mesh_output_path}
            json_data = json.dumps(input)
            headers = {"Content-Type": "application/json"}
            res = requests.post('', data=json_data,headers=headers)
            if res.status_code == 200:
                print("app_autoRig_layer combine Response:", res)
            else:
                print("app_autoRig_layer combine Request failed with status code")
                return None,None,[]

            res = requests.post('', data=json_data,headers=headers)
            if res.status_code == 200:
                print("autoRig_layer auto_rig Response:", res)
            else:
                print("autoRig_layer auto_rig Request failed with status code")
                return None,None,[]
            # auto_rig_layer(mesh_output_path)

            print(mesh_output_path)
            # mesh_output_path = '/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/data/35e40bc3-3adb-55cc-872b-1c7f74a392f3'
            fbx_path = os.path.join(mesh_output_path,"mesh/mesh.fbx")
            gif_path = os.path.join(mesh_output_path,"mesh/mesh_animation.gif")
            input = {"fbx_path":fbx_path,
                    "gif_path":gif_path}
            json_data = json.dumps(input)
            headers = {"Content-Type": "application/json"}
            res = requests.post('', data=json_data,headers=headers)
            if res.status_code == 200:
                print("Response:", res)
            else:
                print("Request failed with status code")
                return None,None,[]
      
            return os.path.join(mesh_output_path,"mesh/mesh.glb"),gif_path,[mesh_output_path]
            # return '/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/data/35e40bc3-3adb-55cc-872b-1c7f74a392f3/mesh.gltf'
              
        model_generate_btn.click(
            fn=model_generation,
            inputs = [save_key1_list,save_mesh1_index,save_key2_list,save_mesh2_index,save_key3_list,save_mesh3_index, \
                    save_key4_list,save_mesh4_index, save_key5_list,save_mesh5_index, save_key6_list,save_mesh6_index],
            outputs=[all_generate_mesh,animation_gif,dislike_folder]
        )   

    launch_args = {"server_port": port}
    
    if listen:
        launch_args["server_name"] = "0.0.0.0"
        launch_args["inbrowser"] = True
    demo.queue().launch(**launch_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--port", type=int, default=8086)
    args = parser.parse_args()

    launch(
        args.port,
        listen=args.listen
    )


   
