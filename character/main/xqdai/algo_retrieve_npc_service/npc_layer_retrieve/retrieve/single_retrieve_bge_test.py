import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, parent_dir)

import argparse

import numpy as np
from PIL import Image
import time
import json
import torch
import threading
import random
sys.path.append("/mnt/aigc_cfs_cq/xiaqiangdai/project/FlagEmbedding/research/visual_bge")
from visual_bge.modeling import Visualized_BGE


num_gpus = torch.cuda.device_count()

data_type = "20241010_bge" #muchangwuyu,readmeplay,muchangwuyu_extend,all

FAISS_GPU_IDX_PATH_ori = "/mnt/aigc_cfs_cq/xiaqiangdai/project/npc_layer_retrieve/retrieve_libs/emb/NPC_faiss_gpu_index_normalised.npy"
KEYS_PATH_ori = '/mnt/aigc_cfs_cq/xiaqiangdai/project/npc_layer_retrieve/retrieve_libs/emb/NPC_keys.pkl'
EMB_2_OBJ_ori = '/mnt/aigc_cfs_cq/xiaqiangdai/project/npc_layer_retrieve/retrieve_libs/emb/NPC_emb_idx_2_obj_idx.pkl'

genders=["male","female"]
part_keys = ['hair','top','trousers','shoe','outfit','others']

FAISS_GPU_IDX_PATH = {}
KEYS_PATH={}
EMB_2_OBJ={}
key_list = []
locks={}
for gender in genders:
    for key in part_keys:
        key_temp = gender+'_'+key
        key_list.append(key_temp)
        FAISS_GPU_IDX_PATH[key_temp] = FAISS_GPU_IDX_PATH_ori.replace('/emb','/emb_'+data_type+'/'+key_temp)
        KEYS_PATH[key_temp] = KEYS_PATH_ori.replace('/emb','/emb_'+data_type+'/'+key_temp)
        EMB_2_OBJ[key_temp] = EMB_2_OBJ_ori.replace('/emb','/emb_'+data_type+'/'+key_temp)
        locks[key_temp] = threading.Lock()


from bge_img_gen.retrieval_bge import (
    load_faiss_index_gpu,
    load_emb_2_index,
    load_obj_keys,
    prepare_img,
    calc_img_emb,
    calc_txt_emb,
    retrive_topk_paths,
    keys_dict
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Visualized_BGE(model_name_bge = "BAAI/bge-m3", model_weight="/aigc_cfs_2/xiaqiangdai/models/Visualized_m3.pth")
   
model.to(device)
model.eval()

json_path ="/aigc_cfs_gdp/xiaqiangdai/retrieve_libs/20241010_daz_decimate_add_ct.json"
global g_json_data
g_json_data = keys_dict(json_path)

# print(FAISS_GPU_IDX_PATH)
# print(EMB_2_OBJ)
# print(KEYS_PATH)
faiss_gpu_index={}
emb2obj={}
obj_keys={}

for id,gender in enumerate(genders):
    for key in part_keys:
        key_temp = gender+'_'+key
        print(key_temp)
        if num_gpus>1:
            faiss_gpu_index[key_temp] = load_faiss_index_gpu(path = FAISS_GPU_IDX_PATH[key_temp],gpu_id=id)
        else:
            faiss_gpu_index[key_temp] = load_faiss_index_gpu(path = FAISS_GPU_IDX_PATH[key_temp],gpu_id=0)
        emb2obj[key_temp] = load_emb_2_index(path = EMB_2_OBJ[key_temp])
        obj_keys[key_temp] = load_obj_keys(path = KEYS_PATH[key_temp])


def shuffle_list(lst):
    n = len(lst)
    for i in range(n-1, 0, -1):
        j = random.randint(0, i)
        lst[i], lst[j] = lst[j], lst[i]
    return lst

def retrive_single_txt(prompt,key_in):
    if ''==prompt or ' '==prompt or None==prompt or 'None'==prompt or key_in not in key_list:
        return [None,None,None]
    query_txt_embed = calc_txt_emb(prompt, model, device)
    locks[key_in].acquire()
    found_keys = retrive_topk_paths(
        faiss_gpu_index[key_in], query_txt_embed,
        emb2obj[key_in], obj_keys[key_in],
        10,
        g_json_data,
        data_type)
    locks[key_in].release()
    
    paths = []
    keys = []
    
    for key in found_keys.keys():
        keys.append(key)
    key_temp = keys.copy()
    # key_temp = shuffle_list(key_temp)
    output_num = 6
    keys = key_temp[:output_num]

    if len(keys) < output_num:
        for i in range(output_num-len(keys)):
            keys.append(None)
    for i in range(output_num):
        if keys[i]!=None:
            print(found_keys[keys[i]]['score'])
    return keys[:output_num]

def webui_retrieve(prompt,gender,part):
    key_in = gender+'_'+part
    keys = retrive_single_txt(prompt,key_in)
    glb_paths = []
    for key in keys:
        if key is not None:
            glb_paths.append(g_json_data[key]['GLB_Mesh'])
        else:
            glb_paths.append(None)
    return glb_paths

def test_performance():
    with open('/mnt/aigc_cfs_cq/xiaqiangdai/project/npc_image_retrieve/lib_generate/test_list.json', 'r') as file:
        list_test = json.load(file)

    with open('/mnt/aigc_cfs_cq/xiaqiangdai/project/npc_image_retrieve/lib_generate/caption_hunyuan_en.json', 'r') as file:
        caption_list = json.load(file)
    
    json_path ="/mnt/aigc_cfs_cq/xiaqiangdai/project/npc_layer_retrieve/20241010_daz_decimate_add_ct.json"
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    g_json_data = keys_dict(json_path)
    folder_keys_dict={}
    for key in json_data['data'].keys():
        for key_1 in json_data['data'][key].keys():
            path = json_data['data'][key][key_1]['ImgDir']
            category = json_data['data'][key][key_1]['Category']
            folder_key = key_1
            # print(key,key_1,key_2)
            gender = json_data['data'][key][key_1]['Gender']
            if gender=="Asexual":
                folder_keys_dict[folder_key]=['male_'+category.lower(),'female_'+category.lower()]
            else:
                folder_keys_dict[folder_key]=[gender.lower()+'_'+category.lower()]


    top5_num = 0
    all_num = 0
    for key in list_test:
        if key not in folder_keys_dict.keys():
            continue
        text_caption = caption_list[key]["caption"]
        text_features = calc_txt_emb(text_caption, model,device)

        for key_in in folder_keys_dict[key]:
            out = retrive_topk_paths(
                faiss_gpu_index[key_in], text_features,
                emb2obj[key_in], obj_keys[key_in],
                10,
                g_json_data,
                data_type="all")
            keys = [key for key in out.keys()][:5]
            if key in keys:
                top5_num+=1
            all_num+=1
            
    
    print(f"top5_num:{top5_num} all_num:{all_num}")

if __name__ == "__main__":

    json_path ="/mnt/aigc_cfs_cq/xiaqiangdai/project/npc_layer_retrieve/20241010_daz_decimate_add_ct.json"
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    g_json_data = keys_dict(json_path)
    folder_keys_dict={}
    for key in json_data['data'].keys():
        for key_1 in json_data['data'][key].keys():
            path = json_data['data'][key][key_1]['ImgDir']
            category = json_data['data'][key][key_1]['Category']
            folder_key = key_1
            # print(key,key_1,key_2)
            gender = json_data['data'][key][key_1]['Gender']
            if gender=="Asexual":
                folder_keys_dict[folder_key]=['male_'+category.lower(),'female_'+category.lower()]
            else:
                folder_keys_dict[folder_key]=[gender.lower()+'_'+category.lower()]


    text_captions_dict = {'business suit':"male_outfit",'polo shirt':"male_top",'red dress with flower pattern':"female_outfit"}
    for text_caption in text_captions_dict.keys():
        text_features = calc_txt_emb(text_caption, model,device)

        key_in = text_captions_dict[text_caption]
        out = retrive_topk_paths(
            faiss_gpu_index[key_in], text_features,
            emb2obj[key_in], obj_keys[key_in],
            10,
            g_json_data,
            data_type="all")
        keys = [key for key in out.keys()][:5]
        print(text_caption,keys)

