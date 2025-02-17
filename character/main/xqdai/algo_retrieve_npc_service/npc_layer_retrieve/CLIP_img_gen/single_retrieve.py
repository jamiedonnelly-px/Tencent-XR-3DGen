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
import clip
import threading
import random

num_gpus = torch.cuda.device_count()
# sys.path.append('/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/bodyfit')
# from smpl_weights.run_auto_rig import auto_rig_layer

data_type = "20241010" #muchangwuyu,readmeplay,muchangwuyu_extend,all

FAISS_GPU_IDX_PATH_ori = "/aigc_cfs_gdp/xiaqiangdai/retrieve_libs/emb/NPC_faiss_gpu_index_normalised.npy"
KEYS_PATH_ori = '/aigc_cfs_gdp/xiaqiangdai/retrieve_libs/emb/NPC_keys.pkl'
EMB_2_OBJ_ori = '/aigc_cfs_gdp/xiaqiangdai/retrieve_libs/emb/NPC_emb_idx_2_obj_idx.pkl'

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


from CLIP_img_gen.retrieval import (
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
model, preprocess = clip.load('ViT-L/14')
model.to(device)

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
    key_temp = shuffle_list(key_temp)
    keys = key_temp[:3]

    if len(keys) < 3:
        for i in range(3-len(keys)):
            keys.append(None)
    for i in range(3):
        if keys[i]!=None:
            print(found_keys[keys[i]]['score'])
    return keys[:3]



if __name__ == "__main__":
    prompt = "zhongshan"
    key_in = "female_outfit"
    results = retrive_single_txt(prompt,key_in)
    for key in results:
        image_path = g_json_data[key]["Preview"]
        print(image_path)

   
