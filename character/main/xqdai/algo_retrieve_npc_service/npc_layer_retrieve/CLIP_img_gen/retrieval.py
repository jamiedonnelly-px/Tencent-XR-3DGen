import os
import argparse
import faiss
import torch
import numpy as np
import pickle
from collections import Counter

import clip
import cv2
from PIL import Image
import json

import ipdb
import time
import sys

import time


FAISS_GPU_IDX_PATH = "/aigc_cfs_gdp/xiaqiangdai/retrieve_libs/emb_all/top/NPC_faiss_gpu_index_normalised.npy"
OBJ_PATH = '/aigc_cfs_gdp/xiaqiangdai/retrieve_libs/emb_all/top/NPC_keys.pkl'
EMB_2_OBJ = '/aigc_cfs_gdp/xiaqiangdai/retrieve_libs/emb_all/top/NPC_emb_idx_2_obj_idx.pkl'

def load_faiss_index_gpu(path=FAISS_GPU_IDX_PATH,gpu_id=0):
    res = faiss.StandardGpuResources()
    faiss_gpu_index = faiss.deserialize_index(np.load(path))
    faiss_gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, faiss_gpu_index)
    
    return faiss_gpu_index

def load_emb_2_index(path=EMB_2_OBJ):
    with open(path, 'rb') as file:
        emb2obj = pickle.load(file)
    return emb2obj

def load_obj_keys(path=OBJ_PATH):
    with open(path, 'rb') as file:
        obj_keys = pickle.load(file)
    return obj_keys

def prepare_img(img):
    if img.max() <= 1.0:
        img = (img * 255).astype('uint8')

    img = Image.fromarray(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return img

def calc_img_emb(img, model, preprocess, device):
    image_input = preprocess(img).unsqueeze(0).to(device)
    print(image_input.shape)
    with torch.no_grad():
        image_embed = model.encode_image(image_input)
    image_embed_norm = image_embed / image_embed.norm()
    
    return image_embed_norm

def calc_txt_emb(text, model, device,cn=False):
    if cn==False:
        text_inputs = torch.cat([clip.tokenize(text)]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        text_features = text_features / text_features.norm()
    else:
        text_features = model.forward_text(text).detach()
    
    return text_features
    

def keys_dict1(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    folder_keys_dict = {}
    for key in json_data['data'].keys():
        for key_1 in json_data['data'][key].keys():
            for key_2 in json_data['data'][key][key_1].keys():
                # path = json_data['data'][key][key_1][key_2]['ImgDir']
                # image_path = json_data['data'][key][key_1][key_2]['Preview']
                # mesh_path = json_data['data'][key][key_1][key_2]['Mesh']
                # obj_path = json_data['data'][key][key_1][key_2]['Obj_Mesh']
                # body_key = json_data['data'][key][key_1][key_2]['body_key']
                
                # folder_keys_dict[folder_key]=[key,key_1,image_path,mesh_path,obj_path,obj_path.replace('.obj','.glb'),body_key]
                folder_key = key_2
                folder_keys_dict[folder_key] = json_data['data'][key][key_1][key_2]
    return folder_keys_dict

def keys_dict(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    folder_keys_dict = {}
    for key in json_data['data'].keys():
        for key_1 in json_data['data'][key].keys():
                folder_key = key_1
                folder_keys_dict[folder_key] = json_data['data'][key][key_1]
    return folder_keys_dict

def retrive_topk_paths(faiss_gpu_index, image_embed_norm, emb2obj, obj_keys, topk,folder_keys_dict,data_type = "muchangwuyu_extend"):
    distances, indices = faiss_gpu_index.search(
        np.array(image_embed_norm.cpu()).astype(np.float32),
        topk)
    
    key_idxes = []
    score={}
    for index,idx in enumerate(indices[0]):
        key_idxes.append(emb2obj[idx])
        score[emb2obj[idx]] = distances[0][index]
    idx_count = Counter(key_idxes)
    # print(score)
    found_keys = {}
    for idx in idx_count.most_common():
        # ipdb.set_trace()
        key = obj_keys[idx[0]]
        key_info = folder_keys_dict[key].copy()
        key_info['score'] = score[idx[0]]
        found_keys[key]=key_info
        
    return found_keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default='/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/data/test.png')
    parser.add_argument('--json_path', default='/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/mcwy_data_withObj_20240202.json')
    parser.add_argument("--topk", default=3)
    parser.add_argument("--dataset_path", default='/mnt/aigc_cfs_gz/layer_avatar_data/readplayerMe/man_sep')

    args, extra = parser.parse_known_args()
    
    start_time = time.time()
    faiss_gpu_index = load_faiss_index_gpu()
    emb2obj = load_emb_2_index()
    obj_keys = load_obj_keys()

    # faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)
    print(f"index.is_trained = {faiss_gpu_index.is_trained}")
    print(f"index.ntotal = {faiss_gpu_index.ntotal}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-L/14')

    print('finished loading clip')

    end_time = time.time()
    print("load cost: {:.2f}s".format(end_time - start_time))
    # start_time = time.time()
    # img = cv2.imread(args.input)
    # img = prepare_img(img)
    # image_embed_norm = calc_img_emb(img, model, preprocess, device)
    # json_data = keys_dict(args.json_path)
    # paths = retrive_topk_paths(
    #     faiss_gpu_index, image_embed_norm,
    #     emb2obj, obj_keys, args.dataset_path,
    #     args.topk,json_data)
    
    # for path in paths:
    #     print(path)
        
    # text_features = calc_txt_emb("white trousers", model, device)
        
    # paths = retrive_topk_paths(
    #     faiss_gpu_index, text_features,
    #     emb2obj, obj_keys, args.dataset_path,
    #     args.topk,args.json_path)
    # print(paths[:3])
    # # for path in paths:
    # #     print(path)
    # end_time = time.time()
    # print("infer cost: {:.2f}s".format(end_time - start_time))

    text_features = calc_txt_emb("white top", model, device)
 
    start_time = time.time()
    json_path ="/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/layer_embedding_20240507_total.json"
    g_json_data = keys_dict(json_path)
    out = retrive_topk_paths(
        faiss_gpu_index, text_features,
        emb2obj, obj_keys,
        10,
        g_json_data,
        data_type="all")
    keys = [key for key in out.keys()]
    print(out[keys[0]])
    # for path in paths:
    #     print(path)
    end_time = time.time()
    print("infer cost: {:.2f}s".format(end_time - start_time))
