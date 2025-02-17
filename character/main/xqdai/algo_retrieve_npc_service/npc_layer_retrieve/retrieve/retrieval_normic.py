import os
import argparse
import faiss
import torch
import numpy as np
import pickle
from collections import Counter
import cv2
from PIL import Image
import json

import ipdb
import time
import sys

import time
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor


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
    inputs = preprocess(img, return_tensors="pt")

    img_emb = model(**inputs).last_hidden_state
    img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
    
    return img_embeddings

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def calc_txt_emb(text, text_model, tokenizer,device):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = text_model(**encoded_input)

    text_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    text_embeddings = F.layer_norm(text_embeddings, normalized_shape=(text_embeddings.shape[1],))
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    
    return text_embeddings
    

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
    parser.add_argument("--topk", default=3)

    args, extra = parser.parse_known_args()
    
    faiss_gpu_index = load_faiss_index_gpu()
    emb2obj = load_emb_2_index()
    obj_keys = load_obj_keys()

    # faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)
    print(f"index.is_trained = {faiss_gpu_index.is_trained}")
    print(f"index.ntotal = {faiss_gpu_index.ntotal}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5')
    text_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
    text_model.eval()

    
    json_path ="/mnt/aigc_cfs_cq/xiaqiangdai/project/npc_layer_retrieve/20241010_daz_decimate_add_ct.json"
    g_json_data = keys_dict(json_path)

    text_captions = ['polo shirt','business suit']
    for text_caption in text_captions:
        text_features = calc_txt_emb(text_caption, text_model, tokenizer,device)

        out = retrive_topk_paths(
            faiss_gpu_index, text_features,
            emb2obj, obj_keys,
            10,
            g_json_data,
            data_type="all")
        keys = [key for key in out.keys()][:5]
        print(text_caption,keys)
