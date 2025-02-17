from sentence_transformers import SentenceTransformer

model = SentenceTransformer('/mnt/aigc_cfs_cq/xiaqiangdai/motion/motion_text_retrieve/bge-large-zh-v1.5.pt')
import numpy as np
import sys
import os
sys.path.append("/apdcephfs/private_xiaqiangdai/workspace/motion/TMR")
from demo.load import load_unit_embeddings, load_splits, load_json
import random


documents = np.load('h3d/humanml3d_texts.npy',allow_pickle=True).tolist() 
idx_to_key = np.load('h3d/humanml3d_idx_to_key.npy',allow_pickle=True).item()

list_path = "/mnt/aigc_cfs_cq/xiaqiangdai/motion/motion_text_retrieve/filenames.txt"
f=open(list_path,'r')
mixm_documents_keys = [line.strip() for line in f.readlines()]
f.close()

description_list_path = "/mnt/aigc_cfs_cq/xiaqiangdai/motion/motion_text_retrieve/filenames_description.txt"
f1=open(description_list_path,'r')
mixm_documents = [line.strip() for line in f1.readlines()]
f1.close()

num_idx_to_key=len(list(idx_to_key.keys()))
print(num_idx_to_key)
print(len(documents))
print(list(idx_to_key.keys())[-1])
index_init = num_idx_to_key

for i,key in enumerate(mixm_documents_keys):
    # print(key,":",mixm_documents[i])
    documents.append(mixm_documents[i])
    idx_to_key[index_init] = key
    index_init+=1
np.save('h3d/humanml3d_mixmo_texts.npy',documents)
np.save('h3d/humanml3d_mixmo_idx_to_key.npy',idx_to_key)


    
    




