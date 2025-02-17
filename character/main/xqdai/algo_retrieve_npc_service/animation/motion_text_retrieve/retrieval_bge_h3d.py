from sentence_transformers import SentenceTransformer

model = SentenceTransformer('/mnt/aigc_cfs_cq/xiaqiangdai/motion/motion_text_retrieve/bge-large-zh-v1.5.pt')
import numpy as np
import sys
import os
sys.path.append("/apdcephfs/private_xiaqiangdai/workspace/motion/TMR")
from demo.load import load_unit_embeddings, load_splits, load_json
import random

DATASET = "humanml3d"
h3d_index = load_json(f"/apdcephfs/private_xiaqiangdai/workspace/motion/TMR/datasets/annotations/{DATASET}/annotations.json")
save_folder = "/mnt/aigc_cfs_cq/xiaqiangdai/data/HumanML3D/models"

models_list = [model_name for model_name in os.listdir(save_folder) if model_name.endswith('.fbx')]
print(len(models_list))
print(len(h3d_index.keys()))  
print(h3d_index[list(h3d_index.keys())[0]]['annotations'][0])

h3d_dict = {}
idx_to_key = {}
documents=[]
index=0
num = 0
for i,key in enumerate(h3d_index):
    h3d_dict[key]=[]
    ll = len(h3d_index[key]['annotations'])
    if "humanact12" in h3d_index[key]['path'] or 'M' in key:
        continue
    num+=1
    for j,ann in enumerate(h3d_index[key]['annotations']):
        h3d_dict[key].append(ann['text'])
        documents.append(ann['text'])
        idx_to_key[index]=key
        index+=1

np.save('h3d/humanml3d_texts.npy',documents)
np.save('h3d/humanml3d_idx_to_key.npy',idx_to_key)
print(len(documents))
print(num)
# documents = np.load('h3d/humanml3d_texts.npy',allow_pickle=True)
# idx_to_key = np.load('h3d/humanml3d_idx_to_key.npy',allow_pickle=True)
# print(documents.shape)
# # print(idx_to_key.item().keys())
# embeddings_2 = model.encode(documents, normalize_embeddings=True)
# querys = ['dance','walk','running','jump','run forward, then back','throw a ball']
# for query in querys:
#     embeddings_1 = model.encode([query], normalize_embeddings=True)

#     score = embeddings_1 @ embeddings_2.T
#     best_index = 0
#     sorted_idxs = np.argsort(-score[0])
#     best_index = sorted_idxs[0]
#     print(best_index)
#     print("output_best:",query,documents[best_index],score[0][best_index]) 
#     print("output_top5:",query,[documents[index] for index in sorted_idxs[:5]],[score[0][index] for index in sorted_idxs[:5]]) 
#     score_top5 = [score[0][index] for index in sorted_idxs[:5]]
#     arr_std = np.sqrt(np.var(score_top5))
#     print(f"arr_std:{arr_std}")
#     if arr_std<0.05:
#         select_index = random.randint(0,4)
#         best_index = sorted_idxs[select_index]
#         print(f"query:{query} select_index:{select_index} best_index:{best_index}")
#     print("output_best:",documents[best_index],score[0][best_index],idx_to_key.item()[best_index]) 


