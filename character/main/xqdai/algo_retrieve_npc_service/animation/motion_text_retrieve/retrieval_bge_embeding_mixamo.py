from sentence_transformers import SentenceTransformer
import numpy as np
import random
import time
import torch
import sys
import os
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

num_gpus = torch.cuda.device_count()
if num_gpus>1:
    device = torch.device('cuda:1')
else:
    device = torch.device('cuda:0')
# model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
model = SentenceTransformer('/aigc_cfs_gdp/xiaqiangdai/bge_models/bge-large-en-v1.5.pt',device=device)
# model.save('bge-large-zh-v1.5.pt')

list_path = os.path.join(parent_directory,"filenames.txt")
f=open(list_path,'r')
documents_obj = [line.strip() for line in f.readlines()]
f.close()

description_list_path = os.path.join(parent_directory,"filenames_description.txt")
f1=open(description_list_path,'r')
documents = [line.strip() for line in f1.readlines()]
f1.close()

embeddings_2 = model.encode(documents, normalize_embeddings=True)

def mixamo_text_retrieve(text_input):
    start_time = time.time()
    instruction = ["Generate a representation for this sentence that can be used to retrieve related articles:"]
    query = [text_input]
    embeddings_1 = model.encode([instruction[0]+query[0]], normalize_embeddings=True)
    score_max = -1000

    score = embeddings_1 @ embeddings_2.T
    best_index = 0
    sorted_idxs = np.argsort(-score[0])
    best_index = sorted_idxs[0]
    score_top5 = [score[0][index] for index in sorted_idxs[:5]]
    print("output_best:",documents[best_index],score[0][best_index]) 
    print("output_top5:",[documents[index] for index in sorted_idxs[:5]],score_top5) 
    arr_std = np.sqrt(np.var(score_top5))
    print(f"arr_std:{arr_std}")
    if arr_std<0.05:
        select_index = random.randint(0,4)
        best_index = sorted_idxs[select_index]
        print(f"select_index:{select_index} best_index:{best_index}")
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print(f"mixamo_text_retrieve Time: {elapsed_time:.2f} seconds")
    return documents_obj[best_index]

if __name__ == '__main__':
    print(mixamo_text_retrieve("running"))

