from sentence_transformers import SentenceTransformer
import numpy as np
import random
import time
import torch
import sys
import os
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

device = torch.device('cuda:0')
# model = SentenceTransformer('BAAI/bge-large-en-v1.5')
model = SentenceTransformer('/aigc_cfs_gdp/xiaqiangdai/bge_models/bge-large-en-v1.5.pt',device=device)
# model.save('bge-large-en-v1.5.pt')

documents = np.load("/aigc_cfs_gdp/xiaqiangdai/data/h3d/humanml3d_texts.npy",allow_pickle=True).tolist() 
idx_to_key = np.load("/aigc_cfs_gdp/xiaqiangdai/data/h3d/humanml3d_idx_to_key.npy",allow_pickle=True).item()
    
embeddings_2 = model.encode(documents, normalize_embeddings=True)

def shuffle_list(lst):
    n = len(lst)
    for i in range(n-1, 0, -1):
        j = random.randint(0, i)
        lst[i], lst[j] = lst[j], lst[i]
    return lst

def mixamo_h3d_text_retrieve(text_input):
    start_time = time.time()
    instruction = ["Generate a representation for this sentence that can be used to retrieve related articles:"]
    query = [text_input]
    embeddings_1 = model.encode([instruction[0]+query[0]], normalize_embeddings=True)

    score = embeddings_1 @ embeddings_2.T
    sorted_idxs = np.argsort(-score[0])
    best_index= sorted_idxs[0]
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
    print(f"mixamo_h3d_text_retrieve Time: {elapsed_time:.2f} seconds")
    return idx_to_key[best_index]

def mixamo_h3d_text_retrieve_multi_output(text_input):
    start_time = time.time()
    instruction = ["Generate a representation for this sentence that can be used to retrieve related articles:"]
    query = [text_input]
    embeddings_1 = model.encode([instruction[0]+query[0]], normalize_embeddings=True)

    score = embeddings_1 @ embeddings_2.T
    sorted_idxs = np.argsort(-score[0])
    best_index= sorted_idxs[0]
    score_top5 = [score[0][index] for index in sorted_idxs[:5]]
    
    print("output_best:",documents[best_index],score[0][best_index]) 
    print(len(documents),sorted_idxs[:5])
    print("output_top5:",[documents[index] for index in sorted_idxs[:5]],score_top5) 

    best_indexs = shuffle_list(sorted_idxs[:6])[:4]
    end_time = time.time()
    elapsed_time = end_time - start_time  
    print(f"mixamo_h3d_text_retrieve_multi_output Time: {elapsed_time:.2f} seconds")
    return [idx_to_key[index] for index in best_indexs]


if __name__ == '__main__':
    print(mixamo_h3d_text_retrieve_multi_output("run"))

