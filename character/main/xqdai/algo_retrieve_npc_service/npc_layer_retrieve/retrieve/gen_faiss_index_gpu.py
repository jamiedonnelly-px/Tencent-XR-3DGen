import os
import faiss
import torch
import numpy as np

import ipdb

EMB_PATH = 'retrieve_libs/emb_all_filter20240407/others/NPC_origin_img_emb.pt'
FAISS_GPU_IDX_PATH = "retrieve_libs/emb_all_filter20240407/others/NPC_faiss_gpu_index_normalised.npy"
part_keys = ['hair','top','bottom','shoe','outfit','others']

if __name__ == "__main__":
    image_embeds = torch.load(EMB_PATH)

    norms = torch.norm(image_embeds, p=2, dim=1, keepdim=True)
    normalised_img_embeds = image_embeds / norms
    normalised_img_embeds.shape

    config = faiss.GpuIndexFlatConfig()
    config.device = 0
    faiss_gpu_index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), normalised_img_embeds.shape[1], config)
    # ipdb.set_trace()
    faiss_gpu_index.add(normalised_img_embeds.cpu().numpy().astype(np.float32))
    
    print(f"faiss_gpu_index.ntotal = {faiss_gpu_index.ntotal}")
    faiss_cpu_index = faiss.index_gpu_to_cpu(faiss_gpu_index)
    
    # os.makedirs("emb", exist_ok=True)
    chunk = faiss.serialize_index(faiss_cpu_index)
    np.save(FAISS_GPU_IDX_PATH, chunk)
    print(f"{FAISS_GPU_IDX_PATH} saved")
    del faiss_gpu_index, faiss_cpu_index
