import os
import faiss
import torch
import numpy as np

import ipdb

EMB_PATH = '/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/retrieve_libs/emb_20240711/hair/NPC_origin_img_emb.pt'
FAISS_GPU_IDX_PATH = "/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/retrieve_libs/emb_20240711/hair/NPC_faiss_gpu_index_normalised.npy"
genders=["male","female"]
part_keys = ['hair','top','trousers','shoe','outfit','others']

if __name__ == "__main__":
    for gender in genders:
        for key in part_keys:
            key_temp = gender+'_'+key
            EMB_PATH_out = EMB_PATH.replace("hair",key_temp)
            FAISS_GPU_IDX_PATH_out = FAISS_GPU_IDX_PATH.replace("hair",key_temp)
            image_embeds = torch.load(EMB_PATH_out)

            norms = torch.norm(image_embeds, p=2, dim=1, keepdim=True)
            normalised_img_embeds = image_embeds / norms
            normalised_img_embeds.shape

            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            res = faiss.StandardGpuResources()
            # res.setAsyncCopyMode(False)
            faiss_gpu_index = faiss.GpuIndexFlatIP(res, normalised_img_embeds.shape[1], config)
            print(normalised_img_embeds.shape[1])
            # faiss_gpu_index = faiss.IndexFlatL2(normalised_img_embeds.shape[1])
            # ipdb.set_trace()
            faiss_gpu_index.add(normalised_img_embeds.cpu().numpy().astype(np.float32))
            
            print(f"faiss_gpu_index.ntotal = {faiss_gpu_index.ntotal}")
            faiss_cpu_index = faiss.index_gpu_to_cpu(faiss_gpu_index)
            
            # os.makedirs("emb", exist_ok=True)
            chunk = faiss.serialize_index(faiss_cpu_index)
            np.save(FAISS_GPU_IDX_PATH_out, chunk)
            print(f"{FAISS_GPU_IDX_PATH_out} saved")
            torch.cuda.empty_cache()
            del faiss_gpu_index, faiss_cpu_index
