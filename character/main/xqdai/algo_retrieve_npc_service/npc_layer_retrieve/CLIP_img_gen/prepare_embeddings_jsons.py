import os
import argparse
# import faiss
import torch
import pickle
import ipdb
from tqdm import tqdm
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', default='/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/mcwy_data.json')
    parser.add_argument('--data_path', default=['/mnt/aigc_cfs_cq/xiaqiangdai/project/retrieve/image_clip_latent_muchangwuyu','/mnt/aigc_cfs_cq/xiaqiangdai/project/retrieve/image_clip_latent_meishuzhongxin'])
    # ,'/mnt/aigc_cfs_cq/xiaqiangdai/project/retrieve/image_clip_latent_meishuzhongxin']
    args = parser.parse_args()
    
    EMB_PATH = 'retrieve_libs/emb_muchangwuyu_extend/NPC_origin_img_emb.pt'
    OBJ_PATH = 'retrieve_libs/emb_muchangwuyu_extend/NPC_keys.pkl'
    EMB_2_OBJ_PATH = 'retrieve_libs/emb_muchangwuyu_extend/NPC_emb_idx_2_obj_idx.pkl'
    

    with open(args.json_path, 'r') as f:
        json_data = json.load(f)

    folder_keys_dict = {}
    for key in json_data['data'].keys():
        for key_1 in json_data['data'][key].keys():
            path = json_data['data'][key][key_1]['ImgDir']
            folder_key = path.split('/')[-2]+"_"+path.split('/')[-1].split('_')[0]
            folder_keys_dict[folder_key]=[key,key_1]
   
    # print(folder_keys_dict.keys())
    objs = []
    embs = []
    embs_to_objs = {}
    
    for path in args.data_path:
        dirs = os.listdir(path)
        for sub_dir in tqdm(dirs):
            if sub_dir not in folder_keys_dict.keys():
                continue
            obj_path = os.path.join(path, sub_dir)
            
            objs.append(sub_dir)
            # print(sub_dir)
            
            for emb_path in os.listdir(obj_path):
                if '.embedding' not in emb_path:
                    continue
                
                emb_full_path = os.path.join(obj_path, emb_path)
                emb = torch.load(emb_full_path).to('cpu')
                # print(emb_path)
                
                embs.append(emb)
                embs_to_objs[len(embs) - 1] = len(objs) - 1
                
            # if len(embs) > 100:
            #     break
    print(len(objs))
    print(len(embs))
    # assert(0)
    # ipdb.set_trace()
    all_embs = torch.concat(embs, dim=0)
    torch.save(all_embs, EMB_PATH)
    
    with open(OBJ_PATH, 'wb') as file:
        pickle.dump(objs, file)
        
    with open(EMB_2_OBJ_PATH, 'wb') as file:
        pickle.dump(embs_to_objs, file)
    