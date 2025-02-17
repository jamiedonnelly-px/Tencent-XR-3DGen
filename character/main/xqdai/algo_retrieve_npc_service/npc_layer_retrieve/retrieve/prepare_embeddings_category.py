import os
import argparse
# import faiss
import torch
import pickle
import ipdb
from tqdm import tqdm
import json
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', default='/mnt/aigc_cfs_cq/xiaqiangdai/project/npc_layer_retrieve/20241010_daz_decimate_add_ct.json')
    parser.add_argument('--data_path', default=['/mnt/aigc_cfs_cq/xiaqiangdai/project/retrieve_lantent/bge_lantent'])
    # ,'/mnt/aigc_cfs_cq/xiaqiangdai/project/retrieve/image_clip_latent_meishuzhongxin']
    args = parser.parse_args()

    with open(args.json_path, 'r') as f:
        json_data = json.load(f)

    folder_keys_dict = {'male_hair':[],'male_top':[],'male_trousers':[],'male_shoe':[],'male_outfit':[],'male_others':[],
                    'female_hair':[],'female_top':[],'female_trousers':[],'female_shoe':[],'female_outfit':[],'female_others':[]}
    print(json_data['data'].keys())
    for key in json_data['data'].keys():
        for key_1 in json_data['data'][key].keys():
                path = json_data['data'][key][key_1]['ImgDir']
                category = json_data['data'][key][key_1]['Category']
                folder_key = key_1
                # print(key,key_1,key_2)
                gender = json_data['data'][key][key_1]['Gender']
                if gender=="Asexual":
                    folder_keys_dict['male_'+category.lower()].append([folder_key,path])
                    folder_keys_dict['female_'+category.lower()].append([folder_key,path])
                else:
                    folder_keys_dict[gender.lower()+'_'+category.lower()].append([folder_key,path])
   
    dirs_all = []
    dirs_full_path_all = []
    for path in args.data_path:
        dirs = os.listdir(path)
        for sub_dir in tqdm(dirs):
            dirs_full_path_all.append(os.path.join(path,sub_dir))
            dirs_all.append(sub_dir)

    part_keys = folder_keys_dict.keys()
    for part_key in part_keys:
        print(f"{part_key}:{len(folder_keys_dict[part_key])}")
    for part_key in part_keys:
        EMB_PATH = f'{parent_dir}/retrieve_libs/emb_20241010_bge/{part_key}/NPC_origin_img_emb.pt'
        OBJ_PATH = f'{parent_dir}/retrieve_libs/emb_20241010_bge/{part_key}/NPC_keys.pkl'
        EMB_2_OBJ_PATH = f'{parent_dir}/retrieve_libs/emb_20241010_bge/{part_key}/NPC_emb_idx_2_obj_idx.pkl'
        
        if not os.path.exists(os.path.dirname(EMB_PATH)):
            os.makedirs(os.path.dirname(EMB_PATH),exist_ok=True)
        if not os.path.exists(os.path.dirname(OBJ_PATH)):
            os.makedirs(os.path.dirname(OBJ_PATH),exist_ok=True)
        if not os.path.exists(os.path.dirname(EMB_2_OBJ_PATH)):
            os.makedirs(os.path.dirname(EMB_2_OBJ_PATH),exist_ok=True)

        objs = []
        embs = []
        embs_to_objs = {}
        
        for folder_key,path in folder_keys_dict[part_key]:
            print(folder_key)
            if folder_key not in dirs_all:
                assert(0)
            
            obj_path = dirs_full_path_all[dirs_all.index(folder_key)]
            objs.append(folder_key)
            for emb_path in os.listdir(obj_path):
                if '.embedding' not in emb_path:
                    continue

                id = emb_path.replace('cam-','').replace('.png.bge.embedding','')
                id = int(id)
                if id%2!=0:
                    continue
                # if id!=6 and id!=46 and id!=22 and id!=24 and id!=26 and id!=28:
                #     continue
                
                print(obj_path,emb_path)
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
        