import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, parent_dir)
sys.path.append(os.path.dirname(current_file_path))

import argparse

import numpy as np
from PIL import Image
import time
import json
import ujson
import torch
import alpha_clip
import clip
import threading
import random
import faiss
import pickle
from torchvision import transforms
from collections import Counter
from calc_clip_img_embedding_multiThread import find_bounding_box,merge_boxes
from calc_clip_img_embedding_multiThread_dinov2 import load_image
import rpyc
import pdb 
import cv2
import uuid
from call_hunyuan_vision import  caption
from call_gpt_vision import gpt_caption
from extract_entity import extract_entity_all
from sentence_transformers import SentenceTransformer
from extract_entity import contains_chinese
import sys
sys.path.append('/mnt/aigc_cfs_cq/xiaqiangdai/project/character_customization')
from character_design import translation_gpt
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
sys.path.append("/mnt/aigc_cfs_cq/xiaqiangdai/project/FlagEmbedding/research/visual_bge")
from visual_bge.modeling import Visualized_BGE


def shuffle_list(lst):
    n = len(lst)
    for i in range(n-1, 0, -1):
        j = random.randint(0, i)
        lst[i], lst[j] = lst[j], lst[i]
    return lst

def create_uuid_folder(base_path="."):
    uuid_folder_name = str(uuid.uuid4())
    folder_path = os.path.join(base_path, uuid_folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def save_numpy_image(image, save_path, image_name="image.png"):
    image_file_path = os.path.join(save_path, image_name)
    print(image_file_path)
    image.save(image_file_path)
    return image_file_path

model_type_list = ['clip','clip_text','alph_clip','alph_clip_text','dinov2','dinov2_text','dinov2_clipText','normic','bge']
class single_retrieve():
    def __init__(self,model_type="dinov2_text",data_type="20240711"):
        if model_type not in model_type_list:
            assert(0)

        print(f"model_type:{model_type}")
        self.model_type = model_type #clip,clip_text,alph_clip,alph_clip_text,dinov2,dinov2_text
        self.data_type = data_type #muchangwuyu,readmeplay,muchangwuyu_extend,all
        self.num_gpus = torch.cuda.device_count()

        if self.model_type == 'clip' or self.model_type == 'clip_text':
            FAISS_GPU_IDX_PATH_ori = "/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/retrieve_libs/emb/NPC_faiss_gpu_index_normalised.npy"
            KEYS_PATH_ori = '/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/retrieve_libs/emb/NPC_keys.pkl'
            EMB_2_OBJ_ori = '/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/retrieve_libs/emb/NPC_emb_idx_2_obj_idx.pkl'
        elif self.model_type == 'normic' or self.model_type == 'bge':
            FAISS_GPU_IDX_PATH_ori = "/mnt/aigc_cfs_cq/xiaqiangdai/project/npc_layer_retrieve/retrieve_libs/emb/NPC_faiss_gpu_index_normalised.npy"
            KEYS_PATH_ori = '/mnt/aigc_cfs_cq/xiaqiangdai/project/npc_layer_retrieve/retrieve_libs/emb/NPC_keys.pkl'
            EMB_2_OBJ_ori = '/mnt/aigc_cfs_cq/xiaqiangdai/project/npc_layer_retrieve/retrieve_libs/emb/NPC_emb_idx_2_obj_idx.pkl'
        else:
            FAISS_GPU_IDX_PATH_ori = "/mnt/aigc_cfs_cq/xiaqiangdai/project/npc_image_retrieve/libs/emb/NPC_faiss_gpu_index_normalised.npy"
            KEYS_PATH_ori = '/mnt/aigc_cfs_cq/xiaqiangdai/project/npc_image_retrieve/libs/emb/NPC_keys.pkl'
            EMB_2_OBJ_ori = '/mnt/aigc_cfs_cq/xiaqiangdai/project/npc_image_retrieve/libs/emb/NPC_emb_idx_2_obj_idx.pkl'

        if self.model_type == "dinov2":
            self.data_type = "20240711_dinov2"
        if self.model_type == "dinov2_text":
            self.data_type = "20240711_dinov2_text"
        if self.model_type == "dinov2_clipText":
            self.data_type = "20241010_dinov2_clipText"
        if self.model_type == "normic":
            self.data_type = "20241010_normic"
        if self.model_type == "clip_text":
            self.data_type = "20241010"
        if self.model_type == "bge":
            self.data_type = "20241010_bge"
        self.genders=["male","female"]
        self.part_keys = ['hair','top','trousers','shoe','outfit','others']
        self.colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255]]

        self.FAISS_GPU_IDX_PATH = {}
        self.KEYS_PATH={}
        self.EMB_2_OBJ={}
        self.key_list = []
        self.locks={}
        for gender in self.genders:
            for key in self.part_keys:
                key_temp = gender+'_'+key
                self.key_list.append(key_temp)
                self.FAISS_GPU_IDX_PATH[key_temp] = FAISS_GPU_IDX_PATH_ori.replace('/emb','/emb_'+self.data_type+'/'+key_temp)
                self.KEYS_PATH[key_temp] = KEYS_PATH_ori.replace('/emb','/emb_'+self.data_type+'/'+key_temp)
                self.EMB_2_OBJ[key_temp] = EMB_2_OBJ_ori.replace('/emb','/emb_'+self.data_type+'/'+key_temp)
                self.locks[key_temp] = threading.Lock()
        
        self.faiss_gpu_index={}
        self.emb2obj={}
        self.obj_keys={}
        self.gpu_res = {}

        for id,gender in enumerate(self.genders):
            for key in self.part_keys:
                key_temp = gender+'_'+key
                print(key_temp)
                if self.num_gpus>1:
                    self.faiss_gpu_index[key_temp],self.gpu_res[key_temp] = self.load_faiss_index_gpu(path = self.FAISS_GPU_IDX_PATH[key_temp],gpu_id=id)
                else:
                    self.faiss_gpu_index[key_temp],self.gpu_res[key_temp] = self.load_faiss_index_gpu(path = self.FAISS_GPU_IDX_PATH[key_temp],gpu_id=0)
                self.emb2obj[key_temp] = self.load_emb_2_index(path = self.EMB_2_OBJ[key_temp])
                self.obj_keys[key_temp] = self.load_obj_keys(path = self.KEYS_PATH[key_temp])
        
        self.mask_transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Resize((224, 224)), # change to (336,336) when using ViT-L/14@336px
                transforms.Normalize(0.5, 0.26)
            ])
        
        json_path ="/mnt/aigc_cfs_cq/xiaqiangdai/project/algo_retrieve_npc_service/npc_layer_retrieve/20241010_daz_decimate_add_ct.json"
        self.g_json_data = self.keys_dict(json_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model_type=='alph_clip' or self.model_type=='alph_clip_text':
            self.model, self.preprocess = alpha_clip.load("ViT-B/16", alpha_vision_ckpt_pth="/mnt/aigc_cfs_cq/xiaqiangdai/project/image_retrieve/models/clip_b16_grit+mim_fultune_4xe.pth", device=self.device)
        elif model_type=='clip' or model_type=='clip_text':
            self.model, self.preprocess = clip.load('ViT-L/14')
        elif model_type=='dinov2':
            self.model = torch.hub.load("/root/.cache/torch/hub/facebookresearch_dinov2_main", "dinov2_vitl14",source='local')
            self.model_dinov2 = self.model
            self.preprocess = load_image
            self.preprocess_dinov2 = self.preprocess
        elif model_type=='dinov2_text':
            self.model = torch.hub.load("/root/.cache/torch/hub/facebookresearch_dinov2_main", "dinov2_vitl14",source='local')
            self.model_dinov2 = self.model
            self.preprocess = load_image
            self.preprocess_dinov2 = self.preprocess
            self.model_bge = SentenceTransformer('/aigc_cfs_2/xiaqiangdai/motion/motion_text_retrieve/bge-large-en-v1.5.pt',device=self.device)
        elif model_type=='dinov2_clipText':
            self.model = torch.hub.load("/root/.cache/torch/hub/facebookresearch_dinov2_main", "dinov2_vitl14",source='local')
            self.model_dinov2 = self.model
            self.preprocess = load_image
            self.preprocess_dinov2 = self.preprocess
            self.model_clip, self.preprocess_clip = clip.load('ViT-L/14')
        elif model_type=="normic":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5')
            self.model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
            self.model.eval()
        elif model_type=='bge':
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model = Visualized_BGE(model_name_bge = "BAAI/bge-m3", model_weight="/aigc_cfs_2/xiaqiangdai/models/Visualized_m3.pth")
            self.model.eval()
        self.model.to(self.device)


    def release(self):
        for gender in self.genders:
            for key in self.part_keys:
                key_temp = gender+'_'+key
                del self.faiss_gpu_index[key_temp]
                del self.gpu_res[key_temp]

    def load_faiss_index_gpu(self,path,gpu_id=0):
        res = faiss.StandardGpuResources()
        faiss_gpu_index = faiss.deserialize_index(np.load(path))
        faiss_gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), gpu_id, faiss_gpu_index)
        
        return faiss_gpu_index,res

    def load_emb_2_index(self,path):
        with open(path, 'rb') as file:
            emb2obj = pickle.load(file)
        return emb2obj

    def load_obj_keys(self,path):
        with open(path, 'rb') as file:
            obj_keys = pickle.load(file)
        return obj_keys



    def calc_img_text_mask_emb(self,img,binary_mask, text):
        if self.model_type=='alph_clip':
            alpha = self.mask_transform((binary_mask).astype(np.uint8))
            alpha = alpha.half().unsqueeze(dim=0).to(self.device)

            image_input = self.preprocess(img).unsqueeze(0).half().to(self.device)

            print(image_input.shape)
            with torch.no_grad():
                image_features = self.model.visual(image_input, alpha)
            # image_embed_norm = image_embed / image_embed.norm()
            image_embed_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        elif self.model_type=='alph_clip_text':
            alpha = self.mask_transform((binary_mask).astype(np.uint8))
            alpha = alpha.half().unsqueeze(dim=0).to(self.device)

            image_input = self.preprocess(img).unsqueeze(0).half().to(self.device)

            print(image_input.shape)
            with torch.no_grad():
                image_features = self.model.visual(image_input, alpha)
                text_features = self.model.encode_text(text = alpha_clip.tokenize(text).to(self.device))
                features = 0.5*image_features+0.5*text_features
                image_embed_norm = features / features.norm(dim=-1, keepdim=True)
        
        elif self.model_type=='clip':
            image_input = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_embed = self.model.encode_image(image_input)
            image_embed_norm = image_embed / image_embed.norm()
        elif self.model_type=='dinov2':
            image_input = self.preprocess(img).to(self.device)
            with torch.no_grad():
                image_embed = self.model(image_input)
            image_embed_norm = image_embed / image_embed.norm()
        
        return image_embed_norm

    def calc_clip_txt_emb(self,text):

        with torch.no_grad():
            text_features = self.model.encode_text(text = clip.tokenize(text).to(self.device))
        # text_features = text_features / text_features.norm()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu()

    def calc_clip_image_txt_emb(self,image ,text):

        with torch.no_grad():
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            image_embed = self.model.encode_image(image_input)
            if text=='' or text==None:
                features = image_embed
            else:
                text_features = self.model.encode_text(text = clip.tokenize(text).to(self.device))
                features = 0.5*image_embed+0.5*text_features
        # text_features = text_features / text_features.norm()
        features = features / features.norm(dim=-1, keepdim=True)

        return features


    def calc_dinov2_bge_emb(self,image ,text):

        print(self.preprocess,text)
        assert(text!='' and text!=None and image!=None and self.preprocess!=None)
        image_input = self.preprocess(image).to(self.device)
        with torch.no_grad():
            image_embed = self.model_dinov2(image_input)
        image_embed_norm = image_embed / image_embed.norm()

        embeddings = self.model_bge.encode(text, normalize_embeddings=True)
        text_embedding = torch.tensor(embeddings).unsqueeze(0)

        emb = torch.cat((image_embed_norm.cpu(), text_embedding.cpu()), dim=1)

        return emb
    
    def calc_dinov2_clipText_emb(self,image ,text):

        assert(image!=None)
        assert(text!='' and text!=None)
        image_input = self.preprocess(image).to(self.device)
        with torch.no_grad():
            image_embed = self.model_dinov2(image_input)
        # image_embed_norm = image_embed / image_embed.norm()
        image_embed_norm = image_embed*0.8

        if text!=None and text!='':
            text_features = self.model_clip.encode_text(text = clip.tokenize(text[:77]).to(self.device))
            text_embedding = torch.tensor(text_features)

            emb = torch.cat((image_embed_norm.cpu(), text_embedding.cpu()), dim=1)

            return emb
        else:
            return image_embed_norm

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def calc_normic_txt_emb(self,text):
        text_input = "search_query:is it "+text+"?"
        encoded_input = self.tokenizer(text_input, padding=True, truncation=True, return_tensors='pt').to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        text_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        text_embeddings = F.layer_norm(text_embeddings, normalized_shape=(text_embeddings.shape[1],))
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        
        return text_embeddings

    def calc_bge_image_txt_emb(self,image,text,device):
        assert(image!=None and text!=None and text!='')
    
        if image is not None:
            image = self.model.preprocess_val(image).unsqueeze(0)

            if text is not None:
                text = self.model.tokenizer(text, return_tensors="pt", padding=True)
                return self.model.encode_mm(image.to(device), text.to(device))
            else:
                return self.model.encode_image(image.to(device))
        else:
            if text is not None:
                text = self.model.tokenizer(text, return_tensors="pt", padding=True)
                return self.model.encode_text(text.to(device))
            else:
                return None

    def keys_dict1(self,json_path):
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

    def keys_dict(self,json_path):
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        folder_keys_dict = {}
        for key in json_data['data'].keys():
            for key_1 in json_data['data'][key].keys():
                    folder_key = key_1
                    folder_keys_dict[folder_key] = json_data['data'][key][key_1]
        return folder_keys_dict

    def retrive_topk_paths(self,faiss_gpu_index, image_embed_norm, emb2obj, obj_keys, topk,folder_keys_dict,data_type = "muchangwuyu_extend"):
        distances, indices = faiss_gpu_index.search(
            np.array(image_embed_norm).astype(np.float32),
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


    def retrive_single_txt(self,prompt,key_in):
        if ''==prompt or ' '==prompt or None==prompt or 'None'==prompt or key_in not in self.key_list:
            return [None,None,None]
        query_txt_embed = self.calc_clip_txt_emb(prompt)
        self.locks[key_in].acquire()
        found_keys = self.retrive_topk_paths(
            self.faiss_gpu_index[key_in], query_txt_embed,
            self.emb2obj[key_in], self.obj_keys[key_in],
            10,
            self.g_json_data,
            self.data_type)
        self.locks[key_in].release()
        
        paths = []
        keys = []
        
        for key in found_keys.keys():
            keys.append(key)
        key_temp = keys.copy()
        keys = key_temp[:3]
        # keys = shuffle_list(keys)

        if len(keys) < 3:
            for i in range(3-len(keys)):
                keys.append(None)
        for i in range(3):
            if keys[i]!=None:
                print(found_keys[keys[i]]['score'])
        return keys[:3]

    def retrive_single_image(self,image,text_caption,mask,key_in):
        print(f"self.model_type:{self.model_type}")
        if key_in not in self.key_list:
            return [None,None,None]
        if self.model_type =='dinov2_text':
            query_embed = self.calc_dinov2_bge_emb(image ,text_caption)
        elif self.model_type =='alph_clip' or self.model_type =='alph_clip_text' or self.model_type =='dinov2' :
            query_embed = self.calc_img_text_mask_emb(image,mask,text_caption)
        elif  self.model_type=='clip':
            query_embed = self.calc_clip_image_txt_emb(image ,text_caption)
        elif self.model_type=='dinov2_clipText':
            query_embed = self.calc_dinov2_clipText_emb(image ,text_caption)
        elif self.model_type=='normic':
            query_embed = self.calc_normic_txt_emb(text_caption)
        elif self.model_type=='clip_text':
            query_embed = self.calc_clip_txt_emb(text_caption)
        elif self.model_type=='bge':
            query_embed = self.calc_bge_image_txt_emb(image,text_caption,self.device).detach().cpu().numpy()
        torch.cuda.empty_cache()

        print(f"query_embed:{query_embed.shape}")

        self.locks[key_in].acquire()
        found_keys = self.retrive_topk_paths(
            self.faiss_gpu_index[key_in], query_embed,
            self.emb2obj[key_in], self.obj_keys[key_in],
            10,
            self.g_json_data,
            self.data_type)
        self.locks[key_in].release()
        
        paths = []
        keys = []
        
        for key in found_keys.keys():
            keys.append(key)
        key_temp = keys.copy()
        # key_temp = shuffle_list(key_temp)
        keys = key_temp[:3]
        # keys = shuffle_list(keys)

        if len(keys) < 3:
            for i in range(3-len(keys)):
                keys.append(None)
        for i in range(3):
            if keys[i]!=None:
                print(found_keys[keys[i]]['score'])
        return keys[:3]


    def retrieve_parts(self,input_image,part_info,gender = 'female'):
    
        # input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        # image =  Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        image =  Image.fromarray(input_image)
        results_output = {}
        result_temp = {}
        for key in part_info.keys():
            result_temp[key] = []

        print(part_info.keys())
        for key in part_info.keys():
            print(key,part_info[key].keys())
            result_temp[key].append([part_info[key]['mask'],part_info[key]['box']])

        h  = input_image.shape[0]
        w  = input_image.shape[1]

        result_temp_merge = {}
        print(f"result_temp:{result_temp}")
        for key in result_temp.keys():
            if len(result_temp[key])==0:
                continue
            mask_temp = np.zeros((h,w),dtype=bool)
            box_temp =None
            num = 0
            for item in result_temp[key]:
               
                # pdb.set_trace()
                mask_temp = mask_temp | item[0]
                if num==0:
                    box_temp = item[1]
                else:
                    box_temp = merge_boxes(box_temp,item[1])
                num+=1
            dict_temp = {'mask':mask_temp,'box':box_temp}
            result_temp_merge[key]=dict_temp
        # pdb.set_trace()

        folder = create_uuid_folder('/aigc_cfs_gdp/xiaqiangdai/t2i')
        print(folder)
        if not os.path.exists(folder):
            print(folder)
            os.makedirs(folder)
        for i,cloth_name in enumerate(list(result_temp_merge.keys())):
            mask = result_temp_merge[cloth_name]['mask']
            mask[mask==True] = 255.0
            mask[mask==False] = 0
            box = result_temp_merge[cloth_name]['box']
            print(box)
            box = box.astype(np.int32)
            
            x_min, y_min, x_max, y_max = box
            cropped_mask = mask[y_min:y_max, x_min:x_max]
            cropped_image = image.crop(box)
            key_in = gender+'_'+cloth_name
            
            print(i,key_in,cropped_image.size)
            img_path = save_numpy_image(cropped_image, folder, image_name=f"{cloth_name}.png")
            
            text_caption = gpt_caption(img_path,gender,cloth_name)
            if contains_chinese(text_caption):
                caption_str_en = translation_gpt(text_caption)
                text_caption = caption_str_en.replace('\[','').replace('\]','')
            print(f"before text_caption:{text_caption}")
            index = self.part_keys.index(cloth_name)
            list_out, description = extract_entity_all(text_caption)
            if list_out[index]!="":
                text_caption = list_out[index]
                # if index==2:
                #     text_caption = "shorts with a prominent brand label"
            print(f"after text_caption:{text_caption}")

            # cropped_image.save(f'/mnt/aigc_cfs_cq/xiaqiangdai/project/image_retrieve/lib_generate/test_images/{names_map[cloth_name]}.png')
            keys = self.retrive_single_image(cropped_image,text_caption,cropped_mask,key_in)
            
            out = {'key':keys,'img_path':img_path,'caption':text_caption}
            results_output[cloth_name] = out
        
        return results_output



    def retrive_image_all(self,input_image:np.ndarray,gender = 'female'):
        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config["sync_request_timeout"] = None
        connection = rpyc.connect('', 0,config=rpyc_config)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
        input_image_json = ujson.dumps(input_image.tolist())
        result = connection.root.mask_predict(input_image_json)
        result = ujson.loads(result)
        masks = np.array(result['mask'])
        h  = masks.shape[-2]
        w  = masks.shape[-1]
        boxes = np.array(result['box'])
        names_map = {"hair":"hair","top cloth":"top","trousers":"trousers","shoe":"shoe","suit":"outfit"}
        
        image =  Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        results_output = {}
        result_temp = {}
        for key in names_map.keys():
            result_temp[key] = []

        print(result['phrase'])
        for i,phrase in enumerate(result['phrase']):
            cloth_name = phrase.split('(')[0]
            result_temp[cloth_name].append([masks[i,0,:,:],boxes[i],phrase])

        result_temp_merge = {}
        for key in result_temp.keys():
            if len(result_temp[key])==0:
                continue
            mask_temp = np.zeros((h,w),dtype=bool)
            box_temp =None
            num = 0
            for item in result_temp[key]:
                # pdb.set_trace()
                mask_temp = mask_temp | item[0]
                if num==0:
                    box_temp = item[1]
                else:
                    box_temp = merge_boxes(box_temp,item[1])
                num+=1
            dict_temp = {'mask':mask_temp,'box':box_temp}
            result_temp_merge[key]=dict_temp
        # pdb.set_trace()

        image_mask_out1 = np.zeros((h,w,3))
        image_mask_out2 = np.zeros((h,w,3))
        for i,cloth_name in enumerate(list(result_temp_merge.keys())):
            index = self.part_keys.index(names_map[cloth_name]) 

            if index<0 and index>4:
                print(f"{cloth_name} index error")
                assert(0)
            color_value = np.array(self.colors[index])
            mask = result_temp_merge[cloth_name]['mask']
            mask_rgb = np.stack((mask,)*3, axis=-1)
            mask_rgb_uint8 = mask_rgb.astype(np.uint8) * 255
            # true_indices = np.where(mask_rgb == [True, True, True])
            # pdb.set_trace()
            if index!=4:
                image_mask_out1[mask_rgb_uint8[:, :, 0] == 255] = color_value
            else:
                image_mask_out2[mask_rgb_uint8[:, :, 0] == 255] = color_value

        folder = create_uuid_folder('/aigc_cfs_gdp/xiaqiangdai/t2i')
        print(folder)
        if not os.path.exists(folder):
            print(folder)
            os.makedirs(folder)
        for i,cloth_name in enumerate(list(result_temp_merge.keys())):
            mask = result_temp_merge[cloth_name]['mask']
            mask[mask==True] = 255.0
            mask[mask==False] = 0
            box = result_temp_merge[cloth_name]['box']
            print(box)
            box = box.astype(np.int32)
            
            x_min, y_min, x_max, y_max = box
            cropped_mask = mask[y_min:y_max, x_min:x_max]
            cropped_image = image.crop(box)
            key_in = gender+'_'+names_map[cloth_name]
            print(i,cloth_name,cropped_image.shape)
            img_path = save_numpy_image(cropped_image, folder, image_name=f"{names_map[cloth_name]}.png")
            
            text_caption = caption(img_path)
            if contains_chinese(text_caption):
                caption_str_en = translation_gpt(text_caption)
                text_caption = caption_str_en.replace('\[','').replace('\]','')
            print(f"text_caption:{text_caption}")

            # cropped_image.save(f'/mnt/aigc_cfs_cq/xiaqiangdai/project/image_retrieve/lib_generate/test_images/{names_map[cloth_name]}.png')
            keys = self.retrive_single_image(cropped_image,text_caption,cropped_mask,key_in)
            out = {'key':keys,'img_path':img_path,'caption':text_caption}
            results_output[names_map[cloth_name]] = out
        
        results_output['mask1'] = image_mask_out1.tolist()
        results_output['mask2'] = image_mask_out2.tolist()
        return results_output



if __name__ == "__main__":

    img_path = "/mnt/aigc_cfs_cq/xiaqiangdai/project/image_retrieve/lib_generate/test_images/top.png"
    image = Image.open(img_path)
    text = caption(img_path)
    retrieve = single_retrieve()
    emb = retrieve.calc_dinov2_bge_emb(image ,text)
    print(emb.shape)
    

   
