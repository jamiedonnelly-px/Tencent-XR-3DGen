# import OpenEXR
import torch
from PIL import Image
import argparse
import cv2
import os
from pathlib import Path
import pandas as pd
import numpy as np
from filelock import FileLock
import random
import json
import Imath
from tqdm import tqdm
import multiprocessing
import torch.multiprocessing as mp
import sys
sys.path.append("/mnt/aigc_cfs_cq/xiaqiangdai/project/FlagEmbedding/research/visual_bge")
from visual_bge.modeling import Visualized_BGE

def rfind_idx(str, cnt, token='_'):
    """
    Reverse find the index of the `cnt`th token.
    Should be faster than str.split or regex, I guess...
    """
    for i in range(len(str) - 1, -1, -1):
        if str[i] == token:
            cnt -= 1

            if cnt == 0:
                return i
    
    return -1

def find_degree_chunk(filename):
    basename = os.path.splitext(filename)[0]
    sidx = rfind_idx(basename, 1, '_')
    return basename[sidx + 1:]

def get_azimuth_polor(chunk):
    return [int(c) for c in chunk.split('#')]

def extract_channel(exr_file, channel_name):
    """
    Read specific channel data and return numpy array
    """

    channel_header = exr_file.header()['channels'][channel_name]
    channel_data = exr_file.channel(channel_name, channel_header.type)
    format = Imath.PixelType(channel_header.type.v)

    if format.v == Imath.PixelType.FLOAT:
        channel_np = np.frombuffer(channel_data, dtype=np.float32)
    elif format.v == Imath.PixelType.HALF:
        channel_np = np.frombuffer(channel_data, dtype=np.float16)
    else:  # UINT
        channel_np = np.frombuffer(channel_data, dtype=np.uint32)
    return channel_np

# def read_exr(path):
#     file = OpenEXR.InputFile(path)

#     # get image size
#     dw = file.header()['dataWindow']
#     width, height = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1

#     # extract RGBA
#     red = extract_channel(file, 'ViewLayer.Combined.R').reshape((height, width)).astype(np.float32)
#     green = extract_channel(file, 'ViewLayer.Combined.G').reshape((height, width)).astype(np.float32)
#     blue = extract_channel(file, 'ViewLayer.Combined.B').reshape((height, width)).astype(np.float32)

#     channels = [red, green, blue]
#     if 'ViewLayer.Combined.A' in file.header()['channels']:
#         alpha = extract_channel(file, 'ViewLayer.Combined.A').reshape((height, width)).astype(np.float32)
#         channels.append(alpha)
#     raw_rgba = cv2.merge(channels)

#     return raw_rgba

def load_seen(path, lock_path):
    seen = set()
    if os.path.exists(path):
        lock = FileLock(lock_path)
        with lock:
            df = pd.read_csv(path)
            seen = set(df['key'])
    else:
        with open(path, 'w') as fo:
            fo.write('key,min,mean,max\n')
    
    return seen

def load_skip(args):
    skip = set()
    if args.skip_list is not None:
        with open(args.skip_list) as fi:
            for line in fi:
                skip.add(line.strip())
            
        print(f'loaded {len(skip)} skip keys')

    return skip

def find_bounding_box(image_path, alpha_threshold=100):
    image = Image.open(image_path)
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    left, top, right, bottom = None, None, None, None

    for y in range(image.height):
        for x in range(image.width):
            r, g, b, a = image.getpixel((x, y))
            if a > alpha_threshold:
                if left is None or x < left:
                    left = x
                if top is None or y < top:
                    top = y
                if right is None or x > right:
                    right = x
                if bottom is None or y > bottom:
                    bottom = y

    if left is None:
        return None

    delta = 10

    left = np.max([0,left-delta])
    top = np.max([0,top-delta])
    right = np.min([image.width-1,right+delta])
    bottom = np.min([image.height-1,bottom+delta])
   
    return left, top, right, bottom

def save_cropped_image(image_path, bounding_box,output_path):
    image = Image.open(image_path)
    cropped_image = image.crop(bounding_box)
    cropped_image.save(output_path)
    
def calc_embedding(model,args,keys):

    # if not torch.cuda.is_available():
    #     raise RuntimeError('CUDA is not available')
    # if not torch.cuda._initialized:
    #     torch.cuda.set_device(0)

    cnt = 0
    #/mnt/aigc_cfs_cq/xiaqiangdai/project/retrieve_lantent/clip_lantent/CN_DR_283_F_A
    for key,sub in keys:
        outdir = os.path.join(args.outdir, key)
        
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        else:
            file_count = sum(os.path.isfile(os.path.join(outdir, f)) for f in os.listdir(outdir))
            if file_count<6:
                print(key,outdir)
                # assert(0)
            
        full_sub = sub+"/color"
        bg_files = [os.path.join(full_sub,file) for file in os.listdir(full_sub) if file.endswith('.png')]
            
        file_size = len(bg_files)
      
        print(f"{cnt} {key} {full_sub} bg_files:{file_size}")
        cnt=cnt+1
        
        imgs = []
        for filepath in bg_files:
            # print(key,filepath)
            filepath = os.path.basename(filepath)

            id = filepath.replace('cam-','').replace('.png','')
            id = int(id)
            # if id!=6 and id!=46 and id!=22 and id!=24 and id!=26 and id!=28:
            #     continue
            if id%2!=0:
                continue

            out_path = os.path.join(outdir, f'{filepath}.bge.embedding')
            if os.path.exists(out_path):
                size_emb = os.path.getsize(out_path)
                if size_emb>0:
                    continue
            
            # chunk = find_degree_chunk(filepath)
            # azimuth, polar = get_azimuth_polor(chunk)

            img_path = os.path.join(full_sub, filepath)
            try:
                # img = cv2.imread(img_path)
                bounding_box = find_bounding_box(img_path)
                image = Image.open(img_path)
                cropped_image = image.crop(bounding_box)

            except:
                print("cropped_image error")
                continue

            img = (np.asarray(cropped_image)).astype('uint8')
            img = Image.fromarray(img)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            with torch.no_grad():
                input_image = model.preprocess_val(img).unsqueeze(0)
                # print("=======",model.device)
        
                image_features = model.encode_image(input_image.to(model.device))

            with open(out_path, 'wb') as fo:
                torch.save(image_features, fo)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", default='/mnt/aigc_cfs_cq/xiaqiangdai/project/npc_layer_retrieve/20241010_daz_decimate_add_ct.json')
    parser.add_argument("--outdir", default='/mnt/aigc_cfs_cq/xiaqiangdai/project/retrieve_lantent/bge_lantent')
    args, extra = parser.parse_known_args()
    
    try:
        with open(args.json_path, 'r') as f:
            json_data = json.load(f)

        keys = []
        for key in json_data['data'].keys():
            for key_1 in json_data['data'][key].keys():
                    ImgDir_path = json_data['data'][key][key_1]['ImgDir']
                    Obj_Mesh_path = json_data['data'][key][key_1]['Obj_Mesh']
                    keys.append([key_1,ImgDir_path])
    except:
        assert(0)
    print(f'found {len(keys)} keys')
    # print(keys)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Visualized_BGE(model_name_bge = "BAAI/bge-m3", model_weight="/aigc_cfs_2/xiaqiangdai/models/Visualized_m3.pth")
   
    model.to(device)
    model.eval()

    random.shuffle(keys)
    thread_num = 1
    num_all = len(keys)
    config=[]

    ctx = torch.multiprocessing.get_context("spawn")
    print(torch.multiprocessing.cpu_count())
    pool = ctx.Pool(8) 
    pool_list = []
    if thread_num==1:
        calc_embedding(model,args,keys)
    else:
        for i in range(thread_num):
            if i!=thread_num-1:
                input_arg = (model,args,keys[num_all//thread_num*i:num_all//thread_num*(i+1)])
            else:
                input_arg = (model,args,keys[num_all//thread_num*i:num_all])
            
            res = pool.apply_async(calc_embedding, args = input_arg)
            pool_list.append(res)
        
    pool.close()
    pool.join()
    for i in pool_list:
        data = i.get()
            