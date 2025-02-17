import OpenEXR
import torch
import clip
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

def read_exr(path):
    file = OpenEXR.InputFile(path)

    # get image size
    dw = file.header()['dataWindow']
    width, height = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1

    # extract RGBA
    red = extract_channel(file, 'ViewLayer.Combined.R').reshape((height, width)).astype(np.float32)
    green = extract_channel(file, 'ViewLayer.Combined.G').reshape((height, width)).astype(np.float32)
    blue = extract_channel(file, 'ViewLayer.Combined.B').reshape((height, width)).astype(np.float32)

    channels = [red, green, blue]
    if 'ViewLayer.Combined.A' in file.header()['channels']:
        alpha = extract_channel(file, 'ViewLayer.Combined.A').reshape((height, width)).astype(np.float32)
        channels.append(alpha)
    raw_rgba = cv2.merge(channels)

    return raw_rgba

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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default='/mnt/aigc_cfs_gz/layer_avatar_data/readplayerMe/render/render_data')
    # parser.add_argument("--input", default='/apdcephfs_cq3/share_2909871/kownseduan/code/text_to_shape_zero123/objaverse_noun_cnt_less_4_clip_02.json')
    parser.add_argument("--input", default='/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/paths_meishuzhongxin.json')
    parser.add_argument("--outdir", default='/mnt/aigc_cfs_cq/xiaqiangdai/project/retrieve/image_clip_latent_meishuzhongxin')
    args, extra = parser.parse_known_args()
    
    keys = json.load(open(args.input))
    print(f'found {len(keys)} keys')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-L/14')
    model.to(device)

    random.shuffle(keys)
    with tqdm(total=len(keys)) as pbar:
        cnt = 0
        for sub in keys:
            pbar.update(1)
            
            outdir = os.path.join(args.outdir, sub.split('/')[-2]+"_"+sub.split('/')[-1].split('_')[0])
            
            if not os.path.exists(outdir):
                os.makedirs(outdir, exist_ok=True)
            else:
                file_count = sum(os.path.isfile(os.path.join(outdir, f)) for f in os.listdir(outdir))
                if file_count==100 or file_count==50:
                    continue
                if file_count>0:
                    os.system("rm "+outdir+"/*")
            print(outdir)
            # full_sub = os.path.join(args.data_root, sub+"/exr")
            full_sub = sub+"/color"
            directory_path = Path(full_sub)
            bg_files = directory_path.glob('cam-*.png')

            # print(f"bg_files:{bg_files}")
                
            # file_size = len(list(bg_files))
         
            imgs = []
            for filepath in bg_files:
                filepath = os.path.basename(filepath)

                id = filepath.replace('cam-','').replace('.png','')
                id = int(id)
                # if id<144 or id >162:
                #     continue
    
                if id%2!=0:
                    continue


                out_path = os.path.join(outdir, f'{filepath}.clip.embedding')
                if os.path.exists(out_path):
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
                    continue

                img = (np.asarray(cropped_image)).astype('uint8')
                img = Image.fromarray(img)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                image_input = preprocess(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    image_features = model.encode_image(image_input)

                with open(out_path, 'wb') as fo:
                    torch.save(image_features, fo)
            