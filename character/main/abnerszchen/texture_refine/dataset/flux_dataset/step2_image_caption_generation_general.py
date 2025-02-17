import json
import argparse
import numpy as np
import requests
import json
import base64
import torch 
import torchvision 
from torchvision.io import read_image 
from torchvision.utils import make_grid 
import os
import time
from io import BytesIO
import random
import traceback
from openai import OpenAI
# pip install venus-api-base from tencent
from venus_api_base.http_client import HttpClient
from venus_api_base.config import Config
import concurrent.futures
import PIL
import sys


current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, save_json, save_lines



def load_rgba_as_rgb(img_path, res=None):
    """load with RGBA and convert to RGB with white backgroud, if is RGB just return

    Args:
        img_path: _description_

    Returns:
        tensor [3, h, w] ---PIL.Image [h, w, 3]
    """
    img = PIL.Image.open(img_path)
    if img.mode == "RGBA":
        background = PIL.Image.new("RGBA", img.size, (127, 127, 127, 255))
        img = PIL.Image.alpha_composite(background, img).convert("RGB")
    if res is not None and isinstance(res, int):
        img = img.resize((res, res))
    img = torch.tensor(np.array(img).transpose(2, 0, 1))
    return img

def save_lines(data_list, out_file):
    with open(out_file, 'w') as f:
        lines = [f"{item.strip()}\n" for item in data_list]
        f.writelines(lines)
# use hunyuan api to translate chinese to english
def translation(sentence):
    url = 'http://url'
    headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer token",
    }
    
    model = 'hunyuan'
    
    history = [{"role": "user", "content": "Translate to english:[{}]".format(sentence)}]
    # print(history)
    
    json_data = {"model":model,"messages": history, }
    input_data = json.dumps(json_data)

    response = requests.post(url, data=input_data.encode(), headers=headers)

    response = json.loads(response.text)
    # print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']

# hunyuan
def caption_global(image_str, use_grid=True):

    url = 'http://url'
    headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer token",
    }
    if use_grid:
        q_text = "你被提供了一个包含同一个3D物体6个正交视角的网格图像。\
                        这个物体是服装或鞋子。\
                        在20个词内用一句话描述这个物体。\
                        不返回类似These orthogonal perspective images show 或 These images showcase之类的前缀，只描述对象。\
                        只描述关于这个物体的整体信息以及部件之间的关系。\
                        使用简洁的语言直接描述物体，不要添加其他内容。不要分行" 
    else:
        q_text = "你被提供了一个服装或鞋子的正面渲染图，灰色背景。\
                        在20个词内用一句话描述这个物体。\
                        不返回太多前缀，直接描述对象。\
                        不要返回背景相关的描述。\
                        只描述关于这个物体的整体信息。\
                        使用简洁的语言直接描述物体，不要添加其他内容。不要分行" 
    # with open(image_path, "rb") as image_file:
    #     encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    json_data = {
        "model": "hunyuan-vision", # 模型名称
        "messages": [
            # {"role": "system", "content": system_prompts},
            {
                "role": "user", # 角色,user或assistant
                "content": [ # 消息内容
                    {
                        "type": "text",
                        "text": q_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,{}".format(image_str)
                        }
                    }
                ]
            }
        ]
    }

    input_data = json.dumps(json_data)

    response = requests.post(url, data=input_data.encode(), headers=headers)

    response = json.loads(response.text)

    # breakpoint()
    return response['choices'][0]['message']['content']


# hunyuan
def caption_local(image_str, use_grid=True):

    url = 'http://url'
    headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer token",
    }
    if use_grid:
        q_text = "你被提供了一个包含同一个3D物体6个正交视角的网格图像。\
                        这个物体是服装或鞋子。\
                        在20个词内用一句话描述这个物体。\
                        只描述关于这个物体的外表细节特征， 例如形状，颜色，纹理等。\
                        不返回类似These orthogonal perspective images show 或 These images showcase之类的前缀，只描述对象。\
                        使用简洁的一句话描述物体，不要添加其他内容。不要分行。" 
    else:
        q_text = "你被提供了一个服装或鞋子的正面渲染图。\
                        在20个词内用一句话描述这个物体。\
                        不返回太多前缀，直接描述对象。\
                        不要返回背景相关的描述。\
                        只描述关于这个物体的外表细节特征， 例如形状，颜色，纹理等。\
                        使用简洁的一句话描述物体，不要添加其他内容。不要分行。" 
    # with open(image_path, "rb") as image_file:
    #     encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    json_data = {
        "model": "hunyuan-vision", # 模型名称
        "messages": [
            # {"role": "system", "content": system_prompts},
            {
                "role": "user", # 角色,user或assistant
                "content": [ # 消息内容
                    {
                        "type": "text",
                        "text": q_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,{}".format(image_str)
                        }
                    }
                ]
            }
        ]
    }

    input_data = json.dumps(json_data)

    response = requests.post(url, data=input_data.encode(), headers=headers)

    response = json.loads(response.text)

    # breakpoint()
    return response['choices'][0]['message']['content']


# Create a client using OpenAI API format.
qwen_client = OpenAI(base_url="http://url:8081/v1", api_key="EMPTY")


def caption_global_qwen(image_str):


    start_time = time.time()
    # Send request to API.
    completion = qwen_client.chat.completions.create(
    model="Qwen-VL-Chat",
    messages=[
        {
            "role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image,"+image_str}},
            {"type": "text", "text": "Given a 3x2 grid image of rthogonally images, describe its overall global of the object and the 3D relationship between components, and its local features using short phrases."},
            ]
        }
    ]
    )
    end_time = time.time()
    print("Time elapsed: ", end_time-start_time)

    # breakpoint()
    return completion.choices[0].message.content


def combine_caption(caption_global, caption_local, use_grid=True):
    secret_id = None
    secret_key = None

    client = HttpClient(config=Config(read_timeout=300), secret_id=secret_id, secret_key=secret_key)

    header = {
        'Content-Type': 'application/json',
    }
    if use_grid:
        q_text = "You are given two discriptions of a garment or shoe. \
                        One describes overall global and the other describes local features. \
                        You must combine them into one english sentence describes this object. \
                        Use less than 30 words.\
                        The object is clothing or shoes.\
                        Do not return anything unrelated. \
                        Do not return prefixes like 'These orthogonal perspective images show' or 'These images showcase', just describe the objects. \
                        Do not return anything related to the grid image, just describe the object. \
                        Return english sentence."
    else:
        q_text = "You are given two discriptions of a garment or shoe. \
                        One describes overall global and the other describes local features. \
                        You must combine them into one english sentence describes this object. \
                        Use less than 30 words.\
                        Do not return anything unrelated. \
                        Do not return prefixes like 'These orthogonal perspective images show' or 'These images showcase', just describe the objects. \
                        Do not return anything related to the grid image, just describe the object. \
                        Return english sentence."
    history = [
        {
            "role": "system",
            "content": q_text
        },
        {"role": "user","content":  "sentenc 1: {}, sentence 2: {}".format(caption_global, caption_local)}
    ]

    body = {
    "appGroupId": 3460,
    # 指定模型
    "model": "gpt-4o-mini",
    # 自定义上下文
    "messages": history,
    "temperature": 0.5,
    "max_tokens": 1000,
    }

    ret = client.post('http://url/', header=header, body=json.dumps(body))
    # print(ret['data']['response'])
    return ret['data']['response']




def process_obj_once(class_name, key_name, view_data):
    class_dict = view_data['data'][class_name]
    value_dict = class_dict[key_name]
    use_grid = False
    if "combined" in value_dict:    # avoid repeat
        return True, key_name
    
    img_path = value_dict['ImgDir']

    if use_grid:
        img_files = ['cam-0000.png', 'cam-0001.png', 'cam-0002.png', 'cam-0003.png', 'cam-0004.png', 'cam-0005.png']
        if 'bak_ImgDir' not in value_dict:
            img_files = ['cam-0022.png', 'cam-0024.png', 'cam-0026.png', 'cam-0028.png', 'cam-0005.png', 'cam-0046.png']

        imgs = [load_rgba_as_rgb(os.path.join(img_path, 'color', img_file)) for img_file in img_files]
        Grid = make_grid(imgs, nrow=3)
    else:
        if 'bak_ImgDir' in value_dict:
            Grid = load_rgba_as_rgb(os.path.join(img_path, 'color', 'cam-0000.png'))
        else:
            Grid = load_rgba_as_rgb(os.path.join(img_path, 'color', 'cam-0022.png'))

    img = torchvision.transforms.ToPILImage()(Grid)
    buffered = BytesIO()
    img.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    start_time = time.time()
    suc_flag = False
    try:
        global_caption = translation(caption_global(img_str, use_grid=use_grid))
        local_caption = translation(caption_local(img_str, use_grid=use_grid))
        final_caption = combine_caption(global_caption, local_caption, use_grid=use_grid)
        print('-----final_caption ', final_caption)

        end_time = time.time()
        print("Time elapsed: ", end_time - start_time)

        view_data["data"][class_name][key_name]['global_hunyuan'] = global_caption
        view_data["data"][class_name][key_name]['local_hunyuan'] = local_caption
        view_data["data"][class_name][key_name]['combined'] = final_caption
        suc_flag = True
    except Exception as e:
        print(f"Failed to process {key_name}: {e}")
        traceback.print_exc()

    return suc_flag, key_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, extras = parser.parse_known_args()

    out_json = '/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/train_flux/add_caption.json'
    
    json_path = '/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/train_flux/source.json'
    name = json_path.split('/')[-1]
    
    view_data, key_pair_list = parse_objs_json(json_path)
    
    amount = len(key_pair_list)
            
    count = 0
    failed_onames = []
    max_workers = 3
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for d_, class_name, key_name in key_pair_list:
            count += 1
            futures.append(executor.submit(process_obj_once, class_name, key_name, view_data))
            # if count == 20:
            #     break
        
        print('futures ', len(futures))
        # breakpoint()
        for future in concurrent.futures.as_completed(futures):
            suc_flag, key_name = future.result()
            count += int(suc_flag)
            if not suc_flag:
                failed_onames.append(key_name)    

    with open(out_json, 'w') as fp:
        json.dump(view_data, fp, indent=2)
    
    save_lines(failed_onames, os.path.join(os.path.dirname(out_json), "failed_caption.txt"))
    print(f'run done, save to {out_json}')