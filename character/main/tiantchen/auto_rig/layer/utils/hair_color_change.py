#coding=utf-8

import json

# pip install venus-api-base from tencent
from venus_api_base.http_client import HttpClient
from venus_api_base.config import Config
import requests

# use hunyuan api to extract required entities from a provided description
def color_detection(sentence):
    url = 'http://url'
    headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer token",
    }
    prompts = "我会给你提供一个描述，根据该描述，提供给我颜色hex编号。不要提供任何其他内容。"
    model = 'hunyuan'

    history = [{"role": "system", "content": prompts}, \
            {"role": "user", "content": "兰花的紫色"}, \
            {"role": "assistant","content": "#9B59B6"}, \
            {"role": "user", "content": "向日葵"}, \
            {"role": "assistant","content": "#FFC300"}, \
            {"role": "user", "content": "太阳，那永恒的光辉之源，悬挂在蔚蓝的天空中，洒下璀璨的金光，滋养着大地万物，驱散阴霾，给予生命无尽的希望与活力。"}, \
            {"role": "assistant","content": "#FFD700"}, \
            {"role": "user", "content": "大海"}, \
            {"role": "assistant","content": "#0000FF"}, \
            {"role": "user", "content": sentence}]
    
    json_data = {"model":model,"messages": history, "temperature": 0.7}

    input_data = json.dumps(json_data)

    response = requests.post(url, data=input_data.encode(), headers=headers)

    response = json.loads(response.text)
    return response['choices'][0]['message']['content']

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_norm(rgb_color):
    return tuple(color/255.0 for color in rgb_color)

def change_hair(prompt):
    return rgb_to_norm(hex_to_rgb(color_detection(prompt)))

if __name__ == '__main__':
    print(change_hair("蓝天白云"))
