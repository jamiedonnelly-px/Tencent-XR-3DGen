#coding=utf-8

import json

from venus_api_base.http_client import HttpClient
from venus_api_base.config import Config
import requests
import sys
import re
import rpyc
import random
import numpy
import ujson
import os
import numpy
import base64


def determine_animal_img(image_path):
    secret_id = ''
    secret_key = ''

    client = HttpClient(config=Config(read_timeout=300), secret_id=secret_id, secret_key=secret_key)
    
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    
    header = {'Content-Type': 'application/json',}
    history = [
        {
            "role": "system",
            "content": "You are provided an image that contains an object, \
                        You should determine does the category of the object belongs to human character. \
                        You must only return true, false"
        },
        {"role": "user","content": [
            {
            "type": "image_url",
            "image_url": {
                "detail": "low",
                "url": "data:image/png;base64,{}".format(encoded_string)
                        }
                    }
          ]}
    ]

    body = {
    "appGroupId": 0,
    "model": "gpt-4-vision-preview",
    "messages": history,
    "temperature": 0.3,
    "max_tokens": 1000,
    }

    ret = client.post('', header=header, body=json.dumps(body))
    return ret['data']['response']



def determine_animal_txt(sentence):
    secret_id = ''
    secret_key = ''

    client = HttpClient(config=Config(read_timeout=300), secret_id=secret_id, secret_key=secret_key)

    header = {
        'Content-Type': 'application/json',
    }

    history = [
        {
            "role": "system",
            "content": "You are provided a sentence that contains an object, \
                        You should determine does the category of the object belongs to human or human-like character. \
                        You must only return true, false"
        },
        {"role": "user", "content": "a doctor"},
        {"role": "assistant","content":  "true"},
        {"role": "user","content":  "plane"},
        {"role": "assistant","content":  "false"},
        {"role": "user","content":  "{}".format(sentence)}
    ]

    body = {
    "appGroupId": ,
    # 指定模型
    "model": "gpt-4o-mini",
    # 自定义上下文
    "messages": history,
    "temperature": 0.1,
    "max_tokens": 1000,
    }

    ret = client.post('', header=header, body=json.dumps(body))
    # print(ret['data']['response'])
    return ret['data']['response']




# use gpt to determine the head-body tatio 
def head_body_rate_classification_gpt(sentence):
    secret_id = ''
    secret_key = ''

    client = HttpClient(config=Config(read_timeout=300), secret_id=secret_id, secret_key=secret_key)

    header = {
        'Content-Type': 'application/json',
    }

    history = [
        {
            "role": "system",
            "content": "你要根据用户输入得到头身比，并返回一个数字。"
        },
        {"role": "user", "content": "2"},
        {"role": "assistant","content":  "2"},
        {"role": "user","content":  "三头身"},
        {"role": "assistant","content":  "3"},
        {"role": "user","content":  "{}".format(sentence)}
    ]

    body = {
    "appGroupId": 0,
    # 指定模型
    "model": "gpt-4o",
    # 自定义上下文
    "messages": history,
    "temperature": 0.1,
    "max_tokens": 1000,
    }

    ret = client.post('', header=header, body=json.dumps(body))
    # print(ret['data']['response'])
    return ret['data']['response']


def sensitive_content_filter(sentence):
    url = ''
    headers = {
    "Content-Type": "application/json",
    "Authorization": "",
    }
    # prompts = "检查以下文本中是否含有关政治，色情，宗教，暴恐，反动等敏感内容。如果有，返回 True。如果没有，返回 False。"
    
    model = 'hunyuan'
    
    history = [{"role": "user", "content": "检查以下文本中是否含明显有关中国现代政治敏感，色情的内容：[{}]。如果有，返回 True。如果没有，返回 False。".format(sentence)}]
    
    json_data = {"model":model,"messages": history}
    input_data = json.dumps(json_data)
    response = requests.post(url, data=input_data.encode(), headers=headers)
    response = json.loads(response.text)
    return response['choices'][0]['message']['content']



# use gpt to determine the color of hair
def attributes_separation_gpt(sentence):
    secret_id = ''
    secret_key = ''

    client = HttpClient(config=Config(read_timeout=300), secret_id=secret_id, secret_key=secret_key)

    header = {
        'Content-Type': 'application/json',
    }

    history = [
        {
            "role": "system",
            "content": "You will be given a description of hair. You should extract the adjectives from the description and classify them into 2 categories: [color, type]. \
                Return results in a string, split different entities by /. Include category classification in bracket following each adjective"
        },
        {"role": "user", "content": "Long black hair"},
        {"role": "assistant","content":  "long (type)/black (color)"},
        {"role": "user","content":  "Long, midnight-blue mane."},
        {"role": "assistant","content":  "long (type)/midnight-blue (color)"},
        {"role": "user","content":  "Bold, fiery red hair."},
        {"role": "assistant","content":  "bold (type)/fiery red (color)"},
        {"role": "user","content":  "Bright red hair."},
        {"role": "assistant","content":  "bright red (color)"},
        {"role": "user","content":  "{}".format(sentence)}
    ]

    body = {
    "appGroupId": 0,
    # 指定模型
    "model": "gpt-4o",
    # 自定义上下文
    "messages": history,
    "temperature": 0.1,
    "max_tokens": 1000,
    }

    ret = client.post('', header=header, body=json.dumps(body))
    # print(ret['data']['response'])
    return ret['data']['response']


# breakpoint()


# use gpt to determine score of users' rating
def score_classification(sentence):
    secret_id = ''
    secret_key = ''

    client = HttpClient(config=Config(read_timeout=300), secret_id=secret_id, secret_key=secret_key)

    header = {
        'Content-Type': 'application/json',
    }

    history = [
        {
            "role": "system",
            "content": "用户会输入针对某事物的评价，请根据输入内容返回一个1到10之间到分数。1是最差，10是最好。不要返回其他任何内容。"
        },
        {"role": "user", "content": "十分"},
        {"role": "assistant","content":  "10"},
        {"role": "user","content":  "还行吧"},
        {"role": "assistant","content":  "6"},
        {"role": "user","content":  "very good"},
        {"role": "assistant","content":  "9"},
        {"role": "user","content":  "10"},
        {"role": "assistant","content":  "10"},
        {"role": "user","content":  "perfect"},
        {"role": "assistant","content":  "10"},
        {"role": "user","content":  "马马虎虎"},
        {"role": "assistant","content":  "5"},
        {"role": "user","content":  "{}".format(sentence)}
    ]

    body = {
    "appGroupId": 0,
    # 指定模型
    "model": "gpt-4o",
    # 自定义上下文
    "messages": history,
    "temperature": 0.1,
    "max_tokens": 1000,
    }

    ret = client.post('', header=header, body=json.dumps(body))
    return ret['data']['response']


# breakpoint()


# print(attributes_separation_gpt('white long hair'))

# breakpoint()


# use gpt to determine the gender of character
def gender_classification_gpt(sentence):
    secret_id = ''
    secret_key = ''

    client = HttpClient(config=Config(read_timeout=300), secret_id=secret_id, secret_key=secret_key)

    header = {
        'Content-Type': 'application/json',
    }

    history = [
        {
            "role": "system",
            "content": "You are a bot that determines gender of a chracter.\
                        You will be given a description of a chracter, \
                        you must determine the most possible gender of this character according to the description. \
                        You only have two choices, \
                        return either male or female. \
                        You must avoid return any other content. "
        },
        {"role": "user", "content": "This virtual character is an elderly woman with big blue wavy long hair. She is wearing a pink short-sleeved blouse and a black long skirt, and on her feet are pink shoes."},
        {"role": "assistant","content":  "female"},
        {"role": "user","content":  "A young boy wearing a blue jacket."},
        {"role": "assistant","content":  "male"},
        {"role": "user","content":  "A chracter with black ponytail."},
        {"role": "assistant","content":  "female"},
        {"role": "user","content":  "{}".format(sentence)}
    ]

    body = {
    "appGroupId": 0,
    # 指定模型
    "model": "gpt-4o",
    # 自定义上下文
    "messages": history,
    "temperature": 0.1,
    "max_tokens": 1000,
    }

    ret = client.post('', header=header, body=json.dumps(body))
    # print(ret['data']['response'])
    return ret['data']['response']


# use hunyuan to determine the gender of character
def gender_classification(sentence):
    url = ''
    headers = {
    "Content-Type": "application/json",
    "Authorization": "",
    }
    prompts = "You are a bot that determines gender of a chracter. \
            You will be given a description, and you must extract all entities and their attributes related to gender, hair, tops, trousers, shoes, outfits, body, or other clothing from this description. \
            Provide the described entities in a string, split different entities by /. If no entities are found in the description, provide an empty string. \
            Ignore person's name. Ignore entities that are not related to gender, hair, tops, trousers, shoes, outfits, or other clothing."
    
    model = 'hunyuan'
    # print('prompts:', prompts)
    # print('model:',  model)
    history = [{"role": "system", "content": prompts}, \
            {"role": "user", "content": "White boots with yellow trim and pointed toe. The boots have a sleek and stylish design, perfect for adding a touch of whimsy to any outfit. The upper is made from a durable material like leather or synthetic, while the trim is made from a bright and cheerful yellow hue. The pointy toe adds an edgy touch to the classic boot shape, making them both functional and fashionable. These boots would complement a variety of outfits, from casual to formal, and can be worn during any season."}, \
            {"role": "assistant","content": "white boots with yellow trim and pointed toe"}, \
            {"role": "user", "content": "The image shows a 3D animated character with long white hair, wearing a light blue dress with a white belt and white shoes. The character is standing with one hand on their hip and the other arm extended forward"}, \
            {"role": "assistant","content": "long, white hair/light blue dress/white belt/white shoes"}, \
            {"role": "user", "content": "A 3D rendering of a young male character with short white hair, blue eyes, and a neutral expression. He is wearing a white t-shirt and shorts with a pattern on the legs. The character is barefoot."}, \
            {"role": "assistant","content": "young male/short, white hair/white t-shirt/shorts with a pattern on the legs/barefoot"}, \
            {"role": "user", "content": "a young male with short white hair. He is wearing a dark blue jacket with a light blue collar and sleeves, dark pants, and black shoes. His arms are outstretched to the sides, and he appears to be standing against a black background."}, \
            {"role": "assistant","content": "young male/short, white hair/dark blue jacket with a light blue collar and sleeves/black shoes"}, \
            {"role": "user", "content": "She had long, golden curly hair and was wearing a white lace dress and high heels, with a silver bracelet on her wrist."}, \
            {"role": "assistant","content": "she/long, golden curly hair/white lace dress/high heels/silver bracelet"}, \
            {"role": "user", "content": "The girl wore a pair of black-framed glasses and had short hair. She was dressed in a blue denim jacket and a black skirt, with a pair of white sneakers on her feet."}, \
            {"role": "assistant","content": "girl/black-Framed glasses/short hair/blue denim jacket/black skirt/white sneakers"}, \
            
            {"role": "user", "content": sentence}]
    
    # json_data = {"model":model,"messages": history, 'temperature':0.2}
    json_data = {"model":model,"messages": history}

    input_data = json.dumps(json_data)

    response = requests.post(url, data=input_data.encode(), headers=headers)

    response = json.loads(response.text)
    # breakpoint()
    # print('实体抽取：', response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']



# use hunyuan api to extract required entities from a provided description
def extract_entity(sentence):
    url = ''
    headers = {
    "Content-Type": "application/json",
    "Authorization": "",
    }
    prompts = "You are part of a team of bots that extracts entities and their attributes from a description. You will be given a description, and you must extract all entities and their attributes related to gender, hair, tops, trousers, shoes, outfits, or other clothing from this description. Provide the described entities in a string, split different entities by /. If no entities are found in the description, provide an empty string. Ignore person's name. Ignore entities that are not related to gender, hair, tops, trousers, shoes, outfits, or other clothing."
    model = 'hunyuan'
    # print('prompts:', prompts)
    # print('model:',  model)
    history = [{"role": "system", "content": prompts}, \
            {"role": "user", "content": "White boots with yellow trim and pointed toe. The boots have a sleek and stylish design, perfect for adding a touch of whimsy to any outfit. The upper is made from a durable material like leather or synthetic, while the trim is made from a bright and cheerful yellow hue. The pointy toe adds an edgy touch to the classic boot shape, making them both functional and fashionable. These boots would complement a variety of outfits, from casual to formal, and can be worn during any season."}, \
            {"role": "assistant","content": "white boots with yellow trim and pointed toe"}, \
            {"role": "user", "content": "The image shows a 3D animated character with long white hair, wearing a light blue dress with a white belt and white shoes. The character is standing with one hand on their hip and the other arm extended forward"}, \
            {"role": "assistant","content": "long, white hair/light blue dress/white belt/white shoes"}, \
            {"role": "user", "content": "A 3D rendering of a young male character with short white hair, blue eyes, and a neutral expression. He is wearing a white t-shirt and shorts with a pattern on the legs. The character is barefoot."}, \
            {"role": "assistant","content": "young male/short, white hair/white t-shirt/shorts with a pattern on the legs/barefoot"}, \
            {"role": "user", "content": "a young male with short white hair. He is wearing a dark blue jacket with a light blue collar and sleeves, dark pants, and black shoes. His arms are outstretched to the sides, and he appears to be standing against a black background."}, \
            {"role": "assistant","content": "young male/short, white hair/dark blue jacket with a light blue collar and sleeves/black shoes"}, \
            {"role": "user", "content": "She had long, golden curly hair and was wearing a white lace dress and high heels, with a silver bracelet on her wrist."}, \
            {"role": "assistant","content": "she/long, golden curly hair/white lace dress/high heels/silver bracelet"}, \
            {"role": "user", "content": "The girl wore a pair of black-framed glasses and had short hair. She was dressed in a blue denim jacket and a black skirt, with a pair of white sneakers on her feet."}, \
            {"role": "assistant","content": "girl/black-Framed glasses/short hair/blue denim jacket/black skirt/white sneakers"}, \
            
            {"role": "user", "content": sentence}]
    
    # json_data = {"model":model,"messages": history, 'temperature':0.2}
    json_data = {"model":model,"messages": history}

    input_data = json.dumps(json_data)

    response = requests.post(url, data=input_data.encode(), headers=headers)

    response = json.loads(response.text)
    # breakpoint()
    # print('实体抽取：', response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']



# use gpt api to extract required entities from a provided description (in use)
def extract_entity_gpt(sentence):
    # print(sentence)
    secret_id = ''
    secret_key = ''

    client = HttpClient(config=Config(read_timeout=300), secret_id=secret_id, secret_key=secret_key)

    header = {
        'Content-Type': 'application/json',
    }

    prompts = "You are part of a team of bots that extracts entities and their attributes from a description. \
        You will be given a description, and you must extract all valid entities and their attributes related to hair, tops, trousers, shoes, outfits, clothing, body, and action from this description. \
        Provide the described entities in an english string, split different entities by /. \
        You must classify each entity into one of the following categories: [hair, top, bottom, shoe, outfit, action, body]. \
        If no entities are found in the description, provide an empty string. \
        Ignore person's name. Ignore entities that are not related to hair, tops, trousers, shoes, outfits, clothing, action and body.\
        Include category classification in bracket following each entity. (dress belongs to outfit) "
        
    # If the entity of [hair, shoe, action] is not valid, like no hair, barefoot, no shoes, or no action, return None. \

    history = [{"role": "system", "content": prompts}, \
            {"role": "user", "content": "The image shows a 3D animated character with long white hair, wearing a light blue dress with a white belt and white shoes. The character is standing with one hand on their hip and the other arm extended forward. She has a slender, athletic build with toned muscles"}, \
            {"role": "assistant","content": "long, white hair (hair)/light blue dress (outfit)/white shoes (shoe)/standing (action)/slender, athletic build with toned muscles (body)"}, \
            {"role": "user", "content": "A young male with short white hair, blue eyes, and a neutral expression. He is wearing a white t-shirt and shorts with a pattern on the legs, without shoes. The character is running. He is tall with lean frame."}, \
            {"role": "assistant","content": "short, white hair (hair)/white t-shirt (top)/shorts with a pattern on the legs (bottom)/running (action)/tall with lean frame (body)/without shoes (shoe)"}, \
            {"role": "user", "content": "She had long, golden curly hair and was wearing a white lace dress and high heels, with a silver bracelet on her wrist. She is fat with round belly with no action."}, \
            {"role": "assistant","content": "long, golden curly hair (hair)/white lace dress (outfit)/high heels (shoe)/fat with round belly (body)"}, \
            {"role": "user", "content": "The girl wore a pair of black-framed glasses and had short hair. She was dressed in a blue denim jacket and a black skirt, with a pair of white sneakers on her feet. She is dancing and has anime-style body."}, \
            {"role": "assistant","content": "short hair (hair)/blue denim jacket (top)/black skirt (bottom)/white sneakers (shoe)/dancing (action)/ anime-style (body)"}, \
            {"role": "user", "content": "A thin man in suits, with no shoes and no hair."}, \
            {"role": "assistant","content": "suits (outfit)/thin (body)/no shoes (shoe)/no hair (hair)"}, \
            {"role": "user", "content": "A tall adult woman in floral dress with barefoot."}, \
            {"role": "assistant","content": "floral dress (outfit)/tall adult (body)/barefoot (shoe)"}, \
                
            {"role": "user", "content": sentence}]
    
    body = {
    "appGroupId": 0,
    # 指定模型
    "model": "gpt-4o",
    # 自定义上下文
    "messages": history,
    "temperature": 0.1,
    "max_tokens": 1000,
    }

    ret = client.post('', header=header, body=json.dumps(body))
    # print(ret['data']['response'])
    return ret['data']['response']


# use gpt api to extract required entities from a provided description, for @chentian
def extract_entity_gpt_2(sentence):
    # print(sentence)
    secret_id = ''
    secret_key = ''

    client = HttpClient(config=Config(read_timeout=300), secret_id=secret_id, secret_key=secret_key)

    header = {
        'Content-Type': 'application/json',
    }
    
    prompts = "You should extract entities and their attributes from a sentence. \
        You will be given a sentence, you must extract all entities and their attributes related to hair, tops, trousers, shoes, outfits, clothing and action. \
        Provide the described entities in a string, split different entities by /. \
        If no entities are found in the description, provide an empty string. \
        Ignore person's name. Ignore entities that are not related to hair, tops, trousers, shoes, outfits, clothing and action.\
        In addition, classify each entity into one of the following categories: [hair, top, bottom, shoe, outfit, action, other]. \
        Include category classification in bracket following each entity. "

    history = [{"role": "system", "content": prompts}, \
            {"role": "user", "content": "sweater"}, \
            {"role": "assistant","content": "sweater (top)"}, \
            {"role": "user", "content": "please make the shoes red"}, \
            {"role": "assistant","content": "red shoes (shoe)"}, \
            {"role": "user", "content": "I need a blue shirt"}, \
            {"role": "assistant","content": "blue shirt (top)"}, \
            {"role": "user", "content": "Please replace the skirt of this digital person with the following image style."}, \
            {"role": "assistant","content": "skirt (bottom)"}, \
            
            {"role": "user", "content": sentence}]

    body = {
    "appGroupId": 0,
    # 指定模型
    "model": "gpt-4o",
    # 自定义上下文
    "messages": history,
    "temperature": 0.1,
    "max_tokens": 1000,
    }

    ret = client.post('', header=header, body=json.dumps(body))
    # print(ret['data']['response'])
    return ret['data']['response']

# use hunyuan api to translate chinese to english
def translation(sentence):
    url = ''
    headers = {
    "Content-Type": "application/json",
    "Authorization": "",
    }
    
    model = 'hunyuan'
    
    
    history = [{"role": "user", "content": "Translate to english:[{}]".format(sentence)}]
    print(history)
    
    json_data = {"model":model,"messages": history, "temperature": 0.2}
    input_data = json.dumps(json_data)

    response = requests.post(url, data=input_data.encode(), headers=headers)

    response = json.loads(response.text)
    print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']


# use gpt api to translate chinese to english
def translation_gpt(sentence):
    secret_id = ''
    secret_key = ''

    client = HttpClient(config=Config(read_timeout=300), secret_id=secret_id, secret_key=secret_key)

    header = {
        'Content-Type': 'application/json',
    }

    # history = [{"role": "user", "content": "Translate to english:[{}]".format(sentence)}]
    # history = [{"role": "user", "content": "Translate the following sentence from Chinese to English: [{}]".format(sentence)}]
    
    history = [{"role": "system", "content": "Translate from Chinese to English. \
                You must only output the translated result, no extra information."},
               {"role": "user", "content": "[{}]".format(sentence)}]
    
    body = {
    "appGroupId": 0,
    # 指定模型
    "model": "gpt-4o",
    # 自定义上下文
    "messages": history,
    "temperature": 0.1,
    "max_tokens": 1000,
    }

    ret = client.post('', header=header, body=json.dumps(body))
    # print(ret['data']['response'])
    return ret['data']['response']

# use gpt api to translate english to chinese
def translation_gpt_e2c(sentence):
    secret_id = ''
    secret_key = ''

    client = HttpClient(config=Config(read_timeout=300), secret_id=secret_id, secret_key=secret_key)

    header = {
        'Content-Type': 'application/json',
    }

    history = [{"role": "user", "content": "翻译为中文:[{}]".format(sentence)}]

    body = {
    "appGroupId": 0,
    # 指定模型
    "model": "gpt-4o",
    # 自定义上下文
    "messages": history,
    "temperature": 0.1,
    "max_tokens": 1000,
    }

    ret = client.post('', header=header, body=json.dumps(body))
    # print(ret['data']['response'])
    return ret['data']['response']



# use gpt api to detect if the chracter needs action
def action_detection_gpt(sentence):
    secret_id = ''
    secret_key = ''

    client = HttpClient(config=Config(read_timeout=300), secret_id=secret_id, secret_key=secret_key)

    header = {
        'Content-Type': 'application/json',
    }

    history = [{"role": "user", "content": "Translate to english:[{}]".format(sentence)}]

    body = {
    "appGroupId": 0,
    # 指定模型
    "model": "gpt-3.5-turbo",
    # 自定义上下文
    "messages": history,
    "temperature": 0.1,
    "max_tokens": 1000,
    }

    ret = client.post('', header=header, body=json.dumps(body))
    # print(ret['data']['response'])
    return ret['data']['response']

# use gpt api to detect if the clothing requires texture changing
def texture_change_gpt(before, after):
    secret_id = ''
    secret_key = ''

    client = HttpClient(config=Config(read_timeout=300), secret_id=secret_id, secret_key=secret_key)

    header = {
        'Content-Type': 'application/json',
    }

    prompts = "You are part of a team of bots that analyze clothing. \
        You will be given a description of two pieces of clothing, A and B. \
            Your task is to determine whether cloth A and cloth B belong to the same clothing sub-category. \
                In other words, you must determine if cloth A can be transformed into cloth B by only changing texture or color. \
                    You must make a binary prediction and return only yes or no."
    
    history = [{"role": "system", "content": prompts},
               {"role": "user", "content": "Does pink trousers can be transformed to pink shorts by only changing texture or color?"},
               {"role": "assistant","content": "no"},
               {"role": "user", "content": "Does blue jeans can be transformed to red jeans by only changing texture or color?"},
               {"role": "assistant","content": "yes"},
               {"role": "user", "content": "Does {} and {} belong to the same clothing sub-category?".format(before, after)}
               ]
    
    breakpoint()
    body = {
    "appGroupId": 0,
    # 指定模型
    "model": "gpt-4o",
    # 自定义上下文
    "messages": history,
    "temperature": 0.1,
    "max_tokens": 1000,
    }

    ret = client.post('', header=header, body=json.dumps(body))
    print(ret['data']['response'])
    return ret['data']['response']



# use gpt api to classify sub clothing category
def entity_classification_gpt(sentence):
    secret_id = ''
    secret_key = ''

    client = HttpClient(config=Config(read_timeout=300), secret_id=secret_id, secret_key=secret_key)

    header = {
        'Content-Type': 'application/json',
    }
    
    history = [
               {"role": "user", "content": "What is the main clothing entity described in this phrase: [{}]. Return the anwser in english word.".format(sentence)},
               ]
    
    # breakpoint()
    body = {
    "appGroupId": 0,
    # 指定模型
    "model": "gpt-4o",
    # 自定义上下文
    "messages": history,
    "temperature": 0.1,
    "max_tokens": 1000,
    }

    ret = client.post('', header=header, body=json.dumps(body))
    print(ret['data']['response'])
    return ret['data']['response']

def get_sparse_result(entities):
    output = {}
    part_keys = ["hair", "top", "bottom", "shoe", "outfit", "action", "body", "others"]
    for key in part_keys:
        output[key] = []
    for entity in entities.split("/"):
        str_description = ' '.join(entity.split(' ')[:-1])
        str_category = entity.split(' ')[-1].replace('(','').replace(')','')
        if str_category in part_keys:
            output[str_category].append(str_description)
        
    # print(output)
    return output

def find_diff_index(list1, list2):
    for i in range(min(len(list1), len(list2))):
        if list1[i] != list2[i]:
            return i
    return -1 

if __name__ == "__main__":
    

    entities_dict = {}
    entities_dict['gender'] = ''
    entities_dict['hair'] = ''
    entities_dict['top'] = ''
    entities_dict['bottom'] = ''
    entities_dict['shoe'] = ''
    entities_dict['outfit'] = ''
    entities_dict['others'] = ''
    key_dict = {"hair":"头发", "top":"上装", "bottom":"下装", "shoe":"鞋", "outfit":"套装", "others":"其他","action":"动作", "gender":'性别', "body":'体型'}
    pre_defined_body_type = ["小孩儿身材", "成年人身材", "肥胖身材", "强壮身材", "匀称身材", "普通身材", "瘦弱身材", "圆滚滚身材", "修长身材"]

    secret_id = ''
    secret_key = ''



    client = HttpClient(config=Config(read_timeout=300), secret_id=secret_id, secret_key=secret_key)

    header = {
        'Content-Type': 'application/json',
    }

    
    entities = ''
    sum_tmp = ''

   

    history = [
        {
            "role": "system",
            "content": "你是一位活泼有创意的角色设计师，可以根据用户提供的信息创造独特的角色。\
                        这些信息涵盖性别、发型、套装或上装和下装、鞋子，体型，动作。\
                        你应该：\
                            1. 与用户关于角色进行聊天。\
                            2. 问题只和以下内容有关：性别，发型，套装或上装和下装，鞋子，体型，动作。 \
                            3. 不进行与角色设计无关的讨论。\
                            4. 言简意赅，每次用不超过10个词的一两个问题提问。\
                            5. 保持活泼可爱的性格 "
        },
        # {"role": "assistant", "content": "请告诉我你想要创建的角色信息！"}
        # {"role": "user", "content": "你好，我想创建一个新的虚拟角色"},
    ]


    
    ret = None
    
    chat_history = ''
    current_history = ''

    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config["sync_request_timeout"] = None
    connection = rpyc.connect('', 0,config=rpyc_config)

    prev_list_out = ['']*8
    out_dict = []
    glb_url=''
    print_enable = True
    enable_api = False
    path_list = []
    key_list = []
    texture_replace=[False, False, False, False, False, False]
    glb_local_path=''
    while True:
        entities = ''
        # sum_tmp =  ''
        gender = None
        change = [0] * 8
        
        if ret is not None:
            history.append({"role": "assistant","content":  ret['data']['response']})
            print(ret['data']['response'])
            
        if glb_url!='':
            print(glb_url)
        # except:
            # pass
        
        new = input('input: ')
        
        try:
            sensitive = sensitive_content_filter(new)
        except:
            sensitive = 'False'
        if sensitive.strip().lower() != 'false':
            # breakpoint()
            print("敏感内容！")
            continue

        if new.isspace():
            continue
        
        if not new:
            continue

        # 体型硬匹配
        cur_body = None
        for cur_body_type in pre_defined_body_type:
            if cur_body_type in new:
                cur_body = cur_body_type
                break
        
        # maximum chat rounds: 40, if over length, we include a sliding window 
        if len(history)>60:
            history.pop(1)
            history.pop(1)
        
        chat_history += 'User: {}\n '.format(new)
        current_history += 'User: {}\n '.format(new)
        
        # print(current_history)
        # if len(history) > 3:
        body_tmp = {
        "appGroupId": 0,
        "model": "gpt-4o",
        "request": 'There is a virtual character now. \
            Here is the known information about this character: [{}] \
            You are a designer of virtual characters and have had a conversation with the user. \
            Here is the conversation: [{}] \
            You need to update the virtual character based on the conversation and known information. \
            Describe the virtual character in one sentence. \
            Do not add additional information.\
            Do not include information that is irrelevant to the character. \
            Do not return anything else.\ '.format(sum_tmp, translation_gpt(current_history)),
        "temperature": 0.05,
        # "top_k": 1,
        "max_tokens": 1000,
        }
        current_history = ''
        ret_tmp = client.post('', header=header, body=json.dumps(body_tmp))
        
        sum_tmp = ret_tmp['data']['response']
        gender = gender_classification_gpt(sum_tmp)
        
        if gender != 'unknown':
            entities_dict['gender'] = gender
                
        entities = extract_entity_gpt(sum_tmp)
        if print_enable:
            print('总结：', ret_tmp['data']['response'])
            print('性别:',  gender)
            print('当前抽取信息：', entities)
            print('---')

        sparse_result = get_sparse_result(entities)

        part_keys = ["hair", "top", "bottom", "shoe", "outfit", "action", "body", "others"]
        list_out = []
        out_dict = []
        for key in part_keys:
            if len(sparse_result[key]) > 0:
                id = random.randint(0, len(sparse_result[key]) - 1)
                if key == 'hair':
                    for tmp_no_hair_name in ['no hair', 'bald', 'hairless', 'bare', 'barehead', 'shaved', 'without']:
                        if tmp_no_hair_name in sparse_result[key][id].lower():
                            list_out.append('---')
                            out_dict.append(key + ': ' + '---')
                            break
                    else:
                        list_out.append(sparse_result[key][id])
                        out_dict.append(key + ': ' + sparse_result[key][id] )

                elif key == 'shoe':
                    for tmp_no_shoe_name in ['no shoe', 'barefoot', 'bare', 'naked', 'without']:
                        if tmp_no_shoe_name in sparse_result[key][id].lower():
                            list_out.append('---')
                            out_dict.append(key + ': ' + '---')
                            break
                    else:
                        list_out.append(sparse_result[key][id])
                        out_dict.append(key + ': ' + sparse_result[key][id] )
                
                elif key == 'body':
                    if cur_body: 
                        list_out.append(cur_body)
                        out_dict.append(key + ': ' + cur_body )
                    else:
                        list_out.append(sparse_result[key][id])
                        out_dict.append(key + ': ' + sparse_result[key][id] )
                
                else:
                    list_out.append(sparse_result[key][id])
                    out_dict.append(key + ': ' + sparse_result[key][id] )
            else:
                list_out.append("")   
                # out_dict.append(key + ': ')         
        
        gender_keys = ["male", "female"]
        if gender not in gender_keys:
            gender_id = random.randint(0, 1)
            gender = gender_keys[gender_id]
        
        list_out.append(gender)
        out_dict.append('gender: ' + gender)
        
        if print_enable:
            print(list_out)
        
        # 上衣下衣和套装互斥
        
        # breakpoint()
        # 得到当前未知信息
        unknown_info = ["头发", "上装", "下装", "鞋", "套装","动作","体型"]
        for out_dict_item in out_dict:
            if key_dict[out_dict_item.split(':')[0]] in unknown_info:
                unknown_info.remove(key_dict[out_dict_item.split(':')[0]])
        
        if (list_out[0] != '' and list_out[1] != '' and list_out[2] != '' and list_out[3] != '' and list_out[5] != '' and list_out[6] != '' and list_out[8] != '') or \
                            (list_out[0] != '' and list_out[3] != '' and list_out[4] != '' and list_out[5] != '' and list_out[6] != '' and list_out[8] != ''):
            
            # hair color（debug用）
            if list_out[0] != '---':
                print('头发颜色检测: ', attributes_separation_gpt(list_out[0]))

            
                
            if enable_api:
                if connection==None:
                    connection = rpyc.connect('', 8081,config=rpyc_config)

                if list_out[0]!="" or  list_out[1]!="" or list_out[2]!="" or list_out[3]!="" or list_out[4]!="":
                    json_str = connection.root.txt_retrive_npc(list_out,sum_tmp[:70])
                    retrieve_list = ujson.loads(json_str)
                    hair_path = retrieve_list[0][0]
                    top_path = retrieve_list[1][0]
                    bottom_path = retrieve_list[2][0]
                    shoe_path = retrieve_list[3][0]
                    outfit_path = retrieve_list[4][0]
                    others_path = retrieve_list[5][0]

                    hair_key = retrieve_list[6][0]
                    top_key = retrieve_list[7][0]
                    bottom_key = retrieve_list[8][0]
                    shoe_key = retrieve_list[9][0]
                    outfit_key = retrieve_list[10][0]
                    others_key = retrieve_list[11][0]

                    gender = retrieve_list[18]
                    suit_enale = retrieve_list[19]
                    if suit_enale==False:
                        path_list = [hair_path,top_path,bottom_path,shoe_path,None,others_path]
                        key_list = [hair_key,top_key,bottom_key,shoe_key,None,others_key]
                    else:
                        path_list = [hair_path,None,None,shoe_path,outfit_path,others_path]
                        key_list = [hair_key,None,None,shoe_key,outfit_key,others_key]
                    
                    json_str = connection.root.retrive_npc_combine(path_list,key_list,texture_replace=texture_replace,gender=gender)
                    combine_result = ujson.loads(json_str)
                    # print(combine_result[1])
                    glb_local_path = combine_result[1]
                    if print_enable:
                        print("glb_local_path:",combine_result[1])

                    glb_url =ujson.loads(connection.root.cos_upload(glb_local_path))
                    print(glb_url)
                
            
            
            # history.append({"role": "user","content": new})
            
            ret = {}
            ret['data'] = {}
            

            ret['data']['response'] = '角色设计已经完成，根据你提供的信息，\
                    我为你设计的角色是：{}\ \n\
                    1. 如果你对当前结果不甚满意，可以点击【重新创造】按钮使用相同提示词重新定制不一样的角色。 \ \n\
                    2. 如果你需要更改当前角色的部件，请告诉我你想要更改成什么新的样式。\ \n\
                    比如： "上衣换成红色外套”，"裤子换成沙滩裤” \ \n\
                    3. 此外，你也可以更换体型，请从以下身材中选择一个输入：["小孩儿身材”, "成年人身材”, "肥胖身材”, "强壮身材”, "匀称身材”, "普通身材”, "瘦弱身材”, "圆滚滚身材”, "修长身材”] \
                    4. 最后，如果你想获得个性化的纹理，请点击【开始纹理替换】按钮。'.format(translation_gpt_e2c(sum_tmp))

           
            
            prev_list_out = list_out
            
          
            
            chat_history += 'Desinger: {}\n '.format(ret['data']['response'])
            current_history += 'Desinger: {}\n '.format(ret['data']['response'])

            # 在这里添加按钮：开始进行纹理替换
            break
        
        else:
            if '套装' not in unknown_info: # 套装已知，不用问上装和下装
                try:
                    unknown_info.remove('上装')
                except:
                    pass
                try:
                    unknown_info.remove('下装')
                except:
                    pass
                current_unknown = unknown_info[0]
            else: # 套装未知
                if '上装' in unknown_info and '下装' in unknown_info: # 上下装都未知，套装也未知，问上装或套装
                    current_unknown = '上装或套装'
                else: # 知道上装或下装，不问套装
                    if '套装' in unknown_info:
                        unknown_info.remove('套装')
                        
                    current_unknown = unknown_info[0]
                
            print('unknown info: ', unknown_info)
            
            # todo: 引入上一轮的对话，增加一点上下文信息
            history = [{
            "role": "system",
            "content": "你是一位有创意的角色设计师，可以根据用户提供的信息创造独特的角色。\
                        这些信息涵盖性别，发型，套装或上装和下装，鞋子，体型，动作。\
                        你应该：\
                            1. 不进行与角色设计无关的讨论。\
                            2. 言简意赅 \
                            3. 保持活泼可爱的性格 \
                        目前你应该收集[{}]的信息。".format(current_unknown)
            }]

            # history.append({"role": "user","content": new})
            
            # print(history)
            body = {
            "appGroupId": 0,
            "model": "gpt-3.5-turbo",
            "messages": history,
            "temperature": 0.05,
            "max_tokens": 1000,
            }

            
            try:
                ret = client.post('', header=header, body=json.dumps(body))
                
            except:

                history = [{
                "role": "system",
                "content": "你是一位有创意的角色设计师，可以根据用户提供的信息创造独特的角色。\
                            这些信息涵盖性别、发型、服装、上装、下装、鞋子，动作。\
                            你应该：\
                                1. 与用户关于角色进行聊天。\
                                2. 问题只和以下内容有关：性别、发型、服装、上装、下装、鞋子，动作。 \
                                3. 不进行与角色设计无关的讨论。\
                                4. 言简意赅，每次用不超过10个词的一两个问题提问。"
            }]
                continue
            
            chat_history += 'Desinger: {}\n '.format(ret['data']['response'])
            current_history += 'Desinger: {}\n '.format(ret['data']['response'])
        
    # 纹理替换
    if (list_out[0] != '' and list_out[1] != '' and list_out[2] != '' and list_out[3] != '' ):
        texture_list = ['上装', '下装', '鞋子']
        texture_list_en = ['top', 'bottom', 'shoe']
        list_out = prev_list_out = [list_out[1], list_out[2], list_out[3]]

    elif (list_out[0] != '' and list_out[3] != '' and list_out[4] != '' ):
        texture_list = ['套装', '鞋子']
        texture_list_en = ['outfit', 'shoe']

        list_out = prev_list_out = [list_out[3], list_out[4]]
        
    # 目前线上版本纹理
    part_keys = texture_list_en
    current_history = ''
    for cloth in texture_list:
        question = '是否要更换{}的纹理？若不需要，请回答否。'.format(cloth)
        print(question)
        history.append({"role": "assistant","content":  question})
        current_history += 'Desinger: {}\n'.format(question)
        new = input('input: ')
        
        try:
            sensitive = sensitive_content_filter(new)
        except:
            sensitive = 'False'
        if sensitive.strip().lower() != 'false':
            print("敏感内容！")
            continue
        
        current_history += 'User: {}\n '.format(new)

        if '否' in new or '不' in new :
            current_history = ''
            continue
        else:
            print(current_history)
            body_tmp = {
            "appGroupId": 0,
            "model": "gpt-4o",
            "request": 'There is a virtual character now. \
                Here is the known information about this character: [{}] \
                You are a designer of virtual characters and have had a conversation with the user. \
                Here is the conversation: [{}] \
                You need to update the virtual character based on the conversation and known information. \
                Describe the virtual character in one sentence. \
                Do not add additional information.\
                Do not include information that is irrelevant to the character. \
                Do not return anything else.\ '.format(sum_tmp, translation_gpt(current_history)),
            "temperature": 0.05,
            "max_tokens": 1000,
            }
            current_history = ''
            ret_tmp = client.post('', header=header, body=json.dumps(body_tmp))
            
            sum_tmp = ret_tmp['data']['response']
            gender = gender_classification_gpt(sum_tmp)

            if gender != 'unknown':
                entities_dict['gender'] = gender
                
            print('性别:',  gender)
            if print_enable:
                print('性别:',  gender)
            entities = extract_entity_gpt(sum_tmp)
            if print_enable:
                print('总结：', ret_tmp['data']['response'])
                print('性别:',  gender)
                print('当前抽取信息：', entities)
                print('---')

            sparse_result = get_sparse_result(entities)

            list_out = []
            out_dict = []
            for key in part_keys:
                if len(sparse_result[key]) > 0:
                    id = random.randint(0, len(sparse_result[key]) - 1)
                    list_out.append(sparse_result[key][id])
                    out_dict.append(key + ': ' + sparse_result[key][id] )
                else:
                    list_out.append("")   
                    # out_dict.append(key + ': ')         

            gender_keys = ["male", "female"]
            if gender not in gender_keys:
                gender_id = random.randint(0, 1)
                gender = gender_keys[gender_id]
                
            # list_out.append(gender)
            out_dict.append('gender: ' + gender)

            if print_enable:
                print(list_out)
            
            print(prev_list_out, '->', list_out)
            
            index = find_diff_index(list_out,prev_list_out)
            prev_list_out = list_out
            if index<0:
                continue
            
            if list_out[index]!='' and enable_api:
                out_mesh_paths_query_key = ujson.loads(connection.root.texReplace(list_out[index],path_list[index],key_list[index]))
                if print_enable:
                    print("texture_replace input:",index,list_out[index],path_list[index],key_list[index])
                    print("texture_replace output:",out_mesh_paths_query_key)
                texture_replace[index]=True
                path_list[index]=out_mesh_paths_query_key.replace('.glb','.obj')

            if enable_api:
                json_str = connection.root.retrive_npc_combine(path_list,key_list,texture_replace=texture_replace,gender=gender)
                combine_result = ujson.loads(json_str)
                
                glb_local_path = combine_result[1]

                glb_url =ujson.loads(connection.root.cos_upload(glb_local_path))
                print(glb_url)
                
    if enable_api:
        connection.root.retrive_npc_text_animation(glb_local_path.replace('mesh/mesh.glb',''),prev_list_out[-2])
        fbx_url =ujson.loads(connection.root.cos_upload(glb_local_path.replace('.glb','_animation.fbx')))
        print("已自动为你添加骨骼并加入动画")
        print(fbx_url)
    
    # 预设问题，直接显示
    print('您认为角色怎么样呢？请从1-10打个分吧。\
          如果您觉得满意，请点击右侧按钮进行发布或收藏。')
    new = input('input:' )
    
    if new:
        score = score_classification(new)
        print('用户评分：', score)
    
    print('感谢你的反馈，先在你可以输入新的提示词定制新的角色。')