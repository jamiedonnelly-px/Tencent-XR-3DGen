# -*- coding: utf-8 -*-
import re
import sys
import os
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_dir,'character_customization'))
from character_design import translation_gpt,extract_entity_gpt,get_sparse_result,gender_classification_gpt,attributes_separation_gpt
import random
import time


def contains_chinese(text):
    pattern = re.compile(r"[\u4e00-\u9fa5]")
    return bool(pattern.search(text))


def extract_entity_all(long_prompt):
    description = long_prompt
    if contains_chinese(description):
        print("=========chinese")
        description = translation_gpt(description)

    # breakpoint()
    entities = extract_entity_gpt(description)
    sparse_result = get_sparse_result(entities)
    part_keys = ["hair", "top", "bottom", "shoe", "outfit", "others","action", "body"]
    list_out = []
    for key in part_keys:
        if len(sparse_result[key]) > 0:
            id = random.randint(0, len(sparse_result[key]) - 1)
            list_out.append(sparse_result[key][id])
        else:
            list_out.append("")
    gender_keys = ["male", "female"]
    gender = gender_classification_gpt(long_prompt)
    if gender not in gender_keys:
        gender_id = random.randint(0, 1)
        gender = gender_keys[gender_id]
    
    hair_entities = attributes_separation_gpt(list_out[0])
    hair_output = {"color": [], "type": []}
    str_description=''
    hair_color=''
    for hair_entity in hair_entities.split("/"):
        hair_str_description = " ".join(hair_entity.split(" ")[:-1])
        hair_str_category = (
            hair_entity.split(" ")[-1].replace("(", "").replace(")", "")
        )

        if hair_str_category in ["color", "type"]:
            hair_output[hair_str_category].append(hair_str_description)
    if len(hair_output["type"]) != 0:
        str_description = hair_output["type"][0]
    else:
        str_description ='hair'
        print("hair attributes_separation_gpt no type")

    if len(hair_output["color"]) != 0:
        hair_color = hair_output["color"][0]
    else:
        hair_color=''
        print("hair attributes_separation_gpt no color")
    # print(list_out)
    list_out[0] = str_description
    list_out.append(hair_color)
    list_out.append(gender)
    print(list_out) #["hair", "top", "bottom", "shoe", "outfit", "others","action", "body","hair_color","gender"]
    return list_out, description





if __name__ == "__main__":

    long_prompt = "她穿着一件粉色的连衣裙，腰间系着一条腰带，穿着一双白色高跟鞋。她的长发被扎成了一个马尾辫，脸上带着一丝甜美的微笑。她手里拿着一本书，显得格外文静。"

    start_time = time.time()
    extract_entity_all(long_prompt)
    end_time = time.time()
    print(f"cost time: {end_time - start_time} s")
    # print(gender_classification_gpt(long_prompt))
