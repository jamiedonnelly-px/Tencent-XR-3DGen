# -*- coding: utf-8 -*-
import re
import sys
sys.path.append('/mnt/aigc_cfs_cq/xiaqiangdai/project/character_customization')
from character_design import translation_gpt,extract_entity_gpt,get_sparse_result,gender_classification_gpt,attributes_separation_gpt
import random


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
    
    list_out.append(gender)
    print(list_out) #["hair", "top", "bottom", "shoe", "outfit", "others","action", "body","gender"]
    return list_out, description





if __name__ == "__main__":

    long_prompt="The boy is dressed in a sleek black suit paired with a crisp white shirt, and his hair is styled in a modern undercut, exuding a sense of sophistication and style."
    extract_entity_all(long_prompt)
    # print(gender_classification_gpt(long_prompt))
