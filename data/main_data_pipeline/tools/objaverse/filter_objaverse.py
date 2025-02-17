import argparse
import json
import os

import objaverse
from tqdm import tqdm


def read_list(in_list_txt):
    if not os.path.exists(in_list_txt):
        print('Cannot find input list txt file ', in_list_txt)
        exit(-1)

    str_list = []
    with open(in_list_txt, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            mesh_path = line.strip()
            if len(mesh_path) > 1:
                str_list.append(mesh_path)
    return str_list


def write_list(path, write_list):
    with open(path, 'w') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4)


def parse_objaverse_list(list_path: str):
    objaverse_list = read_list(list_path)
    objaverse_hash_list = []
    hash_path_map = {}
    for objaverse_name in objaverse_list:
        objaverse_filename = os.path.split(objaverse_name)[1]
        objaverse_hash = os.path.splitext(objaverse_filename)[0]
        objaverse_hash_list.append(objaverse_hash)
        hash_path_map[objaverse_hash] = objaverse_name
    return objaverse_hash_list, hash_path_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch download files to mesh folder')
    parser.add_argument('--objaverse_mesh_list', type=str,
                        help='a list of objaverse mesh paths or objaverse hashes')
    parser.add_argument('--output_json_path', type=str,
                        help='a json of category - hash id map')
    args = parser.parse_args()

    objaverse_hash_list = args.objaverse_mesh_list
    output_json_path = args.output_json_path

    uids, hash_path_map = parse_objaverse_list(objaverse_hash_list)
    print(len(uids))
    annotations = objaverse.load_annotations(uids)
    category_path_map = {}
    for key, attributes in tqdm(annotations.items()):
        if len(attributes['categories']) == 0:
            continue

        for category in attributes['categories']:
            if category['name'] in category_path_map.keys():
                category_path_map[category['name']].append(key)
            else:
                category_path_map[category['name']] = []
                category_path_map[category['name']].append(key)

    for category in category_path_map.keys():
        print(category, len(category_path_map[category]))
    write_json(output_json_path, category_path_map)
