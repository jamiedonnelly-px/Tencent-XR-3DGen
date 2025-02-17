import argparse
import json
import os


def read_json(json_path: str):
    with open(json_path, encoding='utf-8') as f:
        json_struct = json.load(f)
        return json_struct


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4, ensure_ascii=False)


def read_list(in_list_txt):
    if not os.path.exists(in_list_txt):
        print('Cannot find input list txt file ', in_list_txt)
        exit(-1)

    str_list = []
    with open(in_list_txt, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            mesh_path = line.strip()
            if len(mesh_path) > 1:
                str_list.append(mesh_path)
    return str_list


def write_list(path, write_list):
    with open(path, 'w', encoding='utf-8') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


def add_somthing_if_exists(data_struct: dict, data_name: str):
    if data_name in data_struct.keys():
        return data_struct[data_name]
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate management system json')
    parser.add_argument('--data_json_file', type=str,
                        help='input data json file')
    parser.add_argument('--output_json_file', type=str,
                        help='output management system json path')

    args = parser.parse_args()
    data_json_file = args.data_json_file
    output_json_file = args.output_json_file

    result_json_data = {}
    result_json_data["data"] = []
    original_json_data = read_json(data_json_file)

    print("Input management data length is %i" % len(original_json_data["data"]))

    data_struct = original_json_data["data"]
    for single_data_struct in data_struct:
        new_single_data_struct = {}
        new_single_data_struct["Key"] = add_somthing_if_exists(single_data_struct, "Key")
        new_single_data_struct["Category"] = add_somthing_if_exists(single_data_struct, "Category")

        new_single_data_struct["MeshName"] = single_data_struct["MeshName"]
        new_single_data_struct["GameCategory"] = single_data_struct["GameCategory"]
        new_single_data_struct["MeshStyle"] = single_data_struct["MeshStyle"]
        new_single_data_struct["Origin"] = single_data_struct["Origin"]
        new_single_data_struct["index"] = single_data_struct["index"]
        new_single_data_struct["FaceNum"] = single_data_struct["FaceNum"]
        new_single_data_struct["ScrapyUrl"] = single_data_struct["ScrapyUrl"]
        new_single_data_struct["Comment"] = single_data_struct["Comment"]

        new_single_data_struct["SavePaths"] = {}
        new_single_data_struct["SavePaths"]["MeshFilename"] = single_data_struct["MeshFilename"]
        new_single_data_struct["SavePaths"]["MeshThumbnailFilename"] = single_data_struct["MeshThumbnailFilename"]
        new_single_data_struct["SavePaths"]["MeshRenderFolder"] = single_data_struct["MeshRenderFolder"]

        new_single_data_struct["SavePaths"]["MeshFBXFilename"] = add_somthing_if_exists(single_data_struct,
                                                                                        "MeshFBXFilename")
        new_single_data_struct["SavePaths"]["MeshGeometryFolder"] = add_somthing_if_exists(single_data_struct,
                                                                                           "MeshGeometryFolder")
        new_single_data_struct["SavePaths"]["MeshTextureFolder"] = add_somthing_if_exists(single_data_struct,
                                                                                          "MeshTextureFolder")

        new_single_data_struct["IfPropertiesExist"] = {}

        new_single_data_struct["IfPropertiesExist"]["TextureExist"] = add_somthing_if_exists(single_data_struct,
                                                                                             "TextureExist")
        new_single_data_struct["IfPropertiesExist"]["RoughnessExist"] = add_somthing_if_exists(single_data_struct,
                                                                                               "RoughnessExist")
        new_single_data_struct["IfPropertiesExist"]["MetallicExist"] = add_somthing_if_exists(single_data_struct,
                                                                                              "MetallicExist")
        new_single_data_struct["IfPropertiesExist"]["SpecularExist"] = add_somthing_if_exists(single_data_struct,
                                                                                              "SpecularExist")
        new_single_data_struct["IfPropertiesExist"]["NormalExist"] = add_somthing_if_exists(single_data_struct,
                                                                                            "NormalExist")

        new_single_data_struct["Specific"] = {}
        new_single_data_struct["Specific"]["Layer"] = False
        # new_single_data_struct["Specific"]["Hole"] = False
        # new_single_data_struct["Specific"]["Slice"] = False
        # new_single_data_struct["Specific"]["Flat"] = False
        # new_single_data_struct["Specific"]["Building"] = False

        result_json_data["data"].append(new_single_data_struct)

    write_json(output_json_file, result_json_data)
    print("Management data length is %i" % len(result_json_data["data"]))
