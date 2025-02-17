import argparse
import json


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find difference between two lists')
    parser.add_argument('--input_json_path', type=str,
                        help='input json file path')
    parser.add_argument('--output_json_path', type=str,
                        help='output json file path')
    parser.add_argument('--start_index', type=int,
                        help='start index of the list')
    parser.add_argument('--end_index', type=int,
                        help='end index of the list')
    args = parser.parse_args()

    start_index = args.start_index
    end_index = args.end_index
    output_json_path = args.output_json_path

    with open(args.input_json_path, encoding='utf-8') as f:
        json_data = json.load(f)
    new_mesh_data = {}
    new_mesh_data["data"] = {}
    mesh_names = []
    mesh_name_data_name_map = {}

    for data_name in json_data["data"].keys():
        mesh_data = json_data["data"][data_name]
        if data_name not in new_mesh_data["data"].keys():
            new_mesh_data["data"][data_name] = {}

        mesh_names.extend(list(mesh_data.keys()))
        for mesh_name in mesh_data.keys():
            mesh_name_data_name_map[mesh_name] = data_name

    if end_index > len(mesh_names):
        split_mesh_names = mesh_names[start_index:]
        end_index = len(mesh_names)
    else:
        split_mesh_names = mesh_names[start_index:end_index]

    for split_mesh_name in split_mesh_names:
        data_name = mesh_name_data_name_map[split_mesh_name]
        new_mesh_data["data"][data_name][split_mesh_name] = json_data["data"][data_name][split_mesh_name]

    output_json_path = output_json_path + "_{}k_{}k.json".format(int(start_index / 1000), int(end_index / 1000))
    write_json(output_json_path, new_mesh_data)
