import argparse
import json


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4)


def check_data_number(json_struct):
    if "data" not in json_struct.keys():
        return 0
    category = json_struct["data"].keys()
    total_data_number = 0
    for category_name in category:
        total_data_number = total_data_number + \
                            len(json_struct["data"][category_name])
    return total_data_number


def check_individual_number(json_struct):
    if "data" not in json_struct.keys():
        return 0
    category = json_struct["data"].keys()
    number_info = {}
    for category_name in category:
        number_info[category_name] = len(json_struct["data"][category_name])
    return number_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find difference between two lists')
    parser.add_argument('--input_json_path', type=str,
                        help='input json file path')
    parser.add_argument('--input_correspondence_json_path', type=str,
                        help='input correspondence json file path')
    parser.add_argument('--output_json_path', type=str,
                        help='output json file path')
    parser.add_argument('--output_correspondence_json_path', type=str,
                        help='output correspondence json file path')
    args = parser.parse_args()

    with open(args.input_json_path, encoding='utf-8') as f:
        json_data = json.load(f)
    with open(args.input_correspondence_json_path, encoding='utf-8') as f:
        correspondence_json_data = json.load(f)

    mesh_data = json_data["data"]["objaverse"]
    new_mesh_data = {}
    new_mesh_data["data"] = {}
    new_mesh_data["data"]["objaverse"] = {}
    new_correspondence_data = {}
    new_correspondence_data["objaverse"] = {}
    mesh_names = list(mesh_data.keys())
    # split_mesh_names = mesh_names[args.start_index:args.end_index]
    for current_mesh_name in mesh_names:
        if "Mesh" not in mesh_data[current_mesh_name].keys():
            print(current_mesh_name)
            continue
        mesh_path = mesh_data[current_mesh_name]["Mesh"]
        mesh_real_name = mesh_path.split("/")[-3]
        if mesh_real_name not in mesh_data[current_mesh_name]["Manifold"]:
            print(mesh_real_name)
            continue
        current_correspondence = correspondence_json_data["objaverse"][current_mesh_name]
        for original_mesh_name in current_correspondence.keys():
            if mesh_real_name not in list(current_correspondence[original_mesh_name].keys())[0]:
                print(mesh_real_name, original_mesh_name)
                break

        new_correspondence_data["objaverse"][mesh_real_name] = correspondence_json_data["objaverse"][current_mesh_name]
        new_mesh_data["data"]["objaverse"][mesh_real_name] = mesh_data[current_mesh_name]

    write_json(args.output_json_path, new_mesh_data)
    print(check_data_number(new_mesh_data))
    print(check_individual_number(new_mesh_data))
    write_json(args.output_correspondence_json_path, new_correspondence_data)
