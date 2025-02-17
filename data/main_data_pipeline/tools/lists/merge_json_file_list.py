import argparse
import copy
import json


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4, ensure_ascii=False)


def check_individual_number(json_struct):
    if "data" not in json_struct.keys():
        return 0
    category = json_struct["data"].keys()
    number_info = {}
    total_number = 0
    for category_name in category:
        mesh_number = len(json_struct["data"][category_name])
        number_info[category_name] = mesh_number
        total_number = total_number + mesh_number
    number_info["all"] = total_number
    return number_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge json using json file lists')
    parser.add_argument('--json_files', nargs='+',
                        help='abspath of json files')
    parser.add_argument('--original_json_path', type=str, default='',
                        help='original merged json path')
    parser.add_argument('--output_json_path', type=str,
                        help='merged json output path')
    parser.add_argument('--diff_output_json_path', type=str, default='',
                        help='diff json output path (output - original)')
    args = parser.parse_args()

    json_fullpath_list = args.json_files
    original_json_data = {}
    if len(args.original_json_path) > 1:
        with open(args.original_json_path, encoding='utf-8') as f:
            original_json_data = json.load(f)

    # if not os.path.exists(args.json_folder):
    #     exit(-1)

    data_struct = copy.deepcopy(original_json_data)
    if len(data_struct.keys()) == 0:
        data_struct["data"] = {}
    diff_data_struct = {}
    diff_data_struct["data"] = {}

    print(len(json_fullpath_list))

    for json_path in json_fullpath_list:
        with open(json_path, encoding='utf-8') as f:
            json_data = json.load(f)
            print(check_individual_number(json_data))
            for data_name in json_data["data"].keys():
                if data_name not in data_struct["data"].keys():
                    data_struct["data"][data_name] = {}
                if data_name not in diff_data_struct["data"].keys():
                    diff_data_struct["data"][data_name] = {}
                for instance_name in json_data["data"][data_name].keys():
                    data_struct["data"][data_name][instance_name] = json_data["data"][data_name][instance_name]
                    if "data" in original_json_data.keys():
                        if instance_name in original_json_data["data"][data_name].keys():
                            continue
                    diff_data_struct["data"][data_name][instance_name] = json_data["data"][data_name][instance_name]

    write_json(args.output_json_path, data_struct)
    print("Merged data struct length is %s" %
          (str(check_individual_number(data_struct))))
    if len(args.diff_output_json_path) > 1:
        write_json(args.diff_output_json_path, diff_data_struct)
        print("Diff data struct length is %s" %
              (str(check_individual_number(diff_data_struct))))
