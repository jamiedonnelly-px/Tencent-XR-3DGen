import argparse
import os
import time


def write_list(path, write_list):
    with open(path, 'w') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


def check_all_mesh_file_in_folder(folder_path: str, file_type: str, name_list: list, exclude_name_list: list):
    mesh_file_list = []
    potentional_folder_list = []
    if os.path.exists(folder_path):
        if os.path.isdir(folder_path):
            file_names = os.listdir(folder_path)
            for file_name in file_names:
                if file_name[0] != '.':
                    level1_path = os.path.join(folder_path, file_name)
                    level1_file_basename = os.path.splitext(file_name)[0]
                    level1_type_original = os.path.splitext(level1_path)[1]
                    level1_type = level1_type_original.lower()
                    if level1_type == file_type:
                        name_specific_flag = True
                        name_exclude_flag = False
                        if len(name_list) > 0:
                            name_specific_flag = False
                            for name in name_list:
                                if name in level1_file_basename:
                                    name_specific_flag = True
                                    break
                        if len(exclude_name_list) > 0:
                            for exclude_name in exclude_name_list:
                                if exclude_name in level1_file_basename:
                                    name_exclude_flag = True
                                    break
                        if name_exclude_flag:
                            continue
                        if name_specific_flag:
                            mesh_file_list.append(level1_path)

    return mesh_file_list, potentional_folder_list


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Generate mesh list start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(
        description='Generate mesh list from a group of organized mesh folder')
    parser.add_argument('--render_folders', nargs='+',
                        help='mesh folder')
    parser.add_argument('--output_list', type=str, default="",
                        help='output mesh list file path')
    parser.add_argument('--file_type', type=str, default=".fbx",
                        help='file type of mesh file to searched in folder')
    parser.add_argument('--specific_name', nargs='+', default=[],
                        help='only write mesh with this specific name to list, like \'manifold_full\' for \'manifold_full_123.obj\'')
    parser.add_argument('--exclude_name', nargs='+', default=[],
                        help='exclude mesh with this specific name to list, like \'manifold_full\' for \'manifold_full_123.obj\'')
    args = parser.parse_args()

    mesh_name_list = []
    render_folder_list = args.render_folders
    specific_name = args.specific_name
    exclude_name = args.exclude_name

    print(render_folder_list)
    for render_folder in render_folder_list:
        mesh_list, _ = check_all_mesh_file_in_folder(
            render_folder, args.file_type, name_list=specific_name, exclude_name_list=exclude_name)
        mesh_name_list = mesh_name_list + mesh_list
        for root, dirs, files in os.walk(render_folder):
            for dir in dirs:
                print("Check mesh in folder %s" % (os.path.join(root, dir)))
                mesh_list, _ = check_all_mesh_file_in_folder(
                    os.path.join(root, dir), args.file_type, name_list=specific_name, exclude_name_list=exclude_name)
                mesh_name_list = mesh_name_list + mesh_list
    print(args.output_list)
    write_list(args.output_list, mesh_name_list)
