import argparse
import os
import time


def write_list(path, write_list):
    with open(path, 'w') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


def check_all_mesh_file_in_folder(folder_path: str, file_type: str, name: str = "", exclude_name: str = ""):
    folder_list = []
    if os.path.exists(folder_path):
        if os.path.isdir(folder_path):
            file_names = os.listdir(folder_path)
            for file_name in file_names:
                file_fullpath = os.path.join(folder_path, file_name)
                if len(name) > 1:
                    if name in file_name:
                        if len(exclude_name) > 1:
                            if exclude_name in file_name:
                                continue
                        folder_list.append(file_fullpath)
                else:
                    if len(exclude_name) > 1:
                        if exclude_name in file_name:
                            continue
                    folder_list.append(file_fullpath)

    return folder_list


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
    parser.add_argument('--specific_name', type=str, default="",
                        help='only write mesh with this specific name to list, like \'manifold_full\' for \'manifold_full_123.obj\'')
    parser.add_argument('--exclude_name', type=str, default="",
                        help='exclude mesh with this specific name to list, like \'manifold_full\' for \'manifold_full_123.obj\'')
    args = parser.parse_args()

    folder_name_list = []
    render_folder_list = args.render_folders
    print(render_folder_list)
    for render_folder in render_folder_list:
        folder_list, _ = check_all_mesh_file_in_folder(
            render_folder, name=args.specific_name, exclude_name=args.exclude_name)
        folder_name_list = folder_name_list + folder_list
        for root, dirs, files in os.walk(render_folder):
            for dir in dirs:
                print("Check subfolder in folder %s" %
                      (os.path.join(root, dir)))
                folder_list, _ = check_all_mesh_file_in_folder(os.path.join(
                    root, dir), name=args.specific_name, exclude_name=args.exclude_name)
                folder_name_list = folder_name_list + folder_list
    print(args.output_list)
    write_list(args.output_list, folder_name_list)
