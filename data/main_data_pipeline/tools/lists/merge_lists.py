import argparse
import os


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find difference between two lists')
    parser.add_argument('--txt_folder', type=str, default="",
                        help='folder of a group of txt files')
    parser.add_argument('--txt_files', nargs='+',
                        help='abspath of txt files')
    parser.add_argument('--output_txt_path', type=str,
                        help='merged txt list path')
    args = parser.parse_args()

    txt_folder = args.txt_folder
    output_txt_path = args.output_txt_path
    if len(txt_folder) > 1:
        filenames = os.listdir(txt_folder)
        txt_total_datas = []
        for filename in filenames:
            file_fullpath = os.path.join(txt_folder, filename)
            if os.path.isdir(file_fullpath):
                if "pod_" in filename:
                    pod_folder_txt_fullpath = os.path.join(
                        file_fullpath, "success.txt")
                    txt_info = read_list(pod_folder_txt_fullpath)
                    txt_total_datas = txt_total_datas + txt_info
            # else:
            #     file_extention = os.path.splitext(filename)[1]
            #     if file_extention.lower() == ".txt":
            #         txt_fullpath = os.path.join(txt_folder, filename)
            #         txt_info = read_list(txt_fullpath)
            #         txt_total_datas = txt_total_datas+txt_info
    else:
        txt_files = args.txt_files
        txt_total_datas = []
        for txt_fullpath in txt_files:
            txt_info = read_list(txt_fullpath)
            txt_total_datas = txt_total_datas + txt_info

    write_list(output_txt_path, txt_total_datas)
