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
    parser.add_argument('--input_txt_path', type=str,
                        help='input list txt path')
    parser.add_argument('--output_txt_path', type=str,
                        help='output list txt path')
    parser.add_argument('--start_index', type=int,
                        help='start index of the list')
    parser.add_argument('--end_index', type=int,
                        help='end index of the list')
    args = parser.parse_args()

    original_txt_list = read_list(args.txt_path)
    split_txt_list = original_txt_list[args.start_index:args.end_index]
    write_list(args.output_txt_path, split_txt_list)
