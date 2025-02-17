import argparse
import ipaddress
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
    parser = argparse.ArgumentParser(description='Merge a list of ip adresses')
    parser.add_argument('--input_ip_list_path', type=str,
                        help='a list of ip address')
    parser.add_argument('--output_ip_list_path', type=str,
                        help='output list of merged ip')
    args = parser.parse_args()

    input_ip_list_path = args.input_ip_list_path
    output_ip_list_path = args.output_ip_list_path

    ip_list = read_list(input_ip_list_path)
    merged_ip_list = []

    result = ipaddress.collapse_addresses([ipaddress.ip_network(ip) for ip in ip_list])
    for network in result:
        merged_ip_list.append(str(network))

    write_list(output_ip_list_path, merged_ip_list)
