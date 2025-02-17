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
    parser.add_argument('--txt_path1', type=str,
                        help='first list file path')
    parser.add_argument('--txt_path2', type=str,
                        help='second list file path')
    parser.add_argument('--diff_txt', type=str,
                        help='diff file that equals txt1-txt2')
    parser.add_argument('--correct_txt1', type=str,
                        help='correct list of txt1')
    args = parser.parse_args()

    diff_txt = args.diff_txt

    txt_list1 = read_list(args.txt_path1)
    hash_list2 = read_list(args.txt_path2)
    hash_list1 = []
    glb_folder = os.path.split(txt_list1[0])[0]
    if "/apdcephfs_cq3/share_1615605" in glb_folder:
        glb_folder = glb_folder.replace("/apdcephfs_cq3/share_1615605",
                                        "/apdcephfs_cq8/share_1615605/jp_cq3_cephfs")

    correct_txt_list = []
    for glb_path in correct_txt_list:
        if "/apdcephfs_cq3/share_1615605" in glb_path:
            glb_path = glb_path.replace("/apdcephfs_cq3/share_1615605",
                                        "/apdcephfs_cq8/share_1615605/jp_cq3_cephfs")
        glb_path = glb_path + ".glb"
        correct_txt_list.append(glb_path)

    for obj_name in txt_list1:
        obj_hash_id = os.path.split(obj_name)[1]
        hash_list1.append(obj_hash_id)

    hash_set1 = set(hash_list1)
    hash_set2 = set(hash_list2)

    not_processed_hash = hash_set1.difference(hash_set2)
    no_hash_list = list(not_processed_hash)
    output_mesh_path_list = [os.path.join(
        glb_folder, hash_id + ".glb") for hash_id in no_hash_list]
    write_list(diff_txt, output_mesh_path_list)
