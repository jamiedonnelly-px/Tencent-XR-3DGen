import os
import shutil
import imageio
import glob
import numpy as np
import math
import cv2
import argparse

def copy_folders_to_root(root_dir, folder_list):
    cp_dirs = ['opt_mesh']
    cp_files = ['output.gif', 'debug_condi.jpg']
    os.makedirs(root_dir, exist_ok=True)
    
    gif_dir = os.path.join(root_dir, 'merge_gif')
    os.makedirs(gif_dir, exist_ok=True)
    target_gif_list = []
    for folder_path in folder_list:
        oid = os.path.basename(folder_path)
        oid_out_dir = os.path.join(root_dir, oid)
        os.makedirs(oid_out_dir, exist_ok=True)
        
        for cp_file in cp_files:
            item_path = os.path.join(folder_path, cp_file)
            target_path = os.path.join(oid_out_dir, cp_file)
            if not os.path.exists(item_path) or os.path.exists(target_path):
                continue
            shutil.copy(item_path, target_path)
            if '.gif' in cp_file:
                target_gif = os.path.join(gif_dir, f'{oid}.gif')
                shutil.copy(item_path, target_gif)
                target_gif_list.append(target_gif)
        for cp_dir in cp_dirs:
            item_path = os.path.join(folder_path, cp_dir)
            target_path = os.path.join(oid_out_dir, cp_dir)
            if not os.path.exists(item_path) or os.path.exists(target_path):
                continue            
            shutil.copytree(item_path, target_path)
        
        else:
            print(f"Folder {folder_path} does not exist, skipping.")
    return target_gif_list

def ensure_rgb(image):
    print('debug shape ', image.shape)
    if image.shape[2] == 3:
        return image
    elif image.shape[2] >= 4:
        return image[:, :, :3]
    else:
        raise ValueError(f"Unsupported number of channels: {image.shape[2]}")

def concat_images(images, size=512):
    # 计算行和列的数量
    num_images = len(images)
    num_rows = num_cols = int(math.sqrt(num_images))
    
    # 确保图像列表的长度是一个完全平方数
    assert num_rows * num_cols == num_images, "The length of images must be a perfect square."

    if size:
        images = [cv2.resize(image, (size, size)) for image in images]
    # 将数组拆分成num_rows行，每行num_cols个数组
    rows = [images[i*num_cols:(i+1)*num_cols] for i in range(num_rows)]

    # 先将每行的数组拼接在一起
    rows = [np.concatenate(row, axis=1) for row in rows]

    # 然后将所有行拼接在一起
    result = np.concatenate(rows, axis=0)
    
    return result

def vis_gif_list(gif_files, out_gif):
    if not gif_files:
        return
    gifs = [imageio.get_reader(gif_file) for gif_file in gif_files]

    num_frames = len(gifs[0])
    # num_frames = min(len(gif) for gif in gifs)

    new_gif = imageio.get_writer(out_gif, fps=20)

    for i in range(num_frames):
        frames = [ensure_rgb(gif.get_data(i)) for gif in gifs]
        grid = concat_images(frames, 512)
        # frames = [gif.get_data(i) for gif in gifs]
        # grid = np.concatenate(frames, axis=1)
        # grid = np.block([frames[0:2]])
        print('debug frames ', len(frames), frames[0].shape)
        print('debug grid ', grid.shape)
        # grid = np.block([[frames[0:2]], [frames[2:4]], [frames[4:6]], [frames[6:8]]])
        # grid = np.block([[frames[0:4]], [frames[4:8]], [frames[8:12]], [frames[12:16]]])
        
        new_gif.append_data(grid)

    for gif in gifs:
        gif.close()
    new_gif.close()    
    return


def merge_objs_results(in_dir, root_dir, id_list_txt = None):
    if id_list_txt is not None:
        id_dirs = [os.path.join(in_dir, line.strip()) for line in open(id_list_txt, "r").readlines()]
    else:
        opt_mesh_dirs = glob.glob(os.path.join(in_dir, '*/opt_mesh'))
        id_dirs = [os.path.dirname(opt_mesh_dir) for opt_mesh_dir in opt_mesh_dirs]

    if not id_dirs:
        print('can not find id_dirs ')
        return
    
    target_gif_list = copy_folders_to_root(root_dir, id_dirs)
    n_all = len(target_gif_list)
    
    n_sam = int(math.sqrt(n_all)) ** 2
    print(f'sample/target_gif_list {n_sam}/{n_all}')
    sample_gif_list = target_gif_list[:n_sam]
    
    out_gif = os.path.join(root_dir, 'merge_all.gif')
    vis_gif_list(sample_gif_list, out_gif)

    condi_paths = [os.path.join(root_dir, os.path.basename(gif_file).split('.gif')[0], 'debug_condi.jpg') for gif_file in sample_gif_list]
    condi_nps = [cv2.imread(condi_path)  for condi_path in condi_paths]
    cv2.imwrite(os.path.join(root_dir, 'merge_condi.jpg'), concat_images(condi_nps, 512))

# id_list_txt = '/aigc_cfs_3/sz/result/tex_creator/obja_gtD_srender_argum/g8/first_2k_b16a2_nsddpm/for_k1/select.txt'

def main():
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_dir', type=str, default="/aigc_cfs_3/sz/result/tex_creator/obja_gtD_srender_argum/g8/first_2k_b16a2_nsddpm/for_k1/objaverse")
    parser.add_argument('root_dir', type=str, default="/aigc_cfs_3/sz/result/tex_creator/obja_gtD_srender_argum/g8/first_2k_b16a2_nsddpm/for_k1/merge_all1")
    parser.add_argument('--id_list_txt', type=str)
    args = parser.parse_args()

    merge_objs_results(args.in_dir, args.root_dir, args.id_list_txt)

if __name__ == "__main__":
    main()
