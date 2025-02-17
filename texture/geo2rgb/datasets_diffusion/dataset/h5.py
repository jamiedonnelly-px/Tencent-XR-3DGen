import h5py
import json
import numpy as np
import cv2
import os
from .util import image_16bitc1_to_8bitc2, image_8bitc2_to_16bitc1
from multiprocessing import Pool
from tqdm import tqdm
import argparse
from pdb import set_trace as st

def load_images(dir, image_names):
    imgs = []
    for img_name in image_names:
        path = os.path.join(dir, img_name + ".png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 3:
            if img.shape[2] == 4:  # If the image has 4 channels
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 2:
            img = img[...,None] # append channel dimension
        imgs.append(img)
    return np.stack(imgs, axis=0)

def check_h5(h5_path, try_open=False):
    exists = os.path.exists(h5_path)
    if exists and try_open:
        try:
            with h5py.File(h5_path, 'r') as file:
                return len(file.keys()) > 0
        except:
            return False
    return exists


def parse_cam_params(cam_data, resolution=None):
    '''
    returns: c2o as 4x4 camera to object matrix
    intrinsic as 3x4 matrix representing either pinhole or orthographic camera
    '''
    c2o = np.array(cam_data['pose'])
    if cam_data.get("model", "pinhole") in ("pinhole", "perspective"):
        intrinsic = np.zeros(3,4)
        intrinsic[:3,:3] = np.array(cam_data['k'])
    elif cam_data.get("model", "pinhole") == "orthogarphic":
        assert resolution is not None
        scale = cam_data["scale"]
        intrinsic = np.array([
            [resolution/scale, 0, 0, resolution/2],
            [0, resolution/scale, 0, resolution/2],
            [0, 0, 0, 1.0],
        ])
    else:
        raise ValueError(f"unrecognised cambera model {cam_data.get('model')}")
    
    return c2o, intrinsic
    
def dump_h5(inputs):
    
    in_json, view_chunk, images_only, pcd_only, no_overwrite, remove_pcd_npy, try_open = inputs

    geo_pcd_dir = in_json["GeoPcd"]
    tex_pcd_dir = in_json["TexPcd"]
    image_path = in_json["ImgDir"]
    
    if no_overwrite:
        pcd_done = (not pcd_only) or check_h5(os.path.join(geo_pcd_dir, "pcd.h5"), try_open) and check_h5(os.path.join(tex_pcd_dir, "pcd.h5"), try_open)
        h5_done = ((not images_only) and pcd_only) or check_h5(in_json["h5"], try_open)

        if pcd_done and h5_done:
            return True
    
    if (not images_only) or pcd_only: # load pcd if required

        geo_points = np.load(os.path.join(geo_pcd_dir, "pcd_points_300000.npy")).astype(np.float16) # [n_points, 3]
        geo_normals = np.load(os.path.join(geo_pcd_dir, "pcd_normals_300000.npy")).astype(np.float16) # [n_points, 3]

        sdf_points = np.load(os.path.join(geo_pcd_dir, "sdf_points_100000.npy")).astype(np.float16) # [n_points, 3]
        sdf_sdfs = np.load(os.path.join(geo_pcd_dir, "sdf_sdfs_100000.npy")).astype(np.float16) # [n_points, 1]

        tex_points = np.load(os.path.join(tex_pcd_dir, "pcd_points_500000.npy")).astype(np.float16) # [n_points, 3]
        tex_normals = np.load(os.path.join(tex_pcd_dir, "pcd_tex_normals_500000.npy")).astype(np.float16) # [n_points, 3]
        tex_colors = np.load(os.path.join(tex_pcd_dir, "pcd_colors_500000.npy")).astype(np.float16) # [n_points, 3]
        
    if pcd_only:
        try:
            geo_h5_path = os.path.join(geo_pcd_dir, "pcd.h5")
            tex_h5_path = os.path.join(tex_pcd_dir, "pcd.h5")
            
            with h5py.File(geo_h5_path, "w") as h5file:
                geo_data = np.concatenate([geo_points, geo_normals], axis=-1).astype(np.float16) # [n_points, 6]
                h5file.create_dataset('geo', data=geo_data, compression='gzip', compression_opts=5, chunks=geo_data.shape)

                sdf_data = np.concatenate([sdf_points, sdf_sdfs.reshape(-1,1)], axis=-1).astype(np.float16) # [n_points, 4]
                h5file.create_dataset('sdf', data=sdf_data, compression='gzip', compression_opts=5, chunks=sdf_data.shape)
            
            with h5py.File(tex_h5_path, "w") as h5file:
                tex_data = np.concatenate([tex_points, tex_normals, tex_colors], axis=-1).astype(np.float16) # [n_points, 9]
                h5file.create_dataset('tex', data=tex_data, compression='gzip', compression_opts=5, chunks=tex_data.shape)
        except:
            return False
        
        if remove_pcd_npy:
            
            os.remove(os.path.join(geo_pcd_dir, "pcd_points_300000.npy"))
            os.remove(os.path.join(geo_pcd_dir, "pcd_normals_300000.npy"))
            
            os.remove(os.path.join(geo_pcd_dir, "sdf_points_100000.npy"))
            os.remove(os.path.join(geo_pcd_dir, "sdf_sdfs_100000.npy"))
            
            os.remove(os.path.join(tex_pcd_dir, "pcd_points_500000.npy"))
            os.remove(os.path.join(tex_pcd_dir, "pcd_tex_normals_500000.npy"))
            os.remove(os.path.join(tex_pcd_dir, "pcd_colors_500000.npy"))

        if not images_only: # skip packing images unless images_only is set
            return True

    with open(os.path.join(image_path, "cam_parameters.json"), "r") as f:
        cam_config = json.load(f)
    img_names = sorted(cam_config.keys())

    rgb = load_images(os.path.join(image_path, "color"), img_names).astype(np.uint8) # [n_views, h, w, 4]
    albedo = load_images(os.path.join(image_path, "emission", "color"), img_names) # [n_views, h, w, 4]
    depth = image_16bitc1_to_8bitc2(load_images(os.path.join(image_path, "depth"), img_names)) # [n_views, h, w, 2]
    normal = load_images(os.path.join(image_path, "normal"), img_names) # [n_views, h, w, 3]

    cam2world, intrinsic = [], []
    for img_name in img_names:
        c2w, intrn = parse_cam_params(cam_config[img_name], resolution=rgb.shape[1])
        cam2world.append(c2w)
        intrinsic.append(intrn)
        
    intrinsic = np.array(intrinsic).astype(np.float32) # [n_views, 3,4]
    cam2world = np.array(cam2world).astype(np.float32) # [n_views, 4,4]

    try:
        with h5py.File(in_json["h5"], "w") as h5file:


            render_data = np.concatenate([rgb, albedo, depth, normal], axis=-1) # [n_views, h, w, 11]
            h, w, c = render_data.shape[-3:]
            render_chunk_size = (view_chunk, h, w, c) 
            h5file.create_dataset('render', data=render_data, compression='gzip', compression_opts=5, chunks=render_chunk_size)

            h5file.create_dataset('intrinsic', data=intrinsic, chunks=(view_chunk,3,intrinsic.shape[-1]))
            h5file.create_dataset('cam2world', data=cam2world, chunks=(view_chunk,4,4))
            
            if (not images_only) and (not pcd_only): # pack pcd in same h5 as images
                geo_data = np.concatenate([geo_points, geo_normals], axis=-1) # [n_points, 6]
                h5file.create_dataset('geo', data=geo_data, compression='gzip', compression_opts=5, chunks=geo_data.shape)

                sdf_data = np.concatenate([sdf_points, sdf_sdfs.reshape(-1,1)], axis=-1) # [n_points, 4]
                h5file.create_dataset('sdf', data=sdf_data, compression='gzip', compression_opts=5, chunks=sdf_data.shape)

                tex_data = np.concatenate([tex_points, tex_normals, tex_colors], axis=-1) # [n_points, 9]
                h5file.create_dataset('tex', data=tex_data, compression='gzip', compression_opts=5, chunks=tex_data.shape)
                
    except:
        return False

    return True

def load_pcd_only(geo_h5_path, tex_h5_path):
    
    with h5py.File(geo_h5_path, 'r') as h5file:
        geo_data = np.array(h5file['geo']).astype(np.float32)
        sdf_data = np.array(h5file['sdf']).astype(np.float32)
    with h5py.File(tex_h5_path, 'r') as h5file:
        tex_data = np.array(h5file['tex']).astype(np.float32)
        
    geo_points, geo_normals = np.split(geo_data, [3], axis=-1)
    sdf_points, sdf_sdfs = np.split(sdf_data, [3], axis=-1)
    sdf_sdfs = sdf_sdfs.flatten()
    tex_points, tex_normals, tex_colors = np.split(tex_data, [3,6], axis=-1)
    
    return geo_points, geo_normals, sdf_points, sdf_sdfs, \
            tex_points, tex_normals, tex_colors

def load_h5(path, img_idxs, images_only=False):

    with h5py.File(path, 'r') as h5file:
        
        idx_argsort = np.argsort(img_idxs)
        img_idxs = [img_idxs[i] for i in idx_argsort]
        
        render_data = np.asarray(h5file['render'][img_idxs])
        intrinsic = np.asarray(h5file['intrinsic'][img_idxs])
        if intrinsic.shape[-1] == 3: # pinhole/perspective cam
            intrinsic = np.concatenate((intrinsic, np.zeros_like(intrinsic[...,:1])), axis=-1) # make 3x4
        cam2world = np.asarray(h5file['cam2world'][img_idxs])
        
        render_data[idx_argsort] = render_data.copy()
        intrinsic[idx_argsort] = intrinsic.copy()
        cam2world[idx_argsort] = cam2world.copy()

        if not images_only:
            
            try: # try loading pcd from h5
                geo_data = np.array(h5file['geo']).astype(np.float32)
                sdf_data = np.array(h5file['sdf']).astype(np.float32)
                tex_data = np.array(h5file['tex']).astype(np.float32)
            except:
                images_only = True # if errors out, only load images

    rgb, albedo, depth, normal = np.split(render_data, [4, 8, 10], axis=-1)
    depth = image_8bitc2_to_16bitc1(depth)
    
    if images_only:
        return rgb, albedo, depth, normal, intrinsic, cam2world, \
            None, None, None, None, None, None, None

    geo_points, geo_normals = np.split(geo_data, [3], axis=-1)
    sdf_points, sdf_sdfs = np.split(sdf_data, [3], axis=-1)
    sdf_sdfs = sdf_sdfs.flatten()
    tex_points, tex_normals, tex_colors = np.split(tex_data, [3,6], axis=-1)

    return rgb, albedo, depth, normal, intrinsic, cam2world, \
            geo_points, geo_normals, sdf_points, sdf_sdfs, \
            tex_points, tex_normals, tex_colors


def dataset_to_h5(in_json, out_json, parent_dirs, view_chunk, n_workers=16, images_only=False, pcd_only=False, no_overwrite=False, remove_pcd_npy=False, try_open=False):

    if isinstance(parent_dirs, str):
        parent_dirs = [parent_dirs]
    
    with open(in_json) as f:
        data_jsons = json.load(f)['data']

    for parent_dir in parent_dirs:
        for dset in data_jsons:
            os.makedirs(os.path.join(parent_dir, dset), exist_ok=True)


    for dset in data_jsons:

        print(f"processing '{dset}'")

        job_jsons = []
        job_ids = []

        data_json = data_jsons[dset]

        for i, data_obj in enumerate(data_json):
            parent_dir = os.path.join(parent_dirs[i % len(parent_dirs)], dset) # evenly distribute data to multiple folders if possible

            h5file = os.path.join(parent_dir, f"{data_obj}.h5")
            
            if "h5" not in data_json[data_obj]:
                data_json[data_obj]["h5"] = h5file

            job_ids.append(data_obj)
            job_jsons.append((data_json[data_obj], view_chunk, images_only, pcd_only, no_overwrite, remove_pcd_npy, try_open))
        
        if n_workers > 0:
            with Pool(n_workers) as pool:
                success = list(tqdm(pool.imap(dump_h5, job_jsons), total=len(job_jsons))) # multiprocessing
        else:
            success = list(tqdm(map(dump_h5, job_jsons), total=len(job_jsons))) # multiprocessing
        
        out_data_json = {}
        for job_succ, job_id, job_json in zip(success, job_ids, job_jsons):
            if job_succ:
                out_data_json[job_id] = job_json[0]

        data_jsons[dset] = out_data_json
        
    
    with open(out_json, "w+") as f:
        json.dump({'data': data_jsons}, f, indent=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="export dataset to h5 format.")
    parser.add_argument('--input', '-i', type=str, help="input json")
    parser.add_argument('--output', '-o', type=str, help="output json with h5 paths")
    parser.add_argument('--dirs', '-d', nargs='+', type=str, help="paths to save h5 files to")
    parser.add_argument('--chunk_size_n_views', type=int, default=1, help="chunk size of h5 files in view dimension, should be set to number of cameras in the multicam system")
    parser.add_argument('--n_workers', type=int, default=32, help="number of parallel workers")
    parser.add_argument('--images_only', action="store_true", help="to exclude point cloud data when packing")
    parser.add_argument('--pcd_only', action="store_true", help="to only include point cloud data when packing")
    parser.add_argument('--remove_pcd_npy', action="store_true", help="delete old pcd npy files. only applicable when --pcd_only is set")
    parser.add_argument('--no_overwrite', action="store_true", help="skip any h5 files that already exist.")
    parser.add_argument('--try_open', action="store_true", help="only applicable when no_overwrite; this will additionall try to open existing h5 and replace it if corrupted")


    args = parser.parse_args()

    dataset_to_h5(args.input, args.output, args.dirs, args.chunk_size_n_views, args.n_workers, args.images_only, args.pcd_only, args.no_overwrite, args.pcd_only and args.remove_pcd_npy, args.try_open)
    
    
    # python -m dataset.h5 -i /aigc_cfs_11/Asset/list/64view_mmd/fragment3_20240526/part3_120k_180k_color_only_with_point_cloud.json  -o /aigc_cfs_2/zacheng/datajsons/23May19_objaverse_part1_64views_fragment3_cfs.json -d /aigc_cfs_11/23May19_objaverse_64view /aigc_cfs_12/23May19_objaverse_64view --n_workers 12 --images_only
    # python -m dataset.h5 -i /aigc_cfs_11/Asset/list/64view_mmd/fragment4_20240526/part4_180k_240k_color_only_with_point_cloud.json  -o /aigc_cfs_2/zacheng/datajsons/23May19_objaverse_part1_64views_fragment4_cfs.json -d /aigc_cfs_11/23May19_objaverse_64view /aigc_cfs_12/23May19_objaverse_64view --n_workers 12  --images_only


    # python -m dataset.h5 -i /aigc_cfs_2/zacheng/datajsons/23May19_objaverse_part1_64views_fragment"$i"_cfs.json  -o /aigc_cfs_2/zacheng/datajsons/23May19_objaverse_part1_64views_fragment"$i"_cfs.json -d /aigc_cfs_12/23May19_objaverse_64view --n_workers 32 --pcd_only --no_overwrite
    
    
    # python -m dataset.h5 -i /aigc_cfs_11/Asset/list/64view_mmd/avatar_20240531/avatar_color_only_with_point_cloud.json  -o /aigc_cfs_2/zacheng/datajsons/23May19_avatar_part1_64views_fragment1_cfs.json -d /aigc_cfs_12/23May19_avatar_64view --n_workers 8  --images_only --pcd_only
    # python -m dataset.h5 -i /apdcephfs_cq8/share_2909871/Assets/clothes/mvd/render/mcwy/log/valid_color.json  -o /aigc_cfs_2/zacheng/datajsons/23May19_clothes_part1_64views_fragment1_cfs.json -d /aigc_cfs_11/23May19_clothes_64view --n_workers 8  --images_only --pcd_only
    
    # python -m dataset.h5 -i /aigc_cfs_2/zacheng/datajsons/23May19_objaverse_part1_64views_fragment6_cfs.json  -o /aigc_cfs_2/zacheng/datajsons/23May19_objaverse_part1_64views_fragment6_cfs.json -d /aigc_cfs_11/23May19_clothes_64view --n_workers 8  --images_only --pcd_only --try_open --no_overwrite
    
        









        



















