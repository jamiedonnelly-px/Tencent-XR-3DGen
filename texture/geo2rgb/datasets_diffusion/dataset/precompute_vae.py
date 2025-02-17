import json
import torch
from tqdm import tqdm
from .dataset import MVLRMDataset
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import h5py
import os
from torch.utils.data import DataLoader
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from dotmap import DotMap
import yaml
from pdb import set_trace as st
import math
import sys
from omegaconf import OmegaConf
import argparse


class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, dset: MVLRMDataset, total, id):
        self.dset = dset
        self.start_idx = math.floor(id/total * len(self.dset))
        self.end_idx = min(math.ceil((id+1)/total * len(self.dset)), len(self.dset))
    def __len__(self):
        return self.end_idx - self.start_idx
    def get_vae(self):
        return self.dset.get_vae()
    def __getitem__(self, i):
        obj_idx = i+self.start_idx
        colors = []
        albedos = []
        normals = []
        xyzs = []
        view_ids = []
        
        for camsys_idx in range(len(self.dset.cameras_config)):
            color, albedo, normal, xyz, depth, tex_points, tex_colors, tex_normals, \
                sdf_points, sdf_sdfs, geo_points, geo_normals, K, c2w, o2w, n2w, \
                ref_view_id, src_view_id, dataset_name, obj_id = self.dset._parse_item(obj_idx, camsys_idx, None, images_only=True)

            colors.append(color.half())
            albedos.append(albedo.half())
            normals.append(normal.half())
            xyzs.append(xyz.half())
            view_ids.extend(ref_view_id + src_view_id)
            
        view_ids_inv = np.argsort(view_ids)
            
        colors = torch.cat(colors)[view_ids_inv]
        albedos = torch.cat(albedos)[view_ids_inv]
        normals = torch.cat(normals)[view_ids_inv]
        xyzs = torch.cat(xyzs)[view_ids_inv]
        
        return {
            "color": colors[:,:3], 
            "albedo": albedos[:,:3], 
            "normal": normals[:,:3], 
            "xyz": xyzs[:,:3], 
            "dataset_name": dataset_name, 
            "obj_id": obj_id,
        }


@torch.no_grad()
def encode(vae, vae_processor, rgb, chunk_size=32):
    '''
    - inputs: rgb [B,3,H,W]
    - ouputs: rgb [B,4,H,W]
    '''
    rgb_in_torch = rgb.to(torch.float16).cuda() * 2 - 1

    if chunk_size is None or chunk_size >= rgb.shape[0]:
        latent_dist = vae.encode(rgb_in_torch.cuda()).latent_dist
        mean, logvar = latent_dist.mean.cpu().numpy(), latent_dist.logvar.cpu().numpy()
    
    else:
        mean = []
        logvar = []

        for rgb_chunk in torch.split(rgb_in_torch, chunk_size):
            latent_dist = vae.encode(rgb_chunk.cuda()).latent_dist
            mean.append(latent_dist.mean)
            logvar.append(latent_dist.logvar)

        mean = torch.cat(mean).cpu().numpy()
        logvar = torch.cat(logvar).cpu().numpy()

    return mean, logvar

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, help="dataset yaml config file")
    parser.add_argument("--n_ranks", type=int, help="total number of GPUs for processing, world rank", default=1)
    parser.add_argument("--rank", type=int, help="local rank; you need to run this script with every local rank from 0 to (n_gpus-1)", default=0)
    parser.add_argument("--n_load_workers", type=int, help="total number of loading processes", default=8)
    parser.add_argument("--n_save_workers", type=int, help="total number of saving processes", default=4)
    args = parser.parse_args()

    splits = args.n_ranks
    split_id = args.rank

    n_load_workers = args.n_load_workers
    n_save_workers = args.n_save_workers

    FLAGS = OmegaConf.load(args.yaml)
    h5dir = FLAGS.vae_latent_dir

    os.makedirs(h5dir, exist_ok=True)


    dataset = SplitDataset(MVLRMDataset(FLAGS), splits, split_id)
    vae = dataset.get_vae().cuda().to(torch.float16)
    vae.eval()
    vae_processor = VaeImageProcessor()

    dataset = DataLoader(dataset, shuffle=False, num_workers=n_load_workers, pin_memory=True)

    data_len = len(dataset)
    
    print(data_len)

    def to_h5(data):
        
        (dset_name, obj_id, color_mean, color_logvar, albedo_mean, albedo_logvar, xyz_mean, xyz_logvar, normal_mean, normal_logvar) = data

        path = os.path.join(h5dir, dset_name, f"{obj_id}.h5")
        os.makedirs(os.path.join(h5dir, dset_name), exist_ok=True)
        
        vae_res = color_mean.shape[-1]

        with h5py.File(path, "w") as h5file:

            render_chunk_size = (1, 4, vae_res, vae_res) 
            h5file.create_dataset('color_mean', data=color_mean, compression='gzip', compression_opts=5, chunks=render_chunk_size, dtype=np.float16)
            h5file.create_dataset('color_logvar', data=color_logvar, compression='gzip', compression_opts=5, chunks=render_chunk_size, dtype=np.float16)
            h5file.create_dataset('albedo_mean', data=albedo_mean, compression='gzip', compression_opts=5, chunks=render_chunk_size, dtype=np.float16)
            h5file.create_dataset('albedo_logvar', data=albedo_logvar, compression='gzip', compression_opts=5, chunks=render_chunk_size, dtype=np.float16)
            h5file.create_dataset('xyz_mean', data=xyz_mean, compression='gzip', compression_opts=5, chunks=render_chunk_size, dtype=np.float16)
            h5file.create_dataset('xyz_logvar', data=xyz_logvar, compression='gzip', compression_opts=5, chunks=render_chunk_size, dtype=np.float16)
            h5file.create_dataset('normal_mean', data=normal_mean, compression='gzip', compression_opts=5, chunks=render_chunk_size, dtype=np.float16)
            h5file.create_dataset('normal_logvar', data=normal_logvar, compression='gzip', compression_opts=5, chunks=render_chunk_size, dtype=np.float16)
        
        return path

    def vae_data_gen():

        for sample in tqdm(dataset):

            color = sample['color'][0]
            albedo = sample['albedo'][0]
            normal = sample['normal'][0]
            xyz = sample['xyz'][0]
            
            dataset_name = sample['dataset_name'][0]
            obj_id = sample['obj_id'][0]

            color_mean, color_logvar = encode(vae, vae_processor, color)
            albedo_mean, albedo_logvar = encode(vae, vae_processor, albedo)
            xyz_mean, xyz_logvar = encode(vae, vae_processor, xyz)
            normal_mean, normal_logvar = encode(vae, vae_processor, normal)

            yield (dataset_name, obj_id, color_mean, color_logvar, albedo_mean, albedo_logvar, xyz_mean, xyz_logvar, normal_mean, normal_logvar)


    if n_save_workers > 0:
        pool = Pool(n_save_workers)
        gen = pool.imap(to_h5, vae_data_gen())
    else:
        gen = map(to_h5, vae_data_gen())

    pbar = tqdm(gen, total=data_len)
    for i, path in enumerate(pbar):
        pbar.set_description(path)
        if i == data_len - 1:
            exit()

