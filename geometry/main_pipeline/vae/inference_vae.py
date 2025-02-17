import argparse
import os
import json
import time
import trimesh
import torch
import numpy as np
from glob import glob
from huggingface_hub import hf_hub_download
import h5py

import vae
from vae.systems.base import BaseSystem
from vae.utils.config import ExperimentConfig, load_config
import open3d as o3d
import random

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./eval_data", help="Path to the input data",)
    parser.add_argument("--input_mode", type=str, default="z123", help="Path to the input data",)
    parser.add_argument("--output", type=str, default="./eval_outputs", help="Path to the inference results",)
    ############## model and mv model ##############
    parser.add_argument("--model", type=str, default="", help="Path to the image-to-shape diffusion model",)
    parser.add_argument("--mv_model", type=str, default="CRM", help="Path to the multi-view images model",)
    ############## inference ##############
    parser.add_argument("--seed", type=int, default=4, help="Random seed for generating multi-view images",)
    parser.add_argument("--guidance_scale_2D", type=float, default=3, help="Guidance scale for generating multi-view images",)
    parser.add_argument("--step_2D", type=int, default=50, help="Number of steps for generating multi-view images",)
    parser.add_argument("--remesh", type=bool, default=False, help="Remesh the output mesh",)
    parser.add_argument("--target_face_count", type=int, default=2000, help="Target face count for remeshing",)
    parser.add_argument("--guidance_scale_3D", type=float, default=3, help="Guidance scale for 3D reconstruction",)
    parser.add_argument("--step_3D", type=int, default=50, help="Number of steps for 3D reconstruction",)
    parser.add_argument("--octree_depth", type=int, default=7, help="Octree depth for 3D reconstruction",)
    parser.add_argument("--mc_method", type=str, default='old', help="matching cube method, [old, sparse]",)
    ############## data preprocess ##############
    parser.add_argument("--no_rmbg", type=bool, default=False, help="Do NOT remove the background",)
    parser.add_argument("--rm_type", type=str, default="rembg", choices=["rembg", "sam"], help="Type of background removal",)
    parser.add_argument("--bkgd_type", type=str, default="Remove", choices=["Alpha as mask", "Remove", "Original"], help="Type of background",)
    parser.add_argument("--bkgd_color", type=str, default=(127,127,127,255), help="Background color",)
    parser.add_argument("--fg_ratio", type=float, default=1.0, help="Foreground ratio",)
    parser.add_argument("--front_view", type=str, default="", help="Front view of the object",)
    parser.add_argument("--right_view", type=str, default="", help="Right view of the object",)
    parser.add_argument("--back_view", type=str, default="", help="Back view of the object",)
    parser.add_argument("--left_view", type=str, default="", help="Left view of the object",)
    parser.add_argument("--invisible_view_index", action='append', help="view that canno be seen", default=[])
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    time_str_name = time.strftime('%Y-%m-%d-%H:%M:%S')
    args.output = os.path.join(args.output, time_str_name)
    os.makedirs(args.output, exist_ok=True)
    
    # load the shape diffusion model
    if args.model == "":
        ckpt_path = hf_hub_download(repo_id="wyysf/vae", filename="image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae/model.ckpt", repo_type="model")
        config_path = hf_hub_download(repo_id="wyysf/vae", filename="image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae/config.yaml", repo_type="model")
    else:
        ckpt_path = f"{args.model}/model.ckpt"
        config_path = f"{args.model}/config.yaml"   
         
    cfg = load_config(config_path)
    system: BaseSystem = vae.find(cfg.system_type)(
        cfg.system, 
    )
    
    system.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict'])
    model = system.to(device).eval()
        

    
    h5_dir = args.input
    
    with h5py.File(h5_dir, 'r') as f:
        surface = np.array(f["surface_points"])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(surface*2)
        o3d.io.write_point_cloud(os.path.join(args.output, '{}_points_50w.ply'.format(key_name)), pcd)
        
        normal = np.array(f["surface_normals"])
    
    surface = np.concatenate([surface, normal], axis=1)
    
    scale = 1
    noise_sigma = 0
    sample_points = 4096
    
    surface[:, :3] = surface[:, :3] * scale # target scale
    # add noise to input point cloud
    surface[:, :3] += (np.random.rand(surface.shape[0], 3) * 2 - 1) * noise_sigma

    surface = torch.from_numpy(surface).unsqueeze(0).to(device)
    shape_latents, kl_embed, posterior =  model.shape_model.encode(surface)
    
    # save decodeed results:
    kl_embed_truncated = kl_embed

    latents = model.shape_model.decode(kl_embed_truncated)
    
    box_v = 1.1
    mesh_outputs, _ = model.shape_model.extract_geometry(latents,    
        bounds=[-box_v, -box_v, -box_v, box_v, box_v, box_v],
        octree_depth=args.octree_depth,
        method=args.mc_method)

    mesh = trimesh.Trimesh(mesh_outputs[0][0], mesh_outputs[0][1])
    mesh.export(os.path.join(args.output, '{}.obj'.format('vae_res')), include_normals=True)


    

    
    
