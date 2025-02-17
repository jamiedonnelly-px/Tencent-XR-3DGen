import torch
import torch.nn as nn
BCE = nn.BCELoss()
import open3d as o3d

import torch.optim as optim

import  yaml
from easydict import EasyDict as edict


import numpy as np

import copy


from .nets import Deformation_Pyramid
from .loss import compute_truncated_chamfer_distance




def train_cloth_deformer(src_naked_mesh, tgt_naked_mesh):
    # transfer cloth mesh from source to target


    config = {
        "gpu_mode": True,

        "iters": 500,
        "lr": 0.005,
        "max_break_count": 15,
        "break_threshold_ratio": 0.0005,

        "samples": 2000,

        "motion_type":  "Sim3",
        "rotation_format": "euler",

        "m": 10,
        "k0": -8,
        "depth": 3,
        "width": 128,
        "act_fn": "relu",

        "w_reg": 0,
        "w_ldmk": 0,
        "w_cd": 0.1
    }

    config = edict(config)

    if config.gpu_mode:
        config.device = torch.cuda.current_device()
    else:
        config.device = torch.device('cpu')



    """ Src, sample pts"""

    # pcd1 =  src_naked_mesh.sample_points_uniformly(number_of_points=config.samples)
    # pcd1.paint_uniform_color([0, 0.706, 1])
    # src_pcd = np.asarray(pcd1.points, dtype=np.float32)

    # o3d.visualization.draw_geometries([src_naked_mesh])
    # o3d.io.write_triangle_mesh( "/home/rabbityl/workspace/bodyfit/data/hok_example/src_naked_mesh.ply", src_naked_mesh )

    """ Tgt, sample pts"""
    # pcd2 =  tgt_naked_mesh.sample_points_uniformly(number_of_points=config.samples)
    # tgt_pcd = np.asarray(pcd2.points, dtype=np.float32)

    # o3d.visualization.draw_geometries([tgt_naked_mesh])
    # o3d.io.write_triangle_mesh( "/home/rabbityl/workspace/bodyfit/data/hok_example/tgt_naked_mesh.ply", tgt_naked_mesh )


    """load data"""
    # src_pcd, tgt_pcd = map( lambda x: torch.from_numpy(x).to(config.device), [src_pcd, tgt_pcd ] )


    # use landmark
    src_ldmks = torch.from_numpy(np.asarray(src_naked_mesh.vertices, dtype=np.float32)).to(config.device)
    tgt_ldmks = torch.from_numpy(np.asarray(tgt_naked_mesh.vertices, dtype=np.float32)).to(config.device)



    """construct model"""
    NDP = Deformation_Pyramid(depth=config.depth,
                              width=config.width,
                              device=config.device,
                              k0=config.k0,
                              m=config.m,
                              nonrigidity_est=config.w_reg > 0,
                              rotation_format=config.rotation_format,
                              motion=config.motion_type)



    """cancel global translation"""
    # src_mean = src_pcd.mean(dim=0, keepdims=True)
    # tgt_mean = tgt_pcd.mean(dim=0, keepdims=True)
    # src_pcd = src_pcd - src_mean
    # tgt_pcd = tgt_pcd - tgt_mean


    src_mean = src_ldmks.mean(dim=0, keepdims=True)
    tgt_mean = tgt_ldmks.mean(dim=0, keepdims=True)
    src_ldmks = src_ldmks - src_mean
    tgt_ldmks = tgt_ldmks - tgt_mean


    # s_sample = src_pcd
    # t_sample = tgt_pcd

    nsample = len(src_ldmks)


    perm = torch.randperm(src_ldmks.shape[0])
    s_sample_ldmk = src_ldmks[perm[:nsample]]
    t_sample_ldmk = tgt_ldmks[perm[:nsample]]


    for level in range(NDP.n_hierarchy):

        """freeze non-optimized level"""
        NDP.gradient_setup(optimized_level=level)

        optimizer = optim.Adam(NDP.pyramid[level].parameters(), lr=config.lr)

        break_counter = 0
        loss_prev = 1e+6

        """optimize current level"""
        for iter in range(config.iters):



            #ldmk loss
            s_ldmk_warped, data = NDP.warp(s_sample_ldmk, max_level=level, min_level=level)
            loss = torch.mean(torch.sum((s_ldmk_warped - t_sample_ldmk) ** 2, dim=-1))



            if level > 0 and config.w_reg > 0:
                nonrigidity = data[level][1]
                target = torch.zeros_like(nonrigidity)
                reg_loss = BCE(nonrigidity, target)
                loss = loss + config.w_reg * reg_loss


            # early stop
            if loss.item() < 1e-4:
                break
            if abs(loss_prev - loss.item()) < loss_prev * config.break_threshold_ratio:
                break_counter += 1
            if break_counter >= config.max_break_count:
                break
            loss_prev = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # use warped points for next level
        s_sample_ldmk = s_ldmk_warped.detach()



    """warp-original mesh verttices"""
    NDP.gradient_setup(optimized_level=-1)
    mesh_vert = torch.from_numpy(np.asarray(src_naked_mesh.vertices, dtype=np.float32)).to(config.device)
    mesh_vert = mesh_vert - src_mean
    warped_vert, data = NDP.warp(mesh_vert)
    warped_vert = warped_vert + src_mean
    warped_vert = warped_vert.detach().cpu().numpy()

    src_mesh_wrapped = copy.deepcopy(src_naked_mesh)

    src_mesh_wrapped.vertices = o3d.utility.Vector3dVector(warped_vert)
    src_mesh_wrapped.paint_uniform_color([0.5, 0.8, 0])
    # src_naked_mesh.translate([0,2,0])
    # src_mesh_wrapped.translate([-1.5,2,0])
    #
    # tgt_naked_mesh.translate([1.5,2,0])


    # o3d.visualization.draw_geometries([src_naked_mesh, tgt_naked_mesh])

    return NDP, src_mean, tgt_mean


    '''wrap cloth'''
    if False:
        for k,v in cloth_verts.items():
            mesh_vert = torch.from_numpy(np.asarray(v, dtype=np.float32)).to(config.device)
            mesh_vert = mesh_vert - src_mean
            warped_vert, data = NDP.warp(mesh_vert)
            warped_vert = warped_vert + tgt_mean
            warped_vert = warped_vert.detach().cpu().numpy()
            cloth_verts[k] = warped_vert


        # o3d.visualization.draw_geometries([src_naked_mesh, tgt_naked_mesh, src_mesh_wrapped, cloth_mesh, cloth_mesh_deform])


        return cloth_verts



