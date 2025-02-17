import sys

import torch
import trimesh
import os
import pathlib
import torch.nn as nn
BCE = nn.BCELoss()
import open3d as o3d

import torch.optim as optim

import  yaml
from easydict import EasyDict as edict

# from utils.benchmark_utils import setup_seed

import numpy as np

# sys.path.append("./")
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .DeformationPyramid.nets import Deformation_Pyramid
from .DeformationPyramid.loss import compute_truncated_chamfer_distance
import argparse

# setup_seed(0)

config = {
    "gpu_mode": True,
    "iters": 500,
    "lr": 0.01,
    "max_break_count": 15,
    "break_threshold_ratio": 0.001,
    "samples": 6000,
    "motion_type": "Sim3",
    "rotation_format": "euler",
    "m": 9,
    "k0": -8,
    "depth": 3,
    "width": 128,
    "act_fn": "relu",
    "w_reg": 0,
    "w_ldmk": 0,
    "w_cd": 0.1
}


class Deformer():

    def __init__(self, config):
        self.config = config
        if self.config.gpu_mode:
            self.config.device = torch.cuda.current_device()
        else:
            self.config.device = torch.device('cpu')


        # if landmarks is not None:
        #     s_ldmk , t_ldmk = landmarks
        #     self.landmarks = (s_ldmk, t_ldmk)
        # else:
        #     self.landmarks = None


    def train_field(self, src_pcd, tgt_pcd, landmarks=None, cancel_translation=False):
        """
        Args:
            src_pcd: [n,3] numpy array
            tgt_pcd: [n,3] numpy array
        Returns:
        """

        config = self.config



        src_pcd, tgt_pcd = map(lambda x: torch.from_numpy(x).to(config.device), [src_pcd, tgt_pcd])


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
        if cancel_translation:
            src_mean = src_pcd.mean(dim=0, keepdims=True)
            tgt_mean = tgt_pcd.mean(dim=0, keepdims=True)
        else:
            src_mean = 0.
            tgt_mean = 0.


        src_pcd = src_pcd - src_mean
        tgt_pcd = tgt_pcd - tgt_mean

        self.src_mean = src_mean
        self.tgt_mean = tgt_mean


        s_sample = src_pcd
        t_sample = tgt_pcd


        if landmarks is not None:
            src_ldmk = landmarks[0] - src_mean
            tgt_ldmk = landmarks[1] - tgt_mean




        for level in range(NDP.n_hierarchy):

            """freeze non-optimized level"""
            NDP.gradient_setup(optimized_level=level)

            optimizer = optim.Adam(NDP.pyramid[level].parameters(), lr=config.lr)

            break_counter = 0
            loss_prev = 1e+6


            # inpcd = torch.concatenate( [s_sample, x0_plane] , dim=0 )

            """optimize current level"""
            for iter in range(config.iters):

                # s_sample_warped, data = NDP.warp(s_sample, max_level=level, min_level=level)
                # inpcd_warped, data = NDP.warp(inpcd, max_level=level, min_level=level)

                if landmarks is not None:

                    src_pts = torch.cat([src_ldmk, s_sample])
                    warped_pts, data = NDP.warp(src_pts, max_level=level, min_level=level)
                    warped_ldmk = warped_pts[: len(src_ldmk)]
                    s_sample_warped = warped_pts[len(src_ldmk):]
                    loss_ldmk = torch.mean(torch.sum((warped_ldmk - tgt_ldmk) ** 2, dim=-1))
                    loss_CD = compute_truncated_chamfer_distance(s_sample_warped[None], t_sample[None],
                                                                 trunc=config.trunc_cd)

                    loss = loss_ldmk + config.w_cd * loss_CD
                else :
                    s_sample_warped, data = NDP.warp(s_sample, max_level=level, min_level=level)
                    # chamfer distance
                    loss_chamfer = compute_truncated_chamfer_distance(s_sample_warped[None], t_sample[None], trunc=1e+9)
                    loss_ldmk = torch.tensor(0)
                    loss = loss_chamfer

                print(
                    "level-", level ,
                    f"L={loss.item():.4f}, "
                    f"Lcham={loss_ldmk.item():.4f}, "
                    f"Lx0={loss_ldmk.item():.4f}"
                )

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
            # use warped points for next level
            if landmarks is not None:
                src_ldmk = warped_ldmk.detach()

            s_sample = s_sample_warped.detach()

        self.NDP = NDP


    def warp_points(self, points):
        '''
        Args:
            points: [n,3] array
        Returns:
            points: [n,3] numpy array

        '''

        NDP = self.NDP
        config = self.config

        NDP.gradient_setup(optimized_level=-1)

        if isinstance(points, np.ndarray) :
            mesh_vert = torch.from_numpy(np.asarray(points, dtype=np.float32)).to(config.device)
        elif  isinstance(points, torch.Tensor):
            mesh_vert = points.to(config.device)
        else :
            raise  NotImplementedError()

        mesh_vert = mesh_vert - self.src_mean
        warped_vert, data = NDP.warp(mesh_vert)
        warped_vert = warped_vert + self.tgt_mean
        # warped_vert = warped_vert.detach().cpu().numpy()

        return warped_vert

        # src_mesh.vertices = o3d.utility.Vector3dVector(warped_vert)
        # o3d.visualization.draw_geometries([src_mesh])



class Deformer_Mirror ( ):

    def __init__(self, config):
        self.config = config
        # if self.config.gpu_mode:
        #     self.config.device = torch.cuda.current_device()
        # else:
        #     self.config.device = torch.device('cpu')


    def flip_and_average(self, s_sample_warped, nsample):
        # flip point and do average
        left_warped = s_sample_warped[:nsample]
        left_warped_mirror = left_warped.clone()
        left_warped_mirror[:, 0] = left_warped_mirror[:, 0] * -1
        right_warped = s_sample_warped[nsample:]
        right_warped_mirror = right_warped.clone()
        right_warped_mirror[:, 0] = right_warped_mirror[:, 0] * -1
        left_warped = (left_warped + right_warped_mirror) / 2.0
        right_warped = (right_warped + left_warped_mirror) / 2.0
        s_sample_warped = torch.cat([left_warped, right_warped], dim=0)
        return s_sample_warped

    def train_field(self, src_pcd, tgt_pcd, landmarks=None, cancel_translation=False):
        """
        Args:
            src_pcd: [n,3] numpy array
            tgt_pcd: [n,3] numpy array
        Returns:
        """

        config = self.config

        src_pcd, tgt_pcd = map(lambda x: torch.from_numpy(x).to(config.device), [src_pcd, tgt_pcd])

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
        if cancel_translation:
            src_mean = src_pcd.mean(dim=0, keepdims=True)
            tgt_mean = tgt_pcd.mean(dim=0, keepdims=True)
        else:
            src_mean = 0.
            tgt_mean = 0.



        src_pcd = src_pcd - src_mean
        tgt_pcd = tgt_pcd - tgt_mean

        self.src_mean = src_mean
        self.tgt_mean = tgt_mean


        s_sample = src_pcd
        t_sample = tgt_pcd


        if landmarks is not None:
            src_ldmk = landmarks[0] - src_mean
            tgt_ldmk = landmarks[1] - tgt_mean

            # flip landmark
            # src_ldmk_mirror = src_ldmk.clone()
            # tgt_ldmk_mirror = tgt_ldmk.clone()
            # src_ldmk_mirror[:, 0] = src_ldmk_mirror[:, 0] * -1
            # tgt_ldmk_mirror[:, 0] = tgt_ldmk_mirror[:, 0] * -1
            # src_ldmk = torch.cat( [ src_ldmk, src_ldmk_mirror], dim=0 )
            # tgt_ldmk = torch.cat( [ tgt_ldmk, tgt_ldmk_mirror], dim=0 )



        nsample = len(src_pcd)//2
        assert nsample*2 == len(src_pcd)

        for level in range(NDP.n_hierarchy):

            """freeze non-optimized level"""
            NDP.gradient_setup(optimized_level=level)

            optimizer = optim.Adam(NDP.pyramid[level].parameters(), lr=config.lr)

            break_counter = 0
            loss_prev = 1e+6


            # inpcd = torch.concatenate( [s_sample, x0_plane] , dim=0 )

            """optimize current level"""
            for iter in range(config.iters):

                # s_sample_warped, data = NDP.warp(s_sample, max_level=level, min_level=level)
                # inpcd_warped, data = NDP.warp(inpcd, max_level=level, min_level=level)

                if landmarks is not None:

                    src_pts = torch.cat([src_ldmk, s_sample])
                    warped_pts, data = NDP.warp(src_pts, max_level=level, min_level=level)
                    warped_ldmk = warped_pts[: len(src_ldmk)]

                    s_sample_warped = warped_pts[len(src_ldmk):]

                    # flip point and do average
                    s_sample_warped = self.flip_and_average( s_sample_warped, nsample)

                    loss_ldmk = torch.mean(torch.sum((warped_ldmk - tgt_ldmk) ** 2, dim=-1))
                    loss_CD = compute_truncated_chamfer_distance(s_sample_warped[None], t_sample[None],
                                                                 trunc=config.trunc_cd)

                    loss = loss_ldmk + config.w_cd * loss_CD
                else :
                    s_sample_warped, data = NDP.warp(s_sample, max_level=level, min_level=level)
                    s_sample_warped = self.flip_and_average( s_sample_warped, nsample)
                    # chamfer distance
                    loss_chamfer = compute_truncated_chamfer_distance(s_sample_warped[None], t_sample[None], trunc=1e+9)
                    loss_ldmk = torch.tensor(0)
                    loss = loss_chamfer

                # print(
                #     "level-", level ,
                #     f"L={loss.item():.4f}, "
                #     f"Lcham={loss_ldmk.item():.4f}, "
                #     f"Lx0={loss_ldmk.item():.4f}"
                # )

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
            # use warped points for next level
            if landmarks is not None:
                src_ldmk = warped_ldmk.detach()

            s_sample = s_sample_warped.detach()

        self.NDP = NDP


    def warp_points(self, points):
        '''
        Args:
            points: [n,3] array
        Returns:
            points: [n,3] numpy array

        '''

        NDP = self.NDP
        config = self.config

        NDP.gradient_setup(optimized_level=-1)

        if isinstance(points, np.ndarray) :
            mesh_vert = torch.from_numpy(np.asarray(points, dtype=np.float32)).to(config.device)
        elif  isinstance(points, torch.Tensor):
            mesh_vert = points.to(config.device)
        else :
            raise  NotImplementedError()

        mesh_vert = mesh_vert - self.src_mean

        n_verts = len(mesh_vert)

        mesh_vert_flip = mesh_vert.clone()
        mesh_vert_flip[:,0] = mesh_vert_flip[:,0] * -1
        mesh_vert = torch.concatenate ( [ mesh_vert, mesh_vert_flip], dim=0 )


        for level in range(NDP.n_hierarchy):
            mesh_vert, data = NDP.warp(mesh_vert, max_level=level, min_level=level)
            mesh_vert = self.flip_and_average(mesh_vert, n_verts)

        warped_vert = mesh_vert[: n_verts] + self.tgt_mean

        # warped_vert, data = NDP.warp(mesh_vert)
        # warped_vert = warped_vert
        # warped_vert = warped_vert.detach().cpu().numpy()

        return warped_vert

        # src_mesh.vertices = o3d.utility.Vector3dVector(warped_vert)
        # o3d.visualization.draw_geometries([src_mesh])

