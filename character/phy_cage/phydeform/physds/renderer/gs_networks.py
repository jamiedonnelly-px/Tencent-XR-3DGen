#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np 
from torch import nn
import os, pdb
from torch.utils.cpp_extension import load
from typing import NamedTuple
from plyfile import PlyData, PlyElement 
from simple_knn._C import distCUDA2
from renderer.general_utils import * 
from renderer.sh_utils import RGB2SH


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array



class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._sem = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._sem,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._sem,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_semantic(self):
        return self._sem
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        semantics = 0.5 * torch.ones((fused_point_cloud.shape[0], 2), dtype=torch.float, device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._sem = nn.Parameter(semantics.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._sem], 'lr': training_args.sem_lr, "name": "sem"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._sem.shape[1]):
            l.append('sem_{}'.format(i))
        return l

    def save_ply(self, path, i_sem=-1):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        semantics = self._sem.detach().cpu().numpy()

        if i_sem != -1:
            if i_sem >=0:
                _, sem_inds = torch.max(self.get_semantic, 1)
                sem_mask = sem_inds == i_sem
            elif i_sem == -3:
                gau_sem = self.get_semantic
                sem_mask = gau_sem[:, 0] != 1
            elif i_sem == -2:
                gau_sem = self.get_semantic
                sem_mask = gau_sem[:, 0] == gau_sem[:, 1]
            
            sem_mask = sem_mask.detach().cpu().numpy()

            xyz = xyz[sem_mask]
            normals = normals[sem_mask]
            f_dc = f_dc[sem_mask]
            f_rest = f_rest[sem_mask]
            opacities = opacities[sem_mask]
            scale = scale[sem_mask]
            rotation = rotation[sem_mask]
            semantics = semantics[sem_mask]

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, semantics), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        sem_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sem")]
        sem_names = sorted(sem_names, key = lambda x: int(x.split('_')[-1]))
        sems = np.zeros((xyz.shape[0], len(sem_names)))
        for idx, attr_name in enumerate(sem_names):
            sems[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._sem = nn.Parameter(torch.tensor(sems, dtype=torch.float, device="cuda").requires_grad_(True))
        
        self.max_radii2D = torch.zeros((features_extra.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((features_extra.shape[0], 1), device="cuda")
        self.denom = torch.zeros((features_extra.shape[0], 1), device="cuda")

        self.active_sh_degree = self.max_sh_degree
        return self 





    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._sem = optimizable_tensors["sem"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_sem):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "sem": new_sem,}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._sem = optimizable_tensors["sem"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_sem = self._sem[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_sem)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_sem = self._sem[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_sem)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def prune(self, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


from renderer.phyoptimizer import PhyAdam
from submodules.mpm_engine.mpm_solver import MPMSolver
import taichi as ti

# constants
sim_res = 64
infilling_voxel_res = 128
GROUND_Y = 0.05
support_per_particles = 20
youngs_modulus_scale = 0.005 # larger Young’s modulus E indicates higher stiffness
poisson_ratio = 0.48 # a larger poission ratio ν leads to better volume preservation
max_surface_particles = 10000
material_type = MPMSolver.material_elastic

ti.init(arch=ti.cuda, device_memory_GB=8.0)





# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim


class SDFNetwork(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, n_layers, skip_in=(4,), multires=0, bias=0.5,
                 scale=1, geometric_init=True, weight_norm=True, inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs, need_midf=False):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
            
            if need_midf and l==6:
                midf = x.clone()

        if not need_midf:
            return x
        else:
            return x, midf
            # return x, x[..., 1:]

    def sdf(self, x):
        return self.forward(x)[..., :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def sdf_normal(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return y[..., :1].detach(), gradients.detach()




# refer to https://github.com/buaacyw/GaussianEditor/blob/9249f847c57036266a0bfb3a210681f94264ede9/gaussiansplatting/scene/hier_gaussian_model.py
class HierGaussian3DModel(GaussianModel):

    def __init__(self, sh_degree : int, oname: str):
        super().__init__(sh_degree=sh_degree)
        self.target_xyz = torch.empty(0)
        self.target_features_dc = torch.empty(0)
        self.target_features_rest = torch.empty(0)
        self.target_scaling = torch.empty(0)
        self.target_rotation = torch.empty(0)
        self.target_opacity = torch.empty(0)
        self.target_sem = torch.empty(0)

        self.bkg_xyz = torch.empty(0)
        self.bkg_features_dc = torch.empty(0)
        self.bkg_features_rest = torch.empty(0)
        self.bkg_scaling = torch.empty(0)
        self.bkg_rotation = torch.empty(0)
        self.bkg_opacity = torch.empty(0)
        self.bkg_sem = torch.empty(0)
        self.oname = oname

        self.sdf_network = SDFNetwork(d_out=257, d_in=3, d_hidden=256, n_layers=8, skip_in=[4], multires=6, bias=0.5, scale=1.0, geometric_init=True, weight_norm=True).cuda()
        # ckpt = torch.load("../part123/output/neus_frog_sweater1/frog_sweater/ckpt/last.ckpt", map_location='cuda')['state_dict']
        ckpt = torch.load(f"../../imgs23D/imgs2neus/output/{oname}/ckpt/last.ckpt", map_location='cuda')['state_dict']
        init_state = self.sdf_network.state_dict()
        for k in init_state.keys():
            init_state[k] = ckpt['renderer.sdf_network.'+k]
        self.sdf_network.load_state_dict(init_state)
        print("load sdf network")
    

    def query_sdf(self, xyz):
        return self.sdf_network.sdf(xyz)
    
    # direction of - -> +
    def query_gradients(self, x):
        x.requires_grad_(True)
        y = self.sdf_network.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=False,
            retain_graph=False,
            only_inputs=True)[0]
        return gradients.detach()


    @property
    def get_target_scaling(self):
        return self.scaling_activation(self.target_scaling) #.clamp(max=1)
    
    @property
    def get_target_rotation(self):
        return self.rotation_activation(self.target_rotation)
    
    @property
    def get_target_xyz(self):
        return self.target_xyz
    
    @property
    def get_target_features(self):
        features_dc = self.target_features_dc
        features_rest = self.target_features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_target_opacity(self):
        return self.opacity_activation(self.target_opacity)
    
    @property
    def get_target_semantic(self):
        return self.target_sem
    

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_target_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_target_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self.target_xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self.target_features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self.target_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            # {'params': [self.target_opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self.target_scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self.target_rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            # {'params': [self.target_sem], 'lr': training_args.sem_lr, "name": "sem"},
        ]

        self.optimizer = PhyAdam(l, eps=1e-15)
        # self.optimizer = torch.optim.Adam(l, eps=1e-15)
        # pdb.set_trace()
        self.update_mpm()

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_mpm(self):
        self.optimizer.mpm = MPMSolver(
            res=(sim_res, sim_res, sim_res), size=1, max_num_particles=2 ** 19,
            E_scale=youngs_modulus_scale, poisson_ratio=poisson_ratio, unbounded=True
        )
        self.optimizer.mpm.set_gravity((0, 0, 0))

        self.update_whole_scene_property()

        particles = self.target_xyz.detach().cpu().numpy().copy()
        N = particles.shape[0]

        # get idx of rigid particles
        rigid_idx = np.zeros(N, dtype=bool)
        # sems = self.target_sem.detach().cpu().numpy().copy()
        # mask_inpainted = sems[:,0] == sems[:,1]
        # sems_ind = sems.argmax(1)
        # sems_ind[mask_inpainted] = -1
        # rigid_idx = sems_ind != -1

        rigid_flag = rigid_idx.astype(np.int32)

        sdf_dir = f"../../imgs23D/mesh2gs/output/{self.oname}/"
        proxy_sdf = np.load(os.path.join(sdf_dir, "proxy_sdf.npy"))[1:-2,1:-2,1:-2]
        proxy_grad = np.load(os.path.join(sdf_dir, "proxy_grad.npy"))[1:-2,1:-2,1:-2]
        shift_constant = np.load(os.path.join(sdf_dir, "shift_constant.npy"))
        longest_side = np.load(os.path.join(sdf_dir, "longest_side.npy"))

        particles = particles / longest_side + shift_constant

        self.optimizer.mpm.add_sdf_network(proxy_sdf=proxy_sdf, proxy_grad=proxy_grad, coeff=1e-1)
        # self.optimizer.mpm.add_sdf_network(proxy_sdf=proxy_sdf, proxy_grad=proxy_grad, coeff=5e-2)

        self.optimizer.mpm.longest_side = longest_side
        self.optimizer.mpm.shift_constant = shift_constant

        self.optimizer.mpm.add_particles(particles=particles,
            material=material_type,
            color=0xFFFF00, motion_override_flag_arr=rigid_flag
        )


    def update_whole_scene_property(self):
        self._xyz =torch.cat([self.bkg_xyz, self.target_xyz], dim=0)
        self._features_dc=torch.cat([self.bkg_features_dc, self.target_features_dc], dim=0)
        self._features_rest=torch.cat([self.bkg_features_rest, self.target_features_rest], dim=0)
        self._opacity=torch.cat([self.bkg_opacity, self.target_opacity], dim=0)
        self._scaling=torch.cat([self.bkg_scaling, self.target_scaling], dim=0)
        self._rotation=torch.cat([self.bkg_rotation, self.target_rotation], dim=0)
        self._sem=torch.cat([self.bkg_sem, self.target_sem], dim=0)



    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_target_opacity, torch.ones_like(self.get_target_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self.target_opacity = optimizable_tensors["opacity"]
        self.update_whole_scene_property()


    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self.target_xyz = optimizable_tensors["xyz"]
        self.target_features_dc = optimizable_tensors["f_dc"]
        self.target_features_rest = optimizable_tensors["f_rest"]
        self.target_opacity = optimizable_tensors["opacity"]
        self.target_scaling = optimizable_tensors["scaling"]
        self.target_rotation = optimizable_tensors["rotation"]
        self.target_sem = optimizable_tensors["sem"]
        self.update_whole_scene_property()

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_sem):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "sem": new_sem,}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self.target_xyz = optimizable_tensors["xyz"]
        self.target_features_dc = optimizable_tensors["f_dc"]
        self.target_features_rest = optimizable_tensors["f_rest"]
        self.target_opacity = optimizable_tensors["opacity"]
        self.target_scaling = optimizable_tensors["scaling"]
        self.target_rotation = optimizable_tensors["rotation"]
        self.target_sem = optimizable_tensors["sem"]
        self.update_whole_scene_property()

        self.xyz_gradient_accum = torch.zeros((self.get_target_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_target_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_target_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_target_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_target_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_target_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.target_rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_target_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_target_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self.target_rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self.target_features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self.target_features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self.target_opacity[selected_pts_mask].repeat(N,1)
        new_sem = self.target_sem[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_sem)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_target_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self.target_xyz[selected_pts_mask]
        new_features_dc = self.target_features_dc[selected_pts_mask]
        new_features_rest = self.target_features_rest[selected_pts_mask]
        new_opacities = self.target_opacity[selected_pts_mask]
        new_scaling = self.target_scaling[selected_pts_mask]
        new_rotation = self.target_rotation[selected_pts_mask]
        new_sem = self.target_sem[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_sem)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        before = self.target_xyz.shape[0]
        self.densify_and_clone(grads, max_grad, extent)
        clone = self.get_target_xyz.shape[0]
        self.densify_and_split(grads, max_grad, extent)
        split = self.get_target_xyz.shape[0]

        prune_mask = (self.get_target_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_target_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        prune = self.get_target_xyz.shape[0]
        print(f"before: {before} - clone: {clone} - split: {split} - prune: {prune} ")

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, target_N):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[-target_N:][update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    def load_ply_bkg(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        sem_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sem")]
        sem_names = sorted(sem_names, key = lambda x: int(x.split('_')[-1]))
        sems = np.zeros((xyz.shape[0], len(sem_names)))
        for idx, attr_name in enumerate(sem_names):
            sems[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        self.bkg_xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda")).requires_grad_(False)
        self.bkg_features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()).requires_grad_(False)
        self.bkg_features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()).requires_grad_(False)
        self.bkg_opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda")).requires_grad_(False)
        self.bkg_scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda")).requires_grad_(False)
        self.bkg_rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda")).requires_grad_(False)
        self.bkg_sem = nn.Parameter(torch.tensor(sems, dtype=torch.float, device="cuda")).requires_grad_(False)

        self.active_sh_degree = self.max_sh_degree
        return self 

    def load_ply_bkg_target(self, path):
        self.spatial_lr_scale = 1.0
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        sem_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sem")]
        sem_names = sorted(sem_names, key = lambda x: int(x.split('_')[-1]))
        sems = np.zeros((xyz.shape[0], len(sem_names)))
        for idx, attr_name in enumerate(sem_names):
            sems[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        m = sems[:,0] == sems[:,1]
        
        self.bkg_xyz = nn.Parameter(torch.tensor(xyz[~m], dtype=torch.float, device="cuda")).requires_grad_(False)
        self.bkg_features_dc = nn.Parameter(torch.tensor(features_dc[~m], dtype=torch.float, device="cuda").transpose(1, 2).contiguous()).requires_grad_(False)
        self.bkg_features_rest = nn.Parameter(torch.tensor(features_extra[~m], dtype=torch.float, device="cuda").transpose(1, 2).contiguous()).requires_grad_(False)
        self.bkg_opacity = nn.Parameter(torch.tensor(opacities[~m], dtype=torch.float, device="cuda")).requires_grad_(False)
        self.bkg_scaling = nn.Parameter(torch.tensor(scales[~m], dtype=torch.float, device="cuda")).requires_grad_(False)
        self.bkg_rotation = nn.Parameter(torch.tensor(rots[~m], dtype=torch.float, device="cuda")).requires_grad_(False)
        self.bkg_sem = nn.Parameter(torch.tensor(sems[~m], dtype=torch.float, device="cuda")).requires_grad_(False)

        self.target_xyz = nn.Parameter(torch.tensor(xyz[m], dtype=torch.float, device="cuda")).requires_grad_(True)
        self.target_features_dc = nn.Parameter(torch.tensor(features_dc[m], dtype=torch.float, device="cuda").transpose(1, 2).contiguous()).requires_grad_(True)
        self.target_features_rest = nn.Parameter(torch.tensor(features_extra[m], dtype=torch.float, device="cuda").transpose(1, 2).contiguous()).requires_grad_(True)
        self.target_opacity = nn.Parameter(torch.tensor(opacities[m], dtype=torch.float, device="cuda")).requires_grad_(True)
        self.target_scaling = nn.Parameter(torch.tensor(scales[m], dtype=torch.float, device="cuda")).requires_grad_(True)
        self.target_rotation = nn.Parameter(torch.tensor(rots[m], dtype=torch.float, device="cuda")).requires_grad_(True)
        self.target_sem = nn.Parameter(torch.tensor(sems[m], dtype=torch.float, device="cuda")).requires_grad_(True)

        self.active_sh_degree = self.max_sh_degree
        return self 

    def load_ply_target(self, path, load_color=False):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        sem_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sem")]
        sem_names = sorted(sem_names, key = lambda x: int(x.split('_')[-1]))
        sems = np.zeros((xyz.shape[0], len(sem_names)))
        for idx, attr_name in enumerate(sem_names):
            sems[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        self.target_opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))

        
        if not load_color:
            fused_color = torch.tensor(np.random.random((xyz.shape[0], 3)) / 255.0).float().cuda()
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
            self.target_features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self.target_features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
            # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(xyz).float().cuda()), 0.0000001)
            # scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
            # rots = torch.rand((xyz.shape[0], 4), device="cuda")
            # # opacities = self.inverse_opacity_activation(0.1 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))
            # # self.target_opacity = nn.Parameter(opacities.requires_grad_(True))
            # self.target_scaling = nn.Parameter(scales.requires_grad_(True))
            # self.target_rotation = nn.Parameter(rots.requires_grad_(True))
        else:
            self.target_features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self.target_features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.target_scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self.target_rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        

        self.target_xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        
        self.target_sem = nn.Parameter(torch.tensor(sems, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.get_target_xyz.shape[0]), device="cuda")
        return self 

    
    def create_from_pcd_bkg(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        semantics = 0.5 * torch.ones((fused_point_cloud.shape[0], 2), dtype=torch.float, device="cuda")

        self.bkg_xyz = nn.Parameter(fused_point_cloud).requires_grad_(False)
        self.bkg_features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous()).requires_grad_(False)
        self.bkg_features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous()).requires_grad_(False)
        self.bkg_scaling = nn.Parameter(scales).requires_grad_(False)
        self.bkg_rotation = nn.Parameter(rots).requires_grad_(False)
        self.bkg_opacity = nn.Parameter(opacities).requires_grad_(False)
        self.bkg_sem = nn.Parameter(semantics).requires_grad_(False)



    def create_from_pcd_target(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        semantics = 0.5 * torch.ones((fused_point_cloud.shape[0], 2), dtype=torch.float, device="cuda")

        self.target_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self.target_features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self.target_features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.target_scaling = nn.Parameter(scales.requires_grad_(True))
        self.target_rotation = nn.Parameter(rots.requires_grad_(True))
        self.target_opacity = nn.Parameter(opacities.requires_grad_(True))
        self.target_sem = nn.Parameter(semantics.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_target_xyz.shape[0]), device="cuda")
        

    def save_ply_bkg(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self.bkg_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.bkg_features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.bkg_features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.bkg_opacity.detach().cpu().numpy()
        scale = self.bkg_scaling.detach().cpu().numpy()
        rotation = self.bkg_rotation.detach().cpu().numpy()
        semantics = self.bkg_sem.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, semantics), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def save_ply_target(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self.target_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.target_features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.target_features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.target_opacity.detach().cpu().numpy()
        scale = self.target_scaling.detach().cpu().numpy()
        rotation = self.target_rotation.detach().cpu().numpy()
        semantics = self.target_sem.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, semantics), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)