from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import copy
import os
import os.path as osp
import time
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import torch.nn.functional as F
import sys
from ipdb import set_trace as st
import open3d as o3d
import numpy as np
from os import listdir
import math
import json
import argparse
from omegaconf import OmegaConf
import trimesh
from tqdm import tqdm
from matplotlib import pyplot as plt
import networkx as nx
import copy
import numpy as np
import cv2
import subprocess

from utils import *
from models.model import LinNF, VAE
from models.point_transformer_partseg import *

class Fitter:
    @torch.no_grad()
    def __init__(self, args=None):
        self.device = device = torch.device('cuda')
        self.dtype = dtype = torch.float32
        
        """load configs"""
        if args is not None:
            self.__dict__.update(vars(args))
            cfg = OmegaConf.load(args.config)
            self.cfg = cfg
            self.__dict__.update(cfg)

        """ load template """
        self.load_template()

        # """ load pretrained vae or nf """
        # if len(self.prior_dir) > 0:
        #     self.load_prior()

        # """ define parameters """
        # self.global_orient = torch.eye(3, dtype=dtype, device=device)[:2].reshape(-1,6) # 6D rotation representation
        # self.transl = torch.zeros([1,3], dtype=dtype, device=device)
        # self.body_pose = matrix_to_rotation_6d(torch.eye(3)[None].repeat([self.num_joint,1,1])).to(device=device,dtype=dtype)
        # self.log_scale = torch.zeros([self.num_joint, 3], dtype=dtype, device=device)
        
    @torch.no_grad()
    def load_template(self):
        template_dir = self.template_dir
        print(f"load template from {template_dir}")
        
        with open(template_dir + '/config.json','r') as f:
            animal_config = json.load(f)
        self.animal_config = animal_config

        animal_model = np.load(template_dir + '/model.npz')
        self.animal_model = animal_model
        animation_data = np.load(template_dir + '/animation.npz')
        self.animation_data = animation_data

        pose_data = []
        for k, v in animation_data.items():
            if k.endswith('_trans'):
                continue
            pose_data.append(v)
        pose_data = np.concatenate(pose_data, axis=0)
        pose_data_torch = torch.from_numpy(pose_data).to(dtype=self.dtype)
        self.pose_data_torch = pose_data_torch.clone()
        print(f'total number of poses {pose_data.shape[0]}')

        self.joint_name = self.animal_model['joint_name']
        self.parents_name = self.animal_model['parents_name']
        self.faces = self.animal_model['faces']
        v_tmp = self.animal_model['v_template']
        weights = self.animal_model['weights']
        joints = self.animal_model['joints']
        parents = self.animal_model['parents']
        self.v_tmp = torch.from_numpy(v_tmp).to(device=self.device,dtype=self.dtype) # [n, 3]
        self.weights = torch.from_numpy(weights).to(device=self.device,dtype=self.dtype) #[vn, jn]
        self.joints = torch.from_numpy(joints).to(device=self.device,dtype=self.dtype) #[jn 3]
        self.parents = parents.tolist()
        self.template_size = torch.norm(self.v_tmp.max(dim=0)[0] - self.v_tmp.min(dim=0)[0]).item()
        self.num_joint = joints.shape[0]

        self.root_joints = self.animal_config['root_joints'] # assume root joints are always the first few joints
        self.left_joints = np.array(self.animal_config['left_joints'])
        self.right_joints = np.array(self.animal_config['right_joints'])
        parts_tmp = self.animal_config['parts']
        parts = {}
        for k,v in parts_tmp.items():
            parts[k] = np.array(v) #- len(self.root_joints)
        self.parts = parts_tmp
        self.other_joints = np.arange(len(self.root_joints), self.num_joint)

        # get part idx
        parts_tmp = self.animal_config['parts']
        n_parts = len(list(parts_tmp.keys()))
        n_vert = v_tmp.shape[0]
        max_j_idx = np.argmax(weights,axis=1) #[vn, jn]
        part2vid = {}
        for k, v in parts_tmp.items():
            if k not in part2vid.keys():
                part2vid[k] = []
            for jid in v:
               part2vid[k] = part2vid[k] + np.where(max_j_idx == jid)[0].tolist()
        self.part2vid = part2vid

        """load point feature model"""
        self.load_point_feature_model()
        # mesh = trimesh.Trimesh(vertices=v_tmp, faces=self.faces)
        self.v_feat, self.v_part_est, self.template_point, self.template_point_feat, self.template_part = self.get_vert_feat(v_tmp, self.faces) #[vn, feat]

        self.reset_pose()

    def load_point_feature_model(self):
        model_dir = self.animal_config["point_feature_model_config"]["model_dir"]
        model_name = self.animal_config["point_feature_model_config"]["model"]
        
        '''MODEL LOADING'''
        in_channels = 3
        out_feat = 128
        if model_name == 'PointTransformerSeg26':
            model = PointTransformerSeg26(in_channels=in_channels, num_classes=out_feat).cuda() # num_classes is the output channel
        elif model_name == 'PointTransformerSeg38':
            model = PointTransformerSeg38(in_channels=in_channels, num_classes=out_feat).cuda() # num_classes is the output channel
        elif model_name == 'PointTransformerSeg50':
            model = PointTransformerSeg50(in_channels=in_channels, num_classes=out_feat).cuda() # num_classes is the output channel
        model_partseg = torch.nn.Linear(in_features=out_feat, out_features=len(self.animal_config["parts"].keys()), device=self.device, dtype=self.dtype)

        print(f'load from {str(model_dir)}/checkpoints/best_model.pth')
        checkpoint = torch.load(str(model_dir) + '/checkpoints/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model_partseg.load_state_dict(checkpoint['model_partseg_state_dict'])
        self.point_feature_model = model
        self.point_feature_model_partseg = model_partseg
        self.point_feature_model.eval()
        self.point_feature_model_partseg.eval()
        
    @torch.no_grad()    
    # def get_vert_feat(self, mesh):
    def get_vert_feat(self, verts, faces):
        # verts = mesh.vertices
        mesh = trimesh.Trimesh(vertices=verts,faces=faces)
        points_orig = mesh.sample(self.animal_config["point_feature_model_config"]["npoints"])
        vs_max = verts.max(axis=0)[None]
        vs_min = verts.min(axis=0)[None]
        scale = 2/np.linalg.norm(vs_max - vs_min,axis=-1,keepdims=True)
        points = scale * (points_orig - (vs_max+vs_min)/2)
        verts = scale * (verts - (vs_max+vs_min)/2)
        points_torch = torch.from_numpy(points).to(device=self.device,dtype=self.dtype)
        verts = torch.from_numpy(verts).to(device=self.device,dtype=self.dtype)
        n = points_torch.shape[0]
        b = 1
        inp = {
            "coord": points_torch.reshape(n, 3),
            "feat": points_torch.reshape(n, -1),
            "offset": torch.cumsum(torch.ones(b).cuda()*n,dim=0)
        }
        point_feat = self.point_feature_model(inp)
        parts = self.point_feature_model_partseg(point_feat)
        parts = parts.max(dim=-1)[1]
        point_feat = F.normalize(point_feat, dim=-1)
        cdist = torch.cdist(points_torch, verts)
        v2pts_tmp = cdist.min(dim=0)[1]
        # pts_tmp2v = cdist.min(dim=1)[1]

        if False:
            part_col = np.array(part_colors)
            mesh_vis = o3d.geometry.PointCloud()
            mesh_vis.points = o3d.utility.Vector3dVector(points_torch.cpu().data.numpy())
            mesh_vis.colors = o3d.utility.Vector3dVector(part_col[parts.cpu().data.numpy()])
            o3d.io.write_point_cloud(f'{self.out_folder}/fitting/targ_part.ply', mesh_vis)

            mesh_vis = o3d.geometry.PointCloud()
            mesh_vis.points = o3d.utility.Vector3dVector(verts.cpu().data.numpy())
            mesh_vis.colors = o3d.utility.Vector3dVector(part_col[parts[v2pts_tmp].cpu().data.numpy()])
            # mesh.triangles = o3d.utility.Vector3dVector(mesh.faces)
            o3d.io.write_point_cloud(f'{self.out_folder}/fitting/vert_part.ply', mesh_vis)

        return point_feat[v2pts_tmp].detach().clone(), parts[v2pts_tmp].detach().clone(), torch.from_numpy(points_orig).to(device=self.device,dtype=self.dtype), point_feat.detach(), parts.detach()

    # @torch.no_grad()
    # def load_prior(self):
    #     """load data and model"""
    #     prior_dir = self.prior_dir
    #     try:
    #         config = OmegaConf.load(prior_dir + '/vae.yaml')
    #     except:
    #         config = OmegaConf.load(prior_dir + '/nf.yaml')

    #     root_joints = self.animal_config['root_joints'] # assume root joints are always the first few joints
    #     pose_data_torch = self.pose_data_torch[:,len(root_joints):]

    #     n_data = pose_data_torch.shape[0]
    #     data_6d = matrix_to_rotation_6d(pose_data_torch).reshape([n_data,-1])
    #     data_6d = data_6d.reshape([n_data, -1])
    #     mean = torch.mean(data_6d, dim=0)
    #     std = torch.std(data_6d, dim=0)
    #     centered_data = data_6d - mean
    #     standardized_data = centered_data / std
    #     standardized_data = standardized_data.reshape([n_data,-1])
    #     self.mean = mean.to(device=self.device).detach()
    #     self.std = std.to(device=self.device).detach()
        
    #     # the first three joint are roots
    #     # assume using 6D rotation representation
    #     dn, jn = pose_data_torch.shape[0], pose_data_torch.shape[1]
    #     model_conf = config['model_conf']
    #     if model_conf['model_name'] == 'VAE':
    #         model = VAE(data_dim=jn*6, num_layer=model_conf['num_layer'], feat_dim=model_conf['feat_dim']).to(device=self.device, dtype=self.dtype)
    #     else:
    #         model = LinNF(data_dim=jn*6, num_layer=model_conf['num_layer'], with_prelu=model_conf['with_prelu']).to(device=self.device, dtype=self.dtype)
        
    #     state_dict = torch.load(f'{prior_dir}/latest.pth')
    #     model.load_state_dict(state_dict['model_state_dict'])
    #     model.eval()
    #     self.model = model
    #     # get data distribution
    #     xs = torch.split(standardized_data, 10000)
    #     zs = []
        
    #     for x in xs:
    #         x = x.to(device=self.device)
    #         if model_conf['model_name'] == 'VAE':
    #             z = model.encode(x)
    #         else:
    #             z, log_det_jacobian = model(x)
    #         zs.append(z)
    #     zs = torch.cat(zs,dim=0)
    #     mu = zs.mean(dim=0)
    #     zs_cen = zs - mu[None]
    #     cov = torch.mm(zs_cen.T, zs_cen) / (zs_cen.shape[0] - 1)

    #     # convert cov to positive definite
    #     epsilon = 1e-5
    #     eigvals, eigvecs = torch.linalg.eigh(cov)
    #     eigvals = torch.clamp(eigvals, min=epsilon)
    #     cov_new = (eigvecs * eigvals).mm(eigvecs.t())
    #     self.prior = torch.distributions.MultivariateNormal(loc=mu.detach(), covariance_matrix=cov_new.detach())
    #     if model_conf['model_name'] == 'VAE':
    #         self.prior_func = lambda x: (self.prior.log_prob(self.model.encode((x-self.mean)/self.std))).sum(dim=-1)
    #     else:
    #         print('to implement')
        
    #     self.zs = zs

    def reset_pose(self):
        self.global_orient = torch.eye(3, dtype=self.dtype, device=self.device)[:2].reshape(-1,6) # 6D rotation representation
        self.transl = torch.zeros([1,3], dtype=self.dtype, device=self.device)
        self.body_pose = matrix_to_rotation_6d(torch.eye(3)[None].repeat([self.num_joint,1,1])).to(device=self.device,dtype=self.dtype)
        self.log_scale = torch.zeros([self.num_joint, 3], dtype=self.dtype, device=self.device)

    def update_target_mesh(self, mesh_dir):
        self.target_mesh_dir = mesh_dir
        mesh_name = os.path.basename(mesh_dir).split('.')[0]
        print('add:', mesh_dir)
        self.target_mesh_name = mesh_name
        out_folder = f'./output/{mesh_name}' if len(self.out_folder)  <= 0 else self.out_folder
        os.makedirs(out_folder, exist_ok=True)
        os.makedirs(out_folder + '/fitting', exist_ok=True)
        os.makedirs(out_folder + '/rigging', exist_ok=True)
        os.makedirs(out_folder + '/animation', exist_ok=True)
        
        self.out_folder = out_folder
        self.load_target()
        self.reset_pose()

    @torch.no_grad()
    def load_target(self):
        mesh_dir = self.target_mesh_dir
        print(f"load target from {mesh_dir}")
        self.target_mesh = target_mesh = trimesh.load(mesh_dir)
        verts = np.array(self.target_mesh.vertices)
        max_v = verts.max(axis=0)
        min_v = verts.min(axis=0)
        scale = self.template_size/np.linalg.norm(max_v - min_v)
        verts = (verts - (max_v + min_v)[None]/2) * scale
        target_mesh.vertices = verts
        target_mesh.export(f'{self.out_folder}/fitting/target_mesh.obj')
        self.target_verts = torch.from_numpy(verts).to(device=self.device, dtype=self.dtype) #[vn, 3]
        self.target_edges_same, same_vs = add_edge_with_thre(verts, self.device, thres=0.001)
        self.same_vs = same_vs
        self.target_edges = get_edges_from_faces(target_mesh.faces)
        
        # update neighbors
        neighbors_dict = get_neighbour_from_faces(self.target_mesh.faces)
        # _, same_vs = add_edge_with_thre(target_verts.cpu().data.numpy(), self.device, thres=0.001)
        neighbors_dict_new = copy.deepcopy(neighbors_dict)
        for vs in same_vs:
            tmp = []
            for vi in vs:
                tmp += neighbors_dict[vi]
            for vi in vs:
                neighbors_dict_new[vi] = tmp
        self.neighbors_dict = neighbors_dict_new
        self.target_edges_new = get_edges_from_neighbors(neighbors_dict_new) # including the edge for same vertices

        mesh_vis = o3d.geometry.TriangleMesh()
        mesh_vis.vertices = o3d.utility.Vector3dVector(target_mesh.vertices)
        mesh_vis.triangles = o3d.utility.Vector3iVector(target_mesh.faces)
        mesh_vis.vertex_colors = o3d.utility.Vector3dVector(np.ones_like(target_mesh.vertices) * 0.85)
        mesh_vis.compute_vertex_normals()
        o3d.io.write_triangle_mesh(f'{self.out_folder}/fitting/target_mesh_vis.obj',mesh_vis)

        
        self.target_v_feat, self.target_v_parts, self.target_points, self.target_point_feats, self.target_parts = self.get_vert_feat(self.target_mesh.vertices, self.target_mesh.faces) #[vn, feat]

        self.simi = (self.v_feat @ self.target_point_feats.transpose(1,0) + 1)/2 #[vn1, vn2] (0-1)
        
        parts_tmp = self.animal_config['parts']
        self.target_part2vid = {}
        pid = 0
        for k, v in parts_tmp.items():
            self.target_part2vid[k] = torch.where(self.target_v_parts==pid)[0].cpu().data.numpy().tolist()
            pid += 1

        self.target_part2pid = {}
        pid = 0
        for k, v in parts_tmp.items():
            self.target_part2pid[k] = torch.where(self.target_parts==pid)[0].cpu().data.numpy().tolist()
            pid += 1


        if self.is_debug:
            part_col = np.array(part_colors)
            mesh_vis = o3d.geometry.TriangleMesh()
            mesh_vis.vertices = o3d.utility.Vector3dVector(self.target_verts.cpu().data.numpy())
            mesh_vis.vertex_colors = o3d.utility.Vector3dVector(part_col[self.target_v_parts.cpu().data.numpy()])
            mesh_vis.triangles = o3d.utility.Vector3iVector(self.target_mesh.faces)
            o3d.io.write_triangle_mesh(f'{self.out_folder}/fitting/target_v_part.ply', mesh_vis)

            mesh_vis = o3d.geometry.PointCloud()
            mesh_vis.points = o3d.utility.Vector3dVector(self.target_points.cpu().data.numpy())
            mesh_vis.colors = o3d.utility.Vector3dVector(part_col[self.target_parts.cpu().data.numpy()])
            o3d.io.write_point_cloud(f'{self.out_folder}/fitting/target_p_part.ply', mesh_vis)
        
        # # filter the similarity
        # # a -> b -> a
        
        w1 = self.simi * (self.simi > 0.5)
        w1_0 = w1 / (w1.sum(dim=0,keepdims=True) + 1e-10)
        w1_1 = w1 / (w1.sum(dim=1,keepdims=True) + 1e-10)
        w2 = (w1_0 * 1000) * (w1_1 * 1000)
        self.w2 = w2 / (w2.max() + 1e-10) 
        
        # debug 
        if self.is_debug:
            simi = self.simi.cpu().data.numpy()
            # simi = self.w2.cpu().data.numpy()
            # to heatmap
            data_0_255 = np.uint8(255 * simi)
            heatmap = cv2.applyColorMap(data_0_255, cv2.COLORMAP_JET)
            heatmap = (cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0)

            v_src = self.target_points.cpu().data.numpy().reshape([-1,3])
            v_ref = self.v_tmp.cpu().data.numpy().reshape([-1,3])
            # ids = np.random.choice(np.arange(v_src.shape[0]),10)
            ids = np.arange(0, v_src.shape[0],v_src.shape[0]//9)
            for id in ids:
                mesh_source = o3d.geometry.PointCloud()
                mesh_source.points = o3d.utility.Vector3dVector(v_src)
                col = np.ones([v_src.shape[0], 3]) * 0.8
                col[id] = np.array([1.0,0,0])
                mesh_source.colors = o3d.utility.Vector3dVector(col)
                o3d.io.write_point_cloud(f'{self.out_folder}/fitting/{id:03d}_src.ply', mesh_source)

                mesh_ref = o3d.geometry.PointCloud()
                mesh_ref.points = o3d.utility.Vector3dVector(v_ref)
                # col = np.ones([v_ref.shape[0], 3]) * 0.8
                # col[simi[:,id]>0.4,:] = np.array([[0,0,1.0]])
                # col[simi[:,id]>0.6,:] = np.array([[0,1.0,0]])
                # col[simi[:,id]>0.8,:] = np.array([[1.0,0,0]])
                col = heatmap[:,id]
                mesh_ref.colors = o3d.utility.Vector3dVector(col)
                o3d.io.write_point_cloud(f'{self.out_folder}/fitting/{id:03d}_ref.ply', mesh_ref)

    def get_anchor_poses(self, zs, n_cluster):
        # zs [N, dim]
        
        id_remain = np.arange(zs.shape[0])
        id_selected = []
        for i in range(n_cluster):
            if i == 0:
                id_tmp = np.random.choice(id_remain,1)[0]
            else:
                cdist = torch.cdist(zs[id_remain], zs[id_selected]) #[nre,nsel]
                id_tmp = torch.argmax(cdist.min(dim=1)[0]).item()
            id_selected.append(id_remain[id_tmp])
            id_remain = np.setdiff1d(id_remain, id_selected)
        return id_selected, zs[id_selected]
    
    def update_template(self, template_dir):
        self.template_dir = template_dir
        self.load_template()
        if hasattr(self, "target_mesh_dir") and self.target_mesh_dir:
            self.update_target_mesh(self.target_mesh_dir)

    """ fitting """
    def stage0_v2(self):
        stime = time.time()
        stg = 0
        device = self.device
        dtype = self.dtype
        lr = self.lrs[stg]
        maxiters = self.maxiters[stg]
        global_rot = matrix_to_rotation_6d(torch.eye(3).to(device=self.device,dtype=self.dtype))
        global_rot = torch.rand_like(global_rot)
        global_rot = matrix_to_rotation_6d(rotation_6d_to_matrix(global_rot))
        para_glo_rot = nn.Parameter(global_rot[None]) # [1, 6]
        print(f"initial global rot {para_glo_rot}")
        
        para_glo_transl = nn.Parameter(self.transl.detach().clone()) # [1, 3]
        params = [para_glo_rot, para_glo_transl]
        optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999))
        logs = {}
        logs['chamfer'] = []
        progress_bar = tqdm(total=maxiters)
        
        fitting_order = self.animal_config["fitting_order"]["stage0"][0]
        template_vids = []
        target_pids = []
        for pk in fitting_order:
            template_vids += self.part2vid[pk]
            target_pids += self.target_part2pid[pk]

        for i in range(maxiters):
            optimizer.zero_grad()
            v_rot = self.v_tmp @ rotation_6d_to_matrix(para_glo_rot)[0].transpose(1, 0)
            v_rot = v_rot + para_glo_transl
            if i < maxiters//2:
                chamfer_dist = 0
                for pk in fitting_order:
                    # if "root" in pk or "leg" in pk or "spine" in pk or 'fin' in pk:
                    cdist = torch.cdist(v_rot[self.part2vid[pk]], self.target_points[self.target_part2pid[pk]]) #[vn1, vn2]
                    cdist_clone = cdist.clone()
                    cdist_clone[self.simi[self.part2vid[pk]][:,self.target_part2pid[pk]]<0.85] = 1000
                    cd1 = cdist_clone.min(dim=-1)[0]
                    cd2 = cdist_clone.min(dim=-2)[0] # compensate for false segmentation
                    chamfer_dist = chamfer_dist + cd1[cd1<1000].mean() + cd2[cd2<1000].mean()
                    # chamfer_dist += cdist.min(dim=-1)[0].mean() + cdist.min(dim=-2)[0].mean()
                    # cdist_clone = cdist.clone()
                    # cdist_clone[self.simi < 0.9] = 100
            else:
                cdist = torch.cdist(v_rot[template_vids], self.target_points[target_pids], p=1) #[vn1, vn2]
                cdist_clone = cdist.clone()
                cd1 = cdist_clone.min(dim=-1)[0]
                cd2 = cdist_clone.min(dim=-2)[0]
                chamfer_dist = cd1.mean() + cd2.mean()
            # if i < maxiters//2:
            #     loss = (cdist * (self.simi > 0.9)).mean()
            #     loss.backward()
            # else:
            #     (self.chamfer_weights[stg]*chamfer_dist).backward()

            (self.chamfer_weights[stg]*chamfer_dist).backward()
            optimizer.step()
            progress_bar.set_description(f"loss {chamfer_dist.item()*1000:.3f}")
            progress_bar.update(1)
            logs['chamfer'].append(chamfer_dist.item()*1000)
        progress_bar.close()
        if self.is_debug:
            for k, v in logs.items():
                plt.clf()
                plt.cla()
                plt.plot(v)
                plt.ylabel(k)
                plt.xlabel('iter')
                plt.savefig(f'{self.out_folder}/fitting/stage{stg:01d}_{k}.png')
        
        v_rot = self.v_tmp @ rotation_6d_to_matrix(para_glo_rot)[0].transpose(1, 0) + para_glo_transl
        cdist = torch.cdist(v_rot, self.target_verts)
        cd1 = cdist.min(dim=-1)[0]
        cd2 = cdist.min(dim=-2)[0]
        chamfer_dist = cd1.mean() + cd2.mean()
        self.global_orient = para_glo_rot
        self.transl = para_glo_transl
        print(f'best global rotation {self.global_orient}, translation {self.transl}, chamfer {chamfer_dist.item()*1000:.3f}')
    
        if self.is_debug: 
            v_rot = self.v_tmp @ rotation_6d_to_matrix(self.global_orient)[0].transpose(1, 0) + self.transl
            v_rot = v_rot.cpu().data.numpy()
            mesh = trimesh.Trimesh(vertices=v_rot,faces=self.faces)
            mesh.export(f"{self.out_folder}/fitting/mesh_stg{stg:01d}.obj")
        
        print(f'stage 0 finished in {time.time()-stime:.3f}s')

    def stage1_partwise_v2(self):
        stime = time.time()
        stg = 1
        lr = self.lrs[stg]
        maxiters = self.maxiters[stg]
        best_chamfer = 10000
        fitting_order = self.animal_config["fitting_order"]["stage1"]
        fitting_vids = []
        fitting_jids = []
        target_vids = []
        target_pids = []
        for parts in fitting_order:
            vids = []
            jids = []
            tvids = []
            tpids = []
            for pt in parts:
                if not isinstance(pt, list):
                    vids += self.part2vid[pt]
                    jids += self.parts[pt]
                    tvids += self.target_part2vid[pt]
                    tpids += self.target_part2pid[pt]
                else:
                    for ppt in pt:
                        vids += self.part2vid[ppt]
                        jids += self.parts[ppt]
                        tvids += self.target_part2vid[ppt]
                        tpids += self.target_part2pid[ppt]

            fitting_vids.append(vids)
            fitting_jids.append(jids)
            target_vids.append(tvids)
            target_pids.append(tpids)

        
        pose_cand = [
                     [None], 
                     [None], 
                     [None]
                     ]
        
        pose_samples = self.pose_data_torch[::5].clone().to(device=self.device,dtype=self.dtype)
        # pose_cand = matrix_to_rotation_6d(self.pose_cand)
        target_rest_vids = torch.arange(self.target_verts.shape[0]).to(device=self.device,dtype=torch.long)
        for pid, jids, vids, tvids, tpids in zip(list(range(len(fitting_jids))), fitting_jids, fitting_vids, target_vids, target_pids):  
            # st()          
            target_vert = self.target_verts[tvids]
            target_point = self.target_points[tpids]
            simi = self.simi[vids][:,tpids]
            thres = 0.2 #if pid == 0 else 0.4n

            if self.is_debug:
                vtmp = target_point.cpu().data.numpy()
                pcd = trimesh.points.PointCloud(vtmp)
                pcd.export(f"{self.out_folder}/fitting/target_part2opti_{pid:01d}.obj")

                data_0_255 = np.uint8(255 * (simi*(simi>0.85)).cpu().data.numpy())
                heatmap = cv2.applyColorMap(data_0_255, cv2.COLORMAP_JET)
                heatmap = (cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0)
                v_src = self.v_tmp[vids].cpu().data.numpy().reshape([-1,3])
                v_ref = target_point.cpu().data.numpy().reshape([-1,3])
                ids = np.arange(0, v_src.shape[0],v_src.shape[0]//10)
                for id in ids:
                    mesh_source = o3d.geometry.PointCloud()
                    mesh_source.points = o3d.utility.Vector3dVector(v_src)
                    col = np.ones([v_src.shape[0], 3]) * 0.8
                    col[id] = np.array([1.0,0,0])
                    mesh_source.colors = o3d.utility.Vector3dVector(col)
                    o3d.io.write_point_cloud(f'{self.out_folder}/fitting/stg{stg:01d}_part{pid:01d}_{id:03d}_src.ply', mesh_source)

                    mesh_ref = o3d.geometry.PointCloud()
                    mesh_ref.points = o3d.utility.Vector3dVector(v_ref)
                    col = heatmap[id]
                    mesh_ref.colors = o3d.utility.Vector3dVector(col)
                    o3d.io.write_point_cloud(f'{self.out_folder}/fitting/stg{stg:01d}_part{pid:01d}_{id:03d}_ref.ply', mesh_ref)
            best_chamfer = 10000
            for icand, cand in enumerate(pose_cand[pid]):
                # para_body_pose = nn.Parameter(self.body_pose.clone().detach()[self.other_joints])
                # para_body_pose = nn.Parameter(pose_cand[j:j+1].clone().detach().to(device=device,dtype=dtype))
                if cand is None:
                    para_body_pose = nn.Parameter(self.body_pose.clone().detach()[jids]) 
                else:
                    para_body_pose = nn.Parameter(cand.clone().detach()[jids]) 
                # para_pose_latent = nn.Parameter(self.zs_cand[j:j+1].clone().detach().to(device=device,dtype=dtype))
                para_log_scale = nn.Parameter(self.log_scale.clone().detach()[jids])
                para_glo_rot = nn.Parameter(self.global_orient.clone().detach())
                para_transl = nn.Parameter(self.transl.clone().detach())
                if pid == 0:
                    params = [para_body_pose, para_glo_rot, para_transl,para_log_scale]
                else:
                    params = [para_body_pose, para_log_scale]
                # params = [para_pose_latent, para_glo_rot, para_transl, para_log_scale]
                optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999))
                
                scheduler = MultiStepLR(optimizer, milestones=[maxiters//3,maxiters//3*2], gamma=0.1)
                logs = {}
                logs['chamfer'] = []
                logs['pose_prior'] = []
                logs['scale_LR'] = []
                logs['scale_parts'] = []
                progress_bar = tqdm(total=maxiters)
                decay = 0.95
                
                for i in range(maxiters):
                    optimizer.zero_grad()
                    pose = self.body_pose.detach().clone()
                    pose[jids] = para_body_pose
                    pose = rotation_6d_to_matrix(pose)
                    log_scale = self.log_scale.detach().clone()
                    log_scale[jids] = para_log_scale

                    v_posed, _, _, _ = lbs_customer_wscale(pose, self.v_tmp, self.joints, self.parents, self.weights, pose2rot= False, log_scale=log_scale)
                    v_posed = v_posed[0] @ rotation_6d_to_matrix(para_glo_rot)[0].transpose(1,0) + para_transl 
                    chamfer_dist = 0
                    if i < maxiters //2:
                        for pt in fitting_order[pid]:
                            template_vid_tmp = []
                            target_pid_tmp = []
                            if not isinstance(pt,list):
                                template_vid_tmp = self.part2vid[pt]
                                target_pid_tmp = self.target_part2pid[pt]
                            else:
                                for ppt in pt:
                                    template_vid_tmp += self.part2vid[ppt]
                                    target_pid_tmp += self.target_part2pid[ppt]

                            cdist = torch.cdist(v_posed, self.target_points[target_pid_tmp])
                            cdist_clone = cdist.clone()
                            cdist_clone[self.simi[:,target_pid_tmp]<0.85] = 1000
                            cd1 = cdist_clone[template_vid_tmp].min(dim=-1)[0]
                            cd2 = cdist_clone.min(dim=-2)[0] # compensate for false segmentation
                            chamfer_dist = chamfer_dist + cd1[cd1<1000].mean() + cd2[cd2<1000].mean()
                    else:
                        cdist = torch.cdist(v_posed, self.target_points[tpids], p=1)
                        cd1 = cdist[vids].min(dim=-1)[0]
                        cd2 = cdist.min(dim=-2)[0] # compensate for false segmentation
                        chamfer_dist = chamfer_dist + cd1.mean() + cd2.mean()
                    
                    loss_chamfer = chamfer_dist

                    
                    # if self.is_debug:
                    # if i == 0:
                        # pose_tmp = (self.model.decode(self.prior.sample([1])) * self.std) + self.mean
                        # pose_tmp = pose_tmp.reshape([-1,6])
                        # pose = self.body_pose.detach().clone()
                        # pose[self.other_joints] = pose_tmp
                        # pose = rotation_6d_to_matrix(pose)
                        # v_posed, joints_posed, A, T = lbs_customer_wscale(pose, self.v_tmp, self.joints, self.parents, self.weights, pose2rot= False, log_scale=log_scale)
                        # mesh = trimesh.Trimesh(vertices=v_posed.cpu().data.numpy(),faces=self.faces)
                        # mesh.export(f"{self.out_folder}/fitting/mesh_stg{stg:01d}_test.obj")
                    
                    pose_dist = (pose[jids][None] @ pose_samples[:, jids].transpose(2, 3))
                    pose_trace = pose_dist[:,:,0,0] + pose_dist[:,:,1,1] + pose_dist[:,:,2,2] #[nsamp, nj]
                    cos_dist = (1.5 - pose_trace/2)
                    cos_dist = cos_dist.min(dim=0)[0]
                    pose_prior = cos_dist.sum()
                    # pose_prior = torch.zeros_like(loss_chamfer)
                    
                    # left right symmetry
                    scale = torch.exp(log_scale).reshape([-1, 3])
                    scale_left_right = (scale[self.left_joints] - scale[self.right_joints]).pow(2).mean()

                    # parts
                    scale_parts = torch.zeros_like(chamfer_dist)
                    for k, v in self.parts.items():
                        if len(v) < 2:
                            continue
                        scale_parts += torch.std(scale[v],dim=0).pow(2).mean()

                    loss =  self.chamfer_weights[stg] * loss_chamfer + \
                            self.scale_LR_weights[stg] * scale_left_right + \
                            self.scale_parts_weights[stg] * scale_parts + \
                            self.pose_prior_weights[stg] * pose_prior
                    
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    progress_bar.set_description(f"loss {chamfer_dist.item()*1000:.3f}")
                    progress_bar.update(1)
                    logs['chamfer'].append(chamfer_dist.item()*1000)
                    logs['pose_prior'].append(pose_prior.item())
                    logs['scale_LR'].append(scale_left_right.item())
                    logs['scale_parts'].append(scale_parts.item())
                    
                progress_bar.close()

                if False:
                    for k, v in logs.items():
                        plt.clf()
                        plt.cla()
                        plt.plot(v)
                        plt.ylabel(k)
                        plt.xlabel('iter')
                        plt.savefig(f'{self.out_folder}/fitting/stage{stg:01d}_part{pid:02d}_{k}.png')

                pose_final = self.body_pose.clone()
                pose_final[jids] = para_body_pose.data.detach()
                global_orient_final = para_glo_rot.data.detach()
                transl_final = para_transl.data.detach()
                log_scale_final = self.log_scale.clone()
                log_scale_final[jids] = para_log_scale.data.detach()

                with torch.no_grad():
                    pose = pose_final.detach().clone()
                    pose = rotation_6d_to_matrix(pose)
                    log_scale = log_scale_final.detach().clone()
                    v_posed, _, _, _ = lbs_customer_wscale(pose, self.v_tmp, self.joints, self.parents, self.weights, pose2rot= False, log_scale=log_scale)
                    v_posed = v_posed[0] @ rotation_6d_to_matrix(global_orient_final)[0].transpose(1,0) + transl_final
                    cdist = torch.cdist(v_posed, self.target_verts)
                    cd1 = cdist[vids][:, target_rest_vids].min(dim=-1)[0]
                    cd2 = cdist[:,target_rest_vids].min(dim=-2)[0]
                    chamfer_dist = cd1.mean() + cd2.mean()
                    print(f"chamfer dist {chamfer_dist.item():.3f}")
                
                if self.is_debug:
                    v_posed = v_posed.cpu().data.numpy()
                    mesh = trimesh.Trimesh(vertices=v_posed,faces=self.faces)
                    mesh.export(f"{self.out_folder}/fitting/mesh_stg{stg:01d}_part{pid:01d}_cand{icand:01d}.obj")

                if chamfer_dist < best_chamfer:
                    self.body_pose[jids] = para_body_pose.data.detach()
                    self.log_scale[jids] = para_log_scale.data.detach()
                    if pid == 0:
                        self.global_orient = para_glo_rot.data.detach()
                        self.transl = para_transl.data.detach()
                    min_d = cdist[vids][:, target_rest_vids].min(dim=0)[0]
                    target_rest_vids = target_rest_vids[min_d>thres]
                    best_chamfer = chamfer_dist
        print(f'stage 1 finished in {time.time()-stime:.3f}s')

    def weight_smoothing_opti(self, initial_weights, A, target_verts):
        stime = time.time()
        lr = 0.1

        gt_edges = (target_verts[self.target_edges_new[:,0]] - target_verts[self.target_edges_new[:,1]]).norm(dim=-1)
        skip_n = self.pose_data_torch.shape[0] // (self.pose_data_torch.shape[0]//100)
        pose_samples = self.pose_data_torch[::skip_n].detach().clone().to(device=self.device,dtype=self.dtype)
        pose_samples[:,:len(self.root_joints)] = torch.eye(3,device=self.device,dtype=self.dtype)[None]
        
        thre = 1.0
        maxiters = 200
        para_target_weight = nn.Parameter(torch.log(initial_weights * 5 + 1e-5))
        params = [para_target_weight]
        optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999))
        scheduler = MultiStepLR(optimizer, milestones=[maxiters//3,maxiters//3*2], gamma=0.5)
        logs = {}
        logs['edge_loss'] = []
        logs['same_weight_loss'] = []
        logs['neg_entropy'] = []
        logs['weight_smooth'] = []
        progress_bar = tqdm(total=maxiters)
        for i in range(maxiters):
            optimizer.zero_grad()
            target_weights = para_target_weight.softmax(dim=-1)
            # invert lbs to get cano
            target_verts_cano, _ = invert_lbs(target_weights, A, target_verts)

            # edge loss
            cano_edges = (target_verts_cano[self.target_edges_new[:,0]] - target_verts_cano[self.target_edges_new[:,1]]).norm(dim=-1)
            edge_diff = (cano_edges - gt_edges)
            mask = (edge_diff.abs()/gt_edges) > thre
            if mask.sum() > 0:
                edge_loss = (edge_diff[mask] ** 2).mean()
            else:
                edge_loss = torch.tensor(0,dtype=self.dtype,device=self.device)

            if self.is_debug and i % 20 == 0:
                idxs = list(set(self.target_edges_new[mask.cpu().data.numpy()].reshape(-1).tolist()))
                vert_color = np.ones_like(target_verts_cano.cpu().data.numpy()) * 0.8
                vert_color[idxs] = np.array([[1.0,0,0]])
                mesh = trimesh.Trimesh(vertices=target_verts_cano.cpu().data.numpy(),faces=self.target_mesh.faces,vertex_colors=vert_color)
                mesh.export(f"{self.out_folder}/rigging/mesh_weight_opti_{i:03d}.obj")
            
            target_verts_posed, _, _, _ = lbs_customer_wscale(pose_samples, target_verts_cano, self.joints, self.parents, target_weights, pose2rot=False) #[fn,vngt,3]
            cano_edges_posed = (target_verts_posed[:, self.target_edges_new[:,0]] - target_verts_posed[:, self.target_edges_new[:,1]]).norm(dim=-1)
            edge_diff_pose = cano_edges_posed - gt_edges[None]
            mask_pose = (edge_diff_pose.abs()/gt_edges[None]) > thre
            if mask_pose.sum() > 0:
                edge_loss_posed = (edge_diff_pose[mask_pose] ** 2).mean()
            else:
                edge_loss_posed = torch.tensor(0,dtype=self.dtype,device=self.device)

            # same point loss
            same_weight_loss = ((target_weights[self.target_edges_same[:,0]] - target_weights[self.target_edges_same[:,1]]) ** 2).sum(dim=-1).mean()            
            
            neg_entropy = (torch.log(target_weights + 1e-10) * target_weights).sum(dim=-1).mean()
            
            # edges = np.concatenate([self.target_edges, self.target_edges_same],axis=0)
            # edge_2_update = self.target_edges_new[mask.cpu().data.numpy()]
            # weight_smooth = ((target_weights[edge_2_update[:,0]] - target_weights[edge_2_update[:,1]]).norm(dim=-1)**2).mean()

            vid_2_update = set(self.target_edges_new[mask.cpu().data.numpy()].reshape(-1).tolist())
            weight_smooth = torch.tensor(0).to(dtype=self.dtype,device=self.device)
            for vid in vid_2_update:
                try:
                    weight_smooth = weight_smooth + (target_verts_cano[vid] - target_verts_cano[list(self.neighbors_dict[vid])].mean(dim=0)).pow(2).sum()
                except:
                    # st()
                    print(vid)
            weight_smooth = weight_smooth / len(vid_2_update)

            loss =  1000 * edge_loss + \
                    1000 * edge_loss_posed + \
                    5000 * same_weight_loss + \
                    0 * neg_entropy + \
                    5000 * weight_smooth
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            progress_bar.set_description(f"loss {edge_loss.item():.6f}")
            progress_bar.update(1)
            logs['edge_loss'].append(edge_loss.item())
            logs['same_weight_loss'].append(same_weight_loss.item())
            logs['neg_entropy'].append(neg_entropy.item())
            logs['weight_smooth'].append(weight_smooth.item())
        progress_bar.close()

        print(f'optimization finished in {time.time()-stime:.3f}s')
        return para_target_weight.softmax(dim=-1)
    
    def rigging(self):
        # get canonical joint locations
        pose_cano = torch.eye(3).to(device=self.device, dtype=self.dtype)[None].repeat([self.body_pose.shape[0],1,1])
        vert_cano, joint_cano, _, _ = lbs_customer_wscale(pose_cano, self.v_tmp, self.joints, self.parents, self.weights, pose2rot= False, log_scale=self.log_scale)
        if self.is_debug:
            mesh = trimesh.Trimesh(vertices=vert_cano.cpu().data.numpy()[0],faces=self.faces)
            mesh.export(f"{self.out_folder}/rigging/temp_cano.obj")
        pose = rotation_6d_to_matrix(self.body_pose)

        vert_posed, joint_posed, A, T = lbs_customer_wscale(pose, vert_cano[0], joint_cano[0], self.parents, self.weights, pose2rot=False)
        if self.is_debug:
            mesh = trimesh.Trimesh(vertices=vert_posed.cpu().data.numpy()[0],faces=self.faces)
            mesh.export(f"{self.out_folder}/rigging/temp_posed_noglo.obj")

        target_verts = self.target_verts
        target_verts = (target_verts - self.transl) @ rotation_6d_to_matrix(self.global_orient)[0]

        np.savez_compressed(f'{self.out_folder}/rigging/verts_joints.npz', 
                            joints=joint_posed.cpu().data.numpy()[0], 
                            parents=self.parents,
                            parent_names=self.parents_name,
                            joint_names=self.joint_name,
                            verts=target_verts.cpu().data.numpy(),
                            faces=self.target_mesh.faces)

        if self.use_auto_weights:
            command = [
                "/root/blender-4.0.2-linux-x64/blender",
                "-b",
                "--python",
                "./blender/blender_auto_weights.py",
                "--",
                "--data_npz", f'{self.out_folder}/rigging/verts_joints.npz',
                "--out_file", f"{self.out_folder}/rigging/blender_auto_weights.npz"
                ]
            print(' '.join(command))
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            blender_auto_weights = np.load(f"{self.out_folder}/rigging/blender_auto_weights.npz")
            blender_verts = blender_auto_weights['verts']
            blender_weight = blender_auto_weights['weights']
            blender_weight = blender_weight / (blender_weight.sum(axis=-1,keepdims=True) + 1e-5)
            
            cd = torch.cdist(target_verts, torch.from_numpy(blender_verts).to(device=self.device,dtype=self.dtype))
            idx = cd.min(dim=1)[1].cpu().data.numpy()

            weight_auto = blender_weight[idx]
            weight_auto = torch.from_numpy(weight_auto).to(dtype=self.dtype, device=self.device)
            vids = torch.where(weight_auto.sum(dim=-1) < 0.9)[0].cpu().data.numpy().tolist()
            
            # if len(self.same_vs) > 0:
            #     weight_auto = copy_weight_to_repeated_point(weight_auto, self.same_vs, vids)
            #     vids = torch.where(weight_auto.sum(dim=-1) < 0.9)[0].cpu().data.numpy().tolist()

            if len(vids) > 0:
                cols = np.ones([target_verts.shape[0],3])*0.8
                cols[vids] = np.array([[1.0,0,0]])
                cols = (cols * 255).astype(np.uint8)
                mesh = trimesh.Trimesh(vertices=target_verts.cpu().data.numpy(),faces=self.target_mesh.faces,
                                    vertex_colors=cols)
                mesh.export(f"{self.out_folder}/rigging/target_zero_weight.obj")

                weight_auto = weight_smoothing(weight_auto.cpu().data.numpy(), 
                                                target_verts.cpu().data.numpy(), 
                                                self.neighbors_dict, order=1, 
                                                smoothing_type='mean', vids=vids)
                weight_auto = torch.from_numpy(weight_auto).to(device=self.device, dtype=self.dtype)
                target_verts_cano, _ = invert_lbs(weight_auto, A, target_verts)
                if self.is_debug:
                    mesh = trimesh.Trimesh(vertices=target_verts_cano.cpu().data.numpy(),faces=self.target_mesh.faces)
                    mesh.export(f"{self.out_folder}/rigging/target_cano_auto_weight_filled.obj")
                
                weight_auto = self.weight_smoothing_opti(weight_auto, A, target_verts)
                target_verts_cano, _ = invert_lbs(weight_auto, A, target_verts)
                if self.is_debug:
                    mesh = trimesh.Trimesh(vertices=target_verts_cano.cpu().data.numpy(),faces=self.target_mesh.faces)
                    mesh.export(f"{self.out_folder}/rigging/target_cano_auto_weight_smoothed.obj")


        if self.is_debug:
            mesh = trimesh.Trimesh(vertices=target_verts.cpu().data.numpy(),faces=self.target_mesh.faces)
            mesh.export(f"{self.out_folder}/rigging/target_noglo.obj")

        # part-wise rigging
        if self.use_auto_weights:
            weight_auto_new = weight_auto.detach().clone()
            weight_auto_new[:,self.root_joints] = 0
            weight_auto_new[:,self.root_joints[-1]] = weight_auto[:, self.root_joints].sum(dim=-1)
            weights_smoothed = weight_auto_new.clone()
        else:
            chamfer_weights = self.simi.transpose(1,0)
            cdist_weights = (1 - chamfer_weights) * 10
            # cdist_weights[chamfer_weights>0.8] = 1
            weights_merge = interp_lbs_weights(target_verts, vert_posed[0], self.weights, K=6, weight=100, chamfer_weights=cdist_weights)
            target_verts_cano, _ = invert_lbs(weights_merge, A, target_verts)
            if self.is_debug:
                mesh = trimesh.Trimesh(vertices=target_verts_cano.cpu().data.numpy(),faces=self.target_mesh.faces)
                mesh.export(f"{self.out_folder}/rigging/target_cano.obj")
            
            i = 0
            weights_smoothed = weights_merge.detach().clone()
            edge_gt = (target_verts[self.target_edges[:,0]] -  target_verts[self.target_edges[:,1]]).norm(dim=-1)
            edge_mean = edge_gt.mean(dim=0)
            vidlen = []
            thres = 2.0
            steps = 100
            
            while True:
                target_verts_cano, _ = invert_lbs(weights_smoothed, A, target_verts)
                # weights_smoothed = interp_lbs_weights(target_verts_cano, vert_cano[0], self.weights, K=6, weight=100)
                # target_verts_cano, _ = invert_lbs(weights_smoothed, A, target_verts)
                # if self.is_debug and i % 2 == 0:
                #     mesh = trimesh.Trimesh(vertices=target_verts_cano.cpu().data.numpy(),faces=self.target_mesh.faces)
                #     mesh.export(f"{self.out_folder}/rigging/target_cano_smoothed_{i:02d}.obj")
                
                edge_cano = (target_verts_cano[self.target_edges[:,0]] -  target_verts_cano[self.target_edges[:,1]]).norm(dim=-1)
                relative_err = ((edge_cano - edge_gt).abs()/edge_gt).cpu().data.numpy()
                eidx = np.logical_and(relative_err > thres, relative_err > (edge_mean.item() * 0.5))
                edge2opti = (self.target_edges[eidx])
                vid2opti = list(set(edge2opti.reshape(-1).tolist()))
                
                if self.is_debug and i % 5 == 0:
                    cols = np.ones([target_verts_cano.shape[0],3])*0.8
                    cols[vid2opti] = np.array([[1.0,0,0]])
                    cols = (cols * 255).astype(np.uint8)
                    mesh = trimesh.Trimesh(vertices=target_verts_cano.cpu().data.numpy(),faces=self.target_mesh.faces,
                                        vertex_colors=cols)
                    mesh.export(f"{self.out_folder}/rigging/target_cano_smoothed_{i:02d}.obj")

                print(f'step {i:03d}, thres {thres:.2f} number of verts {len(vid2opti):03d}!!')
                if len(vid2opti) in vidlen[-5:] and thres > 1.0:
                    thres = thres/2
                if (len(vid2opti) < 1 or len(vid2opti) in vidlen[-5:] or i > steps) and i > 20:
                    break
                weights_smoothed = weight_smoothing(weights_smoothed.cpu().data.numpy(), target_verts.cpu().data.numpy(), self.neighbors_dict, order=1, smoothing_type='mean', vids=vid2opti)
                weights_smoothed = torch.from_numpy(weights_smoothed).to(device=self.device, dtype=self.dtype)
                # weights_smoothed = repeat_point_consist(target_verts,weights_smoothed)
                vidlen.append(len(vid2opti))
                i += 1
                # thres = thres - 0.5 / steps
            print(f'need {i:03d} steps')
        target_verts_cano, _ = invert_lbs(weights_smoothed, A, target_verts)
        if self.is_debug:
            mesh = trimesh.Trimesh(vertices=target_verts_cano.cpu().data.numpy(),faces=self.target_mesh.faces)
            mesh.export(f"{self.out_folder}/rigging/target_cano_smoothed.obj")
        
        weights_smoothed = repeat_point_consist(target_verts,weights_smoothed)
        target_verts_cano, _ = invert_lbs(weights_smoothed, A, target_verts)
        if self.is_debug:
            mesh = trimesh.Trimesh(vertices=target_verts_cano.cpu().data.numpy(),faces=self.target_mesh.faces)
            mesh.export(f"{self.out_folder}/rigging/target_cano_final.obj")

        mesh_vis = o3d.geometry.TriangleMesh()
        mesh_vis.vertices = o3d.utility.Vector3dVector(target_verts_cano.cpu().data.numpy())
        mesh_vis.vertex_colors = o3d.utility.Vector3dVector(np.ones([target_verts_cano.shape[0],3])*0.8)
        mesh_vis.triangles = o3d.utility.Vector3iVector(self.target_mesh.faces)
        mesh_vis.compute_vertex_normals()
        o3d.io.write_triangle_mesh(f'{self.out_folder}/rigging/target_cano_final_vis.obj',mesh_vis)

        np.savez_compressed(f"{self.out_folder}/rigging/rigging.npz", 
                            weights=weights_smoothed.cpu().data.numpy(),
                            joint_cano=joint_cano.cpu().data.numpy()[0],
                            verts_cano=target_verts_cano.cpu().data.numpy(),
                            parents = self.parents,
                            joint_name = self.joint_name,
                            faces=self.target_mesh.faces    
                            )

    def fit(self):
        if self.is_fitting:
            self.stage0_v2()
            self.stage1_partwise_v2()
            np.savez_compressed(f'{self.out_folder}/fitting/fitting.npz', body_pose=self.body_pose.cpu().data.numpy(),
                                global_orient=self.global_orient.cpu().data.numpy(), transl=self.transl.cpu().data.numpy(),
                                log_scale=self.log_scale.cpu().data.numpy())
        else:
            fitting_data = np.load(f'{self.out_folder}/fitting/fitting.npz')
            self.body_pose = torch.from_numpy(fitting_data['body_pose']).to(device=self.device, dtype=self.dtype)
            self.global_orient = torch.from_numpy(fitting_data['global_orient']).to(device=self.device,dtype=self.dtype)
            self.transl = torch.from_numpy(fitting_data['transl']).to(device=self.device,dtype=self.dtype)
            self.log_scale = torch.from_numpy(fitting_data['log_scale']).to(device=self.device,dtype=self.dtype)

        if self.is_rigging:
            self.rigging()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/fitting.yaml')
    parser.add_argument('--target_file', type=str, default='/aigc_cfs_2/weimao/non-smalfit/data/dolphin.obj')
    parser.add_argument('--target_folder', type=str, default='')
    # parser.add_argument('--data_file', type=str, default='')
    # parser.add_argument('--data_folder', type=str, default='/aigc_cfs_2/weimao/non-smalfit/data/')
    parser.add_argument('--out_folder', type=str, default='')
    parser.add_argument('--is_debug', type=bool, default=True)
    parser.add_argument('--is_fitting', type=bool, default=True)
    parser.add_argument('--is_rigging', type=bool, default=True)
    parser.add_argument('--use_auto_weights', type=bool, default=True)
    # parser.add_argument('--is_animation', type=bool, default=False)
    # args = parser.parse_args()

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)
    

    if len(args.target_file) > 0:
        fitter = Fitter(args)
        # fitter.mesh_dir = args.data_file
        fitter.update_target_mesh(args.target_file)
        fitter.fit()
    else:
        mesh_names = sorted(listdir(args.target_folder))
        for mesh_name in mesh_names:
            fitter = Fitter(args)
            print('processing:', mesh_name)
            fitter.update_target_mesh(args.target_folder + '/' + mesh_name)
            fitter.fit()
    