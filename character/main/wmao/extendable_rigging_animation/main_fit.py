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

        self.joint_name = self.animal_model['joint_name']
        self.parents_name = self.animal_model['parents_name']
        self.faces = self.animal_model['faces']
        v_tmp = self.animal_model['v_template']
        weights = self.animal_model['weights']
        joints = self.animal_model['joints']
        parents = self.animal_model['parents']
        self.v_tmp = torch.from_numpy(v_tmp).to(device=device,dtype=dtype) # [n, 3]
        self.weights = torch.from_numpy(weights).to(device=device,dtype=dtype) #[vn, jn]
        self.joints = torch.from_numpy(joints).to(device=device,dtype=dtype) #[jn 3]
        self.parents = parents.tolist()
        self.template_size = torch.norm(self.v_tmp.max(dim=0)[0] - self.v_tmp.min(dim=0)[0]).item()
        self.num_joint = joints.shape[0]

        # # get data config
        # try:
        #     prior_config = OmegaConf.load(cfg.prior_dir + '/vae.yaml')
        # except:
        #     prior_config = OmegaConf.load(cfg.prior_dir + '/nf.yaml')
        # with open(prior_config.data_dir + '/config.json','r') as f:
        #     animal_config = json.load(f)
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
        # vidx2part = np.zeros([n_vert])
        # pi = 0
        #     for jid in range(n_vert):
        #             if max_j_idx[jid] in v:
        #                 vidx2part[jid] = pi
        #     pi += 1
        # vidx2part = torch.from_numpy(vidx2part).to(device=device, dtype=torch.long)

        """load point feature model"""
        self.load_point_feature_model()
        mesh = trimesh.Trimesh(vertices=v_tmp, faces=self.faces)
        self.v_feat, self.v_part_est = self.get_vert_feat(mesh) #[vn, feat]

        # # get cand pose
        # id_sel, zs_sel= self.get_anchor_poses(self.zs, 1)
        # self.pose_cand = self.pose_data_torch[:,len(self.root_joints):][id_sel].clone()
        # self.zs_cand = zs_sel.clone()

        """ load pretrained vae or nf """
        if len(self.prior_dir) > 0:
            self.load_prior()

        """ define parameters """
        self.global_orient = torch.eye(3, dtype=dtype, device=device)[:2].reshape(-1,6) # 6D rotation representation
        self.transl = torch.zeros([1,3], dtype=dtype, device=device)
        self.body_pose = matrix_to_rotation_6d(torch.eye(3)[None].repeat([self.num_joint,1,1])).to(device=device,dtype=dtype)
        self.log_scale = torch.zeros([self.num_joint, 3], dtype=dtype, device=device)
        
    @torch.no_grad()
    def load_template(self):
        data_dir = self.data_dir
        animal_model = np.load(data_dir + '/model.npz')
        self.animal_model = animal_model
        animation_data = np.load(data_dir + '/animation.npz')
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

        with open(data_dir + '/config.json','r') as f:
            animal_config = json.load(f)
        self.animal_config = animal_config

        # # get all meshes
        # verts = []
        # for i in range(self.pose_data_torch.shape[0]):
        #     pose_tmp = self.pose_data_torch[i].to(device=self.device)
        #     pose_tmp[:len(self.root_joints)] = torch.eye(3)[None].to(device=self.device,dtype=self.dtype)
        #     vert_tmp, _, _, _ = lbs_customer_wscale(pose_tmp, self.v_tmp, self.joints, self.parents, self.weights)            
        #     verts.append(vert_tmp)
        # verts = torch.cat(verts, dim=0)
        # self.target_verts_tmp_torch = verts.clone()

    def load_point_feature_model(self):
        model_dir = self.point_feature_model_config.model_dir
        model_name = self.point_feature_model_config.model
        
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
        
    @torch.no_grad()    
    def get_vert_feat(self, mesh):
        verts = mesh.vertices
        points = mesh.sample(self.point_feature_model_config.npoints)
        vs_max = verts.max(axis=0)[None]
        vs_min = verts.min(axis=0)[None]
        scale = 2/np.linalg.norm(vs_max - vs_min,axis=-1,keepdims=True)
        points = scale * (points - (vs_max+vs_min)/2)
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
        return point_feat[v2pts_tmp], parts[v2pts_tmp] #[vn, feat]

    @torch.no_grad()
    def load_prior(self):
        """load data and model"""
        prior_dir = self.prior_dir
        try:
            config = OmegaConf.load(prior_dir + '/vae.yaml')
        except:
            config = OmegaConf.load(prior_dir + '/nf.yaml')

        root_joints = self.animal_config['root_joints'] # assume root joints are always the first few joints
        pose_data_torch = self.pose_data_torch[:,len(root_joints):]

        n_data = pose_data_torch.shape[0]
        data_6d = matrix_to_rotation_6d(pose_data_torch).reshape([n_data,-1])
        data_6d = data_6d.reshape([n_data, -1])
        mean = torch.mean(data_6d, dim=0)
        std = torch.std(data_6d, dim=0)
        centered_data = data_6d - mean
        standardized_data = centered_data / std
        standardized_data = standardized_data.reshape([n_data,-1])
        self.mean = mean.to(device=self.device).detach()
        self.std = std.to(device=self.device).detach()
        
        # for elephant the first three joint are roots
        # assume using 6D rotation representation
        dn, jn = pose_data_torch.shape[0], pose_data_torch.shape[1]
        model_conf = config['model_conf']
        if model_conf['model_name'] == 'VAE':
            model = VAE(data_dim=jn*6, num_layer=model_conf['num_layer'], feat_dim=model_conf['feat_dim']).to(device=self.device, dtype=self.dtype)
        else:
            model = LinNF(data_dim=jn*6, num_layer=model_conf['num_layer'], with_prelu=model_conf['with_prelu']).to(device=self.device, dtype=self.dtype)
        
        state_dict = torch.load(f'{prior_dir}/latest.pth')
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()
        self.model = model
        # get data distribution
        xs = torch.split(standardized_data, 10000)
        zs = []
        
        for x in xs:
            x = x.to(device=self.device)
            if model_conf['model_name'] == 'VAE':
                z = model.encode(x)
            else:
                z, log_det_jacobian = model(x)
            zs.append(z)
        zs = torch.cat(zs,dim=0)
        mu = zs.mean(dim=0)
        zs_cen = zs - mu[None]
        cov = torch.mm(zs_cen.T, zs_cen) / (zs_cen.shape[0] - 1)

        # convert cov to positive definite
        epsilon = 1e-5
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = torch.clamp(eigvals, min=epsilon)
        cov_new = (eigvecs * eigvals).mm(eigvecs.t())
        self.prior = torch.distributions.MultivariateNormal(loc=mu.detach(), covariance_matrix=cov_new.detach())
        if model_conf['model_name'] == 'VAE':
            self.prior_func = lambda x: (self.prior.log_prob(self.model.encode((x-self.mean)/self.std))).sum(dim=-1)
        else:
            print('to implement')
        
        self.zs = zs

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

        
        self.target_v_feat, self.target_v_parts = self.get_vert_feat(self.target_mesh) #[vn, feat]
        self.simi = (self.v_feat @ self.target_v_feat.transpose(1,0) + 1)/2 #[vn1, vn2] (0-1)
        
        parts_tmp = self.animal_config['parts']
        self.target_part2vid = {}
        pid = 0
        for k, v in parts_tmp.items():
            self.target_part2vid[k] = torch.where(self.target_v_parts==pid)[0].cpu().data.numpy().tolist()
            pid += 1

        if self.is_debug:
            part_col = np.array(part_colors)
            mesh_vis = o3d.geometry.TriangleMesh()
            mesh_vis.vertices = o3d.utility.Vector3dVector(self.target_mesh.vertices)
            mesh_vis.vertex_colors = o3d.utility.Vector3dVector(part_col[self.target_v_parts.cpu().data.numpy()])
            mesh_vis.triangles = o3d.utility.Vector3iVector(self.target_mesh.faces)
            o3d.io.write_triangle_mesh(f'{self.out_folder}/fitting/target_part.ply', mesh_vis)
        
        # # filter the similarity
        # # a -> b -> a
        
        w1 = self.simi * (self.simi > 0.5)
        w1_0 = w1 / (w1.sum(dim=0,keepdims=True) + 1e-10)
        w1_1 = w1 / (w1.sum(dim=1,keepdims=True) + 1e-10)
        w2 = (w1_0 * 1000) * (w1_1 * 1000)
        self.w2 = w2 / (w2.max() + 1e-10) 
        # idx1 = self.simi.max(dim=0)[1] #[vn_gt]
        # idx2 = self.simi.max(dim=1)[1] #[vn_tmp]
        # v1 = self.target_verts[idx2[idx1]].clone().detach()
        # d1 = (v1 - self.target_verts).norm(dim=-1) #[vn_gt]
        

        
        # debug 
        if self.is_debug:
            simi = self.simi.cpu().data.numpy()
            # simi = self.w2.cpu().data.numpy()
            # to heatmap
            data_0_255 = np.uint8(255 * simi)
            heatmap = cv2.applyColorMap(data_0_255, cv2.COLORMAP_JET)
            heatmap = (cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0)

            v_src = self.target_verts.cpu().data.numpy().reshape([-1,3])
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
    
    """ fitting """
    def stage0_fit(self):
        stime = time.time()
        stg = 0
        device = self.device
        dtype = self.dtype
        lr = self.lrs[stg]
        maxiters = self.maxiters[stg]
        print('try to guess the initial orientation!')
        global_rot_cands = np.array([[0,0,0],
                                    [math.pi/2,0,0],
                                    [math.pi,0,0],
                                    [math.pi*3/2,0,0],
                                    [0,math.pi/2,0],
                                    [0,math.pi,0],
                                    [0,math.pi*3/2,0],
                                    [0,0,math.pi/2],
                                    [0,0,math.pi],
                                    [0,0,math.pi*3/2]])
        global_rot_cands = torch.from_numpy(global_rot_cands).to(device=device,dtype=dtype)
        global_rot_cands = matrix_to_rotation_6d(batch_rodrigues(global_rot_cands))            
        # global_rot_cands = np.array([[math.pi*3/2,0,0]])
        best_chamfer = 10000
        best_value = None
        for gi in range(global_rot_cands.shape[0]):
            
            global_rot = global_rot_cands[gi]
            para_glo_rot = nn.Parameter(global_rot[None]) # [1, 6]
            if gi == 0:
                para_glo_transl = nn.Parameter(self.transl.detach().clone()) # [1, 3]
            params = [para_glo_rot, para_glo_transl]
            optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999))
            logs = {}
            logs['chamfer'] = []
            progress_bar = tqdm(total=maxiters)
            for i in range(maxiters):
                optimizer.zero_grad()
                v_rot = self.v_tmp @ rotation_6d_to_matrix(para_glo_rot)[0].transpose(1, 0)
                v_rot = v_rot + para_glo_transl
                cdist = torch.cdist(v_rot, self.target_verts)
                cd1 = cdist.min(dim=-1)[0]
                cd2 = cdist.min(dim=-2)[0]
                chamfer_dist = cd1.mean() + cd2.mean()
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
                    plt.savefig(f'{self.out_folder}/fitting/stage{stg:01d}_{k}_{gi:02d}.png')

            
            v_rot = self.v_tmp @ rotation_6d_to_matrix(para_glo_rot)[0].transpose(1, 0) + para_glo_transl
            cdist = torch.cdist(v_rot, self.target_verts)
            cd1 = cdist.min(dim=-1)[0]
            cd2 = cdist.min(dim=-2)[0]
            chamfer_dist = cd1.mean() + cd2.mean()
            print(f'global rotation {global_rot}, chamfer {chamfer_dist.item()*1000:.3f}')
            if chamfer_dist < best_chamfer:
                self.global_orient = matrix_to_rotation_6d(rotation_6d_to_matrix(para_glo_rot.clone().detach()))
                self.transl = para_glo_transl.clone().detach()
                best_chamfer = chamfer_dist.clone().detach()
                if chamfer_dist * 1000 < 100:
                    break
        print(f'best global rotation {self.global_orient}, translation {self.transl}, chamfer {best_chamfer.item()*1000:.3f}')
        
        if self.is_debug: 
            v_rot = self.v_tmp @ rotation_6d_to_matrix(self.global_orient)[0].transpose(1, 0) + self.transl
            v_rot = v_rot.cpu().data.numpy()
            mesh = trimesh.Trimesh(vertices=v_rot,faces=self.faces)
            mesh.export(f"{self.out_folder}/fitting/mesh_stg{stg:01d}.obj")

        print(f'stage 0 finished in {time.time()-stime:.3f}s')

    def stage0_with_pointfeat(self):
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
        
        for i in range(maxiters):
            optimizer.zero_grad()
            v_rot = self.v_tmp @ rotation_6d_to_matrix(para_glo_rot)[0].transpose(1, 0)
            v_rot = v_rot + para_glo_transl
            cdist = torch.cdist(v_rot, self.target_verts, p=1) #[vn1, vn2]
            if i < maxiters//2:
                cdist_clone = cdist.clone()
                cdist_clone[self.simi < 0.9] = 100
            else:
                cdist_clone = cdist.clone()    
            cd1 = cdist_clone.min(dim=-1)[0]
            cd2 = cdist_clone.min(dim=-2)[0]
            chamfer_dist = cd1[cd1<100].mean() + cd2[cd2<100].mean()
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
        
        # not good
        # with torch.no_grad():
        #     # initialize pose to the closest template pose
        #     v_rot = self.target_verts_tmp_torch @ rotation_6d_to_matrix(self.global_orient).transpose(1, 2) + self.transl[None]
        #     cds = []
        #     v_rots = torch.split(v_rot, 100, dim=0)
        #     for vr in v_rots:
        #         cdist = torch.cdist(self.target_verts[None], vr)
        #         dist = cdist.min(dim=-1)[0].mean(dim=-1) + cdist.min(dim=-2)[0].mean(dim=-1)
        #         cds.append(dist)
        #     cds = torch.cat(cds,dim=0)
        #     idx = dist.min(dim=0)[1]
        #     pose_init = self.pose_data_torch[idx].to(device=self.device,dtype=self.dtype)
        #     pose_init[:len(self.root_joints)] = torch.eye(3)[None].to(device=self.device,dtype=self.dtype)
        #     self.body_pose = matrix_to_rotation_6d(pose_init)
        #     if self.is_debug:
        #         v_tmp =  v_rot[idx].cpu().data.numpy()
        #         mesh = trimesh.Trimesh(vertices=v_tmp,faces=self.faces)
        #         mesh.export(f"{self.out_folder}/fitting/mesh_init.obj")
        print(f'stage 0 finished in {time.time()-stime:.3f}s')

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
        
        for i in range(maxiters):
            optimizer.zero_grad()
            v_rot = self.v_tmp @ rotation_6d_to_matrix(para_glo_rot)[0].transpose(1, 0)
            v_rot = v_rot + para_glo_transl
            if i < maxiters:
                chamfer_dist = 0
                for pi, pk in enumerate(self.animal_config["parts"].keys()):
                    if "root" in pk or "leg" in pk or "spine" in pk:
                        cdist = torch.cdist(v_rot[self.part2vid[pk]], self.target_verts[self.target_v_parts==pi], p=1) #[vn1, vn2]
                        chamfer_dist += cdist.min(dim=-1)[0].mean() + cdist.min(dim=-2)[0].mean()
                    # cdist_clone = cdist.clone()
                    # cdist_clone[self.simi < 0.9] = 100
            else:
                cdist = torch.cdist(v_rot, self.target_verts, p=1) #[vn1, vn2]
                cdist_clone = cdist.clone()    
                cd1 = cdist_clone.min(dim=-1)[0]
                cd2 = cdist_clone.min(dim=-2)[0]
                chamfer_dist = cd1[cd1<100].mean() + cd2[cd2<100].mean()
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
        
        
        # not good
        # with torch.no_grad():
        #     # initialize pose to the closest template pose
        #     v_rot = self.target_verts_tmp_torch @ rotation_6d_to_matrix(self.global_orient).transpose(1, 2) + self.transl[None]
        #     cds = []
        #     v_rots = torch.split(v_rot, 100, dim=0)
        #     for vr in v_rots:
        #         cdist = torch.cdist(self.target_verts[None], vr)
        #         dist = cdist.min(dim=-1)[0].mean(dim=-1) + cdist.min(dim=-2)[0].mean(dim=-1)
        #         cds.append(dist)
        #     cds = torch.cat(cds,dim=0)
        #     idx = dist.min(dim=0)[1]
        #     pose_init = self.pose_data_torch[idx].to(device=self.device,dtype=self.dtype)
        #     pose_init[:len(self.root_joints)] = torch.eye(3)[None].to(device=self.device,dtype=self.dtype)
        #     self.body_pose = matrix_to_rotation_6d(pose_init)
        #     if self.is_debug:
        #         v_tmp =  v_rot[idx].cpu().data.numpy()
        #         mesh = trimesh.Trimesh(vertices=v_tmp,faces=self.faces)
        #         mesh.export(f"{self.out_folder}/fitting/mesh_init.obj")
        print(f'stage 0 finished in {time.time()-stime:.3f}s')

    def stage1_fit(self):
        stime = time.time()
        stg = 1
        device = self.device
        dtype = self.dtype
        lr = self.lrs[stg]
        maxiters = self.maxiters[stg]
        
        para_body_pose = nn.Parameter(self.body_pose.clone().detach()[self.other_joints])
        para_pose_latent = nn.Parameter(self.prior.sample([1]))
        para_glo_rot = nn.Parameter(self.global_orient.clone().detach())
        para_transl = nn.Parameter(self.transl.clone().detach())
        para_log_scale = nn.Parameter(self.log_scale.clone().detach()[self.other_joints])
        # params = [para_body_pose, para_glo_rot, para_transl,para_log_scale]
        params = [para_pose_latent, para_glo_rot, para_transl, para_log_scale]
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
            # pose[self.other_joints] = para_body_pose
            body_pose = self.model.decode(para_pose_latent) * self.std + self.mean
            pose[self.other_joints] = body_pose.reshape([-1,6])
            pose = rotation_6d_to_matrix(pose)
            log_scale = self.log_scale.detach().clone()
            log_scale[self.other_joints] = para_log_scale

            if True:
                for i in range(pose_sel.shape[0]):
                    pose[self.other_joints] = pose_sel[i]
                    v_posed, _, _, _ = lbs_customer_wscale(pose, self.v_tmp, self.joints, self.parents, self.weights, pose2rot= False, log_scale=log_scale)
                    v_posed = v_posed[0].cpu().data.numpy()
                    mesh = trimesh.Trimesh(vertices=v_posed,faces=self.faces)
                    mesh.export(f"{self.out_folder}/fitting/test_{i:02d}.obj")
                    


            
            v_posed, joints_posed, A, T = lbs_customer_wscale(pose, self.v_tmp, self.joints, self.parents, self.weights, pose2rot= False, log_scale=log_scale)
            v_posed = v_posed[0] @ rotation_6d_to_matrix(para_glo_rot)[0].transpose(1,0) + para_transl 

            cdist = torch.cdist(v_posed, self.target_verts)
            cd1 = cdist.min(dim=-1)[0]
            cd2 = cdist.min(dim=-2)[0]
            chamfer_dist = cd1.mean() + cd2.mean()

            if i < 0:
                loss_chamfer = (cdist * (self.simi > 0.9)).mean()
            else:
                loss_chamfer = chamfer_dist

            
            # if self.is_debug:
            if False:
                pose_tmp = (self.model.decode(self.prior.sample([1])) * self.std) + self.mean
                pose_tmp = pose_tmp.reshape([-1,6])
                pose = self.body_pose.detach().clone()
                pose[self.other_joints] = pose_tmp
                pose = rotation_6d_to_matrix(pose)
                v_posed, joints_posed, A, T = lbs_customer_wscale(pose, self.v_tmp, self.joints, self.parents, self.weights, pose2rot= False, log_scale=log_scale)
                mesh = trimesh.Trimesh(vertices=v_posed[0].cpu().data.numpy(),faces=self.faces)
                mesh.export(f"{self.out_folder}/fitting/mesh_stg{stg:01d}.obj")

            # pose_prior = self.prior_func(matrix_to_rotation_6d(rotation_6d_to_matrix(para_body_pose)).reshape([1,-1]))
            
            pose_prior = self.prior.log_prob(para_pose_latent).sum()
            
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
                    self.pose_prior_weights[stg] * pose_prior + \
                    self.scale_LR_weights[stg] * scale_left_right + \
                    self.scale_parts_weights[stg] * scale_parts
            
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

        if self.is_debug:
            for k, v in logs.items():
                plt.clf()
                plt.cla()
                plt.plot(v)
                plt.ylabel(k)
                plt.xlabel('iter')
                plt.savefig(f'{self.out_folder}/fitting/stage{stg:01d}_{k}.png')

        # self.body_pose[self.other_joints] = para_body_pose.data.detach()
        body_pose = self.model.decode(para_pose_latent) * self.std + self.mean
        self.body_pose[self.other_joints] = body_pose.reshape([-1,6])
        self.global_orient = para_glo_rot.data.detach()
        self.transl = para_transl.data.detach()
        self.log_scale[self.other_joints] = para_log_scale.data.detach()

        if self.is_debug:
            pose = self.body_pose.detach().clone()
            pose = rotation_6d_to_matrix(pose)
            log_scale = self.log_scale.detach().clone()
            v_posed, _, _, _ = lbs_customer_wscale(pose, self.v_tmp, self.joints, self.parents, self.weights, pose2rot= False, log_scale=log_scale)
            v_posed = v_posed[0] @ rotation_6d_to_matrix(self.global_orient)[0].transpose(1,0) + para_transl 
            v_posed = v_posed.cpu().data.numpy()
            mesh = trimesh.Trimesh(vertices=v_posed,faces=self.faces)
            mesh.export(f"{self.out_folder}/fitting/mesh_stg{stg:01d}.obj")

        print(f'stage 1 finished in {time.time()-stime:.3f}s')
   
    def stage1_with_cand(self):
        stime = time.time()
        stg = 1
        device = self.device
        dtype = self.dtype
        lr = self.lrs[stg]
        maxiters = self.maxiters[stg]
        

        # if self.is_debug:
        #     pose_sel = self.pose_cand.to(device=device)
        #     for i in range(pose_sel.shape[0]):
        #         pose[self.other_joints] = pose_sel[i]
        #         v_posed, _, _, _ = lbs_customer_wscale(pose, self.v_tmp, self.joints, self.parents, self.weights, pose2rot= False, log_scale=log_scale)
        #         v_posed = v_posed[0].cpu().data.numpy()
        #         mesh = trimesh.Trimesh(vertices=v_posed,faces=self.faces)
        #         mesh.export(f"{self.out_folder}/fitting/test_{i:02d}.obj")
        best_chamfer = 10000
        
        pose_cand = torch.from_numpy(self.animation_data['Elephant@IdleLookAroundEat_rot'][123][len(self.root_joints):][None]).to(device=self.device,dtype=self.dtype)
        pose_cand = matrix_to_rotation_6d(pose_cand)
        # pose_cand = matrix_to_rotation_6d(self.pose_cand)
        for j in range(self.zs_cand.shape[0]):
            # para_body_pose = nn.Parameter(self.body_pose.clone().detach()[self.other_joints])
            para_body_pose = nn.Parameter(pose_cand[j:j+1].clone().detach().to(device=device,dtype=dtype))
            para_pose_latent = nn.Parameter(self.zs_cand[j:j+1].clone().detach().to(device=device,dtype=dtype))
            para_glo_rot = nn.Parameter(self.global_orient.clone().detach())
            para_transl = nn.Parameter(self.transl.clone().detach())
            para_log_scale = nn.Parameter(self.log_scale.clone().detach()[self.other_joints])
            params = [para_body_pose, para_glo_rot, para_transl,para_log_scale]
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
                pose[self.other_joints] = para_body_pose
                # body_pose = self.model.decode(para_pose_latent) * self.std + self.mean
                # pose[self.other_joints] = body_pose.reshape([-1,6])
                pose = rotation_6d_to_matrix(pose)
                log_scale = self.log_scale.detach().clone()
                log_scale[self.other_joints] = para_log_scale

                
                v_posed, joints_posed, A, T = lbs_customer_wscale(pose, self.v_tmp, self.joints, self.parents, self.weights, pose2rot= False, log_scale=log_scale)
                v_posed = v_posed[0] @ rotation_6d_to_matrix(para_glo_rot)[0].transpose(1,0) + para_transl 

                cdist = torch.cdist(v_posed, self.target_verts)
                cd1 = cdist.min(dim=-1)[0]
                cd2 = cdist.min(dim=-2)[0]
                chamfer_dist = cd1.mean() + cd2.mean()

                # if i < maxiters//3:
                # loss_chamfer = (cdist * self.simi.detach())[self.simi>0.9].mean()
                # else:
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

                # pose_prior = self.prior_func(matrix_to_rotation_6d(rotation_6d_to_matrix(para_body_pose)).reshape([1,-1]))
                
                pose_prior = self.prior.log_prob(para_pose_latent).sum()
                
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
                        self.scale_parts_weights[stg] * scale_parts
                
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

            if self.is_debug:
                for k, v in logs.items():
                    plt.clf()
                    plt.cla()
                    plt.plot(v)
                    plt.ylabel(k)
                    plt.xlabel('iter')
                    plt.savefig(f'{self.out_folder}/fitting/stage{stg:01d}_cand{j:02d}_{k}.png')

            self.body_pose[self.other_joints] = para_body_pose.data.detach()
            # body_pose = self.model.decode(para_pose_latent) * self.std + self.mean
            # self.body_pose[self.other_joints] = body_pose.reshape([-1,6])
            self.global_orient = para_glo_rot.data.detach()
            self.transl = para_transl.data.detach()
            self.log_scale[self.other_joints] = para_log_scale.data.detach()

            if self.is_debug:
                pose = self.body_pose.detach().clone()
                pose = rotation_6d_to_matrix(pose)
                log_scale = self.log_scale.detach().clone()
                v_posed, _, _, _ = lbs_customer_wscale(pose, self.v_tmp, self.joints, self.parents, self.weights, pose2rot= False, log_scale=log_scale)
                v_posed = v_posed[0] @ rotation_6d_to_matrix(self.global_orient)[0].transpose(1,0) + para_transl 
                v_posed = v_posed.cpu().data.numpy()
                mesh = trimesh.Trimesh(vertices=v_posed,faces=self.faces)
                mesh.export(f"{self.out_folder}/fitting/mesh_stg{stg:01d}_cand{j:02d}.obj")

        print(f'stage 1 finished in {time.time()-stime:.3f}s')
    
    def stage1_partwise(self):
        stime = time.time()
        stg = 1
        lr = self.lrs[stg]
        maxiters = self.maxiters[stg]
        best_chamfer = 10000
        fitting_order = self.animal_config["fitting_order"]
        fitting_vids = []
        fitting_jids = []
        for parts in fitting_order:
            vids = []
            jids = []
            for pt in parts:
                vids += self.part2vid[pt]
                jids += self.parts[pt]
            fitting_vids.append(vids)
            fitting_jids.append(jids)
        # smoothl1 = nn.SmoothL1Loss(beta=0.5)
        
        pose_tmp = [
                self.body_pose.detach().clone(),
                self.body_pose.detach().clone(),
                self.body_pose.detach().clone()
                ]
        jid_trunk = [26, 27, 28, 29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]
        
        # ang_x = torch.arange(len(jid_trunk)) * (-np.pi/4/len(jid_trunk))
        # rots = torch.from_numpy(np.array([[1, 0, 0]])) * ang_x[:,None]
        # rots = rots.to(device=self.device, dtype=self.dtype)
        # rots = matrix_to_rotation_6d(batch_rodrigues(rots))
        pose_tmp[1][jid_trunk] = matrix_to_rotation_6d(torch.from_numpy(self.animation_data["Elephant@IdleLookAroundEat_rot"][120]).to(device=self.device,dtype=self.dtype))[jid_trunk]
        # pose_tmp[1][jid_trunk] = rots
        pose_tmp[2][jid_trunk] = matrix_to_rotation_6d(torch.from_numpy(self.animation_data["Elephant@IdleLookAroundEat_rot"][160]).to(device=self.device,dtype=self.dtype))[jid_trunk]
        # pose_cand = [[None], 
        #              [None], 
        #              [
        #                 matrix_to_rotation_6d(torch.from_numpy(self.animation_data["Elephant@IdleLookAroundEat_rot"][80]).to(device=self.device,dtype=self.dtype)),
        #                 matrix_to_rotation_6d(torch.from_numpy(self.animation_data["Elephant@IdleLookAroundEat_rot"][120]).to(device=self.device,dtype=self.dtype)),
        #              ]
        #              ]

        pose_cand = [[None], 
                     [None], 
                     pose_tmp
                     ]

        
        pose_samples = self.pose_data_torch[::5].clone().to(device=self.device,dtype=self.dtype)
        # pose_cand = matrix_to_rotation_6d(self.pose_cand)
        target_vids = torch.arange(self.target_verts.shape[0]).to(device=self.device,dtype=torch.long)
        
        for pid, jids, vids in zip(list(range(len(fitting_jids))), fitting_jids, fitting_vids):            
            target_vert = self.target_verts[target_vids]
            simi = self.simi[vids][:,target_vids]
            thres = 0.2 #if pid == 0 else 0.4

            if self.is_debug:
                vtmp = target_vert.cpu().data.numpy()
                pcd = trimesh.points.PointCloud(vtmp)
                pcd.export(f"{self.out_folder}/fitting/target_part2opti_{pid:01d}.obj")
                data_0_255 = np.uint8(255 * (simi*(simi>0.85)).cpu().data.numpy())
                heatmap = cv2.applyColorMap(data_0_255, cv2.COLORMAP_JET)
                heatmap = (cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0)
                v_src = self.v_tmp[vids].cpu().data.numpy().reshape([-1,3])
                v_ref = target_vert.cpu().data.numpy().reshape([-1,3])
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
                para_glo_rot = nn.Parameter(self.global_orient.clone().detach())
                para_transl = nn.Parameter(self.transl.clone().detach())
                para_log_scale = nn.Parameter(self.log_scale.clone().detach()[jids])
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
                    # body_pose = self.model.decode(para_pose_latent) * self.std + self.mean
                    # pose[self.other_joints] = body_pose.reshape([-1,6])
                    pose = rotation_6d_to_matrix(pose)
                    log_scale = self.log_scale.detach().clone()
                    log_scale[jids] = para_log_scale

                    
                    v_posed, _, _, _ = lbs_customer_wscale(pose, self.v_tmp, self.joints, self.parents, self.weights, pose2rot= False, log_scale=log_scale)
                    v_posed = v_posed[0] @ rotation_6d_to_matrix(para_glo_rot)[0].transpose(1,0) + para_transl 

                    # v_posed_sel = v_posed[vids]
                    # v_posed_sel = v_posed_sel[:,None].repeat([1, target_vert.shape[0],1])
                    # cdist = smoothl1(v_posed_sel, target_vert, reduction=None)
                    
                    cdist = torch.cdist(v_posed, self.target_verts, p=1)
                    cdist_clone = cdist.clone()
                    if pid == (len(fitting_order)-1) and i < maxiters//5:
                        cdist_clone[self.simi<0.85] = 1000
                    cd1 = cdist_clone[vids][:, target_vids].min(dim=-1)[0]
                    cd2 = cdist_clone[:,target_vids].min(dim=-2)[0]
                    # if pid == 3:
                    #     chamfer_dist = cd1[cd1<1000].mean() + cd2[cd2<1000].mean()
                    # else:
                    chamfer_dist = cd1[cd1<1000].mean() + cd2[cd2<1000].mean()

                    # if i < maxiters//3:
                    # loss_chamfer = (cdist * self.simi.detach())[self.simi>0.9].mean()
                    # else:
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
                    
                    # pose_prior = self.prior_func(matrix_to_rotation_6d(pose[len(self.root_joints):]).reshape([1,-1]))
                    # pose_prior = self.prior.log_prob(para_pose_latent).sum()
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
                    cd1 = cdist[vids][:, target_vids].min(dim=-1)[0]
                    cd2 = cdist[:,target_vids].min(dim=-2)[0]
                    chamfer_dist = cd1.mean() + cd2.mean()
                
                if self.is_debug:
                    v_posed = v_posed.cpu().data.numpy()
                    mesh = trimesh.Trimesh(vertices=v_posed,faces=self.faces)
                    mesh.export(f"{self.out_folder}/fitting/mesh_stg{stg:01d}_part{pid:01d}_cand{icand:01d}.obj")

                if chamfer_dist < best_chamfer:
                    self.body_pose[jids] = para_body_pose.data.detach()
                    # body_pose = self.model.decode(para_pose_latent) * self.std + self.mean
                    # self.body_pose[self.other_joints] = body_pose.reshape([-1,6])
                    self.log_scale[jids] = para_log_scale.data.detach()
                    if pid == 0:
                        self.global_orient = para_glo_rot.data.detach()
                        self.transl = para_transl.data.detach()
                    min_d = cdist[vids][:, target_vids].min(dim=0)[0]
                    target_vids = target_vids[min_d>thres]
                    best_chamfer = chamfer_dist
            
            # if True:
            #     # initialize pose to the closest template pose
            #     pose = self.pose_data_torch[::5].to(device=self.device)
            #     jid_done = [item for sublist in fitting_jids[:-1] for item in sublist]
            #     pose_done = rotation_6d_to_matrix(self.body_pose[jid_done])
            #     pose[:,jid_done] = pose_done
            #     pose[:, :len(self.root_joints)] = torch.eye(3, device=self.device,dtype=self.dtype)[None]
            #     log_scale = self.log_scale.detach().clone()[None].repeat([pose.shape[0], 1, 1])
            #     vs_gt, _, _, _ = lbs_customer_wscale(pose, self.v_tmp, self.joints, self.parents, self.weights, pose2rot= False, log_scale=log_scale)
            #     vs_gt = vs_gt @ rotation_6d_to_matrix(self.global_orient).transpose(2,1) + self.transl[None] 
            #     cds = []
            #     vs_gts = torch.split(vs_gt, 100, dim=0)
            #     for vs in vs_gts:
            #         cdist = torch.cdist(self.target_verts[None], vs)
            #         dist = cdist.min(dim=-1)[0].mean(dim=-1) + cdist.min(dim=-2)[0].mean(dim=-1)
            #         cds.append(dist)
            #     cds = torch.cat(cds,dim=0)
            #     idx = cds.min(dim=0)[1]
            #     pose_init = pose[idx].to(device=self.device,dtype=self.dtype)
            #     if self.is_debug:
            #         v_tmp =  vs_gt[idx].cpu().data.numpy()
            #         mesh = trimesh.Trimesh(vertices=v_tmp,faces=self.faces)
            #         mesh.export(f"{self.out_folder}/fitting/mesh_init.obj")
        print(f'stage 1 finished in {time.time()-stime:.3f}s')

    def stage1_partwise_v2(self):
        stime = time.time()
        stg = 1
        lr = self.lrs[stg]
        maxiters = self.maxiters[stg]
        best_chamfer = 10000
        fitting_order = self.animal_config["fitting_order"]
        fitting_vids = []
        fitting_jids = []
        target_vids = []
        for parts in fitting_order:
            vids = []
            jids = []
            tvids = []
            for pt in parts:
                vids += self.part2vid[pt]
                jids += self.parts[pt]
                tvids += self.target_part2vid[pt]
            fitting_vids.append(vids)
            fitting_jids.append(jids)
            target_vids.append(tvids)
        
        # pose_tmp = [
        #         self.body_pose.detach().clone(),
        #         self.body_pose.detach().clone(),
        #         self.body_pose.detach().clone()
        #         ]
        # jid_trunk = [26, 27, 28, 29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]
        # ang_x = torch.arange(len(jid_trunk)) * (-np.pi/4/len(jid_trunk))
        # rots = torch.from_numpy(np.array([[1, 0, 0]])) * ang_x[:,None]
        # rots = rots.to(device=self.device, dtype=self.dtype)
        # rots = matrix_to_rotation_6d(batch_rodrigues(rots))
        # pose_tmp[1][jid_trunk] = matrix_to_rotation_6d(torch.from_numpy(self.animation_data["Elephant@IdleLookAroundEat_rot"][120]).to(device=self.device,dtype=self.dtype))[jid_trunk]
        # pose_tmp[1][jid_trunk] = rots
        # pose_tmp[2][jid_trunk] = matrix_to_rotation_6d(torch.from_numpy(self.animation_data["Elephant@IdleLookAroundEat_rot"][160]).to(device=self.device,dtype=self.dtype))[jid_trunk]
        # pose_cand = [[None], 
        #              [None], 
        #              [
        #                 matrix_to_rotation_6d(torch.from_numpy(self.animation_data["Elephant@IdleLookAroundEat_rot"][80]).to(device=self.device,dtype=self.dtype)),
        #                 matrix_to_rotation_6d(torch.from_numpy(self.animation_data["Elephant@IdleLookAroundEat_rot"][120]).to(device=self.device,dtype=self.dtype)),
        #              ]
        #              ]

        pose_cand = [
                     [None], 
                     [None], 
                     [None]
                     ]

        pose_samples = self.pose_data_torch[::5].clone().to(device=self.device,dtype=self.dtype)
        # pose_cand = matrix_to_rotation_6d(self.pose_cand)
        target_rest_vids = torch.arange(self.target_verts.shape[0]).to(device=self.device,dtype=torch.long)
        for pid, jids, vids, tvids in zip(list(range(len(fitting_jids))), fitting_jids, fitting_vids, target_vids):  
                      
            target_vert = self.target_verts[tvids]
            simi = self.simi[vids][:,tvids]
            thres = 0.2 #if pid == 0 else 0.4

            if self.is_debug:
                vtmp = target_vert.cpu().data.numpy()
                pcd = trimesh.points.PointCloud(vtmp)
                pcd.export(f"{self.out_folder}/fitting/target_part2opti_{pid:01d}.obj")
                data_0_255 = np.uint8(255 * (simi*(simi>0.85)).cpu().data.numpy())
                heatmap = cv2.applyColorMap(data_0_255, cv2.COLORMAP_JET)
                heatmap = (cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0)
                v_src = self.v_tmp[vids].cpu().data.numpy().reshape([-1,3])
                v_ref = target_vert.cpu().data.numpy().reshape([-1,3])
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
                    for pt in fitting_order[pid]:
                        cdist = torch.cdist(v_posed, self.target_verts[self.target_part2vid[pt]])
                        cdist_clone = cdist.clone()
                        cdist_clone[self.simi[:,self.target_part2vid[pt]]<0.85] = 1000
                        cd1 = cdist_clone[self.part2vid[pt]].min(dim=-1)[0]
                        cd2 = cdist_clone.min(dim=-2)[0] # compensate for false segmentation
                        chamfer_dist = chamfer_dist + cd1[cd1<1000].mean() + cd2[cd2<1000].mean()

                    # cdist = torch.cdist(v_posed, self.target_verts[tvids], p=1)
                    # cdist_clone = cdist.clone()
                    # # if pid == (len(fitting_order)-1) and i < maxiters//5:
                    # #     cdist_clone[self.simi<0.85] = 1000
                    # cd1 = cdist_clone[vids].min(dim=-1)[0]
                    # cd2 = cdist_clone.min(dim=-2)[0] # compensate for false segmentation
                    # # if pid == 3:
                    # #     chamfer_dist = cd1[cd1<1000].mean() + cd2[cd2<1000].mean()
                    # # else:
                    # chamfer_dist = cd1[cd1<1000].mean() + cd2[cd2<1000].mean()

                    # if i < maxiters//3:
                    # loss_chamfer = (cdist * self.simi.detach())[self.simi>0.9].mean()
                    # else:
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
                    
                    # pose_prior = self.prior_func(matrix_to_rotation_6d(pose[len(self.root_joints):]).reshape([1,-1]))
                    # pose_prior = self.prior.log_prob(para_pose_latent).sum()
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


    def stage2(self):
        stime = time.time()
        stg = 2
        lr = self.lrs[stg]
        maxiters = self.maxiters[stg]
        best_chamfer = 10000
        
        with torch.no_grad():
            # read initial pose, logscale, global orientation, weights
            body_pose = self.body_pose.detach().clone()
            log_scale = self.log_scale.detach().clone()
            global_orient = self.global_orient.detach().clone()
            transl = self.transl.detach().clone()
            target_verts = (self.target_verts - transl) @ rotation_6d_to_matrix(global_orient)[0]
            v_posed, _, _, _ = lbs_customer_wscale(rotation_6d_to_matrix(body_pose), self.v_tmp, self.joints, self.parents, self.weights, pose2rot= False, log_scale=log_scale)
            # v_posed = v_posed[0] @ rotation_6d_to_matrix(global_orient)[0].transpose(1,0) + transl 
            # cdist = torch.cdist(target_verts,v_posed[0]) ** 2
            # mutual_weight = -10*cdist
            
            # K = 3
            # idxs, dists = knn(target_verts, v_posed[0], k=6)
            # neighbs_weight = torch.exp(-100*dists)
            # neighbs_weight = neighbs_weight / neighbs_weight.sum(-1, keepdim=True)
            
            # mutual_weight = torch.zeros([target_verts.shape[0],v_posed.shape[1]],device=self.device,dtype=self.dtype)
            # id1 = torch.arange(target_verts.shape[0],device=self.device,dtype=torch.long)[:,None].repeat([1,K]).reshape([-1])
            # mutual_weight[id1, idxs.reshape(-1)] = neighbs_weight.reshape(-1)
            # mutual_weight = torch.log(mutual_weight * 10 + 1e-5)
            
            # get initial weights
            target_weights0 = interp_lbs_weights(target_verts, v_posed[0], self.weights, K=6, weight=100, chamfer_weights=self.simi)

            pose_samples = self.pose_data_torch[::5].detach().clone().to(device=self.device,dtype=self.dtype)
            pose_samples[:,:len(self.root_joints)] = torch.eye(3,device=self.device,dtype=self.dtype)[None]

            pose_cano = torch.eye(3).to(device=self.device, dtype=self.dtype)[None].repeat([body_pose.shape[0],1,1])
            vert_cano, joint_cano, _, _ = lbs_customer_wscale(pose_cano, self.v_tmp, self.joints, self.parents, self.weights, pose2rot= False, log_scale=self.log_scale)
            if self.is_debug:
                mesh = trimesh.Trimesh(vertices=vert_cano.cpu().data.numpy()[0],faces=self.faces)
                mesh.export(f"{self.out_folder}/rigging/temp_cano.obj")
            vert_samples, joint_samples, _, _ = lbs_customer_wscale(pose_samples, vert_cano[0], joint_cano[0], self.parents, self.weights, pose2rot=False)

        para_body_pose = nn.Parameter(body_pose) 
        # para_mutual_weight = nn.Parameter(mutual_weight)
        para_target_weight = nn.Parameter(torch.log(target_weights0 + 1e-5))
        params = [para_target_weight]
        optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999))
        scheduler = MultiStepLR(optimizer, milestones=[maxiters//3,maxiters//3*2], gamma=0.1)
        logs = {}
        logs['chamfer'] = []
        logs['pose_prior'] = []
        logs['scale_LR'] = []
        logs['scale_parts'] = []
        progress_bar = tqdm(total=maxiters)
        for i in range(maxiters):
            optimizer.zero_grad()
            pose = rotation_6d_to_matrix(para_body_pose)
            # target_weights = para_mutual_weight.softmax(dim=-1) @ self.weights #[vngt, jn]
            target_weights = para_target_weight.softmax(dim=-1)
            # invert lbs to get cano
            _, _, A, _ = lbs_customer_wscale(pose, vert_cano[0], joint_cano[0], self.parents, self.weights, pose2rot=False)
            target_verts_cano, _ = invert_lbs(target_weights, A, target_verts)
            cdist = torch.cdist(target_verts_cano,vert_cano[0])
            loss_chamfer = cdist.min(dim=-1)[0].mean() + cdist.min(dim=-2)[0].mean()

            if self.is_debug and i % 50 == 0:
                mesh = trimesh.Trimesh(vertices=target_verts_cano.cpu().data.numpy(),faces=self.target_mesh.faces)
                mesh.export(f"{self.out_folder}/rigging/mesh_cano_{i:03d}.obj")

            fidx = np.random.choice(np.arange(pose_samples.shape[0]),20)
            target_verts_posed, _, _, _ = lbs_customer_wscale(pose_samples[fidx], target_verts_cano, self.joints, self.parents, target_weights, pose2rot=False) #[fn,vngt,3]
            

            
            # mutual_weight = target_weights @ self.weights.transpose(1,0)
            cdist = torch.cdist(target_verts_posed, vert_samples[fidx]) ** 2
            # cd1 = (para_mutual_weight.softmax(dim=-1)[None] * cdist).sum(dim=-1)
            # cd2 = (para_mutual_weight.softmax(dim=-2)[None] * cdist).sum(dim=-2)
            # loss_chamfer = cd1.mean() + cd2.mean()
            loss_chamfer1 = cdist.min(dim=-1)[0].mean(dim=-1).mean() + cdist.min(dim=-2)[0].mean(dim=-1).mean()

            # pose_dist = (pose[None] @ pose_samples.transpose(2, 3))
            # pose_trace = pose_dist[:,:,0,0] + pose_dist[:,:,1,1] + pose_dist[:,:,2,2] #[nsamp, nj]
            # cos_dist = (1.5 - pose_trace/2)
            # cos_dist = cos_dist.min(dim=0)[0]
            # pose_prior = cos_dist.sum()

            neg_entropy = (torch.log(target_weights + 1e-10) * target_weights).sum(dim=-1).mean()
            # pose_prior = torch.zeros_like(loss_chamfer)
            
            # edge loss
            edge_loss = (((target_verts_cano[self.target_edges[:,0]] - target_verts_cano[self.target_edges[:,1]]).norm(dim=-1) - \
                (target_verts[self.target_edges[:,0]] - target_verts[self.target_edges[:,1]]).norm(dim=-1)) ** 2).mean()
            
            # edge loss
            edge_loss2 = (((target_verts_cano[self.target_edges_same[:,0]] - target_verts_cano[self.target_edges_same[:,1]]).norm(dim=-1) - \
                (target_verts[self.target_edges_same[:,0]] - target_verts[self.target_edges_same[:,1]]).norm(dim=-1)) ** 2).mean()
            
            edges = np.concatenate([self.target_edges,self.target_edges_same],axis=0)
            weight_smooth = ((target_weights[edges[:,0]] - target_weights[edges[:,1]]).norm(dim=-1)**2).mean()

            # # left right symmetry
            # scale = torch.exp(log_scale).reshape([-1, 3])
            # scale_left_right = (scale[self.left_joints] - scale[self.right_joints]).pow(2).mean()

            # # parts
            # scale_parts = torch.zeros_like(chamfer_dist)
            # for k, v in self.parts.items():
            #     if len(v) < 2:
            #         continue
            #     scale_parts += torch.std(scale[v],dim=0).pow(2).mean()

            loss =  self.chamfer_weights[stg] * loss_chamfer + \
                    self.chamfer_weights[stg] * loss_chamfer1 + \
                    10 * neg_entropy + \
                    10 * weight_smooth
                    # 10000 * edge_loss + \
                    # 100000 * edge_loss2
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            progress_bar.set_description(f"loss {loss_chamfer.item():.6f}")
            progress_bar.update(1)
            logs['chamfer'].append(loss_chamfer.item()*1000)
            # logs['pose_prior'].append(pose_prior.item())
            # logs['scale_LR'].append(scale_left_right.item())
            # logs['scale_parts'].append(scale_parts.item())
            
        progress_bar.close()

        print(f'stage 2 finished in {time.time()-stime:.3f}s')

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

            weight_auto = np.load(f"{self.out_folder}/rigging/blender_auto_weights.npz")['weights']
            weight_auto = weight_auto / (weight_auto.sum(axis=-1,keepdims=True) + 1e-5) 
            weight_auto = torch.from_numpy(weight_auto).to(dtype=self.dtype, device=self.device)

            vids = torch.where(weight_auto.sum(dim=-1) < 0.9)[0].cpu().data.numpy().tolist()
            # st()
            if len(self.same_vs) > 0:
                weight_auto = copy_weight_to_repeated_point(weight_auto, self.same_vs, vids)
                vids = torch.where(weight_auto.sum(dim=-1) < 0.9)[0].cpu().data.numpy().tolist()

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
            weights_smoothed = weight_auto.clone()
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
            # self.stage0_with_pointfeat()
            self.stage0_v2()
            # self.stage1_partwise()
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
    parser.add_argument('--data_file', type=str, default='/aigc_cfs_2/weimao/non-smalfit/data/elephant2.obj')
    parser.add_argument('--data_folder', type=str, default='')
    # parser.add_argument('--data_file', type=str, default='')
    # parser.add_argument('--data_folder', type=str, default='/aigc_cfs_2/weimao/non-smalfit/data/')
    parser.add_argument('--out_folder', type=str, default='')
    parser.add_argument('--is_debug', type=bool, default=True)
    parser.add_argument('--is_fitting', type=bool, default=False)
    parser.add_argument('--is_rigging', type=bool, default=True)
    parser.add_argument('--use_auto_weights', type=bool, default=True)
    # parser.add_argument('--is_animation', type=bool, default=False)
    # args = parser.parse_args()

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)
    

    if len(args.data_file) > 0:
        fitter = Fitter(args)
        # fitter.mesh_dir = args.data_file
        fitter.update_target_mesh(args.data_file)
        fitter.fit()
    else:
        mesh_names = sorted(listdir(args.data_folder))
        for mesh_name in mesh_names:
            fitter = Fitter(args)
            print('processing:', mesh_name)
            fitter.update_target_mesh(args.data_folder + '/' + mesh_name)
            fitter.fit()
    