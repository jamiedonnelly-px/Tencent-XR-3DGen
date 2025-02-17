import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import numpy as np
import open3d as o3d
import trimesh
import math
# sys.path.append('/usr/lib/python3.8/site-packages/pointops-1.0-py3.8-linux-x86_64.egg')
from pathlib import Path
from tqdm import tqdm
from ipdb import set_trace as st
import matplotlib.pyplot as plt
import trimesh
import miniball
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import subprocess
from omegaconf import OmegaConf
import json
from torch import nn
import glob

from models.point_transformer_partseg import *
from utils import *
from models.model import LinNF, VAE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

device = torch.device('cuda:0')
dtype = torch.float32

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size, pn, _ = target.shape
        _, pred = output.topk(maxk, 1, True, True)
        # pred = pred.transpose(-1,-2)
        _, targ = target.topk(1,1,True,True)
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = pred.eq(targ)
        res = []
        
        for k in topk:
            correct_k = (correct[:,:k].float().sum(dim=1) > 0).float().sum()
            res.append(correct_k * (100.0 / (batch_size*pn)))
        return res


def accuracy2(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size, pn, _ = output.shape
        _, pred = output.topk(maxk, 2, True, True)
        # pred = pred.transpose(-1,-2)
        # _, targ = target.topk(1,1,True,True)
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = pred.eq(target[...,None])
        res = []
        
        for k in topk:
            correct_k = (correct[..., :k].float().sum(dim=-1) > 0).float().sum()
            res.append(correct_k * (100.0 / (batch_size*pn)))
        return res

def optimal_rotation_svd(X, Y):
    # Center the point clouds
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)

    # Compute the covariance matrix
    cov_matrix = np.dot(X_centered.T, Y_centered)

    # Perform Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(cov_matrix)

    # Compute the optimal rotation matrix
    R = np.dot(Vt.T, U.T)

    # Correct for reflections
    if np.linalg.det(R) < 0:
        Vt[-1] = -Vt[-1]
        R = np.dot(Vt.T, U.T)

    return R

def load_data(data_dir):
    """load data and model"""
    # try:
    #     config = OmegaConf.load(prior_dir + '/vae.yaml')
    # except:
    #     config = OmegaConf.load(prior_dir + '/nf.yaml')
    # data_dir = config['data_dir']
    # with open(f'{data_dir}/config.json','r') as f:
    #     data_config = json.load(f)
    animal_model = np.load(data_dir + '/model.npz')
    
    joint_name = animal_model['joint_name']
    parents_name = animal_model['parents_name']
    faces = animal_model['faces']
    v_tmp = animal_model['v_template']
    weights = animal_model['weights']
    joints = animal_model['joints']
    parents = animal_model['parents']
    v_tmp = torch.from_numpy(v_tmp).to(device=device,dtype=dtype) # [n, 3]
    weights = torch.from_numpy(weights).to(device=device,dtype=dtype) #[vn, jn]
    joints = torch.from_numpy(joints).to(device=device,dtype=dtype) #[jn 3]
    parents = parents.tolist()
    
    animation_data = np.load(data_dir + '/animation.npz')
    pose_data = []
    for k, v in animation_data.items():
        if k.endswith('_trans'):
            continue
        pose_data.append(v)
    pose_data = np.concatenate(pose_data, axis=0)
    pose_data_torch = torch.from_numpy(pose_data[::2]).to(dtype=dtype, device=device)
    print(f'total number of poses {pose_data.shape[0]}')
    with open(data_dir + '/config.json','r') as f:
        animal_config = json.load(f)

    '''get part seg'''
    parts_tmp = animal_config['parts']
    n_parts = len(list(parts_tmp.keys()))
    n_vert = v_tmp.shape[0]
    max_j_idx = np.argmax(weights.cpu().data.numpy(),axis=1)
    vidx2part = np.zeros([n_vert])
    pi = 0
    for k, v in parts_tmp.items():
        for jid in range(n_vert):
                if max_j_idx[jid] in v:
                    vidx2part[jid] = pi
        pi += 1
    vidx2part = torch.from_numpy(vidx2part).to(device=device, dtype=torch.long)

    root_joints = animal_config['root_joints'] # assume root joints are always the first few joints
    left_joints = animal_config['left_joints']
    right_joints = animal_config['right_joints']

    pose_data_torch[:, :len(root_joints), :, :] = 0.0
    pose_data_torch[:, :len(root_joints), 0, 0] = 1.0
    pose_data_torch[:, :len(root_joints), 1, 1] = 1.0
    pose_data_torch[:, :len(root_joints), 2, 2] = 1.0
    n_data = pose_data_torch.shape[0]
    vs = []
    pts = []
    vids = []
    parts = []
    
    num_repeats = 6
    print("loading data")
    parts_aug = animal_config['parts_aug']
    for i in tqdm(range(n_data)):
        pose_tmp = pose_data_torch[i].clone()
        pose_tmp = pose_tmp[None].repeat([num_repeats,1,1,1])
        rand_tmp = torch.rand_like(pose_tmp[:,:,:,0])
        
        log_scale = torch.zeros_like(pose_tmp[:,:,:,0])
        log_scale[rand_tmp<0.5] = rand_tmp[rand_tmp<0.5].clone() * 0.5 + 0.5
        log_scale[rand_tmp>=0.5] = rand_tmp[rand_tmp>=0.5].clone() + 1.0
        log_scale = torch.log(log_scale)
        log_scale_new = log_scale.clone()
        for pid in parts_aug.values():
            log_scale_new[:,pid] = log_scale[:,pid].mean(dim=1,keepdims=True)
        # make sure left and right is consistent
        scale_sym = (log_scale[:,left_joints] + log_scale[:,right_joints])/2
        log_scale_new[:,left_joints] = scale_sym
        log_scale_new[:,right_joints] = scale_sym
        log_scale_new[0,:] = 0
        
        v_posed, _, _, _ = lbs_customer_wscale(pose_tmp, v_tmp, joints, parents, weights, pose2rot=False, log_scale=log_scale_new)
        # v_posed = v_posed.cpu().data.numpy()
        
        if True:
            vs_vis = v_posed[3].cpu().data.numpy()
            col = np.array(part_colors)[vidx2part.cpu().data.numpy()]
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vs_vis)
            mesh.vertex_colors = o3d.utility.Vector3dVector(col)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            o3d.io.write_triangle_mesh(f'{data_dir}/parts.ply', mesh)

        for ii in range(num_repeats):
            mesh = trimesh.Trimesh(vertices=v_posed[ii].cpu().data.numpy(),faces=faces)
            pt = mesh.sample(10000)
            pt_torch = torch.from_numpy(pt).to(device=device,dtype=dtype)
            
            cdist = torch.cdist(pt_torch, v_posed[ii])
            vid = torch.min(cdist,dim=-1)[1].cpu().data.numpy()
            pts.append(pt[None])
            vs.append(v_posed[ii:ii+1].cpu().data.numpy())
            vids.append(vid[None])
            parts.append(vidx2part[vid][None].cpu().data.numpy())
    vs = np.concatenate(vs, axis=0)
    pts = np.concatenate(pts,axis=0)
    vids = np.concatenate(vids,axis=0)
    parts = np.concatenate(parts,axis=0)

    return animal_model, vs, pts, vids, parts, v_tmp, faces

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def parse_args():
    parser = argparse.ArgumentParser('Model')

    # model config
    parser.add_argument('--model', type=str, default='PointTransformerSeg38', help='choose from: PointTransformerSeg26, PointTransformerSeg38, PointTransformerSeg50') # 
    parser.add_argument('--is_moco', action='store_true', default=False, help='use moco') # 
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')

    parser.add_argument('--data_dir', type=str, default='/aigc_cfs_2/weimao/non-smalfit/dataset/SK_Dolphin', help='dataset dir')
    parser.add_argument('--temp_mesh_dir', type=str, default='/aigc_cfs_2/weimao/non-smalfit/dataset/SK_Dolphin/mesh_template.obj', help='template mesh dir')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--test_data_dir', type=str, default='/aigc_cfs_2/weimao/non-smalfit/data', help='')
    
    # training config
    # parser.add_argument('--pretrained_dir', type=str, default='/aigc_cfs_2/weimao/non-smalfit/output/SK_Elephant/contrast_rot_aug_PointTransformerSeg38/2024-06-27_18-56-19', help='consume training')
    parser.add_argument('--pretrained_dir', type=str, default='none', help='consume training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch Size during training')
    parser.add_argument('--epoch', default=500, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--is_eval', type=bool, default=False, help='decay rate for lr decay')
    parser.add_argument('--grad_acc_step', type=int, default=4, help='gradient accumulate steps')
    parser.add_argument('--rot_aug', action='store_true', default=False, help='whether apply random rotation or not')
    parser.add_argument('--scale_aug', action='store_true', default=False, help='whether apply random scale or not')
    parser.add_argument('--noise_aug', action='store_true', default=False, help='whether apply random noise or not')
    parser.add_argument('--trans_aug', action='store_true', default=False, help='whether apply random translation or not')

    # # evaluation config
    # parser.add_argument('--num_clusters', default=20, type=int, help='epoch to run')
    # parser.add_argument('--num_furthest', default=100, type=int, help='epoch to run')
    # parser.add_argument('--use_manual_template', action='store_true', default=False, help='whether to use manual templates')
    
    # log config
    parser.add_argument('--log_dir', type=str, default='./log', help='log path')

    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    animal_model, vs, pts, vids, parts, v_tmp, faces = load_data(args.data_dir)
    
    # scale to uni-sphere 
    vs_max = vs.max(axis=1)[:,None]
    vs_min = vs.min(axis=1)[:,None]
    scale = 2 / np.linalg.norm(vs_max - vs_min,axis=-1,keepdims=True)
    vs = scale * (vs - (vs_max+vs_min)/2)
    pts = scale * (pts - (vs_max+vs_min)/2)
    animal_name = args.data_dir.split('/')[-1]
    n_data = vs.shape[0]
    
    """save part segmentation for debug"""
    # if True:
    pid = 100
    part_col = np.array(part_colors)
    mesh = o3d.geometry.PointCloud()
    mesh.points = o3d.utility.Vector3dVector(pts[pid])
    mesh.colors = o3d.utility.Vector3dVector(part_col[parts[pid]])
    o3d.io.write_point_cloud(f'./output/{animal_name}/parts_test.ply', mesh)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if args.pretrained_dir == 'none':
        exp_dir = f"./output/{animal_name}/contrast_{'moco_' if args.is_moco else ''}{'rot_' if args.rot_aug else ''}{'scale_' if args.scale_aug else ''}{'trans_' if args.trans_aug else ''}{'noise_' if args.noise_aug else ''}{'aug_' if args.rot_aug or args.trans_aug or args.scale_aug or args.noise_aug else ''}{args.model}/{timestr}"
    else:
        exp_dir = args.pretrained_dir
    
    checkpoints_dir = f'{exp_dir}/checkpoints/'
    os.makedirs(checkpoints_dir, exist_ok=True)
    log_dir = f'{exp_dir}/logs/'
    os.makedirs(log_dir, exist_ok=True)

    '''read template mesh'''
    # temp_mesh = trimesh.load(args.temp_mesh_dir)
    # temp_verts = temp_mesh.vertices
    # temp_verts_torch = torch.from_numpy(temp_verts).to(device=device, dtype=dtype) #[vn, 3]
    temp_verts_torch = v_tmp.to(device=device, dtype=dtype)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    log_string("The number of training data is: %d" % vs.shape[0])

    '''MODEL LOADING'''
    in_channels = 3
    if args.normal:
        in_channels += 3
    out_feat = 128
    if args.model == 'PointTransformerSeg26':
        model = PointTransformerSeg26(in_channels=in_channels, num_classes=out_feat).cuda() # num_classes is the output channel
    elif args.model == 'PointTransformerSeg38':
        model = PointTransformerSeg38(in_channels=in_channels, num_classes=out_feat).cuda() # num_classes is the output channel
    elif args.model == 'PointTransformerSeg50':
        model = PointTransformerSeg50(in_channels=in_channels, num_classes=out_feat).cuda() # num_classes is the output channel
    model_partseg = torch.nn.Linear(in_features=out_feat, out_features=parts.max()+1, device=device, dtype=dtype)

    total_num_para, train_num_para = count_parameters(model)
    log_string(f'Trainable/Total number of parameters {train_num_para/1000000:.3f}M/{total_num_para/1000000:.3f}M')
    # try:
    if os.path.exists(str(exp_dir) + '/checkpoints/best_model.pth'):
        print(f'load from {str(exp_dir)}/checkpoints/best_model.pth')
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        model_partseg.load_state_dict(checkpoint['model_partseg_state_dict'])
        best_test_acc1 = checkpoint['test_acc1']
        global_epoch = start_epoch
        log_string(f'Use pretrain model epoch {start_epoch:03d}')
    else:
        log_string('No existing model, starting training from scratch...')
        best_test_acc1 = 0
        start_epoch = 0
        global_epoch = 0

    if args.is_eval:
        eval_other_dataset(model, model_partseg, args.test_data_dir, args, results_path=exp_dir + '/evaluation/')
        return

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(model_partseg.parameters()),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(list(model.parameters()) + list(model_partseg.parameters()), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoch//25, gamma=0.7)
    criterion = nn.CrossEntropyLoss().cuda()

    log_loss = []
    log_acc1 = []
    log_acc5 = []
    log_acc1_part = []
    
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch)) 
        model.train()
        scheduler.step()
        log_string('Learning rate:%f' % optimizer.param_groups[0]['lr'])
        
        acc1s = 0
        acc5s = 0
        acc1s_part = 0
        loss_all = 0
        num_pair = 2 if args.rot_aug else 1
        shuffled_idx = np.arange(n_data)
        np.random.shuffle(shuffled_idx)
        n_data = args.batch_size * (n_data//args.batch_size)
        shuffled_idx = shuffled_idx[:n_data]
        idxs = np.split(shuffled_idx, n_data//args.batch_size)
        cov = 0.06
        """training epoch """ 
        for i, id in tqdm(enumerate(idxs), total=len(idxs), smoothing=0.9):
            
            # verts = torch.from_numpy(vs[id]).to(device=device, dtype=dtype) #[bs, vn, 3]

            with torch.no_grad():
                shuffled_point_idx = np.arange(pts.shape[1])
                np.random.shuffle(shuffled_point_idx)
                shuffled_point_idx = shuffled_point_idx[:args.npoint]
                points_orig = torch.from_numpy(pts[id][:,shuffled_point_idx]).to(device=device, dtype=dtype) #[bs, pn, 3]
                b, n, _ = points_orig.shape
                pts_vid = torch.from_numpy(vids[id][:,shuffled_point_idx]).to(device=device, dtype=torch.long) #[bs, pn]
                pts_part = torch.from_numpy(parts[id][:,shuffled_point_idx]).to(device=device, dtype=torch.long) #[bs, pn]
                
                assert args.rot_aug or args.scale_aug or args.noise_aug or args.trans_aug
                
                if args.rot_aug or args.scale_aug or args.noise_aug or args.trans_aug:
                    shuffled_idx_tmp = np.arange(points_orig.shape[0])
                    np.random.shuffle(shuffled_idx_tmp)
                    points = torch.cat([points_orig, points_orig.clone(), points_orig.clone()[shuffled_idx_tmp]], dim=0)
                    label1 = torch.arange(points_orig.shape[1])[None].repeat([b, 1]).to(device=device,dtype=torch.long)#[bs, pn]
                    
                    tmp_v1 = temp_verts_torch[pts_vid]
                    tmp_v2 = temp_verts_torch[pts_vid[shuffled_idx_tmp]]
                    tmp_cdist = torch.cdist(tmp_v1, tmp_v2) #[bs, pn, 3]
                    label2 = tmp_cdist.min(dim=-1)[1] #[bs, pn]
                    label = torch.cat([label1,label2], dim=0) #[bb, pn]
                    label_part = torch.cat([pts_part,pts_part.clone(),pts_part.clone()[shuffled_idx_tmp]], dim=0)

                    # debug
                    if False:
                        jjs = [10,20]
                        pps = [100,1000]
                        for jj in jjs:
                            for pp in pps:
                                mesh_source = o3d.geometry.PointCloud()
                                mesh_source.points = o3d.utility.Vector3dVector(points_orig[jj].reshape([-1,3]).cpu().data.numpy())
                                col = np.ones([n, 3]) * 0.8
                                col[pp] = np.array([1.0,0,0])
                                mesh_source.colors = o3d.utility.Vector3dVector(col)
                                o3d.io.write_point_cloud(f'./output/{animal_name}/{jj:03d}_{pp:03d}.ply', mesh_source)

                                mesh_ref = o3d.geometry.PointCloud()
                                mesh_ref.points = o3d.utility.Vector3dVector(points_orig.clone()[shuffled_idx_tmp][jj].reshape([-1,3]).cpu().data.numpy())
                                col = np.ones([n, 3]) * 0.8
                                col[label2[jj,pp].item()] = np.array([1.0,0,0])
                                mesh_ref.colors = o3d.utility.Vector3dVector(col)
                                o3d.io.write_point_cloud(f'./output/{animal_name}/{jj:03d}_{pp:03d}_ref.ply', mesh_ref)

                bb = points.shape[0]

                if args.rot_aug:
                    rot_rand = torch.randn([points.shape[0],3]).float().cuda()
                    rot_rand = F.normalize(rot_rand, dim=1) * torch.rand([points.shape[0],3]).float().cuda() * math.pi
                    rot_rand = axis_angle_to_matrix(rot_rand) #[bs*2, 3, 3]
                    points = points @ rot_rand

                # # rand scale
                if args.scale_aug:
                    scale = torch.rand([points_orig.shape[0], 1, 1]).float().cuda() * 0.2 + 0.9
                    points = points * scale

                # # rand translate
                if args.trans_aug:
                    trans = torch.rand([points_orig.shape[0], 1, 1]).float().cuda() * 0.2 - 0.1
                    points = points + trans

                # # rand noise
                if args.noise_aug:
                    noise = torch.randn_like(points) * 0.05
                    points = points + noise

            
            inp = points
            inp = {
                "coord": points.reshape(bb*n, 3),
                "feat": inp.reshape(bb*n, -1),
                "offset": torch.cumsum(torch.ones(bb).cuda()*n,dim=0)
            }

            seg_pred = model(inp) # [b*2*n, out_feat]
            part_logit = model_partseg(seg_pred)
            seg_pred = F.normalize(seg_pred, dim=-1)
            seg_pred = seg_pred.reshape([bb, n, -1])
            q = seg_pred[:b]

            # compute the regular contrastive loss
            ks = torch.split(seg_pred[b:], b, dim=0)
            label_list = torch.split(label, b, dim=0)
            loss = 0
            logits = []
            
            for k, label in zip(ks,label_list):
                ''' contrastive loss '''
                # compute logits
                logit = q @ k.transpose(2, 1)/0.07 #[b, n, n]
                loss += criterion(logit.reshape([b*n,-1]), label.reshape([-1]))
                logits.append(logit)


            # compute part segmentation loss
            loss += 0.1*criterion(part_logit,label_part.reshape([-1]))

            if i % args.grad_acc_step == 0:
                optimizer.zero_grad()
            loss.backward()
            if (i+1) % args.grad_acc_step == 0 or (i+1) == len(idxs):
                optimizer.step()

            loss_all += loss.item()
            
            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # measure accuracy and record loss
            acc1, acc5 = accuracy2(torch.cat(logits,dim=0), torch.cat(label_list,dim=0), topk=(1, 5))
            acc1_part = accuracy2(part_logit[None], label_part.reshape([1,-1]), topk=(1,))
            
            acc1s += acc1.item()
            acc5s += acc5.item()
            acc1s_part += acc1_part[0].item()

        acc1s = acc1s/len(idxs)
        acc5s = acc5s/len(idxs)
        acc1s_part = acc1s_part/len(idxs)
        loss_all = loss_all/len(idxs)
        log_string(f'Loss is {loss_all:.5f}, Train acc1 is: {acc1s:.1f}, acc5 is: {acc5s:.1f}, Part Seg acc1 is: {acc1s_part:.1f}')

        log_acc1.append(acc1s)
        log_acc5.append(acc5s)
        log_acc1_part.append(acc1s_part)
        log_loss.append(loss_all)
        
        if (acc1s >= best_test_acc1):
            best_test_acc1 = acc1s
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'test_acc1': acc1s,
                'test_acc5': acc5s,
                'test_acc1_part': acc1s_part,
                'model_state_dict': model.state_dict(),
                'model_partseg_state_dict': model_partseg.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')
            
        global_epoch += 1

        savepath = str(checkpoints_dir) + '/last_model.pth'
        log_string('Saving at %s' % savepath)
        state = {
            'epoch': epoch,
            'test_acc1': acc1s,
            'test_acc5': acc5s,
            'test_acc1_part': acc1s_part,
            'model_state_dict': model.state_dict(),
            'model_partseg_state_dict': model_partseg.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)

    plt.clf()
    plt.cla()
    plt.plot(log_loss)
    plt.savefig(f"{exp_dir}/logs/loss.jpg")
    plt.clf()
    plt.cla()
    plt.plot(log_acc1)
    plt.savefig(f"{exp_dir}/logs/acc1.jpg")
    plt.clf()
    plt.cla()
    plt.plot(log_acc5)
    plt.savefig(f"{exp_dir}/logs/acc5s.jpg")
    plt.clf()
    plt.cla()
    plt.plot(log_acc1_part)
    plt.savefig(f"{exp_dir}/logs/acc1_part.jpg")
    
     
def eval_epoch(model, testDataLoader, args, is_aug=True, save_results=False, results_path=''):
    model.eval()
    # test_rot_dist_all = 0
    # test_pts_dist_all = 0
    acc1s = 0
    acc5s = 0
    q_ids = []
    q_feats = [] # store the normalized feature output
    q_points = []
    rot_rands = []
    q_feats2 = [] # store the normalized feature output
    q_points2 = []
    rot_rands2 = []
    
    with torch.no_grad():
        for batch_id, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            points_orig = data['pts'].float().cuda()
            b, n, _ = points_orig.shape
            norms_orig = data['norms'].float().cuda()
            # target = data['semantic'].to(torch.int64).cuda()
            label = torch.zeros(points_orig.shape[0]).float().cuda()

            points_pair = data['pts_pair'].float().cuda()
            norms_pair = data['norms_pair'].float().cuda()
            # target_pair = data['semantic_pair'].to(torch.int64).cuda()
            cdist = torch.cdist(points_orig, points_pair) #[b, n, n] [0, 2]
            weight = torch.exp(-cdist*10) #[0, 1] the closer the higher

            # rot_rand = torch.randn([points_orig.shape[0],3]).float().cuda()
            # rot_rand = axis_angle_to_matrix(rot_rand)
            # points = torch.matmul(rot_rand, points_orig.transpose(2,1))
            # norms = torch.matmul(rot_rand, norms_orig.transpose(2,1))

            points = torch.cat([points_orig.transpose(2,1), points_pair.transpose(2,1)], dim=0)
            norms = torch.cat([norms_orig.transpose(2,1), norms_pair.transpose(2,1)], dim=0)
            if is_aug:
                if args.rot_aug:
                    rot_rand = torch.randn([points.shape[0],3]).float().cuda()
                    rot_rand = F.normalize(rot_rand, dim=1) * torch.rand([points.shape[0],3]).float().cuda() * math.pi
                    # rot_rand[-b:] = 0
                    # idxs = np.random.choice(b,b//20) # add some identities
                    # rot_rand[idxs] = rot_rand[idxs]*0
                    rot_rand = axis_angle_to_matrix(rot_rand)
                    
                    points = torch.matmul(rot_rand, points)
                    norms = torch.matmul(rot_rand, norms)
                    rot_rands.append(rot_rand[:b].cpu().data.numpy())
                    rot_rands2.append(rot_rand[b:].cpu().data.numpy())
                # # rand scale
                if args.scale_aug:
                    scale = torch.rand([points_orig.shape[0], 1, 1]).float().cuda() * 0.2 + 0.9
                    points = points * scale

                # # rand translate
                if args.trans_aug:
                    trans = torch.rand([points_orig.shape[0], 1, 1]).float().cuda() * 0.2 - 0.1
                    points = points + trans

                # # rand noise
                if args.noise_aug:
                    noise = torch.randn_like(points) * 0.05
                    points = points + noise
                    noise = torch.randn_like(norms) * 0.025
                    norms = norms + noise
                    norms = F.normalize(norms, dim=1)

            
            q_ids += data['key']
            q_points.append(points[:b].transpose(2, 1).cpu().data.numpy())
            q_points2.append(points[b:b*2].transpose(2, 1).cpu().data.numpy())
            if not args.is_moco:
                inp = points
                if args.normal:
                    inp = torch.cat([inp,norms],dim=1)
                inp = {
                    "coord": points.transpose(2, 1).reshape(b*n*2, 3),
                    "feat": inp.transpose(2, 1).reshape(b*n*2, -1),
                    "offset": torch.cumsum(torch.ones(b*2).cuda()*n,dim=0)
                }

                seg_pred = model(inp) # [b*2, n, out_feat]
                seg_pred = F.normalize(seg_pred, dim=-1)
                seg_pred = seg_pred.reshape(b*2,n,-1)
                q = seg_pred[:b]
                q_feats.append(q.cpu().data.numpy())
                q_feats2.append(seg_pred[b:b*2].cpu().data.numpy())
                ks = [seg_pred[b*ii:b*(ii+1)] for ii in range(1, 2)]
                loss = 0
                logits = []
                labels = []
                for k in ks:
                    ''' contrastive loss '''
                    # compute logits
                    logit = q @ k.transpose(2, 1)/0.07 #[b, n, n]
                    # label = torch.arange(n,dtype=torch.long).cuda()
                    label = torch.argmax(weight,dim=-1)
                    # label = label[None].repeat(b,1) # [b, n]
                    logit = logit.reshape([b*n, n])
                    label = label.reshape(b*n)
                    logits.append(logit)
                    labels.append(label)
            else:
                inp_q = points[:b]
                if args.normal:
                    inp_q = torch.cat([inp_q,norms[:b]],dim=1)
                inp_q = {
                    "coord": points[:b].transpose(2, 1).reshape(b*n, 3),
                    "feat": inp_q.transpose(2, 1).reshape(b*n, -1),
                    "offset": torch.cumsum(torch.ones(b).cuda()*n,dim=0)
                }
                inp_k = points[b:]
                if args.normal:
                    inp_k = torch.cat([inp_k,norms[b:]],dim=1)
                inp_k = {
                    "coord": points[b:].transpose(2, 1).reshape(b*n, 3),
                    "feat": inp_k.transpose(2, 1).reshape(b*n, -1),
                    "offset": torch.cumsum(torch.ones(b).cuda()*n,dim=0)
                }
                logits, labels, q = model(inp_q, [inp_k], is_train=False)
                q_feats.append(q.cpu().data.numpy())

            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # measure accuracy and record loss
            try:
                acc1, acc5 = accuracy(logits[0], labels[0], topk=(1, 5))
                acc1s += acc1.item()
                acc5s += acc5.item()
            except Exception as e:
                print(f'error when computing accuracy')

        q_feats = np.concatenate(q_feats,axis=0)
        q_points = np.concatenate(q_points,axis=0)
        q_feats2 = np.concatenate(q_feats2,axis=0)
        q_points2 = np.concatenate(q_points2,axis=0)
        if is_aug:
            rot_rands = np.concatenate(rot_rands,axis=0)
            rot_rands2 = np.concatenate(rot_rands2,axis=0)

        acc1s = acc1s/len(testDataLoader)
        acc5s = acc5s/len(testDataLoader)
            
    return acc1s, acc5s, (q_ids, q_feats, q_points, rot_rands, q_feats2, q_points2, rot_rands2)


def eval_other_dataset(model, model_partseg, data_dir, args, save_results=False, results_path=''):
    os.makedirs(results_path,exist_ok=True)
    model.eval()
    model_partseg.eval()
    
    # load template
    temp_mesh = trimesh.load(args.temp_mesh_dir)
    verts = temp_mesh.vertices
    vs_max = verts.max(axis=0)[None]
    vs_min = verts.min(axis=0)[None]
    scale = 2/np.linalg.norm(vs_max - vs_min,axis=-1,keepdims=True)
    pt_tmp = temp_mesh.sample(args.npoint)
    pt_tmp = scale * (pt_tmp - (vs_max+vs_min)/2)
    verts_tmp = scale * (verts - (vs_max+vs_min)/2)
    pt = torch.from_numpy(pt_tmp).to(device=device,dtype=dtype)
    verts_torch = torch.from_numpy(verts_tmp).to(device=device,dtype=dtype)
    
    def run_model(pts):
        n = pts.shape[0]
        b = 1
        inp = {
            "coord": pts.reshape(n, 3),
            "feat": pts.reshape(n, -1),
            "offset": torch.cumsum(torch.ones(b).cuda()*n,dim=0)
        }
        
        seg_pred = model(inp)
        temp_part = model_partseg(seg_pred)
        temp_part = temp_part.max(dim=-1)[1].cpu().data.numpy()
        seg_pred = F.normalize(seg_pred, dim=-1)
        temp_feat = seg_pred.reshape(n, -1)
        return temp_feat, temp_part
    
    temp_feat, temp_part = run_model(pt)
    part_col = np.array(part_colors)
    mesh = o3d.geometry.PointCloud()
    mesh.points = o3d.utility.Vector3dVector(pt_tmp)
    mesh.colors = o3d.utility.Vector3dVector(part_col[temp_part])
    o3d.io.write_point_cloud(f'{results_path}/temp_part.ply', mesh)

    
    # save for verts
    temp_feat_vert, temp_part_vert = run_model(verts_torch)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_tmp)
    mesh.vertex_colors = o3d.utility.Vector3dVector(part_col[temp_part_vert])
    mesh.triangles = o3d.utility.Vector3iVector(temp_mesh.faces)
    o3d.io.write_triangle_mesh(f'{results_path}/temp_part_mesh.ply', mesh)

    fdirs = sorted(glob.glob(f'{data_dir}/dolphin*.obj'))
    q_ids = []
    q_feats = [] # store the normalized feature output
    q_points = []

    cols_max = np.array([[1,0,0.]])
    cols_min = np.array([[0,0,1.]])
    with torch.no_grad():
        for fdir in fdirs:
            key = fdir.split('/')[-1]
            mesh = trimesh.load(fdir)
            points = mesh.sample(args.npoint)
            verts = mesh.vertices
            vs_max = verts.max(axis=0)[None]
            vs_min = verts.min(axis=0)[None]
            scale = 2/np.linalg.norm(vs_max - vs_min, axis=-1, keepdims=True)
            points = scale * (points - (vs_max + vs_min)/2)
            verts = scale * (verts - (vs_max + vs_min)/2)
            
            points_orig = torch.from_numpy(points).float().cuda()
            n, _ = points_orig.shape
            b = 1
            points = points_orig
            
            seg_pred, part_seg = run_model(points)

            # inp = points
            # inp = {
            #     "coord": points.reshape(b*n, 3),
            #     "feat": inp.reshape(b*n, -1),
            #     "offset": torch.cumsum(torch.ones(b).cuda()*n,dim=0)
            # }
            # seg_pred = model(inp)
            # part_seg = model_partseg(seg_pred)
            # part_seg = part_seg.max(dim=-1)[1].cpu().data.numpy()
            # seg_pred = F.normalize(seg_pred, dim=-1)
            # seg_pred = seg_pred.reshape(n, -1)
            # # np.save(f'{results_path}/{key}.npy', seg_pred.cpu().data.numpy())
            # 
            simi = (seg_pred @ temp_feat.transpose(1,0) + 1)/2
            simi = simi.cpu().data.numpy()
            ids = np.random.choice(np.arange(n),10)
            for id in ids:
                mesh_source = o3d.geometry.PointCloud()
                mesh_source.points = o3d.utility.Vector3dVector(points.reshape([-1,3]).cpu().data.numpy())
                mesh_source.colors = o3d.utility.Vector3dVector(part_col[part_seg])
                o3d.io.write_point_cloud(f'{results_path}/{key}_part.ply', mesh_source)

                mesh_source = o3d.geometry.PointCloud()
                mesh_source.points = o3d.utility.Vector3dVector(points.reshape([-1,3]).cpu().data.numpy())
                col = np.ones([n, 3]) * 0.8
                col[id] = np.array([1.0,0,0])
                mesh_source.colors = o3d.utility.Vector3dVector(col)
                o3d.io.write_point_cloud(f'{results_path}/{key}_{id:03d}.ply', mesh_source)

                mesh_ref = o3d.geometry.PointCloud()
                mesh_ref.points = o3d.utility.Vector3dVector(pt_tmp.reshape([-1,3]))
                col = np.ones([n, 3]) * 0.8
                col[simi[id]>0.7,:] = np.array([[0,0,1.0]])
                col[simi[id]>0.8,:] = np.array([[0,1.0,0]])
                col[simi[id]>0.9,:] = np.array([[1.0,0,0]])
                # col = cols_max * simi[id][:,None] + cols_min * (1-simi[id][:,None])
                mesh_ref.colors = o3d.utility.Vector3dVector(col)
                o3d.io.write_point_cloud(f'{results_path}/{key}_{id:03d}_ref.ply', mesh_ref)

            # for verts 
            
            verts_torch = torch.from_numpy(verts).float().cuda()
            seg_pred_vert, part_seg_vert = run_model(verts_torch)

            mesh_source = o3d.geometry.TriangleMesh()
            mesh_source.vertices = o3d.utility.Vector3dVector(verts)
            mesh_source.vertex_colors = o3d.utility.Vector3dVector(part_col[part_seg_vert])
            mesh_source.triangles = o3d.utility.Vector3iVector(mesh.faces)
            o3d.io.write_triangle_mesh(f'{results_path}/{key}_part_mesh.ply', mesh_source)

            simi = (seg_pred_vert @ temp_feat_vert.transpose(1,0) + 1)/2
            simi = simi.cpu().data.numpy()
            ids = np.random.choice(np.arange(n),10)
            for id in ids:
                mesh_source = o3d.geometry.PointCloud()
                mesh_source.points = o3d.utility.Vector3dVector(verts.reshape([-1,3]))
                col = np.ones([verts.shape[0], 3]) * 0.8
                col[id] = np.array([1.0,0,0])
                mesh_source.colors = o3d.utility.Vector3dVector(col)
                o3d.io.write_point_cloud(f'{results_path}/{key}_{id:03d}_mesh.ply', mesh_source)

                mesh_ref = o3d.geometry.PointCloud()
                mesh_ref.points = o3d.utility.Vector3dVector(verts_tmp.reshape([-1,3]))
                col = np.ones([verts_tmp.shape[0], 3]) * 0.8
                col[simi[id]>0.7,:] = np.array([[0,0,1.0]])
                col[simi[id]>0.8,:] = np.array([[0,1.0,0]])
                col[simi[id]>0.9,:] = np.array([[1.0,0,0]])
                mesh_ref.colors = o3d.utility.Vector3dVector(col)
                o3d.io.write_point_cloud(f'{results_path}/{key}_{id:03d}_ref_mesh.ply', mesh_ref)

    # return q_ids, q_points, q_feats


def eval_weapons_2800(model, data_json, args, save_results=False, results_path=''):
    model.eval()
    
    with open(data_json,'r') as f:
        data_json = json.load(f)
    
    q_ids = []
    q_feats = [] # store the normalized feature output
    q_points = []
    cats = []
    with torch.no_grad():
        for key in tqdm(data_json):
            obj_id = key.split('_')[0]
            fdir = f'/apdcephfs_cq8/share_2909871/Assets/objaverse/render_free/models/axisaligned/common_230k/models/{obj_id}/{obj_id}_manifold_full.obj'
            mesh = o3d.io.read_triangle_mesh(fdir)
            verts = np.array(mesh.vertices)
            norms = np.array(mesh.vertex_normals)
            verts, scale, cen = normalize_2_sphere(verts)
        
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            pcd = mesh.sample_points_uniformly(number_of_points=args.npoint)
            pts = np.array(pcd.points)
            norms = np.array(pcd.normals)

            points_orig = torch.from_numpy(pts).float().cuda()[None]
            b, n, _ = points_orig.shape
            norms_orig = torch.from_numpy(norms).float().cuda()[None]
            
            # rot_rand = torch.randn([points_orig.shape[0],3]).float().cuda()
            # rot_rand = axis_angle_to_matrix(rot_rand)
            # points = torch.matmul(rot_rand, points_orig.transpose(2,1)).transpose(2,1).contiguous()
            # norms = torch.matmul(rot_rand, norms_orig.transpose(2,1)).transpose(2,1).contiguous()
            points = points_orig
            norms = norms_orig

            inp = points
            if args.normal:
                inp = torch.cat([inp,norms],dim=1)

            inp = {
                "coord": points.reshape(b*n, 3),
                "feat": inp.reshape(b*n, -1),
                "offset": torch.cumsum(torch.ones(b).cuda()*n,dim=0)
            }
            
            seg_pred = model(inp)
            seg_pred = F.normalize(seg_pred, dim=-1)
            seg_pred = seg_pred.reshape(b, n, -1)
            q_ids.append(key)
            q_points.append(points.cpu().data.numpy())
            q_feats.append(seg_pred.cpu().data.numpy())
            cats.append(data_json[key]['category'] if 'category' in data_json[key] else 'unknown')
            
        q_feats = np.concatenate(q_feats,axis=0)
        q_points = np.concatenate(q_points,axis=0)

    return q_ids, q_points, q_feats, cats


if __name__ == '__main__':
    args = parse_args()
    main(args)
