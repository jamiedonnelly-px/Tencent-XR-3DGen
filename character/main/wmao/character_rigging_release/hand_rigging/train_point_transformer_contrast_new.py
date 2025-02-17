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

def majority_element(array):
    unique, counts = np.unique(array, return_counts=True)
    return unique[np.argmax(counts)]

def load_data(data_dir, is_augment=True, mode='train'):
    test_character = ['Ch06_nonPBR', 'Ch05_nonPBR',
        'Paladin WProp J Nordstrom', 'Maw J Laygo', 'Aj',
       'Ch17_nonPBR', 'Ch41_nonPBR', 'Uriel A Plotexia',
       'Ch25_nonPBR', 'Ty', 'Ch42_nonPBR', 'Ch45_nonPBR',
       'Ch37_nonPBR', 'exo_red', 'Ch19_nonPBR',
       'Ch21_nonPBR', 'Sporty Granny', 'Ch06_nonPBR',
       'Ely By K.Atienza', 'Castle Guard 02']

    """load data and model"""
    points = []
    parts = []
    npt = 20480
    file_names = []
    for fn in os.listdir(data_dir):
        if mode == 'train':
            if fn.split('_left.npz')[0].split('_right.npz')[0] in test_character:
                continue
        elif mode == 'test':
            if not fn.split('_left.npz')[0].split('_right.npz')[0] in test_character:
                continue
        
        file_path = os.path.join(data_dir, fn)
        file_names.append(fn.split('.npz')[0])
        data = np.load(file_path)
        v = data['verts']
        # normalize
        v = v - v.mean(axis=0)
        pmax = v.max(axis=0)
        pmin = v.min(axis=0)
        scale = 2.0 / np.linalg.norm(pmax-pmin)
        v = v * scale
        
        f = np.int32(data['faces'])
        p = data['vert_parts']
        fp = p[f]
        fp = np.apply_along_axis(majority_element, axis=-1, arr=fp)
        
        mesh = trimesh.Trimesh(vertices=v, faces=f)
        try:
            pts, face_indices = trimesh.sample.sample_surface(mesh, npt)
        except:
            st()
        # cdist = np.linalg.norm(v[None]-pts[:,None],axis=-1)
        # p2vid = np.argmin(cdist, axis=1)
        # part = p[p2vid]
        part = fp[face_indices]

        parts.append(part[None])
        points.append(pts[None])
        if is_augment:
            p_max = part.max() + 1
            pid2remain = [[0, 1, 2, 3, 4], [0, 1, 2, 3], [0, 1, 2], [0, 2]]
            pid2remain = pid2remain[6-p_max:]
            for pids in pid2remain:
                pids = np.setdiff1d(np.arange(p_max), np.array(pids))
                face2remove = []
                for pid in pids:
                    face2remove.append(np.where(fp==pid)[0])
                face2remove = np.concatenate(face2remove)
                face_remain = np.setdiff1d(np.arange(f.shape[0]),face2remove)
                vert2remove = np.unique(f[face2remove].reshape(-1))
                vert_remain = np.setdiff1d(np.arange(v.shape[0]), vert2remove)
                new_vert_idxs = np.arange(len(vert_remain))
                
                old2new = np.zeros((v.shape[0],)) - 1
                old2new[vert_remain] = new_vert_idxs
                old2new = np.int32(old2new)
                
                vnew = v[vert_remain]
                fnew = f[face_remain]
                fnew = old2new[fnew]
                fnew_remain = np.where(np.all(fnew>=0,axis=-1))[0]
                fnew =fnew[fnew_remain]
                fpnew = fp[face_remain][fnew_remain]
                mesh = trimesh.Trimesh(vertices=vnew, faces=fnew)
                try:
                    pts, face_indices = trimesh.sample.sample_surface(mesh, npt)
                except:
                    st()
                part = fpnew[face_indices]
                parts.append(part[None])
                points.append(pts[None])

            
            # for _ in range(5):
            #     # random remove one or 2 parts
            #     pids = np.random.choice(np.arange(fp.max()+1), 2 if np.random.rand(1)[0] > 0.5 else 1)
            #     face2remove = []
            #     for pid in pids:
            #         face2remove.append(np.where(fp==pid)[0])
            #     face2remove = np.concatenate(face2remove)
            #     face_remain = np.setdiff1d(np.arange(f.shape[0]),face2remove)
            #     vert2remove = np.unique(f[face2remove].reshape(-1))
            #     vert_remain = np.setdiff1d(np.arange(v.shape[0]), vert2remove)
            #     new_vert_idxs = np.arange(len(vert_remain))
                
            #     old2new = np.zeros((v.shape[0],)) - 1
            #     old2new[vert_remain] = new_vert_idxs
            #     old2new = np.int32(old2new)
                
            #     vnew = v[vert_remain]
            #     fnew = f[face_remain]
            #     fnew = old2new[fnew]
            #     fnew_remain = np.where(np.all(fnew>=0,axis=-1))[0]
            #     fnew =fnew[fnew_remain]
            #     fpnew = fp[face_remain][fnew_remain]
            #     mesh = trimesh.Trimesh(vertices=vnew, faces=fnew)
            #     try:
            #         pts, face_indices = trimesh.sample.sample_surface(mesh, npt)
            #     except:
            #         st()
            #     part = fpnew[face_indices]
            #     parts.append(part[None])
            #     points.append(pts[None])

    points = np.concatenate(points,axis=0)
    parts = np.concatenate(parts,axis=0)
    return points, parts, file_names

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

    parser.add_argument('--data_dir', type=str, default='/aigc_cfs_2/weimao/hand_rigging/data/npz', help='dataset dir')
    parser.add_argument('--temp_mesh_dir', type=str, default='', help='template mesh dir')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--test_data_dir', type=str, default='', help='')
    
    # training config
    # parser.add_argument('--pretrained_dir', type=str, default='/aigc_cfs_2/weimao/non-smalfit/output/SK_Elephant/contrast_rot_aug_PointTransformerSeg38/2024-06-27_18-56-19', help='consume training')
    parser.add_argument('--pretrained_dir', type=str, default='none', help='consume training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch Size during training')
    parser.add_argument('--epoch', default=1000, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--decay_rate', type=float, default=1e-5, help='weight decay')
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
    os.makedirs('./output', exist_ok=True)

    pts, parts, file_names = load_data(args.data_dir, is_augment=True, mode='train')
    pts_test, parts_test, file_names_test = load_data(args.data_dir, is_augment=True, mode='test')
    print(f'data size {pts.shape[0]}')
    
    """save part segmentation for debug"""
    # if True:
    pid = 10
    part_col = np.array(part_colors)
    mesh = o3d.geometry.PointCloud()
    mesh.points = o3d.utility.Vector3dVector(pts[pid])
    mesh.colors = o3d.utility.Vector3dVector(part_col[parts[pid]])
    o3d.io.write_point_cloud(f'./output/parts_test.ply', mesh)
    
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if args.pretrained_dir == 'none':
        exp_dir = f"./output/hand_part_seg_{'moco_' if args.is_moco else ''}{'rot_' if args.rot_aug else ''}{'scale_' if args.scale_aug else ''}{'trans_' if args.trans_aug else ''}{'noise_' if args.noise_aug else ''}{'aug_' if args.rot_aug or args.trans_aug or args.scale_aug or args.noise_aug else ''}{args.model}/{timestr}"
    else:
        exp_dir = args.pretrained_dir
    
    checkpoints_dir = f'{exp_dir}/checkpoints/'
    os.makedirs(checkpoints_dir, exist_ok=True)
    log_dir = f'{exp_dir}/logs/'
    os.makedirs(log_dir, exist_ok=True)

    
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
    log_string("The number of training data is: %d" % pts.shape[0])

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
        best_test_acc1 = checkpoint['test_acc1_part']
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

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoch//(args.epoch//20), gamma=0.7)
    criterion = nn.CrossEntropyLoss().cuda()

    log_loss = []
    log_acc1 = []
    log_acc5 = []
    log_acc1_part = []
    log_test_acc1_part = []
    n_data = pts.shape[0]
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch)) 
        model.train()
        model_partseg.train()
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
                pts_part = torch.from_numpy(parts[id][:,shuffled_point_idx]).to(device=device, dtype=torch.long) #[bs, pn]
                
                shuffled_point_idx2 = np.arange(pts.shape[1])
                np.random.shuffle(shuffled_point_idx2)
                shuffled_point_idx2 = shuffled_point_idx2[:args.npoint]
                points2 = torch.from_numpy(pts[id][:,shuffled_point_idx2]).to(device=device, dtype=dtype) #[bs, pn, 3]
                pts_part2 = torch.from_numpy(parts[id][:,shuffled_point_idx2]).to(device=device, dtype=torch.long) #[bs, pn]

                b, n, _ = points_orig.shape
                
                assert args.rot_aug or args.scale_aug or args.noise_aug or args.trans_aug
                
                if args.rot_aug or args.scale_aug or args.noise_aug or args.trans_aug:
                    points = torch.cat([points_orig, points_orig.clone(), points2], dim=0)
                    label1 = torch.arange(points_orig.shape[1])[None].repeat([b, 1]).to(device=device,dtype=torch.long)#[bs, pn]
                    
                    tmp_cdist = torch.cdist(points_orig, points2) #[bs, pn, 3]
                    label2 = tmp_cdist.min(dim=-1)[1] #[bs, pn]
                    label = torch.cat([label1,label2], dim=0) #[bb, pn]
                    label_part = torch.cat([pts_part,pts_part.clone(),pts_part2.clone()], dim=0)
                    
                    # debug
                    if False:
                        jjs = [5,9]
                        pps = [100,1000]
                        for jj in jjs:
                            for pp in pps:
                                mesh_source = o3d.geometry.PointCloud()
                                mesh_source.points = o3d.utility.Vector3dVector(points_orig[jj].reshape([-1,3]).cpu().data.numpy())
                                col = np.ones([n, 3]) * 0.8
                                col[pp] = np.array([1.0,0,0])
                                mesh_source.colors = o3d.utility.Vector3dVector(col)
                                o3d.io.write_point_cloud(f'./output/{jj:03d}_{pp:03d}.ply', mesh_source)

                                mesh_ref = o3d.geometry.PointCloud()
                                mesh_ref.points = o3d.utility.Vector3dVector(points2.clone()[jj].reshape([-1,3]).cpu().data.numpy())
                                col = np.ones([n, 3]) * 0.8
                                col[label2[jj,pp].item()] = np.array([1.0,0,0])
                                mesh_ref.colors = o3d.utility.Vector3dVector(col)
                                o3d.io.write_point_cloud(f'./output/{jj:03d}_{pp:03d}_ref.ply', mesh_ref)
                
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
            #     loss += criterion(logit.reshape([b*n,-1]), label.reshape([-1]))
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


        acc1s_part_eval = eval_epoch(model, model_partseg, pts_test, parts_test, args)
        acc1s = acc1s/len(idxs)
        acc5s = acc5s/len(idxs)
        acc1s_part = acc1s_part/len(idxs)
        loss_all = loss_all/len(idxs)
        log_string(f'Loss is {loss_all:.5f}, Test part seg acc1 is: {acc1s_part_eval:.1f}  Train part seg acc1 is: {acc1s_part:.1f}')

        log_acc1.append(acc1s)
        log_acc5.append(acc5s)
        log_acc1_part.append(acc1s_part)
        log_test_acc1_part.append(acc1s_part_eval)
        log_loss.append(loss_all)
        
        if (acc1s_part_eval >= best_test_acc1):
            best_test_acc1 = acc1s_part_eval
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc1_part': acc1s_part,
                'test_acc1_part': acc1s_part_eval,
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
            'train_acc1_part': acc1s_part,
            'test_acc1_part': acc1s_part_eval,
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
    plt.clf()
    plt.cla()
    plt.plot(log_test_acc1_part)
    plt.savefig(f"{exp_dir}/logs/test_acc1_part.jpg")
    
     
def eval_epoch(model, model_partseg, pts, parts, args, save_results=False, results_path=''):
    model.eval()
    model_partseg.eval()
    n_data = pts.shape[0]
    acc1s_part = 0
    idxs = torch.split(torch.arange(n_data), 10)
    cov = 0.06
    """training epoch """ 
    for i, id in tqdm(enumerate(idxs), total=len(idxs), smoothing=0.9):
        # verts = torch.from_numpy(vs[id]).to(device=device, dtype=dtype) #[bs, vn, 3]
        with torch.no_grad():
            shuffled_point_idx = np.arange(pts.shape[1])
            np.random.shuffle(shuffled_point_idx)
            shuffled_point_idx = shuffled_point_idx[:args.npoint]
            points = torch.from_numpy(pts[id][:,shuffled_point_idx]).to(device=device, dtype=dtype) #[bs, pn, 3]
            pts_part = torch.from_numpy(parts[id][:,shuffled_point_idx]).to(device=device, dtype=torch.long) #[bs, pn]
        
            b, n, _ = points.shape
    
            inp = points
            inp = {
                "coord": points.reshape(b*n, 3),
                "feat": inp.reshape(b*n, -1),
                "offset": torch.cumsum(torch.ones(b).cuda()*n,dim=0)
            }
            seg_pred = model(inp) # [b*2*n, out_feat]
            part_logit = model_partseg(seg_pred)
            acc1_part = accuracy2(part_logit[None], pts_part.reshape([1,-1]), topk=(1,))
            acc1s_part += acc1_part[0].item()

    acc1s_part = acc1s_part/len(idxs)        
    return acc1s_part


def eval_other_dataset(model, model_partseg, data_dir, args, save_results=False, results_path=''):
    os.makedirs(results_path,exist_ok=True)
    model.eval()
    model_partseg.eval()
    part_col = np.array(part_colors)

    def run_model(pts):
        n = pts.shape[0]
        b = 1
        inp = {
            "coord": pts.reshape(n, 3),
            "feat": pts.reshape(n, -1),
            "offset": torch.cumsum(torch.ones(b).to(dtype=torch.int32).cuda()*n,dim=0)
        }
        
        seg_pred = model(inp)
        temp_part = model_partseg(seg_pred)
        temp_part = temp_part.max(dim=-1)[1].cpu().data.numpy()
        seg_pred = F.normalize(seg_pred, dim=-1)
        temp_feat = seg_pred.reshape(n, -1)
        return temp_feat, temp_part
    
    pts, parts, file_names = load_data(data_dir, is_augment=False, mode='test')

    q_ids = []
    q_feats = [] # store the normalized feature output
    q_points = []

    cols_max = np.array([[1,0,0.]])
    cols_min = np.array([[0,0,1.]])
    with torch.no_grad():
        shuffled_point_idx = np.arange(pts.shape[1])
        np.random.shuffle(shuffled_point_idx)
        shuffled_point_idx = shuffled_point_idx[:args.npoint]
        points_orig = torch.from_numpy(pts[:,shuffled_point_idx]).to(device=device, dtype=dtype) #[bs, pn, 3]
        pts_part = torch.from_numpy(parts[:,shuffled_point_idx]).to(device=device, dtype=torch.long) #[bs, pn]
        b, n, _ = points_orig.shape
        bids = torch.split(torch.arange(b),10)
        for ids in bids:
            points = points_orig[ids]
            parts = pts_part[ids]
            bb, n, _ = points.shape
            inp = points
            inp = {
                "coord": points.reshape(bb*n, 3),
                "feat": inp.reshape(bb*n, -1),
                "offset": torch.cumsum(torch.ones(bb).cuda()*n,dim=0)
            }
            seg_pred = model(inp)
            part_logit = model_partseg(seg_pred)
            acc1_part = accuracy2(part_logit[None], parts.reshape([1,-1]), topk=(1,))

            temp_part = part_logit.max(dim=-1)[1].cpu().data.numpy().reshape((bb,-1))
            for id in np.arange(ids.shape[0]):
                part_seg = temp_part[id]
                mesh_source = o3d.geometry.PointCloud()
                mesh_source.points = o3d.utility.Vector3dVector(points[id].reshape([-1,3]).cpu().data.numpy())
                mesh_source.colors = o3d.utility.Vector3dVector(part_col[part_seg])
                o3d.io.write_point_cloud(f'{results_path}/{file_names[ids[id].item()]}_part.ply', mesh_source)


if __name__ == '__main__':
    args = parse_args()
    main(args)
