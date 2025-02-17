import argparse

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import trimesh
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from skimage.io import imsave
from tqdm import tqdm

import mcubes

from renderer.renderer import NeuSRenderer, DEFAULT_SIDE_LENGTH
from util import instantiate_from_config, read_pickle, output_points

import os, pdb
import copy
import open3d as o3d 


CLUSTER_COLOR_MAP_40 = {-1: (0., 0., 0.), 0: (174., 199., 232.), 1: (152., 223., 138.), 2: (31., 119., 180.), 3: (255., 187., 120.), 4: (188., 189., 34.), 5: (140., 86., 75.),
                        6: (255., 152., 150.), 7: (214., 39., 40.), 8: (197., 176., 213.), 9: (148., 103., 189.), 10: (196., 156., 148.), 11: (23., 190., 207.), 12: (247., 182., 210.), 
                        13: (219., 219., 141.), 14: (255., 127., 14.), 15: (158., 218., 229.), 16: (44., 160., 44.), 17: (112., 128., 144.), 18: (227., 119., 194.), 19: (82., 84., 163.), 
                        20: (232., 199., 174.), 21: (138., 223., 152.), 22: (180., 119., 31.), 23: (120., 187., 255.), 24: (34., 189., 188.), 25: (75., 86., 140.),
                        26: (150., 152., 255.), 27: (40., 39., 214.), 28: (213., 176., 197.), 29: (189., 103., 148.), 30: (148., 156., 196.), 31: (207., 190., 23.), 32: (210., 182., 247.), 
                        33: (141., 219., 219.), 34: (14., 127., 255.), 35: (229., 218., 158.), 36: (44., 44., 160.), 37: (144., 128., 112.), 38: (194., 119., 227.), 39: (163., 84., 82.)}


class ResumeCallBacks(Callback):
    def __init__(self):
        pass

    def on_train_start(self, trainer, pl_module):
        pl_module.optimizers().param_groups = pl_module.optimizers()._optimizer.param_groups


def extract_fields(bound_min, bound_max, resolution, query_func, batch_size=64, outside_val=1.0):
    N = batch_size
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                    val = query_func(pts).detach()
                    outside_mask = torch.norm(pts,dim=-1)>=1.0
                    val[outside_mask]=outside_val
                    val = val.reshape(len(xs), len(ys), len(zs)).cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_fields_sem(bound_min, bound_max, resolution, query_func, batch_size=64, outside_val=-1.0):
    N = batch_size
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts.numpy())
                    outside_mask = torch.norm(pts,dim=-1)>=1.0
                    val[outside_mask]=outside_val
                    val = val.reshape(len(xs), len(ys), len(zs))
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, color_func, partseg_func, output_dir, outside_val=1.0):

    u = extract_fields(bound_min, bound_max, resolution, query_func, outside_val=outside_val)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    # change surface normal direction
    triangles_ = np.copy(triangles)
    for idt in range(triangles.shape[0]):
        triangles_[idt][1] = triangles[idt][2]
        triangles_[idt][2] = triangles[idt][1]
    triangles = triangles_
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    vertex_colors = color_func(vertices.copy())

    # visualize part segmentation
    y_pred = partseg_func(vertices.copy())
    
    partseg_colors = np.array([CLUSTER_COLOR_MAP_40[label % 40] for label in y_pred.copy()])
    partseg_colors = partseg_colors / 255

    return vertices, triangles, vertex_colors, partseg_colors, y_pred



def extract_mesh(model, output, obj_name, resolution=512):
    # if not isinstance(model.renderer, NeuSRenderer): return
    
    bbox_min = -torch.ones(3)*DEFAULT_SIDE_LENGTH
    bbox_max = torch.ones(3)*DEFAULT_SIDE_LENGTH
    with torch.no_grad():
        vertices, triangles, vertex_colors, partseg_colors, label_pred = extract_geometry(bbox_min, bbox_max, resolution, 0, lambda x: model.renderer.sdf_network.sdf(x), lambda x: model.renderer.get_vertex_colors(x), lambda x: model.renderer.get_vertex_partseg_feats(x), output)

    # # output geometry
    # mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)
    # mesh.export(str(f'{output}/test_mesh.ply'))

    mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=(partseg_colors*255.).astype(np.uint8), process=False)
    mesh.export(str(f'{output}/samauto_vis.ply'))

    mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors, process=False)
    mesh.export(str(f'{output}/samauto_vis.obj'))

    pcd0 = trimesh.PointCloud(vertices[label_pred == 0])
    pcd0.export(str(f'{output}/pcd_0.ply'))
    pcd1 = trimesh.PointCloud(vertices[label_pred == 1])
    pcd1.export(str(f'{output}/pcd_1.ply'))

    
    omesh = mesh.as_open3d
    omesh1 = omesh.select_by_index(np.arange(len(vertices))[label_pred == 1])
    o3d.io.write_triangle_mesh(str(f'{output}/omesh_1.obj'), omesh1)

    omesh0 = omesh.select_by_index(np.arange(len(vertices))[label_pred == 0])
    o3d.io.write_triangle_mesh(str(f'{output}/omesh_0.obj'), omesh0)
        
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-b', '--base', type=str, default='configs/neus_cw.yaml')
    parser.add_argument('-s', '--seed', type=int, default=6033)
    parser.add_argument('-g', '--gpus', type=str, default='0,')
    parser.add_argument('-r', '--resume', action='store_true', default=False, dest='resume')
    parser.add_argument('--fp16', action='store_true', default=False, dest='fp16')
    opt = parser.parse_args()
    # seed_everything(opt.seed)

    # estimate_partnum(opt.name)

    # configs
    cfg = OmegaConf.load(opt.base)
    name = opt.name
    log_dir, ckpt_dir = Path("output") / name, Path("output") / name / 'ckpt'
    cfg.model.params['image_path'] = os.path.join("../../single2multi/output/mvimgs", opt.name, "mvout.png")
    cfg.model.params['log_dir'] = log_dir

    # setup
    log_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    trainer_config = cfg.trainer
    callback_config = cfg.callbacks
    model_config = cfg.model
    data_config = cfg.data

    data_config.params.seed = opt.seed
    data = instantiate_from_config(data_config)
    data.prepare_data()
    data.setup('fit')

    model = instantiate_from_config(model_config,)
    model.cpu()
    model.learning_rate = model_config.base_lr

    # logger
    logger = TensorBoardLogger(save_dir=log_dir, name='tensorboard_logs')
    callbacks=[]
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, filename="{epoch:06}", verbose=True, save_last=True, every_n_train_steps=callback_config.save_interval))

    # trainer
    trainer_config.update({
        "accelerator": "cuda", "check_val_every_n_epoch": None,
        "benchmark": True, "num_sanity_val_steps": 0,
        "devices": 1, "gpus": opt.gpus,
    })
    if opt.fp16:
        trainer_config['precision']=16

    if opt.resume:
        callbacks.append(ResumeCallBacks())
        trainer_config['resume_from_checkpoint'] = str(ckpt_dir / 'last.ckpt')
    else:
        if (ckpt_dir / 'last.ckpt').exists():
            raise RuntimeError(f"checkpoint {ckpt_dir / 'last.ckpt'} existing ...")
    trainer = Trainer.from_argparse_args(args=argparse.Namespace(), **trainer_config, logger=logger, callbacks=callbacks)

    trainer.fit(model, data)

    model = model.cuda().eval()

    extract_mesh(model, log_dir, name, resolution=256)

if __name__=="__main__":
    main()