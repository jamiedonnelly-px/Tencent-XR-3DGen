import argparse

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from skimage.io import imsave
from tqdm import tqdm

import mcubes
import pickle

from util import instantiate_from_config, read_pickle, output_points

import os, pdb
from renderer.gs_renderer import MiniCam
from scipy.spatial.transform import Rotation as R
from renderer.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh, Mesh
import open3d as o3d
import math

COLOR_MAP_20 = {-1: (0., 0., 0.), 0: (174., 199., 232.), 1: (152., 223., 138.), 2: (31., 119., 180.), 3: (255., 187., 120.), 4: (188., 189., 34.), 5: (140., 86., 75.),
                6: (255., 152., 150.), 7: (214., 39., 40.), 8: (197., 176., 213.), 9: (148., 103., 189.), 10: (196., 156., 148.), 11: (23., 190., 207.), 12: (247., 182., 210.), 
                13: (219., 219., 141.), 14: (255., 127., 14.), 15: (158., 218., 229.), 16: (44., 160., 44.), 17: (112., 128., 144.), 18: (227., 119., 194.), 19: (82., 84., 163.)}


class ResumeCallBacks(Callback):
    def __init__(self):
        pass

    def on_train_start(self, trainer, pl_module):
        pl_module.optimizers().param_groups = pl_module.optimizers()._optimizer.param_groups

def render_images(model, output, num_imgs, i_sem=-1):
    # render from model
    # K, _, _, _, poses = read_pickle(f'meta_info/camera-16.pkl')
    K, _, _, _, poses = read_pickle(f'../../meta_info/camera-{num_imgs}.pkl')
    logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)
    
    h, w = 256, 256
    default_size = 256
    K = np.diag([w/default_size,h/default_size,1.0]) @ K
    imgs = []
    deps = []
    alps = []
    nors = []
    sems = []
    # n = 16
    n = num_imgs
    for ni in tqdm(range(n)):
        pose = poses[ni]

        cam = MiniCam(w2c=pose, width=w, height=h, K=K)
        with torch.no_grad():
            # pdb.set_trace()
            outs = model.renderer.render(cam, i_sem=i_sem)
            process = lambda x: (x.cpu().numpy() * 255).astype(np.uint8)
            rgb = torch.clamp(outs['image'].permute(1,2,0), max=1.0, min=0.0)
            mask = torch.clamp(outs['alpha'].permute(1,2,0), max=1.0, min=0.0)
            mask = torch.repeat_interleave(mask, 3, dim=-1)
            depth = outs['depth'].permute(1,2,0)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
            depth = torch.repeat_interleave(depth, 3, dim=-1)

        imgs.append(process(rgb))
        deps.append(process(depth))
        alps.append(process(mask))
        
    imgs = np.concatenate(imgs, 1)
    deps = np.concatenate(deps, 1)
    alps = np.concatenate(alps, 1)
    if i_sem == -1:
        imsave(f'{output}/save_depth.png', deps)
        imsave(f'{output}/save_alpha.png', alps)
        imsave(f'{output}/save_image.png', imgs)
    else:
        os.makedirs(os.path.join(output, f"sem_{i_sem:01d}"), exist_ok=True)
        imsave(f'{output}/sem_{i_sem:01d}/save_depth.png', deps)
        imsave(f'{output}/sem_{i_sem:01d}/save_alpha.png', alps)
        imsave(f'{output}/sem_{i_sem:01d}/save_image.png', imgs)



def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def look_at(campos, target, opengl=True):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    if not opengl:
        # camera forward aligns with -z
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


# elevation & azimuth to pose (cam2world) matrix
def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
    # radius: scalar
    # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    # return: [4, 4], camera pose matrix
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    # T = np.eye(4, dtype=np.float32)
    # T[:3, :3] = look_at(campos, target, opengl)
    # T[:3, 3] = campos
    T = np.eye(4, dtype=np.float32)
    RT = look_at(campos, target, opengl).T

    # rectify
    T[:3, 0:1] = RT[:3, 2:3]
    T[:3, 1:2] = RT[:3, 0:1]
    T[:3, 2:3] = -RT[:3, 1:2]
    T[:3, 3] = -RT @ campos
    T[2, :] *= -1 
    return T[:3]


def orbit_camera_fibonacci(num_samples):
    K, azs, els, dists, poses = read_pickle(f'../../meta_info/camera-{16}.pkl')
    render_resolution=256
    cameras = []
    # for campos in cam_positions: 
    for azimuth in np.arange(0,360,360/num_samples):
        elevation = 30 # np.arcsin(campos[1] / radius)  # y
        # azimuth =  #np.arctan2(campos[2], campos[0])  # z, x

        # if is_degree:
        #     elevation = np.rad2deg(elevation)
        #     azimuth = np.rad2deg(azimuth)
        
        camera_matrix = orbit_camera(elevation, azimuth, dists[0])
 
        cur_cam = MiniCam(
            camera_matrix, render_resolution, render_resolution,  K
        )
        cameras.append(cur_cam)
    return cameras


def extract_geo(model, output, i_sem=-1):
    gaussExtractor = GaussianExtractor(model.renderer)
    n_fames = 100
    # pdb.set_trace()
    cameras = orbit_camera_fibonacci(n_fames)  
    gaussExtractor.reconstruction(cameras, i_sem)

    os.makedirs(os.path.join(output, f"imgs{n_fames}_{i_sem}"), exist_ok=True)

    from PIL import Image
    imgs = (gaussExtractor.rgbmaps.cpu().numpy()*255.).astype(np.uint8)
    for i in range(n_fames):
        Image.fromarray(imgs[i].transpose(1,2,0)).save(os.path.join(output, f"imgs{n_fames}_{i_sem}/{i:05d}.png"))


    return
    # mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=0.05, depth_trunc=depth_trunc)
    # if i_sem == -1:
    #     o3d.io.write_triangle_mesh(f'{output}/fused.ply', mesh) 
    # else:
    #     os.makedirs(os.path.join(output, f"sem_{i_sem:01d}"), exist_ok=True)
    #     o3d.io.write_triangle_mesh(f'{output}/sem_{i_sem:01d}/fused.ply', mesh) 




def render_physical(model, output, i_sem):
    # render from model
    # K, _, _, _, poses = read_pickle(f'meta_info/camera-16.pkl')
    K, _, _, _, poses = read_pickle(f'../../meta_info/camera-{16}.pkl')
    logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)

    with open(os.path.join(output, "editing_modifier.pkl"), "rb") as f:
        emdict = pickle.load(f)

    # pdb.set_trace()
    trajs = emdict['particles_trajectory_tn3']

    phy_path = os.path.join(output, f"phy_{i_sem}")
    os.makedirs(phy_path, exist_ok=True)
    
    h, w = 256, 256
    default_size = 256
    K = np.diag([w/default_size,h/default_size,1.0]) @ K
    imgs = []
    deps = []
    alps = []
    nors = []
    sems = []
    # n = 16

    # import json 
    # json_dir = "output/skeleton1/motion_data/"
    # deltas = []
    # for ni in range(100):
        # with open(os.path.join(json_dir, f"motion_data_{ni:06d}.json")) as fjson:
            # deltas.append(json.load(fjson)['2']['center_of_mass'])
    # deltas = torch.tensor(deltas).cuda()
    # deltas = deltas - deltas[0]
    # n = len(deltas)

    n = len(trajs)
    pose = poses[0]
    for ni in tqdm(range(n)):
        traj = trajs[ni]
        cam = MiniCam(w2c=pose, width=w, height=h, K=K)
        # model.renderer.gaussians._xyz = torch.from_numpy(traj).float().cuda()
        # if ni>=1:
            # model.renderer.gaussians._xyz[model.renderer.gaussians._sem[:,1]==1] += deltas[ni] - deltas[ni-1]
        # pdb.set_trace()
        with torch.no_grad():
            outs = model.renderer.render(cam, i_sem=i_sem, update=False) # set update==False is important
            process = lambda x: (x.cpu().numpy() * 255).astype(np.uint8)
            rgb = torch.clamp(outs['image'].permute(1,2,0), max=1.0, min=0.0)
            imsave(f'{phy_path}/{ni:05d}.png', process(rgb))
            



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-l', '--log', type=str, default='output/')
    parser.add_argument('-s', '--seed', type=int, default=6033)
    parser.add_argument('-g', '--gpus', type=str, default='0,')
    parser.add_argument('--numimgs', type=int, default=16)
    parser.add_argument('-r', '--resume', action='store_true', default=False, dest='resume')
    parser.add_argument('-p', '--phy', action='store_true', default=False, dest='phy')
    parser.add_argument('--fp16', action='store_true', default=False, dest='fp16')
    parser.add_argument('--rd360', action='store_true', default=False, dest='rd360')
    parser.add_argument('-t', '--prompt', type=str, default='')
    parser.add_argument('-m', '--negative_prompt', type=str, default='')
    opt = parser.parse_args()
    # seed_everything(opt.seed)

    # configs
    cfg = OmegaConf.load(f"configs/sds3d_cw.yaml")
    name = opt.name
    log_dir, ckpt_dir = Path(opt.log) / name, Path(opt.log) / name / 'ckpt'
    cfg.model.params['image_path'] = f"../../single2multi/output/mvimgs/{opt.name}/mvout.png"
    cfg.model.params['log_dir'] = log_dir
    cfg.model.params['oname'] = opt.name
    cfg.model.params['opt_dir'] = f"configs/sds3d_cw.yaml"
    cfg.model.params['prompt'] = opt.prompt
    cfg.model.params['negative_prompt'] = opt.negative_prompt

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



    if opt.rd360:
        for i_sem in [-3,-2,1]:
            extract_geo(model, log_dir, i_sem)
        return

    if opt.phy:
        # pdb.set_trace()
        for i_sem in [-3,1]:
            render_physical(model, log_dir, i_sem)
        return

    # model.renderer.gaussians.save_ply_bkg(os.path.join(model.log_dir, "target.ply"))

    render_images(model, log_dir, opt.numimgs)

    # extract_geo(model, log_dir)

    for i_sem in [-3,-2,1]:
        render_images(model, log_dir, opt.numimgs, i_sem)
        model.renderer.gaussians.save_ply(os.path.join(model.log_dir, f"sem_{i_sem:01d}", "model.ply"), i_sem)

    # for i_sem in range(2):
    #     extract_geo(model, log_dir, i_sem)

if __name__=="__main__":
    main()