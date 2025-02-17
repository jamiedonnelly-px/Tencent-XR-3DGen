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
import cv2

COLOR_MAP_20 = {-1: (0., 0., 0.), 0: (174., 199., 232.), 1: (152., 223., 138.), 2: (31., 119., 180.), 3: (255., 187., 120.), 4: (188., 189., 34.), 5: (140., 86., 75.),
                        6: (255., 152., 150.), 7: (214., 39., 40.), 8: (197., 176., 213.), 9: (148., 103., 189.), 10: (196., 156., 148.), 11: (23., 190., 207.), 12: (247., 182., 210.), 
                        13: (219., 219., 141.), 14: (255., 127., 14.), 15: (158., 218., 229.), 16: (44., 160., 44.), 17: (112., 128., 144.), 18: (227., 119., 194.), 19: (82., 84., 163.)}


class ResumeCallBacks(Callback):
    def __init__(self):
        pass

    def on_train_start(self, trainer, pl_module):
        pl_module.optimizers().param_groups = pl_module.optimizers()._optimizer.param_groups

def render_images(model, output, num_imgs):
    K, _, _, _, poses = read_pickle(f'../../meta_info/camera-{num_imgs}.pkl')
    
    h, w = 256, 256
    default_size = 256
    K = np.diag([w/default_size,h/default_size,1.0]) @ K
    imgs = []
    deps = []
    alps = []
    nors = []
    sems = []
    sem_colors = []
    # n = 16
    n = num_imgs
    for ni in tqdm(range(n)):
        pose = poses[ni]
        pose_ = torch.from_numpy(pose.astype(np.float32)).unsqueeze(0)
        K_ = torch.from_numpy(K.astype(np.float32)).unsqueeze(0) # [1,3,3]

        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
        coords = coords.float()[None, :, :, :].repeat(1, 1, 1, 1)  # imn,h,w,2
        coords = coords.reshape(1, h * w, 2)
        coords = torch.cat([coords, torch.ones(1, h * w, 1, dtype=torch.float32)], 2)  # imn,h*w,3

        # imn,h*w,3 @ imn,3,3 => imn,h*w,3
        rays_d = coords @ torch.inverse(K_).permute(0, 2, 1)
        R, t = pose_[:, :, :3], pose_[:, :, 3:]
        rays_d = rays_d @ R
        rays_d_unnorm = rays_d.clone()
        rays_d = F.normalize(rays_d, dim=-1)
        rays_o = -R.permute(0, 2, 1) @ t  # imn,3,3 @ imn,3,1
        rays_o = rays_o.permute(0, 2, 1).repeat(1, h * w, 1)  # imn,h*w,3

        ray_batch = {
            'rays_o': rays_o.reshape(-1,3).cuda(),
            'rays_d': rays_d.reshape(-1,3).cuda(),
            'rays_d_unnorm': rays_d_unnorm.reshape(-1,3).cuda(),
        }
        with torch.no_grad():
            outs = model.renderer.render(ray_batch,False,5000)
            depmap = outs['depth'].reshape(h,w).cpu().numpy()
            depmap = depmap / depmap.max() * 255.

            alpha_map = outs['mask'].reshape(h,w).cpu().numpy()
            alpha_map = alpha_map.clip(0.0, 1.0) * 255.

            rgbmap = outs['rgb'].reshape(h,w,-1).cpu().numpy()
            rgbmap = rgbmap.clip(0.0, 1.0) * 255.

            normal_map = outs['normal'].reshape(h,w,-1).cpu().numpy()
            normal_map = ((normal_map + 1) / 2).clip(0.0, 1.0) * 255.

            semmap = outs['partseg'].reshape(h,w,-1).cpu().numpy().argmax(2)
            # pdb.set_trace()
            semmap[alpha_map<0.01*255.] = -1
            sem_color = np.zeros((h,w,3))
            for idp in [-1, 0, 1]:
                sem_color[semmap == idp] = np.array(COLOR_MAP_20[idp])

        # image = (image.cpu().numpy() * 255).astype(np.uint8)
        # imgs.append(image)
        deps.append(depmap)
        alps.append(alpha_map)
        nors.append(normal_map[:,:,[2,1,0]])
        sem_colors.append(sem_color[:,:,[2,1,0]])
        sems.append(semmap)
        imgs.append(rgbmap[:,:,[2,1,0]])

    depmap = np.concatenate(deps, 1)
    alpha_map = np.concatenate(alps, 1)
    normal_map = np.concatenate(nors, 1)
    sem_color = np.concatenate(sem_colors, 1)
    semmap = np.concatenate(sems, 1)
    rgbmap = np.concatenate(imgs, 1)
    cv2.imwrite(f'{output}/save_depth.png', depmap)
    cv2.imwrite(f'{output}/save_alpha.png', alpha_map)
    cv2.imwrite(f'{output}/save_normal.png', normal_map)
    cv2.imwrite(f'{output}/save_sem.png', sem_color)
    cv2.imwrite(f'{output}/save_rgb.png', rgbmap)
    np.save(f'{output}/save_sem.npy', semmap)
    np.save(f'{output}/save_depth.npy', depmap)
    np.save(f'{output}/save_alpha.npy', alpha_map)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-b', '--base', type=str, default='configs/neus_cw.yaml')
    parser.add_argument('-s', '--seed', type=int, default=6033)
    parser.add_argument('-g', '--gpus', type=str, default='0,')
    parser.add_argument('--numimgs', type=int, default=16)
    parser.add_argument('-r', '--resume', action='store_true', default=False, dest='resume')
    parser.add_argument('--fp16', action='store_true', default=False, dest='fp16')
    opt = parser.parse_args()
    # seed_everything(opt.seed)

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

    render_images(model, log_dir, opt.numimgs)

if __name__=="__main__":
    main()