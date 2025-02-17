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



import torch.nn as nn

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




class ResumeCallBacks(Callback):
    def __init__(self):
        pass

    def on_train_start(self, trainer, pl_module):
        pl_module.optimizers().param_groups = pl_module.optimizers()._optimizer.param_groups

def render_images(model, output, num_imgs, i_sem=-1):
    # render from model
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
    sem_colors = []
    # n = 16
    n = num_imgs
    for ni in tqdm(range(n)):
        pose = poses[ni]

        cam = MiniCam(w2c=pose, width=w, height=h, K=K)
        # pdb.set_trace()
        with torch.no_grad():
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


def fibonacci_sphere(samples=1, randomize=True):
    rnd = 1.
    if randomize:
        rnd = np.random.random() * samples

    points = []
    offset = 2. / samples
    increment = np.pi * (3. - np.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y * y)

        phi = ((i + rnd) % samples) * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        points.append([x, y, z])
    return np.array(points)



def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

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



def getProjectionMatrix(znear, zfar, tanfovx, tanfovy):

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanfovx
    P[1, 1] = 1 / tanfovy
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCamOLD:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVx = fovx 
        self.FoVy = fovy
        self.tanfovx = math.tan((fovx / 2))
        self.tanfovy = math.tan((fovy / 2))
        self.znear = znear
        self.zfar = zfar

        fx = width / (2 * self.tanfovx)
        fy = height / (2 * self.tanfovy)
        self.K = np.array(
            [[fx, 0., width/2.],
            [0., fy, height/2.],
            [0., 0., 1.0]]
        )

        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, tanfovx=self.tanfovx, tanfovy=self.tanfovy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()



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
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T


def orbit_camera_fibonacci(num_samples, radius=2.5, is_degree=True, target=None, opengl=True, render_resolution=512, fov=49.1):
    cam_positions = fibonacci_sphere(num_samples, randomize=False)   

    cameras = []
    for campos in cam_positions: 
        elevation = np.arcsin(campos[1] / radius)  # y
        azimuth = np.arctan2(campos[2], campos[0])  # z, x

        if is_degree:
            elevation = np.rad2deg(elevation)
            azimuth = np.rad2deg(azimuth)
        
        camera_matrix = orbit_camera(elevation, azimuth, radius, is_degree, target, opengl)
 
        cur_cam = MiniCamOLD(
            camera_matrix, render_resolution, render_resolution,  fov,  fov,  0.1, 100
        )
        cameras.append(cur_cam)
    return cameras


def extract_geo(model, output, i_sem=-1):
    gaussExtractor = GaussianExtractor(model.renderer)
    n_fames = 200 
    radius = 2 
    fov = 49.1 
    voxel_size = 0.008
    depth_trunc = 4.0
    cameras = orbit_camera_fibonacci(n_fames, render_resolution=512, fov=49.1)  
    gaussExtractor.reconstruction(cameras, i_sem)
    mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=0.05, depth_trunc=depth_trunc)
    if i_sem == -1:
        o3d.io.write_triangle_mesh(f'{output}/fused.ply', mesh) 
    else:
        os.makedirs(os.path.join(output, f"sem_{i_sem:01d}"), exist_ok=True)
        o3d.io.write_triangle_mesh(f'{output}/sem_{i_sem:01d}/fused.ply', mesh) 





def render_physical(model, output):
    # render from model
    K, _, _, _, poses = read_pickle(f'../../meta_info/camera-{16}.pkl')
    logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)

    with open(os.path.join(output, "editing_modifier.pkl"), "rb") as f:
        emdict = pickle.load(f)
    trajs = emdict['objects'][0]['actions'][0]['particles_trajectory_tn3']

    phy_path = os.path.join(output, "phy")
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
    n = len(trajs)
    pose = poses[3]
    for ni in tqdm(range(n)):
        traj = trajs[ni]
        cam = MiniCam(w2c=pose, width=w, height=h, K=K)
        model.renderer.gaussians._xyz = torch.from_numpy(traj).float().cuda()
        # pdb.set_trace()
        with torch.no_grad():
            outs = model.renderer.render(cam, i_sem=-1)
            process = lambda x: (x.cpu().numpy() * 255).astype(np.uint8)
            rgb = torch.clamp(outs['image'].permute(1,2,0), max=1.0, min=0.0)
            imsave(f'{phy_path}/{ni:05d}.png', process(rgb))

import open3d as o3d

def post_process_sem(model, k=19):
    from torch_cluster import knn   
    sems = model.renderer.gaussians._sem
    xyzs=  model.renderer.gaussians.get_xyz
    idxs = knn(xyzs, xyzs, k)[1].reshape(-1, k)
    sele = sems[idxs]
    newsems = sele.mean(1)
    model.renderer.gaussians._sem = sele.mean(1)


def query_sdf(sdf_network, xyz):
    return sdf_network.sdf(torch.from_numpy(xyz).cuda()).detach().cpu().numpy()

# direction of - -> +
def query_gradients(sdf_network, xyz):
    x = torch.from_numpy(xyz).cuda()
    x.requires_grad_(True)
    y = sdf_network.sdf(x)
    d_output = torch.ones_like(y, requires_grad=False, device=y.device)
    gradients = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=d_output,
        create_graph=False,
        retain_graph=False,
        only_inputs=True)[0]
    return gradients.detach().cpu().numpy()


def compute_sdf_grid(model, log_dir, sdf_dir):
    if os.path.exists(os.path.join(log_dir, "proxy_sdf.npy")) and os.path.exists(os.path.join(log_dir, "proxy_grad.npy")) and os.path.exists(os.path.join(log_dir, "shift_constant.npy")) and os.path.exists(os.path.join(log_dir, "longest_side.npy")):
        print("EXISTS...")
        return
    sdf_network = SDFNetwork(d_out=257, d_in=3, d_hidden=256, n_layers=8, skip_in=[4], multires=6, bias=0.5, scale=1.0, geometric_init=True, weight_norm=True).cuda()
    ckpt = torch.load(sdf_dir, map_location='cuda')['state_dict']
    init_state = sdf_network.state_dict()
    for k in init_state.keys():
        init_state[k] = ckpt['renderer.sdf_network.'+k]
    sdf_network.load_state_dict(init_state)
    print("load sdf network")

    
    sim_res = 64

    particles = model.renderer.gaussians._xyz.detach().cpu().numpy()
    # pdb.set_trace()
    particle_max = particles.max(axis=0)
    particle_max[2] += 0.05
    particle_min = particles.min(axis=0)
    particle_min[2] -= 0.05

    longest_side = max(particle_max - particle_min)

    particles[:, 0] /= longest_side
    particles[:, 1] /= longest_side
    particles[:, 2] /= longest_side

    # Align centers of x and z to 0.5
    # Set the bottom of the object to 0
    shift_constant = np.array([
        -particles[:,0].mean() + 0.5,
        -particles[:,1].mean() + 0.5,
        -particle_min[2] / longest_side
    ])

    particles += shift_constant
    
    proxy_xyz = np.mgrid[-1:sim_res+2,-1:sim_res+2,-1:sim_res+2].transpose(1,2,3,0).astype(np.float32) / sim_res # [r,r,r,3]
    proxy_xyz -= shift_constant
    proxy_xyz *= longest_side

    proxy_sdf = query_sdf(sdf_network, proxy_xyz.reshape(-1,3)).reshape(sim_res+3,sim_res+3,sim_res+3).astype(np.float32)
    proxy_grad = query_gradients(sdf_network, proxy_xyz.reshape(-1,3)).reshape(sim_res+3,sim_res+3,sim_res+3,3).astype(np.float32)

    np.save(os.path.join(log_dir, "proxy_sdf.npy"), proxy_sdf)
    np.save(os.path.join(log_dir, "proxy_grad.npy"), proxy_grad)
    np.save(os.path.join(log_dir, "shift_constant.npy"), shift_constant)
    np.save(os.path.join(log_dir, "longest_side.npy"), longest_side)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-b', '--base', type=str, default='configs/gs_cw.yaml')
    parser.add_argument('-l', '--log', type=str, default='output')
    parser.add_argument('-s', '--seed', type=int, default=6033)
    parser.add_argument('-g', '--gpus', type=str, default='0,')
    parser.add_argument('--numimgs', type=int, default=16)
    parser.add_argument('-r', '--resume', action='store_true', default=False, dest='resume')
    parser.add_argument('-p', '--phy', action='store_true', default=False, dest='phy')
    parser.add_argument('--sdf', action='store_true', default=False, dest='sdf')
    parser.add_argument('--fp16', action='store_true', default=False, dest='fp16')
    opt = parser.parse_args()
    # seed_everything(opt.seed)

    # configs
    cfg = OmegaConf.load(opt.base)
    name = opt.name
    xyz_init = os.path.join("../imgs2neus/output", opt.name)
    log_dir, ckpt_dir = Path(opt.log) / name, Path(opt.log) / name / 'ckpt'
    cfg.model.params['image_path'] = os.path.join("../../single2multi/output/mvimgs", opt.name, "mvout.png")
    cfg.model.params['log_dir'] = log_dir
    cfg.model.params['xyz_init'] = xyz_init

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

    # pdb.set_trace()

    
    compute_sdf_grid(model, log_dir, sdf_dir=os.path.join(xyz_init, "ckpt/last.ckpt"))
    if opt.sdf:
        return 

    # pdb.set_trace()

    # post_process_sem(model)

    if opt.phy:
        render_physical(model, log_dir)
        return

    # pdb.set_trace()

    render_images(model, log_dir, opt.numimgs)

    # # extract_geo(model, log_dir)

    for i_sem in range(2):
        render_images(model, log_dir, opt.numimgs, i_sem)
        # extract_geo(model, log_dir, i_sem)
        model.renderer.gaussians.save_ply(os.path.join(log_dir, f"sem_{i_sem:01d}", "model.ply"), i_sem)

    

if __name__=="__main__":
    main()