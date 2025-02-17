import abc
import os, pdb
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from skimage.io import imread, imsave
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR

from util import instantiate_from_config, read_pickle, concat_images_list
from info_nce import InfoNCE
import copy
import cc3d

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from renderer.gs_networks import GaussianModel, BasicPointCloud, HierGaussian3DModel
from renderer.sh_utils import eval_sh, SH2RGB
from renderer.general_utils import depth_to_normal
from renderer.cam_utils import OrbitCamera, fov2focal 
from renderer.loss_utils import l1_loss, ssim
import math
import imageio
# from pytorch3d.loss import chamfer_distance

DEFAULT_RADIUS = np.sqrt(3)/2
DEFAULT_SIDE_LENGTH = 0.6

COLOR_MAP_20 = {-1: (0., 0., 0.), 0: (174., 199., 232.), 1: (152., 223., 138.), 2: (31., 119., 180.), 3: (255., 187., 120.), 4: (188., 189., 34.), 5: (140., 86., 75.),
                        6: (255., 152., 150.), 7: (214., 39., 40.), 8: (197., 176., 213.), 9: (148., 103., 189.), 10: (196., 156., 148.), 11: (23., 190., 207.), 12: (247., 182., 210.), 
                        13: (219., 219., 141.), 14: (255., 127., 14.), 15: (158., 218., 229.), 16: (44., 160., 44.), 17: (112., 128., 144.), 18: (227., 119., 194.), 19: (82., 84., 163.)}

# borrowed from SAM3D
def num_to_natural(group_ids):
    '''
    Change the group number to natural number arrangement
    '''
    if np.all(group_ids == -1):
        return group_ids
    array = copy.deepcopy(group_ids)
    unique_values = np.unique(array[array != -1])
    mapping = np.full(np.max(unique_values) + 2, -1)
    mapping[unique_values + 1] = np.arange(len(unique_values))
    array = mapping[array + 1]
    return array

def remove_small_group(group_ids, th):
    unique_elements, counts = np.unique(group_ids, return_counts=True)
    result = group_ids.copy()
    for i, count in enumerate(counts):
        if count < th:
            result[group_ids == unique_elements[i]] = -1
    
    return result


def sample_pdf(bins, weights, n_samples, det=True):
    device = bins.device
    dtype = bins.dtype
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples, dtype=dtype, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], dtype=dtype, device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def near_far_from_sphere(rays_o, rays_d, radius=DEFAULT_RADIUS):
    a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
    b = torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    mid = -b / a
    near = mid - radius
    far = mid + radius
    return near, far


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
def orbit_camera(elevation, azimuth, radius=1.5, is_degree=True, target=None, opengl=True):
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
    campos = np.array([x,y,z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    RT = look_at(campos, target, opengl).T

    # rectify
    T[:3, 0:1] = RT[:3, 2:3]
    T[:3, 1:2] = RT[:3, 0:1]
    T[:3, 2:3] = -RT[:3, 1:2]
    T[:3, 3] = -RT @ campos
    T[2, :] *= -1 
    return T[:3]


def getProjectionMatrix(znear, zfar, tanfovx, tanfovy):
    # tanHalfFovY = math.tan((fovY / 2))
    # tanHalfFovX = math.tan((fovX / 2))
    tanHalfFovY = tanfovy
    tanHalfFovX = tanfovx

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, w2c, width, height, K, znear=0.01, zfar=100):
        ###NOTE:hyan not c2w (refer to https://github.com/liuyuan-pal/SyncDreamer/blob/bcad0f11b72c027cc4d0464918a7f6bb07ac26e5/blender_script.py#L202

        self.image_width = width
        self.image_height = height
        # self.FoVy = fovy
        # self.FoVx = fovx
        self.tanfovy = width / (2 * K[1][1])
        self.tanfovx = height / (2 * K[0][0])
        self.K = K
        self.znear = znear
        self.zfar = zfar

        R, t = w2c[:3,:3], w2c[:3,3]
        self.camera_center = torch.tensor((-R.T @ t).astype(np.float32)).cuda()
        w2c = np.concatenate([w2c, np.array([[0., 0., 0., 1.]])])

        self.world_view_transform = torch.tensor(w2c.astype(np.float32)).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, tanfovx=self.tanfovx, tanfovy=self.tanfovy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        
 


class BackgroundRemoval:
    def __init__(self, device='cuda'):
        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)
        image = self.interface([image])[0]
        image = np.array(image)
        return image


class GS3DRenderer(nn.Module):
    def __init__(self, train_batch_num, test_batch_num, lambda_eikonal_loss=0.1, use_mask=True,
                 lambda_rgb_loss=1.0, lambda_mask_loss=0.0, lambda_contra_loss=0.02, mvgen_backbone='syncdreamer', rgb_loss='soft_l1', coarse_sn=64, fine_sn=64,
                 oname=""):
        # super().__init__(train_batch_num, test_batch_num)
        super().__init__()
        self.n_samples = coarse_sn
        self.n_importance = fine_sn
        self.up_sample_steps = 4
        self.anneal_end = 200
        self.use_mask = use_mask
        self.lambda_eikonal_loss = lambda_eikonal_loss
        self.lambda_rgb_loss = lambda_rgb_loss
        self.lambda_mask_loss = lambda_mask_loss
        self.lambda_contra_loss = lambda_contra_loss
        self.mvgen_backbone = mvgen_backbone
        self.rgb_loss = rgb_loss

        print(f"Set contrastive loss weight to : {self.lambda_contra_loss}")
        self.default_dtype = torch.float32

        #################
        self.sh_degree = 3
        self.white_background = True 
        self.gaussians = HierGaussian3DModel(3, oname)

        self.bg_color = torch.tensor(
            [1, 1, 1] if True else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )


    def initialize_bkg_target(self, input=None):
        # load checkpoint
        self.gaussians.load_ply_bkg_target(input)

    def render(
        self, 
        viewpoint_camera,  
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None, 
        compute_cov3D_python=False,
        convert_SHs_python=False, 
        i_sem=-1,
        update=True
    ):

        """
        Render the scene.  
        """
        if update:
            self.gaussians.update_whole_scene_property()

        target_N = self.gaussians.get_target_semantic.shape[0]

        if i_sem == -1:
            sem_mask = torch.ones(len(self.gaussians.get_xyz)).bool()
        elif i_sem == -2:
            sem3D = self.gaussians.get_semantic # [5000,2]
            sem_mask = sem3D[:, 0] == sem3D[:, 1]
        elif i_sem == -3:
            sem3D = self.gaussians.get_semantic # [5000,2]
            sem_mask = sem3D[:, 0] != 1
        elif i_sem == -4:
            sem3D = self.gaussians.get_semantic # [5000,2]
            sem_mask = sem3D[:, 0] != sem3D[:, 1]
        else:
            sem3D = self.gaussians.get_semantic # [5000,2]
            # _, sem_inds = torch.max(sem3D, 1)
            # sem_mask = sem_inds == i_sem
            # sem_mask[-target_N:] = True
            sem_mask = sem3D[:, i_sem] == 1

            
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(
            self.gaussians.get_xyz[sem_mask], 
            dtype=self.gaussians.get_xyz.dtype, 
            requires_grad=True, 
            device="cuda"
            ).contiguous() + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
 
        # Set up rasterization configuration
        tanfovx = viewpoint_camera.tanfovx # math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = viewpoint_camera.tanfovy # math.tan(viewpoint_camera.FoVy * 0.5)

        patch_size = [float('inf'), float('inf')] 

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform, 
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,  
        )
 

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz[sem_mask]
        means2D = screenspace_points
        opacity = self.gaussians.get_opacity[sem_mask]
        semantics = self.gaussians.get_semantic[sem_mask]

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling[sem_mask]
            rotations = self.gaussians.get_rotation[sem_mask]

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree+1 )** 2
                )
                dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(self.gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_features[sem_mask]
        else:
            colors_precomp = override_color
            
        
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            # semantics = semantics,
        )
  
        rendered_image = rendered_image.clamp(0, 1)

        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            # "rend_sem": rendered_sem,
        }
 


class SDSRendererTrainer(pl.LightningModule):
    def __init__(self, image_path, total_steps, warm_up_steps, log_dir, train_batch_fg_num=0,
                 use_cube_feats=False, cube_ckpt=None, cube_cfg=None, cube_bound=0.5,
                 train_batch_num=4096, test_batch_num=8192, use_warm_up=True, use_mask=True,
                 lambda_rgb_loss=1.0, lambda_mask_loss=0.0, renderer='neus', lambda_contra_loss=0.02,
                 # used for different backbones
                 mvbackbone='syncdreamer', num_mvimgs=16,  
                 # used in neus
                 lambda_eikonal_loss=0.1,
                 oname="", opt_dir="", prompt="", negative_prompt="",
                 coarse_sn=64, fine_sn=64):
        super().__init__()
        # self.num_images = 16
        self.num_images = num_mvimgs
        self.mvbackbone = mvbackbone
        self.image_size = 256
        self.log_dir = log_dir
        (Path(log_dir)/'images').mkdir(exist_ok=True, parents=True)
        self.train_batch_num = train_batch_num
        self.train_batch_fg_num = train_batch_fg_num
        self.test_batch_num = test_batch_num
        self.image_path = image_path
        self.total_steps = total_steps
        self.warm_up_steps = warm_up_steps
        self.use_mask = use_mask
        self.lambda_eikonal_loss = lambda_eikonal_loss
        self.lambda_rgb_loss = lambda_rgb_loss
        self.lambda_mask_loss = lambda_mask_loss
        self.lambda_contra_loss = lambda_contra_loss
        self.use_warm_up = use_warm_up
        self.oname = oname
        self.xyz_init = os.path.join("../../imgs23D/imgs2gs/output", oname)

        
        self.automatic_optimization = False ### NOTEL:hyan
        self.opt = OmegaConf.load(opt_dir)

        ###############

        self.W = self.opt.W
        self.H = self.opt.H
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image
        device = torch.device("cuda")

        self.guidance_sd = None

        self.enable_sd = True


        ######################

        self.use_cube_feats, self.cube_cfg, self.cube_ckpt = use_cube_feats, cube_cfg, cube_ckpt

        self._init_dataset()
        if renderer=='3dgs':
            self.renderer = GS3DRenderer(train_batch_num, test_batch_num,
                                         lambda_rgb_loss=lambda_rgb_loss,
                                         lambda_eikonal_loss=lambda_eikonal_loss,
                                         lambda_mask_loss=lambda_mask_loss,
                                         lambda_contra_loss=lambda_contra_loss,
                                         mvgen_backbone=mvbackbone,
                                         coarse_sn=coarse_sn, fine_sn=fine_sn,
                                         oname=oname)
        else:
            raise NotImplementedError
        self.validation_index = 0


        # input text
        assert len(prompt)>0
        self.prompt = prompt + ", 3d model"
        self.negative_prompt = negative_prompt

        # training stuff
        self.step = 0

        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
        
        # # override prompt from cmdline
        # if self.opt.prompt is not None:
        #     self.prompt = self.opt.prompt
        # if self.opt.negative_prompt is not None:
        #     self.negative_prompt = self.opt.negative_prompt

        # override if provide a checkpoint
        # if self.opt.load is not None:
            # initialize gaussians to a blob
        self.renderer.initialize_bkg_target(os.path.join(self.xyz_init, "model.ply"))
        self.renderer.gaussians.update_whole_scene_property()
        
        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree


        ####################

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(device, ckpt_path="/aigc_cfs/model/MVDream/sd-v2.1-base-4view.pt")
                print(f"[INFO] loaded MVDream!")
            elif self.opt.imagedream:
                print(f"[INFO] loading ImageDream...")
                from guidance.imagedream_utils import ImageDream
                self.guidance_sd = ImageDream(device)
                print(f"[INFO] loaded ImageDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(device, t_range=[0.02, 0.1])#, lora_path=os.path.join("../output/lora_ckpts", oname))
                print(f"[INFO] loaded SD!")


        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt], self.opt.subprompts) ### NOTE:hyan

        print(self.prompt)
        ####################

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        # refer to https://github.com/Harry-Zhi/semantic_nerf/blob/main/SSR/training/trainer.py
        self.logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)


    @staticmethod
    def load_model(cfg, ckpt):
        config = OmegaConf.load(cfg)
        model = instantiate_from_config(config.model)
        print(f'loading model from {ckpt} ...')
        ckpt = torch.load(ckpt)
        model.load_state_dict(ckpt['state_dict'])
        model = model.cuda().eval()
        return model

    def _init_dataset(self):
        mask_predictor = BackgroundRemoval()
        self.K, self.azs, self.els, self.dists, self.poses = read_pickle(f'../../meta_info/camera-{self.num_images}.pkl')

        self.images_info = {'images': [] ,'masks': [], 'Ks': [], 'poses':[], 'sam_masks': []}  # add part-segment 2D mask

        img = imread(self.image_path)

        # part-seg, sam mask load
        sam_mask = np.load(self.image_path.replace('.png', '.npy'))  
        self.img_part_info = []  # ni,
        vis_sam_masks = []


        for index in range(self.num_images):
            rgb = np.copy(img[:,index*self.image_size:(index+1)*self.image_size,:])
            # predict mask
            if self.use_mask:
                imsave(f'{self.log_dir}/input-{index}.png', rgb)
                masked_image = mask_predictor(rgb)
                imsave(f'{self.log_dir}/masked-{index}.png', masked_image)
                mask = masked_image[:,:,3].astype(np.float32)/255
            else:
                h, w, _ = rgb.shape
                mask = np.zeros([h,w], np.float32)
            
            # part-seg, sam mask load
            sam_singl_mask = np.copy(sam_mask[:,index*self.image_size:(index+1)*self.image_size])  
            sam_singl_mask[mask < 0.5] = -1  # set background as invalid mask
            # self.img_part_info.append([np.max([np.min(sam_singl_mask), 0]), np.max(sam_singl_mask)]) ###NOTE:hyan.
            self.img_part_info.append(np.unique(sam_singl_mask[sam_singl_mask!=-1])) ###NOTE:hyan.

            # SAM mask visualization (for checking bg removal)
            img_color = np.zeros((sam_singl_mask.shape[0], sam_singl_mask.shape[1], 3))
            for idp in range(0, np.max(sam_singl_mask)+1):
                color_mask = np.array(COLOR_MAP_20[idp % 20])
                img_color[sam_singl_mask==idp] = color_mask
            vis_sam_masks.append(img_color.astype(np.uint8))
            ##################################################

            rgb = rgb.astype(np.float32)/255
            K, pose = np.copy(self.K), self.poses[index]
            self.images_info['images'].append(torch.from_numpy(rgb.astype(np.float32))) # h,w,3
            self.images_info['masks'].append(torch.from_numpy(mask.astype(np.float32))) # h,w for bg [0.,1.]
            self.images_info['Ks'].append(torch.from_numpy(K.astype(np.float32)))
            self.images_info['poses'].append(torch.from_numpy(pose.astype(np.float32)))
            # part-seg, sam mask load
            self.images_info['sam_masks'].append(torch.from_numpy(sam_singl_mask.astype(np.int64))) # h,w for sam -1/0/1/...

        
        # SAM mask visualization: saving to log dir
        vis_sam_masks = np.concatenate(vis_sam_masks, axis=1)
        cv2.imwrite(f'{self.log_dir}/vis_sam_mask.png', vis_sam_masks[...,[2,1,0]])

        for k, v in self.images_info.items(): self.images_info[k] = torch.stack(v, 0) # stack all values

        self.mv_cameras = self.get_cameras(self.poses, self.K)
        self.mv_images = self.images_info['images'].permute(0,3,1,2).cuda()
        self.mv_masks = self.images_info['masks'].unsqueeze(1).cuda()
        self.mv_sems = self.images_info['sam_masks'].cuda()
        self._shuffle_train_mv_batch()


    @torch.no_grad()
    def get_cameras(self, poses, K):
        """
        poses: np.array [N,3,4]
        K: np.array [3,3]
        """
        cameras = []
        for w2c in poses:
            cameras.append(MiniCam(w2c=w2c, width=self.image_size, height=self.image_size, K=K))
        return cameras

    def _shuffle_train_mv_batch(self):
        self.train_batch_mv_list = np.arange(16)
        np.random.shuffle(self.train_batch_mv_list)
        self.mv_idx = 0

    def training_step(self, batch, batch_idx):
        self.step = self.global_step + 1
        # self.renderer.gaussians.update_learning_rate(self.step)
        # if self.step % 1000 == 0:
        #     self.renderer.gaussians.oneupSHdegree()

        ###################

        

        ### novel view (manual batch)
        render_resolution = self.image_size
        # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
        # min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
        # max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)

        # render random view
        # ver = np.random.randint(min_ver, max_ver)
        hor = np.random.randint(-180, 180)
        w2c = orbit_camera(self.opt.elevation, hor, self.opt.radius)
        cur_cam = MiniCam(w2c, render_resolution, render_resolution, self.K)
        # bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
        bg_color = torch.rand(3).float().cuda()

        sub_images = []
        sub_outs = []
        sd_loss = 0
        step_ratio = min(1, self.step / self.opt.iters)
        # for i_sem in range(1):
        #     sub_out = self.renderer.render(cur_cam, bg_color=bg_color, i_sem=i_sem)
        #     sub_images.append(sub_out["image"].unsqueeze(0))
        #     sub_outs.append(sub_out)

        #     sd_loss = sd_loss + 0.5 * self.opt.lambda_sd * self.guidance_sd.train_step(
        #         sub_images[i_sem], 
        #         step_ratio=step_ratio if self.opt.anneal_timestep else None,
        #         sub=i_sem)

        # sub_out = self.renderer.render(cur_cam, bg_color=bg_color)
        # sub_image, sub_alpha, sub_rend_sem = sub_out["image"], sub_out["alpha"], sub_out["rend_sem"]

        # pdb.set_trace()

        idx = int((hor + 180) / 22.5 + 0.5) # nearest training view
        # idx = 0 # force the front view
        if idx == 16: idx = 0
        cam, gt_image, gt_mask, gt_sem = self.mv_cameras[idx], self.mv_images[idx], self.mv_masks[idx] , self.mv_sems[idx]
        
        # sub_out0 = self.renderer.render(cam, bg_color=bg_color, i_sem=0)
        sub_out0 = self.renderer.render(cam, i_sem=-2, bg_color=bg_color)
        sub_image0, sub_alpha0 = sub_out0["image"], sub_out0["alpha"]

        # if idx in [0, 1, 2, 15]:
        #     hors = 0
        # elif idx in [3,4,5,12,13,14]:
        #     hors = 100
        # else:
        #     hors = 150
        # pdb.set_trace()

        # ture sd loss
        # pdb.set_trace()
        sd_loss = sd_loss + self.opt.lambda_sd * self.guidance_sd.train_step(
            sub_image0.unsqueeze(0), 
            step_ratio=step_ratio if self.opt.anneal_timestep else None)

        target_xyz = self.renderer.gaussians.target_xyz.detach()
        sdf = self.renderer.gaussians.query_sdf(target_xyz)
        sdf_normal = self.renderer.gaussians.query_gradients(target_xyz)
        sdf_normal /= torch.norm(sdf_normal, dim=1, keepdim=True)
        
        sdf_normal *= sdf
        sdf_normal[sdf[:,0]<=0] = 0.
        
        phy_xyz = (target_xyz - sdf_normal).detach()

        phy_loss = (phy_xyz - self.renderer.gaussians.target_xyz).abs().sum()

        # inpaint_xyz = self.renderer.gaussians.target_xyz.unsqueeze(0)
        # cloth_xyz = self.renderer.gaussians.bkg_xyz[self.renderer.gaussians.bkg_sem[:,1]==1].unsqueeze(0)
        # cd_loss = chamfer_distance(cloth_xyz, inpaint_xyz, point_reduction="sum")[0] #* 10000

        # pdb.set_trace()

        # sd_loss += phy_loss * 1.0

        sub_images.append(sub_image0.unsqueeze(0))
            


        ###NOTE:hyan
        if self.step % 50 == 0 or self.step == 1:
            os.makedirs(f'{self.log_dir}/sd/', exist_ok=True)
            final = (torch.cat(sub_images, dim=-1)[0].permute(1,2,0).detach().cpu().clamp(0.0, 1.0).numpy() * 255.).astype(np.uint8)
            imageio.imwrite(f'{self.log_dir}/sd/{self.step}.jpg', final)


        ###################
        
        
        gs_out0 = self.renderer.render(cam, i_sem=-3) 
        image0, alpha0 = gs_out0["image"], gs_out0["alpha"]

        # pdb.set_trace()
        
        valid_image = image0.permute(1,2,0) # [gt_sem == 0]
        valid_gt_image = gt_image.permute(1,2,0) # [gt_sem == 0]
        rgb_loss = l1_loss(valid_image, valid_gt_image)
        # ssim_loss = 1.0 - ssim(image0 * (gt_sem == 0), gt_image * (gt_sem == 0))
        ssim_loss = 1.0 - ssim(image0, gt_image)


        # mask_loss = (alpha0[0][~(gt_sem == 1)] * (1 - gt_mask[0][~(gt_sem == 1)])).mean() 
        mask_loss = (alpha0[0] * (1 - gt_mask[0])).mean() 

        loss_batch = {
            'rgb': rgb_loss,
            'ssim': ssim_loss,
            'mask': mask_loss,
            "sd": sd_loss,
            "idx": idx,
            'phy':phy_loss,
            # "cd": cd_loss
        }

        # if self.step > 500:
        #     regs_loss = self.renderer.gaussians.get_target_scaling.sum()
        #     loss += regs_loss * 1e-4
        #     loss_batch['regs'] = regs_loss

        #     rego_loss = (1 - self.renderer.gaussians.get_target_opacity).sum()
        #     loss += rego_loss * 1e-4
        #     loss_batch['rego'] = rego_loss
        
        loss = 0.

        loss += sd_loss * 1e-2

        # if self.step < 100:
        #     loss += sd_loss
        # elif self.step < 200:
        #     loss += sd_loss * 1e-1
        # else:
        #     loss += sd_loss * 1e-2
        
        
        loss += (1.0 - self.opt.lambda_dssim) * rgb_loss + self.opt.lambda_dssim * ssim_loss + self.opt.lambda_mask * mask_loss



        self.manual_backward(loss)

        self.optimizers().step()
        self.optimizers().zero_grad()

        with torch.no_grad():
            # Densification
            # if self.step < self.opt.density_end_iter:
            #     target_N = self.renderer.gaussians.get_target_xyz.shape[0]
            #     viewspace_point_tensor, visibility_filter, radii = sub_out0["viewspace_points"], sub_out0["visibility_filter"][-target_N:], sub_out0["radii"][-target_N:]
            #     self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, target_N)

            #     if self.step >= self.opt.density_start_iter and self.step % self.opt.densification_interval == 0:
            #         size_threshold = 20 if self.step > self.opt.opacity_reset_interval else None
            #         self.renderer.gaussians.densify_and_prune(
            #             self.opt.densify_grad_threshold, 
            #             min_opacity=self.opt.densify_min_opacity, 
            #             extent=self.opt.densify_extent,  
            #             max_screen_size=size_threshold, 
            #             )
        
            #     if self.step % self.opt.opacity_reset_interval == 0:
            #         self.renderer.gaussians.reset_opacity()
            # if self.step % 50 == 0:
            #     prune_mask = (self.renderer.gaussians.get_target_opacity < self.opt.densify_min_opacity).squeeze()
            #     size_threshold = 20 #if self.step > self.opt.opacity_reset_interval else None
            #     if size_threshold:
            #         big_points_vs = self.renderer.gaussians.max_radii2D > size_threshold
            #         big_points_ws = self.renderer.gaussians.get_target_scaling.max(dim=1).values > 0.1 * self.opt.densify_extent
            #         prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            #     self.renderer.gaussians.prune_points(prune_mask)

            # pdb.set_trace()
            
            self.renderer.gaussians.optimizer.step()
            self.renderer.gaussians.optimizer.zero_grad(set_to_none = True)
        
        ################################
        self.log_dict(loss_batch, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)

        self.log('step', self.global_step, prog_bar=True, on_step=True, on_epoch=False, logger=False, rank_zero_only=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.global_rank==0:
                # we output an rendering image
                idx = self.validation_index
                self.validation_index += 1
                self.validation_index %= self.num_images
                cam, gt_image, gt_mask, gt_sem = self.mv_cameras[idx], self.mv_images[idx], self.mv_masks[idx] , self.mv_sems[idx]

                gs_out = self.renderer.render(cam)

                process = lambda x: (x.cpu().numpy() * 255).astype(np.uint8)
                h, w = self.image_size, self.image_size
                rgb = torch.clamp(gs_out['image'].permute(1,2,0), max=1.0, min=0.0)
                mask = torch.clamp(gs_out['alpha'].permute(1,2,0), max=1.0, min=0.0)
                mask = torch.repeat_interleave(mask, 3, dim=-1)
                depth = gs_out['depth'].permute(1,2,0)
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                depth = torch.repeat_interleave(depth, 3, dim=-1)

                # sem = gs_out['rend_sem'].permute(1,2,0) # [h,w,2]
                # bg_mask = torch.norm(sem, p=2, dim=-1) == 0
                # sem = self.logits_2_label(sem)
                # sem[bg_mask] = -1
                # sem = sem.cpu().numpy()
                # sem_color = np.zeros((h, w, 3))
                # for idp in range(0, np.max(sem)+1):
                #     color_mask = np.array(COLOR_MAP_20[idp % 20])
                #     sem_color[sem==idp] = color_mask

                output_image = concat_images_list(process(rgb), process(mask), process(depth))

                # save images
                imsave(f'{self.log_dir}/images/{self.global_step}.jpg', output_image)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.AdamW([{"params": self.renderer.parameters(), "lr": lr},], lr=lr)

        def schedule_fn(step):
            total_step = self.total_steps
            warm_up_step = self.warm_up_steps
            warm_up_init = 0.02
            warm_up_end = 1.0
            final_lr = 0.02
            interval = 10   ### NOTE:hyan
            times = total_step // interval
            ratio = np.power(final_lr, 1/times)
            if step<warm_up_step:
                learning_rate = (step / warm_up_step) * (warm_up_end - warm_up_init) + warm_up_init
            else:
                learning_rate = ratio ** (step // interval) * warm_up_end
            return learning_rate

        if self.use_warm_up:
            scheduler = [{
                    'scheduler': LambdaLR(opt, lr_lambda=schedule_fn),
                    'interval': 'step',
                    'frequency': 1
                }]
        else:
            scheduler = []
        return [opt], scheduler

    def on_load_checkpoint(self, checkpoint) -> None:
        r"""
        Called by Lightning to restore your model.
        If you saved something with :meth:`on_save_checkpoint` this is your chance to restore this.

        Args:
            checkpoint: Loaded checkpoint

        Example::

            def on_load_checkpoint(self, checkpoint):
                # 99% of the time you don't need to implement this method
                self.something_cool_i_want_to_save = checkpoint['something_cool_i_want_to_save']

        Note:
            Lightning auto-restores global step, epoch, and train state including amp scaling.
            There is no need for you to restore anything regarding training.
        """
        # pdb.set_trace()
        self.renderer.gaussians.load_ply_target(checkpoint["target_ply_path"], load_color=True)
        self.renderer.gaussians.load_ply_bkg(checkpoint["bkg_ply_path"])
        self.renderer.gaussians.update_whole_scene_property()

    def on_save_checkpoint(self, checkpoint) -> None:
        r"""
        Called by Lightning when saving a checkpoint to give you a chance to store anything
        else you might want to save.

        Args:
            checkpoint: The full checkpoint dictionary before it gets dumped to a file.
                Implementations of this hook can insert additional data into this dictionary.

        Example::

            def on_save_checkpoint(self, checkpoint):
                # 99% of use cases you don't need to implement this method
                checkpoint['something_cool_i_want_to_save'] = my_cool_pickable_object

        Note:
            Lightning saves all aspects of training (epoch, global step, etc...)
            including amp scaling.
            There is no need for you to store anything about training.

        """
        checkpoint["bkg_ply_path"] = os.path.join(self.log_dir, "bkg.ply")
        checkpoint["target_ply_path"] = os.path.join(self.log_dir, "target.ply")
        checkpoint["ply_path"] = os.path.join(self.log_dir, "model.ply")
        self.renderer.gaussians.update_whole_scene_property()
        self.renderer.gaussians.save_ply(checkpoint["ply_path"])
        self.renderer.gaussians.save_ply_target(checkpoint["target_ply_path"])
        self.renderer.gaussians.save_ply_bkg(checkpoint["bkg_ply_path"])
