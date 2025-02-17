'''
 # @ Copyright: Copyright 2022 Tencent Inc
 # @ Author: shenzhou
 # @ Create Time: 2023-08-31 11:00:00
 # @ Description: hybrid 3d(pcd, sdf, normal) and 2d(render img) dataset
 '''

import os
import glob
import json
import collections

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import random
import time
from . import util, h5, hdri, sample_pcd
# import util

import imageio as iio

from pdb import set_trace as st
import h5py
import numba
from .data_augmentation import MVLRMAugmentor

import trimesh

try:
    from diffusers import AutoencoderKL
except:
    pass

from .transform import transform_pcd

###############################################################################
# NERF image based dataset (synthetic)
###############################################################################


@numba.njit
def _render_pcd(image, u,v,rgb):
    for i in range(len(u)):
        image[int(np.round(v[i])), int(np.round(u[i]))] = rgb[i]
    
def render_pcd(image, u,v,d,rgb):
    ord = np.argsort(-d)
    u = u[ord]
    v = v[ord]
    rgb = rgb[ord]
    image = image + 0 # copy
    _render_pcd(image, u,v,rgb)
    return image

@torch.no_grad()
def vae_decode(vae, latents):
    '''
    latents: [B, 4, H,W]
    returns: [B, H,W,3]
    '''
    imgs = []
    for l in torch.split(latents, 1, dim=0):

        imgs.append((vae.decode(l.cuda()).sample + 1).permute(0,2,3,1).cpu().numpy().clip(0,2) * 0.5)

    return np.concatenate(imgs, axis=0)

def _process_img(img, scaled=True, srgb=True):

    if img.dtype != np.float32:  # LDR image
        if img.dtype == np.uint8:
            scale = 255
            img = torch.from_numpy(img).float()
            if srgb:
                img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3]/scale) * scale
        elif img.dtype == np.uint16:
            scale = 65535
            img = torch.from_numpy(img * 1.0).float()
            if srgb:
                img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3]/scale) * scale
        else:
            raise ValueError
    else:
        img = torch.from_numpy(img).float()
    if scaled:
        img = img / scale
    return img


def _load_img(path, scaled=True, srgb=True, suffixs=['png', 'jpg']):
    if path.split('.')[-1] in suffixs:
        img_name = path
    else:
        files = glob.glob(path + '.*')
        assert len(
            files
        ) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
        img_name = files[0]
    # img = util.load_image_raw(img_name)
    # print('debug img ', img.shape, img.dtype)
    
    img = iio.imread(img_name)
    if img.ndim == 2:
        img = img[..., None] # append channel dimension

    if img is None:
        print("path", path, "img_name", img_name)
        st()
    # img[..., :3] = img[..., :3][..., ::-1]

    return _process_img(img, scaled, srgb)

def _zero_image(img_size, channels):
    return torch.zeros(tuple(img_size) + (channels,), dtype=torch.float32)

def get_test_camera_embeddings(FLAGS, ref_view_id=0):
    '''
    returns camera embedding 
    '''
    with open(FLAGS['data_json']) as j:
        camera_embeddings = json.load(j)['camsys_config']['embeddings']
    
    ref_cam_elevations = torch.tensor(camera_embeddings['ref'])[ref_view_id:ref_view_id+1] # [n_ref_cams]
    src_cam_elevations = torch.tensor(camera_embeddings['src'])[:,0] # [n_src_cams]
    src_cam_azimuths = torch.tensor(camera_embeddings['src'])[:,1] # [n_src_cams]

    elevations = torch.cat([ref_cam_elevations, src_cam_elevations])
    azimuths = torch.cat([torch.zeros_like(ref_cam_elevations), src_cam_azimuths])
    n_views = 1 + len(src_cam_elevations)

    elevations = elevations.unsqueeze(1)
    azimuths = azimuths.unsqueeze(1)
    elevations_cond = ref_cam_elevations.expand_as(elevations)

    all_embeddings = []

    for class_embeddings in FLAGS['classe_embeddings']:

        assert len(class_embeddings) == 1, "format error"

        class_embeddings = list(class_embeddings.values())[0]

        class_embeddings = torch.tensor(class_embeddings).float().expand(n_views,-1)
        class_embeddings = torch.cat((elevations_cond, elevations, azimuths, class_embeddings), dim=1)
        all_embeddings.append(class_embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)

    return all_embeddings

class MVLRMDataset(Dataset):

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

        self.parse_data_json(self.FLAGS)
        
        self.n_objs_all = len(self.meta_pairs)
        if self.n_objs_all < 1:
            print('cannot load data from ', FLAGS['data_json'])
            raise TypeError
        
        print(f'[DatasetHybrid] will use {self.n_objs_all} objs.')
        
        self.augmentor = MVLRMAugmentor(FLAGS)
        self.hdri_cache = hdri.HDRIcache(FLAGS.get('hdri_size', (32,64)), FLAGS.get('hdri_sh_dim', 0))
    
    
    @staticmethod
    def make_data_json(import_paths, cameras_config, camera_embeddings, includes, excludes, out_json):
        '''
        Inputs:
            - import_paths: dicionary of following format:
              {
                  "npu": ["path_to_dataset_json1", "path_to_dataset_json2", ...],
                  "cfs": ["path_to_dataset_json1", "path_to_dataset_json2", ...],
                  "lanjing": ["path_to_dataset_json1", "path_to_dataset_json2", ...],
              }
            - cameras_config: list of camera systems, each system is a dictionary of following format:
                {
                    "ref": [ref_id1, ref_id2, ...],
                    "src": [src_id1, src_id2, ...],
                }
            - camera_embeddings: dictionary of following format:
                {
                    "ref": [elevation1, elevation2, ...],
                    "src": [[elevation1, azimuth1], [elevation2, azimuth2], ...],
                }
            - includes: list of tuples of following format:
                [
                    ("dataset_name", "object_id", {"camsys": [camsys_id1, camsys_id2, ...], ...}),
                ]
            - out_json: path to save the output json file
        '''
        
        camsys_config = {'cameras': cameras_config, 'embeddings': camera_embeddings}
        includes_data = {}
        excludes_data = {}
        
        if includes == "ALL":
            includes_data = "ALL"
        else:            
            for dset, obj_id, settings in includes:
                if dset not in includes_data:
                    includes_data[dset] = {}
                includes_data[dset][obj_id] = settings

        if excludes is not None:
            for dset, obj_id, settings in excludes:
                if dset not in excludes_data:
                    excludes_data[dset] = {}
                excludes_data[dset][obj_id] = settings
        
        dataset_json = {'import_paths': import_paths, 'camsys_config': camsys_config, 'includes': includes_data, 'excludes': excludes_data}
        
        with open(out_json, 'w') as f:
            json.dump(dataset_json, f, indent=2)
    

    def parse_data_json(self, FLAGS):
        """Read the input JSON file and parse the (name, rendering path, camera parameters, point cloud path, triplane path, and octree path) for each obj.

        Args:
            data_json: [data][dtype][dict of oname:meta], meta is dict example: data/merge/0926_raw.json
            FLAGS: config

        Returns:
            (name, rendering path, camera parameters, point cloud path, triplane path, and octree path) for each obj.  
            meta_pairs: list of tupe (dtype, oname)
            render_dirs: list of str [n]
            geom_dicts: list of geom dict('point':[N,3]) with surface points xyz [n(dict)]
            tri_steps_ckpts_list, oct_steps_ckpts_list: list of steps_ckpts, where steps_ckpts is a list of paths with length step_cnt or None. [n(4)]
        """
        dataset_json = json.load(open(FLAGS['data_json'], 'r'))
        cluster = FLAGS['cluster']
        assert cluster in dataset_json['import_paths'], f"data json does not contain any import path for cluster '{cluster}'"
        
        self.data_json_list = dataset_json['import_paths'][cluster]
        
        data_dict = {}
        for data_path in dataset_json['import_paths'][cluster]:
            loaded_data = json.load(open(data_path, 'r'))['data']
            for dset in loaded_data:
                if dset not in data_dict:
                    data_dict[dset] = {}
                data_dict[dset].update(loaded_data[dset])
        
        camsys_config = dataset_json['camsys_config']
        
        self.cameras_config = camsys_config['cameras']
        self.n_ref_views = len(self.cameras_config[0]['ref'])
        
        includes = dataset_json['includes']
        excludes = dataset_json.get('excludes', {})
        
        if includes == "ALL":
            includes = data_dict
            
        includes = util.exclude_from(includes, excludes)

        # Load all objs cfg
        # self.meta_pairs, self.render_dirs, self.geopcd_dirs, self.texpcd_dirs, self.h5_paths, self.cam_systems
        
        self.meta_pairs = []
        self.render_dirs = []
        self.geopcd_dirs = []
        self.texpcd_dirs = []
        self.manifold_mesh_paths = []
        self.manifold_transformation_paths = []
        self.h5_paths = []
        self.cam_systems = []
        
        all_camsys = list(range(len(self.cameras_config)))
        
        expected_n_objs = sum([len(includes[dtype]) for dtype in includes])
        pbar = tqdm(total=expected_n_objs)

        for dtype in includes:
            for oname in includes[dtype]:
                
                try:
                    if dtype not in data_dict or oname not in data_dict[dtype]:
                        continue
                    
                    if self.FLAGS.get("view_select", True):
                        camsys = includes[dtype][oname].get("camsys", all_camsys)
                    else:
                        camsys = all_camsys
                        
                    
                    meta = data_dict[dtype][oname]
                    render_dir = meta['ImgDir']
                    gep_pcd_dir = meta['GeoPcd']
                    transform_path = meta.get("Transformation", os.path.join(os.path.dirname(meta['GeoPcd']), 'transformation.txt'))
                    manifold_path = meta['Manifold']
                    
                    
                    self.meta_pairs.append((dtype, oname))
                    self.cam_systems.append(camsys)
                    self.render_dirs.append(render_dir)  
                    self.geopcd_dirs.append(gep_pcd_dir)   
                    self.texpcd_dirs.append(meta.get('TexPcd', None))
                    
                    self.manifold_mesh_paths.append(manifold_path)
                    self.manifold_transformation_paths.append(transform_path)
                    
                    self.h5_paths.append(meta.get("h5", None))   
                except Exception as e:
                    print(f"error loading {dtype}.{oname} from {self._get_raw_json_path(dset=dtype, oid=oname)}: {type(e).__name__} {e}")
                
                pbar.update(1)
                
        pbar.close()
        
        print(f"Loaded {len(self.meta_pairs)} multiview images.")
        print(f"Expected {expected_n_objs} objs, got {len(self.render_dirs)} objs.")
        print(f"An average of {np.mean([len(sys) for sys in self.cam_systems]):.2f} camera poses were selected per obj, out of {len(self.cameras_config)} available.")
    
    def _get_raw_json_path(self, obj_idx=None, dset=None, oid=None):
        if obj_idx is not None:
            dset, oid = self._get_obj_id(obj_idx)
        for json_path in reversed(self.data_json_list):
            with open(json_path) as json_f:
                json_data = json.load(json_f)['data']
                if dset in json_data and oid in json_data[dset]:
                    return json_path

    def _process_pcd(self, *pcds, refc2o):
        '''
        Inputs:
            - pcds: tensors of shape [Npoints, 3], in object space
            - refc2o: tensor of shape [4,4] or [1,4,4], ref camera pose

        Returns:
            - list of tensors of same shape as pcds, transformed in reference xyz system
        '''
        
        _, o2w, _ = self._get_transforms(refc2o, refc2o)
        o2w_T = o2w.transpose(-1,-2)[:3,:3]
        
        return_pcds = []
        for pcd in pcds:
            return_pcds.append(pcd @ o2w_T)
            
        return transform_pcd(self.FLAGS.get("pcd_coordinate_system", "world,xyz"), o2w, *return_pcds)

    def _get_transforms(self, c2o, refc2o):
        '''
        Inputs:
            - c2o: tensor of shape [Nviews, 4,4], camera pose
            - refc2o: tensor of shape [4,4] or [1,4,4], ref camera pose

        Returns:
            - c2w: tensor of shape [Nviews, 4,4], camera pose
            - o2w: tensor of shape [4,4], transform matrix
            - n2w: tensor of shape [4,4], rotation matrix
        '''
        w2n = torch.tensor([
            [0,1,0,0],
            [0,0,1,0],
            [1,0,0,0],
            [0,0,0,1]], dtype=c2o.dtype, device=c2o.device)

        w2refc_R = torch.tensor([
            [0,1,0],
            [0,0,-1],
            [-1,0,0]], dtype=c2o.dtype, device=c2o.device)

        o2w_R = torch.inverse(refc2o[...,:3,:3] @ w2refc_R.reshape(3,3)) # object to world 
        o2w = torch.eye(4, device=o2w_R.device, dtype=o2w_R.dtype)
        o2w[:3,:3] = o2w_R
        c2w = o2w @ c2o # camera to world
        
        n2w = w2n.transpose(-1,-2)
        
        return c2w, o2w, n2w

    def _process_geometry_imgs(self, normal, depth, intrinsic, c2o, refc2o):
        '''
        Inputs:
            - normal: tensor of shape [Nviews, 3,H,W], normals in object space
            - depth: tensor of shape [Nviews, 1,H,W], depth images
            - intrinsic: tensor of shape [Nviews, 3,4], camera proj
            - c2o: tensor of shape [Nviews, 4,4], camera pose
            - refc2o: tensor of shape [4,4] or [1,4,4], ref camera pose

        Returns:
            - normal_image: tensor of shape [Nviews, 4,H,W], normal images in reference normal system
              with values in [0,1], last channel is alpha, background is grey
            - xyz_images: tensor of shape [Nviews, 4,H,W], coordinate maps in reference xyz system
              with values in [0,1], last channel is alpha, background is black
            - distance_images: tensor of shape [Nviews, 1,H,W], distance maps from near plane of viewing frustum, 
              with values in [0,inf), background is black
        '''
        c2w, o2w, n2w = self._get_transforms(c2o, refc2o)
        o2n = torch.inverse(n2w) @ o2w
        
        xyz_alpha = torch.logical_and((depth > 1e-3), (depth < 30)).to(dtype=depth.dtype)
        depth[xyz_alpha == False] = 0
        xyz_image = (util.depth_to_world(depth, intrinsic, c2w) + 1) / 2 # [Nviews,3,H,W]
        xyz_image = xyz_image * xyz_alpha
        xyz_image = torch.cat([xyz_image, xyz_alpha], dim=1).clip(0,1)

        normal = normal * 2 - 1
        normal = torch.flip(normal, dims=(1,))
        normal_image = (torch.einsum('vohw,no->vnhw', normal, o2n[:3,:3]) + 1) / 2
        normal_alpha = (torch.linalg.vector_norm(normal, dim=1, keepdim=True) * 1.1 - 0.05).clip(0,1)
        normal_image = torch.cat([normal_image, normal_alpha], dim=1).clip(0,1)
        
        distance_image = util.depth_to_distance(depth, intrinsic)
        distance_image[xyz_alpha==False] = 0

        return normal_image, xyz_image, distance_image
    
    def _get_background_color(self):
        
        if self.FLAGS.get('white_background', False):
            assert self.FLAGS.get('background', None) is None
            bg_color = 1.1
        elif self.FLAGS.get('background', None) is not None:
            if self.FLAGS['background'] == 'white':
                bg_color = 1.1
            elif self.FLAGS['background'] == 'black':
                bg_color = 0.0
            elif self.FLAGS['background'] == 'random_grey':
                bg_color = random.random()
            elif self.FLAGS['background'] == 'random_bright':
                bg_color = (torch.rand(3) * 0.025 + 0.9 + torch.rand(1) * 0.1).clip(0.9, 1.1)
            elif self.FLAGS['background'] == 'grey':
                bg_color = 0.5
            elif isinstance(self.FLAGS['background'], collections.Sequence) and len(self.FLAGS['background']) == 2:
                r = random.random()
                bg_color = self.FLAGS['background'][0] * r + (1-r) * self.FLAGS['background'][1]
            elif isinstance(self.FLAGS['background'], float) or isinstance(self.FLAGS['background'], int):
                bg_color = self.FLAGS['background']
            else:
                raise ValueError(f"Unknown background option '{self.FLAGS['background']}'")
        else:
            bg_color = None
        
        return bg_color
    
    def _apply_background(self, *images):
        '''
        images are tensors of shape [...,4,h,w] where last channel is alpha
        
        returns list tensors of same shape
        '''
        returns = []
        
        bg_color = self._get_background_color()
        
        if bg_color is None:
            return [im.clip(0,1) for im in images]
        
        if isinstance(bg_color, torch.Tensor):
            bg_color = bg_color.reshape(3,1,1)
        else:
            bg_color = torch.tensor(bg_color)
        
        for im in images:
            if bg_color is not None:
                im_bg = torch.cat(((im[...,:3,:,:] - bg_color.to(im.dtype)) * im[...,3:,:,:] + bg_color.to(im.dtype) , im[...,3:,:,:]), dim=1)
            else:
                im_bg = im
            returns.append(im_bg.clip(0,1))
            
        return returns
    
    def _reapply_background(self, bg_color, *images):
        '''
        images are tensors of shape [...,4,h,w] where last channel is alpha
        
        returns list tensors of same shape
        '''
        returns = []
        
        for im in images:
            if bg_color is not None:
                im_bg = torch.cat(((im[...,:3,:,:] - bg_color) * im[...,3:,:,:] + bg_color , im[...,3:,:,:]), dim=1)
            else:
                im_bg = im
            returns.append(im_bg.clip(0,1))
            
        return returns
    
    def _resize_images(self, *images, intrinsic, align_corners=True, img_size=None):
        '''
        intrinsic: either 3x3 or 3x4, if 3x4, can be either perspective or orthographic projection
        align_corners (bool, optional): Whether the intrinsic's image origin is the corner (or centre) of top-left pixel.
        this function does not change above convention, but it needs to know which one it is.
        '''
        resize_images = []
        raw_img_size = images[0].shape[-2:]
        intrinsic = intrinsic.clone() # make copy

        if img_size is None:
            img_size = self.FLAGS['img_size']
            
        for img in images:
            assert raw_img_size == img.shape[-2:], "image size mismatch"
            if (img_size,img_size) == img.shape[-2:]:
                resize_images.append(img)
            else:
                resize_images.append(torch.nn.functional.interpolate(img, (img_size, img_size), mode='area'))
        
        width_scale = img_size / raw_img_size[1]
        height_scale = img_size / raw_img_size[0]
        intrinsic[:, 0] *= width_scale  
        intrinsic[:, 1] *= height_scale 
        if not align_corners:
            half = 0.5 * intrinsic[:, 2].sum(-1, keepdim=True)
            intrinsic[:, 0, 2:] += half * (width_scale - 1)
            intrinsic[:, 1, 2:] += half * (height_scale - 1)
            
        return resize_images + [intrinsic]
    
    def _get_obj_id(self, obj_idx):
        
        dataset_name, obj_id = self.meta_pairs[obj_idx]
        
        return dataset_name, obj_id
    
    def _get_obj_idx(self, dataset_name, obj_id):
        for obj_idx, (_dataset_name, _obj_id) in enumerate(self.meta_pairs):
            if (dataset_name is None or dataset_name == _dataset_name) and (_obj_id == obj_id):
                return obj_idx
        return -1
    
    @staticmethod
    def _load_pcd_npy(geopcd_dir, texpcd_dir):
        
        tex_pts_path = os.path.join(texpcd_dir, "pcd_points_500000.npy")
        tex_rgb_path = os.path.join(texpcd_dir, "pcd_colors_500000.npy")
        tex_nrm_path = os.path.join(texpcd_dir, "pcd_tex_normals_500000.npy")
        
        sdf_pts_path = os.path.join(geopcd_dir, "sdf_points_100000.npy")
        sdf_sdf_path = os.path.join(geopcd_dir, "sdf_sdfs_100000.npy")
        geo_pts_path = os.path.join(geopcd_dir, "pcd_points_300000.npy")
        geo_nrm_path = os.path.join(geopcd_dir, "pcd_normals_300000.npy")
        
        assert os.path.exists(tex_pts_path) and os.path.exists(tex_rgb_path) and os.path.exists(tex_nrm_path) and \
            os.path.exists(sdf_pts_path) and os.path.exists(sdf_sdf_path) and os.path.exists(geo_pts_path) and os.path.exists(geo_nrm_path)
            
        tex_points = torch.from_numpy(np.load(tex_pts_path)).float()
        tex_colors = torch.from_numpy(np.load(tex_rgb_path)).float()
        tex_normals = torch.from_numpy(np.load(tex_nrm_path)).float()

        sdf_points = torch.from_numpy(np.load(sdf_pts_path)).float()
        sdf_sdfs = torch.from_numpy(np.load(sdf_sdf_path)).float()
        geo_points = torch.from_numpy(np.load(geo_pts_path)).float() 
        geo_normals = torch.from_numpy(np.load(geo_nrm_path)).float()
            
        return tex_points, tex_colors, tex_normals, sdf_points, sdf_sdfs, geo_points, geo_normals
    
    @staticmethod
    def _load_pcd_h5(geo_pcd_dir, tex_pcd_dir):
        
        geo_h5_path = os.path.join(geo_pcd_dir, "pcd.h5")
        tex_h5_path = os.path.join(tex_pcd_dir, "pcd.h5")
        
        assert os.path.exists(geo_h5_path) and os.path.exists(tex_h5_path)
        
        geo_points, geo_normals, sdf_points, sdf_sdfs, \
            tex_points, tex_normals, tex_colors = h5.load_pcd_only(geo_h5_path, tex_h5_path)
            
        tex_points = torch.from_numpy(tex_points).float()
        tex_colors = torch.from_numpy(tex_colors).float()
        tex_normals = torch.from_numpy(tex_normals).float()

        sdf_points = torch.from_numpy(sdf_points).float()
        sdf_sdfs = torch.from_numpy(sdf_sdfs).float()
        geo_points = torch.from_numpy(geo_points).float() 
        geo_normals = torch.from_numpy(geo_normals).float()
        
        return tex_points, tex_colors, tex_normals, sdf_points, sdf_sdfs, geo_points, geo_normals
    
    @staticmethod
    def _load_pcd(geo_pcd_dir, tex_pcd_dir, preference="h5"):
        assert preference in ("h5", "npy")
        if preference=="h5":
            try:
                return MVLRMDataset._load_pcd_h5(geo_pcd_dir, tex_pcd_dir)
            except:
                return MVLRMDataset._load_pcd_npy(geo_pcd_dir, tex_pcd_dir)
        else:
            try:
                return MVLRMDataset._load_pcd_npy(geo_pcd_dir, tex_pcd_dir)
            except:
                return MVLRMDataset._load_pcd_h5(geo_pcd_dir, tex_pcd_dir)
    
    @staticmethod
    def _yup2zup_pcd(*pcds):
        ret_pcds = []
        yup2zup = torch.tensor([
            [1,0,0],
            [0,0,-1],
            [0,1,0]
        ], dtype=torch.float32)
        for pcd in pcds:
            pcd = pcd @ yup2zup.transpose(-1,-2)
            ret_pcds.append(pcd)
        return ret_pcds
            
    def _load_pcd_new(self, geo_pcd_dir, refc2o=None):
         
        assert self.FLAGS.n_surface_pts <= 500000, "n_surface_pts must not exceed 500000."
        assert self.FLAGS.n_near_surface_pts <= 500000, "n_near_surface_pts must not exceed 500000."
        assert self.FLAGS.n_space_pts <= 500000, "n_space_pts must not exceed 500000."
        
        surface_samples = np.sort(np.random.choice(500000, self.FLAGS.n_surface_pts))
        near_surface_samples = np.sort(np.random.choice(500000, self.FLAGS.n_near_surface_pts))
        space_samples = np.sort(np.random.choice(500000, self.FLAGS.n_space_pts))

        surface_pts = torch.from_numpy(np.load(os.path.join(geo_pcd_dir, "surface_point_500000.npy"))[surface_samples]).float()
        surface_norms = torch.from_numpy(np.load(os.path.join(geo_pcd_dir, "surface_normal_500000.npy"))[surface_samples]).float()
        
        near_surface_pts = torch.from_numpy(np.load(os.path.join(geo_pcd_dir, "near_surface_point_500000.npy"))[near_surface_samples]).float()
        near_surface_sdf = -torch.from_numpy(np.load(os.path.join(geo_pcd_dir, "near_surface_sdf_500000.npy"))[near_surface_samples]).float().reshape(-1,1)
        near_surface_vis = 1 - torch.from_numpy(np.load(os.path.join(geo_pcd_dir, "near_surface_occupancy_500000.npy"))[near_surface_samples]).float().reshape(-1,1)
        
        space_pts = torch.from_numpy(np.load(os.path.join(geo_pcd_dir, "space_point_500000.npy"))[space_samples]).float()
        space_sdf = -torch.from_numpy(np.load(os.path.join(geo_pcd_dir, "space_sdf_500000.npy"))[space_samples]).float().reshape(-1,1)
        space_vis = 1 - torch.from_numpy(np.load(os.path.join(geo_pcd_dir, "space_occupancy_500000.npy"))[space_samples]).float().reshape(-1,1)
        
        surface_pts, surface_norms, near_surface_pts, space_pts = self._yup2zup_pcd(surface_pts, surface_norms, near_surface_pts, space_pts)
        
        if refc2o is not None:
            surface_pts, surface_norms, near_surface_pts, space_pts = \
                self._process_pcd(surface_pts, surface_norms, near_surface_pts, space_pts, refc2o=refc2o)
        
        return surface_pts, surface_norms, near_surface_pts, near_surface_sdf, near_surface_vis, space_pts, space_sdf, space_vis
    
    def _load_pcd_new_h5(self, geo_pcd_dir, refc2o=None):
        
        assert self.FLAGS.n_surface_pts <= 500000, "n_surface_pts must not exceed 500000."
        assert self.FLAGS.n_near_surface_pts <= 500000, "n_near_surface_pts must not exceed 500000."
        assert self.FLAGS.n_space_pts <= 500000, "n_space_pts must not exceed 500000."
        
        assert self.FLAGS.get("offline_sample_pcd_strategy", "consecutive") in ["consecutive", "uniform"]
        
        if self.FLAGS.get("offline_sample_pcd_strategy", "consecutive") == "consecutive": # much faster but not completely random
            start_idx = np.random.randint(500000 + 1 - self.FLAGS.n_surface_pts)
            surface_samples = np.arange(start_idx, start_idx+self.FLAGS.n_surface_pts)
            start_idx = np.random.randint(500000 + 1 - self.FLAGS.n_near_surface_pts)
            near_surface_samples = np.arange(start_idx, start_idx+self.FLAGS.n_near_surface_pts)
            start_idx = np.random.randint(500000 + 1 - self.FLAGS.n_space_pts)
            space_samples = np.arange(start_idx, start_idx+self.FLAGS.n_space_pts)
        else: # much slower but completely random
            surface_samples = np.sort(np.random.choice(500000, self.FLAGS.n_surface_pts))
            near_surface_samples = np.sort(np.random.choice(500000, self.FLAGS.n_near_surface_pts))
            space_samples = np.sort(np.random.choice(500000, self.FLAGS.n_space_pts))
        
        with h5py.File(os.path.join(geo_pcd_dir, "sample.h5")) as f:
            
            surface_pts = torch.from_numpy(np.array(f["surface_points"][surface_samples])).float()
            surface_norms = torch.from_numpy(np.array(f["surface_normals"][surface_samples])).float()
            
            near_surface_pts = torch.from_numpy(np.array(f["near_surface_points"][near_surface_samples])).float()
            near_surface_sdf = -torch.from_numpy(np.array(f["near_surface_sdf"][near_surface_samples])).float()
            near_surface_vis = 1 - torch.from_numpy(np.array(f["near_surface_occupancy"][near_surface_samples])).float()
            
            space_pts = torch.from_numpy(np.array(f["space_points"][space_samples])).float()
            space_sdf = -torch.from_numpy(np.array(f["space_sdf"][space_samples])).float()
            space_vis = 1 - torch.from_numpy(np.array(f["space_occupancy"][space_samples])).float()
        
        surface_pts, surface_norms, near_surface_pts, space_pts = self._yup2zup_pcd(surface_pts, surface_norms, near_surface_pts, space_pts)
        
        if refc2o is not None:
            surface_pts, surface_norms, near_surface_pts, space_pts = \
                self._process_pcd(surface_pts, surface_norms, near_surface_pts, space_pts, refc2o=refc2o)
        
        return surface_pts, surface_norms, near_surface_pts, near_surface_sdf, near_surface_vis, space_pts, space_sdf, space_vis    
            
        

    def _parse_h5_item(self, obj_idx, camsys_idx, refview_idx=None, images_only=False):
        
        dataset_name, obj_id = self._get_obj_id(obj_idx)
        camera = self.cameras_config[camsys_idx]

        if refview_idx is None:
            refview_idx = list(range(len(camera['ref']))) # default to all views
        elif isinstance(refview_idx, int):
            refview_idx = [refview_idx]

        ref_view_id = [camera['ref'][i] for i in refview_idx]
        src_view_id = camera['src']
        
        idx_img_list = ref_view_id + src_view_id
            
            
        img_list, albedo_list, depth_list, normal_list, intrinsic_list, c2o_list, \
        geo_points, geo_normals, sdf_points, sdf_sdfs, \
        tex_points, tex_normals, tex_colors = h5.load_h5(self.h5_paths[obj_idx], idx_img_list, images_only)
        
        if (not images_only) and (geo_points is None): # did not load pcd from h5, try loading npy instead
            
            geopcd_dir = self.geopcd_dirs[obj_idx]
            texpcd_dir = self.texpcd_dirs[obj_idx]
            
            tex_points, tex_colors, tex_normals, sdf_points, sdf_sdfs, geo_points, geo_normals = self._load_pcd(geopcd_dir, texpcd_dir)
            

        color = torch.stack([_process_img(img, srgb=False) for img in img_list], dim=0).permute(0,3,1,2)
        albedo = torch.stack([_process_img(img, srgb=False) for img in albedo_list], dim=0).permute(0,3,1,2)
        depth = torch.stack([_process_img(img, scaled=False, srgb=False) / 1000.0 for img in depth_list], dim=0).permute(0,3,1,2)
        normal = torch.stack([_process_img(img, srgb=False) for img in normal_list], dim=0).permute(0,3,1,2)

        def to_torch(*args):
            return [(torch.from_numpy(arg) if isinstance(arg, np.ndarray) else arg) for arg in args ]

        intrinsic, c2o, geo_points, geo_normals, sdf_points, sdf_sdfs, \
        tex_points, tex_normals, tex_colors = to_torch(intrinsic_list, c2o_list, \
        geo_points, geo_normals, sdf_points, sdf_sdfs, \
        tex_points, tex_normals, tex_colors)
        refc2o = c2o[0] # [4,4]
        
        # coordinate transform
        normal, xyz, distance = self._process_geometry_imgs(normal, depth, intrinsic, c2o, refc2o) 
        
        c2w, o2w, n2w = self._get_transforms(c2o, refc2o)
        
        if 'custom_shading' in self.FLAGS:
            shaded = self._custom_shader(albedo, normal)
            # add background
            color, albedo, shaded, normal, xyz = self._apply_background(color, albedo, shaded, normal, xyz)
            #resize
            ref_color, ref_intrinsic = self._resize_images(color[:1], intrinsic=intrinsic[:1], img_size=self.FLAGS.get('ref_img_size', None))
            color, albedo, shaded, normal, xyz, depth, distance, intrinsic = self._resize_images(color, albedo, shaded, normal, xyz, depth, distance, intrinsic=intrinsic)
        else:
            # add background
            color, albedo, normal, xyz = self._apply_background(color, albedo, normal, xyz)
            #resize
            ref_color, ref_intrinsic = self._resize_images(color[:1], intrinsic=intrinsic[:1], img_size=self.FLAGS.get('ref_img_size', None))
            color, albedo, normal, xyz, depth, distance, intrinsic = self._resize_images(color, albedo, normal, xyz, depth, distance, intrinsic=intrinsic)
            shaded = None
            
        if not images_only:
            tex_points, sdf_points, geo_points, tex_normals, geo_normals = self._process_pcd(tex_points, sdf_points, geo_points, tex_normals, geo_normals, refc2o=refc2o)
          
        return ref_color, color, albedo, shaded, normal, xyz, depth, distance, tex_points, tex_colors, tex_normals, \
               sdf_points, sdf_sdfs, geo_points, geo_normals, ref_intrinsic, intrinsic, c2w, o2w, n2w, \
               ref_view_id, src_view_id, dataset_name, obj_id
    
    @staticmethod
    def parse_cam_params(cam_data, resolution=None):
        '''
        returns: c2o as 4x4 camera to object matrix
        intrinsic as 3x4 matrix representing either pinhole or orthographic camera
        '''
        c2o = np.array(cam_data['pose'])
        if cam_data.get("model", "pinhole") in ("pinhole", "perspective"):
            intrinsic = np.zeros(3,4)
            intrinsic[:3,:3] = np.array(cam_data['k'])
        elif cam_data.get("model", "pinhole") == "orthographic":
            assert resolution is not None
            scale = cam_data["scale"]
            intrinsic = np.array([
                [resolution/scale, 0, 0, resolution/2],
                [0, resolution/scale, 0, resolution/2],
                [0, 0, 0, 1.0],
            ])
        else:
            raise ValueError(f"unrecognised cambera model {cam_data.get('model')}")
        
        return torch.from_numpy(c2o).float(), torch.from_numpy(intrinsic).float()
    
    def _parse_raw_frame(self, obj_dir, cam_param_json_path, idx_img_list):
        
        img_suffix = '.png'
        
        img_modalities = self.FLAGS.get("image_modalities", ["color", "normal", "depth", "xyz"])
        
        with open(cam_param_json_path, 'r') as cam_json_file:
            cam_datas = json.load(cam_json_file)  
            cam_names = list(cam_datas.keys())
        
        imgs, albedos, normals, depths, c2os, intrinsics = [], [], [], [], [], []
        
        for idx_img in idx_img_list:
            
            cam_name = cam_names[idx_img]
            cam_data = cam_datas[cam_name]

            try:
                assert 'color' in img_modalities or 'rgb' in img_modalities or 'rgba' in img_modalities
                img = _load_img(os.path.join(obj_dir, 'color', cam_name + img_suffix), srgb=False)
            except:
                img = None
            try:
                assert 'albedo' in img_modalities
                albedo = _load_img(os.path.join(obj_dir, 'emission', 'color', cam_name + img_suffix), srgb=False)
            except:
                albedo = None
            try:
                assert 'normal' in img_modalities
                normal = _load_img(os.path.join(obj_dir, 'normal', cam_name + img_suffix), srgb=False)
            except:
                normal = None
            try:
                assert 'depth' in img_modalities or 'xyz' in img_modalities
                depth = _load_img(os.path.join(obj_dir, 'depth', cam_name + img_suffix), scaled=False, srgb=False) / 1000.0 # mm to metres
            except:
                depth = None
            
            assert any(i is not None for i in [img, albedo, normal, depth]), "load images all failed"
            
            img_size = next(i.shape[:2] for i in [img, albedo, normal, depth] if i is not None)
            img_size = tuple(img_size)
            
            if img is None:
                img = _zero_image(img_size, 4)
            if albedo is None:
                albedo = _zero_image(img_size, 4)
            if normal is None:
                normal = _zero_image(normal, 3) + 0.5
            if depth is None:
                depth = _zero_image(depth, 4)
            
            c2o, intrinsic = self.parse_cam_params(cam_data, img.shape[0])

            imgs.append(img)
            albedos.append(albedo)
            normals.append(normal)
            depths.append(depth)
            c2os.append(c2o)
            intrinsics.append(intrinsic)
        
        return imgs, albedos, normals, depths, c2os, intrinsics

    def _parse_raw_item(self, obj_idx, camsys_idx, refview_idx=None, images_only=False):
        """parse one obj 

        Args:
            idx_obj: _description_
            view_all: if true, parse all imgs and all points. Defaults to False.

        Returns:
            img [1, nv, h, w, 3] -> will be batch [no, nv, h, w, 3]
            mv [1, nv, 4, 4] 
            mvp [1, nv, 4, 4] 
            campos [1, nv, 3] 
            points [1, N, 3]   N = Nsurf + Nrand
            masks, normals, sdf  [1, N], [1, N, 3], [1, N, 1]
        """
        
        dataset_name, obj_id = self._get_obj_id(obj_idx)
        camera = self.cameras_config[camsys_idx]

        if refview_idx is None:
            refview_idx = list(range(len(camera['ref']))) # default to all views
        elif isinstance(refview_idx, int):
            refview_idx = [refview_idx]

        ref_view_id = [camera['ref'][i] for i in refview_idx]
        src_view_id = camera['src']
        
        idx_img_list = ref_view_id + src_view_id

        render_dir = self.render_dirs[obj_idx]
        geopcd_dir = self.geopcd_dirs[obj_idx]
        texpcd_dir = self.texpcd_dirs[obj_idx]
        
        # parse_obj_cfg
        img_list, albedo_list, normal_list, depth_list, c2o_list, intrinsic_list = \
            self._parse_raw_frame(render_dir, os.path.abspath(os.path.join(render_dir, 'cam_parameters.json')), idx_img_list)

        color = torch.stack(list(img_list), dim=0).permute(0,3,1,2) # [n_ref+n_source, 4, H, W]
        albedo = torch.stack(list(albedo_list), dim=0).permute(0,3,1,2) # [n_ref+n_source, 4, H, W]
        normal = torch.stack(list(normal_list), dim=0).permute(0,3,1,2) # [n_ref+n_source, 3, H, W]
        depth = torch.stack(list(depth_list), dim=0).permute(0,3,1,2) # [n_ref+n_source, 1, H, W]
        c2o = torch.stack(list(c2o_list), dim=0) # [n_ref+n_source, 4,4]
        intrinsic = torch.stack(list(intrinsic_list), dim=0) # [n_ref+n_source, 3,4]
        refc2o = c2o[0] # [4,4]
        
        # coordinate transform
        normal, xyz, distance = self._process_geometry_imgs(normal, depth, intrinsic, c2o, refc2o) 
        
        c2w, o2w, n2w = self._get_transforms(c2o, refc2o)
        
        
        if 'custom_shading' in self.FLAGS:
            shaded = self._custom_shader(albedo, normal)
            # add background
            color, albedo, shaded, normal, xyz = self._apply_background(color, albedo, shaded, normal, xyz)
            #resize
            ref_color, ref_intrinsic = self._resize_images(color[:1], intrinsic=intrinsic[:1], img_size=self.FLAGS.get('ref_img_size', None))
            color, albedo, shaded, normal, xyz, depth, distance, intrinsic = self._resize_images(color, albedo, shaded, normal, xyz, depth, distance, intrinsic=intrinsic)
        
        else:
            # add background
            color, albedo, normal, xyz = self._apply_background(color, albedo, normal, xyz)
            #resize
            ref_color, ref_intrinsic = self._resize_images(color[:1], intrinsic=intrinsic[:1], img_size=self.FLAGS.get('ref_img_size', None))
            color, albedo, normal, xyz, depth, distance, intrinsic = self._resize_images(color, albedo, normal, xyz, depth, distance, intrinsic=intrinsic)
            shaded = None
            
        if not images_only:
            tex_points, tex_colors, tex_normals, sdf_points, sdf_sdfs, geo_points, geo_normals = self._load_pcd(geopcd_dir, texpcd_dir)

            tex_points, sdf_points, geo_points, tex_normals, geo_normals = self._process_pcd(tex_points, sdf_points, geo_points, tex_normals, geo_normals, refc2o=refc2o)
        
        else:
             tex_points, tex_colors, tex_normals, \
               sdf_points, sdf_sdfs, geo_points, geo_normals = None, None, None, None, None, None, None
            
          
        return ref_color, color, albedo, shaded, normal, xyz, depth, distance, tex_points, tex_colors, tex_normals, \
               sdf_points, sdf_sdfs, geo_points, geo_normals, ref_intrinsic, intrinsic, c2w, o2w, n2w, \
               ref_view_id, src_view_id, dataset_name, obj_id
    
    def _parse_item(self, obj_idx, camsys_idx, refview_idx, images_only=False):
        
        if self.h5_paths[obj_idx] is None:
            ref_color, color, albedo, shaded, normal, xyz, depth, distance, tex_points, tex_colors, tex_normals, \
               sdf_points, sdf_sdfs, geo_points, geo_normals, ref_intrinsic, intrinsic, c2w, o2w, n2w, \
               ref_view_id, src_view_id, dataset_name, obj_id = self._parse_raw_item(obj_idx, camsys_idx, refview_idx, images_only)
        else:
            try:
                ref_color, color, albedo, shaded, normal, xyz, depth, distance, tex_points, tex_colors, tex_normals, \
                sdf_points, sdf_sdfs, geo_points, geo_normals, ref_intrinsic, intrinsic, c2w, o2w, n2w, \
                ref_view_id, src_view_id, dataset_name, obj_id = self._parse_h5_item(obj_idx, camsys_idx, refview_idx, images_only)
            except:
                ref_color, color, albedo, shaded, normal, xyz, depth, distance, tex_points, tex_colors, tex_normals, \
                sdf_points, sdf_sdfs, geo_points, geo_normals, ref_intrinsic, intrinsic, c2w, o2w, n2w, \
                ref_view_id, src_view_id, dataset_name, obj_id = self._parse_raw_item(obj_idx, camsys_idx, refview_idx, images_only)
                
        
        return ref_color, color, albedo, shaded, normal, xyz, depth, distance, tex_points, tex_colors, tex_normals, \
               sdf_points, sdf_sdfs, geo_points, geo_normals, ref_intrinsic, intrinsic, c2w, o2w, n2w, \
               ref_view_id, src_view_id, dataset_name, obj_id

    def _sample_pcd_from_mesh(self, obj_idx, refc2o, n_surface_pts, n_near_surface_pts, n_space_pts, near_surface_std, n_visibility_random_samples=128):
        
        manifold_mesh_path = self.manifold_mesh_paths[obj_idx]
        manifold_transform_path = self.manifold_transformation_paths[obj_idx]
        
        surface_pts, surface_norms, near_surface_pts, near_surface_sdf, near_surface_visibility, space_pts, space_sdf, space_visibility \
            = sample_pcd.sample_pcd(manifold_mesh_path, manifold_transform_path, n_surface_pts, n_near_surface_pts, n_space_pts, near_surface_std, n_visibility_random_samples)
        
        def to_torch(*args):
            return [(torch.from_numpy(arg) if isinstance(arg, np.ndarray) else arg) for arg in args ]
        
        surface_pts, surface_norms, near_surface_pts, near_surface_sdf, near_surface_visibility, space_pts, space_sdf, space_visibility \
            = to_torch(surface_pts, surface_norms, near_surface_pts, near_surface_sdf, near_surface_visibility, space_pts, space_sdf, space_visibility)
            
        surface_pts, surface_norms, near_surface_pts, space_pts = self._process_pcd(surface_pts, surface_norms, near_surface_pts, space_pts, refc2o=refc2o)
        
        near_surface_sdf = near_surface_sdf.reshape(-1,1)
        near_surface_visibility = near_surface_sdf.reshape(-1,1)
        space_sdf = space_sdf.reshape(-1,1)
        space_visibility = space_visibility.reshape(-1,1)
        
        return surface_pts, surface_norms, near_surface_pts, near_surface_sdf, near_surface_visibility, space_pts, space_sdf, space_visibility
    
    def _process_mesh(self, obj_idx, o2w, process_verts=False):
        
        manifold_mesh_path = self.manifold_mesh_paths[obj_idx]
        manifold_transform_path = self.manifold_transformation_paths[obj_idx]
        
        manifold_verts, manifold_faces = sample_pcd.load_mesh(manifold_mesh_path, manifold_transform_path)
        
        if process_verts:
            manifold_mesh = trimesh.Trimesh(manifold_verts, manifold_faces)
            manifold_verts, manifold_faces = manifold_mesh.vertices, manifold_mesh.faces
        
        manifold_verts = torch.from_numpy(manifold_verts).float()
        manifold_faces = torch.from_numpy(manifold_faces).int()
        
        o2w = o2w / o2w[-1,-1]
        manifold_verts = manifold_verts @ o2w[:3,:3].transpose(0,1) + o2w[:3,3]
        
        return manifold_verts, manifold_faces    
    
    def _get_camera_embeddings(self, c2w):
        '''
        c2w: of shape [..., Nviews, 4,4]
        returns: [..., Nviews, 3], last dim is (ref_elevation, elevation, azimuth)
        '''
        lookfrom = -c2w[...,:3,2] # use negative z axis as lookfrom
        azimuth = torch.arctan2(lookfrom[...,1], lookfrom[...,0]) # [...]
        elevation = torch.arctan2(lookfrom[...,2], torch.linalg.vector_norm(lookfrom[...,:2], dim=-1)) # [...]
        ref_elevation = elevation[...,0:1].expand_as(elevation)
        azimuth_negative_mask = (azimuth < -1e-3)
        azimuth[azimuth_negative_mask] = azimuth[azimuth_negative_mask] + 2 * np.pi
        return torch.stack([ref_elevation, elevation, azimuth], dim=-1)
        
        
    def _get_task_embeddings(self, n_views):
        
        out = {}
        for class_embeddings in self.FLAGS['classe_embeddings']:

            assert len(class_embeddings) == 1, "format error"

            class_name = list(class_embeddings.keys())[0]
            class_embeddings = list(class_embeddings.values())[0]

            out[f"{class_name}_task_embeddings"] = torch.tensor(class_embeddings).float().expand(n_views,-1)
        
        return out
    
    def _get_vae_cache(self, obj_idx, view_idx):
        dataset_name, obj_id = self.meta_pairs[obj_idx]
        
        vae_path = os.path.join(self.FLAGS['vae_latent_dir'], dataset_name, f"{obj_id}.h5")
        
        out = {}
        
        with h5py.File(vae_path) as h5file:
            if 'view_ids' in h5file.keys():
                cached_view_ids = np.asarray(h5file['view_ids']).tolist()
                cached_view_idx = []
                for view_i in view_idx:
                    assert view_i in cached_view_ids, f"view {view_i} was not cached for {obj_idx}-th dataset item {vae_path}"
                    cached_view_idx.append(cached_view_ids.index(view_i))
            else:
                cached_view_idx = view_idx
                
            out['color_mean'] = torch.from_numpy(np.array(h5file['color_mean'][cached_view_idx])).float()
            out['color_logvar'] = torch.from_numpy(np.array(h5file['color_logvar'][cached_view_idx])).float()
            out['albedo_mean'] = torch.from_numpy(np.array(h5file['albedo_mean'][cached_view_idx])).float()
            out['albedo_logvar'] = torch.from_numpy(np.array(h5file['albedo_logvar'][cached_view_idx])).float()
            out['xyz_mean'] = torch.from_numpy(np.array(h5file['xyz_mean'][cached_view_idx])).float()
            out['xyz_logvar'] = torch.from_numpy(np.array(h5file['xyz_logvar'][cached_view_idx])).float()
            out['normal_mean'] = torch.from_numpy(np.array(h5file['normal_mean'][cached_view_idx])).float()
            out['normal_logvar'] = torch.from_numpy(np.array(h5file['normal_logvar'][cached_view_idx])).float()

        return out
    
    def _get_random_cam_id(self, obj_idx):
        
        camsys_idx = random.choice(self.cam_systems[obj_idx])
        refview_idx = random.randint(0, len(self.cameras_config[camsys_idx]['ref'])-1)
        
        return camsys_idx, refview_idx
    
    def get_vae(self):
        return AutoencoderKL.from_pretrained(self.FLAGS['vae_checkpoint_path'], subfolder="vae")  
    
    def _get_ndc_projection(self, h, w, intrinsic, c2w, align_corners=True, ndc_convention='nvdiffrast', n=None, f=None):
        """
        Compute the normalized device coordinates (NDC) projection matrix.

        Args:
            - h (int): Height of the image.
            - w (int): Width of the image.
            - intrinsic (torch.Tensor): Intrinsic matrix of shape (Nviews, 3, 4).
              representing the camera projection matrix, either perspective or orthographic.
            - c2w (torch.Tensor): Camera-to-world transformation matrix of shape (Nviews, 4, 4).
            - align_corners (bool, optional): Whether the intrinsic's image origin is the corner (or centre) of top-left pixel.
            - ndc_convention: string, 'nvdiffrast' or 'opengl'. nvdiffrast is different from opengl in the y-axis direction.
            - n (float, optional): Near plane distance. generally this is a positive value. If None, computes near and far from c2w
            - f (float, optional): Far plane distance. generally this is a positive value greater than near. If None, computes near and far from c2w

        Returns:
            torch.Tensor: w2dnc, the NDC projection matrix of shape (Nviews, 4, 4).
            
        both intrinsic and c2w follow opencv convention with +x being right, +y being down and +z being forward in camera coordinates.
        and +u being right, +v being down for image coordinates.
        
        the 3x4 intrinsic can represent either perspective or orthographic cameras, it MUST follow below format: 
        - if perspective:
            [fx, skew, cx, 0]
            [0,   fy,  cy, 0]
            [0,   0,   1,  0]
        - if orthographic:
            [sx, skew, 0, cx]
            [0,   sy,  0, cy]
            [0,   0,   0,  1]
        and if align_corners is set to true, the intrinsic is assumed to have the image origin at the top-left corner of the top-left pixel.
        otherwise the image origin is at the centre of the top-left pixel.
        
        the output w2ndc describes the transformation from world coordinates to normalized device coordinates where the view frustum is defined in [-1,1]^3.
        the convention of the NDC is either 'nvdiffrast' or 'opengl'.
        - for 'nvdiffrast', the NDC is defined as +x right, +y down, +z forward. and (x=-1 y=-1) defines the top left corner of the top left pixel, (x=1,y=1) defines
          the bottom right corner of bottom right pixel. z=-1 is the near plane and z=1 is the far plane. in this case the returned matrix is the 'mvp' matrix used in nvdiff* libraries.
        - for 'opengl', the NDC is defined as +x right, +y up, +z forward, effectively a left-handed system. and (x=-1 y=1) defines the top left corner of the top left pixel, (x=1,y=-1) defines
          the bottom right corner of bottom right pixel. z=-1 is the near plane and z=1 is the far plane.
        
        """
        assert ndc_convention in ['nvdiffrast', 'opengl'], "ndc_convention must be 'nvdiffrast' or 'opengl'"
        
        w2c = torch.inverse(c2w)
        o_depth = w2c[:,2,3] / w2c[:,3,3] # [Nviews] depth of world origin point in camera
        n = (o_depth - 1.1).clip(min=0.1)
        f = (o_depth + 1.1).clip(min=n+0.1)
        
        intrinsic = torch.cat((intrinsic, intrinsic[:,-1:]), dim=1) # [Nviews, 4,4]
        t_persp = intrinsic[:,2,2].clone() # for perspective cameras, this is one
        t_ortho = intrinsic[:,2,3].clone() # for orthographic cameras, this is one
        intrinsic[:,2,2] = (f+n)/(f-n) * t_persp + 2/(f-n) * t_ortho
        intrinsic[:,2,3] = 2*f*n/(n-f) * t_persp + (n+f)/(n-f) * t_ortho
        
        y_sign = 1 if ndc_convention == 'nvdiffrast' else -1
        
        if align_corners:
            cv2ndc = torch.tensor([
                [2/w, 0, 0, -1],
                [0, y_sign*2/h, 0, y_sign*(-1)],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=intrinsic.dtype, device=intrinsic.device) # [4,4]
        else:
            cv2ndc = torch.tensor([
                [2/w, 0, 0, -1+1/w],
                [0, y_sign*2/h, 0, y_sign*(-1+1/h)],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=intrinsic.dtype, device=intrinsic.device)
            
        w2ndc = cv2ndc @ intrinsic @ torch.inverse(c2w)
        
        return w2ndc, n, f
    
    def _custom_shader(self, albedo, normal):
        light_directions = torch.tensor(self.FLAGS['custom_shading']['light_directions']) # [num_lights, 3]
        light_intensities = torch.tensor(self.FLAGS['custom_shading']['light_intensities']) # [num_lights]
        lights = light_directions * light_intensities.unsqueeze(dim=-1) # [num_lights, 3]
        
        shadings = torch.einsum('vxhw,lx->vlhw', normal[:,:3]*2-1, -lights) # [num_views, num_lights, h, w]
        shadings = shadings.clip(min=0).sum(dim=1,keepdim=True) # [num_views, 1, h, w]
        shaded_rgb = shadings * albedo[:,:3]
        shaded_alpha = normal[:,3:4] * albedo[:,3:4]
        shaded = torch.cat((shaded_rgb, shaded_alpha), dim=1)
        return shaded
    
    def _parse_render_config(self, obj_idx):
        with open(os.path.join(self.render_dirs[obj_idx], 'config.json')) as render_json:
            return json.load(render_json)
        
    def _build_data(self, obj_idx, camsys_idx, refview_idx, images_only=False, online_sample_pcd=False):
        
        ref_color, color, albedo, shaded, normal, xyz, depth, distance, tex_points, tex_colors, tex_normals, \
            sdf_points, sdf_sdfs, geo_points, geo_normals, ref_intrinsic, intrinsic, c2w, o2w, n2w, \
            ref_view_id, src_view_id, dataset_name, obj_id = self._parse_item(obj_idx, camsys_idx, refview_idx, images_only=True)
        
        if self.FLAGS.get("load_mesh", True):
            manifold_mesh_verts, manifold_mesh_faces = self._process_mesh(obj_idx, o2w, process_verts=False)
        else:
            manifold_mesh_verts, manifold_mesh_faces = None, None
        
        if not images_only:
            refc2o = torch.inverse(o2w) @ c2w[0]
            if online_sample_pcd:
                surface_pts, surface_norms, near_surface_pts, near_surface_sdf, near_surface_visibility, space_pts, space_sdf, space_visibility = \
                    self._sample_pcd_from_mesh(obj_idx, refc2o, self.FLAGS.get("online_sample_n_surface_pts"), self.FLAGS.get("online_sample_n_near_surface_pts"), self.FLAGS.get("online_sample_n_space_pts"), self.FLAGS.get("online_sample_near_surface_std"), self.FLAGS.get("online_sample_n_visibility_rays", 128))
            else:
                surface_pts, surface_norms, near_surface_pts, near_surface_sdf, near_surface_visibility, space_pts, space_sdf, space_visibility = \
                    self._load_pcd_new_h5(self.geopcd_dirs[obj_idx], refc2o)      
               
        view_id = ref_view_id + src_view_id
               
        mask = color[:,-1:].clone().contiguous()
        
        augmented_color, augmented_albedo, augmented_normal, augmented_xyz, augmented_mask = color.clone(), albedo.clone(), normal.clone(), xyz.clone(), mask.clone()

        augmentation_params = self.augmentor(augmented_color, augmented_albedo, augmented_normal, augmented_xyz, augmented_mask)
        
        # # re-apply background if color was adjusted
        # bg_color = self._get_background_color()
        # if augmentation_params['rgb']['adjusted']:
        #     color, = self._reapply_background(bg_color, augmented_color)
        # if augmentation_params['albedo']['adjusted']:
        #     albedo, = self._reapply_background(bg_color, augmented_albedo)
        # if augmentation_params['normal']['adjusted']:
        #     normal, = self._reapply_background(bg_color, augmented_normal)
        # if augmentation_params['xyz']['adjusted']:
        #     xyz, = self._reapply_background(bg_color, augmented_xyz)
        
        camera_embeddings = self._get_camera_embeddings(c2w)
        class_embeddings = self._get_task_embeddings(color.shape[0])
        
        
        w2ndc_nv, near, far = self._get_ndc_projection(color.shape[-2], color.shape[-1], intrinsic, c2w, ndc_convention='nvdiffrast')
        
        
        try:
            assert not self.FLAGS.get("ignore_hdri", False)
            render_config = self._parse_render_config(obj_idx)
            hdri_paths = [render_config["hdr"]["paths"][f"cam-{i:04}"] for i in view_id]
            assert all(hdri_path==hdri_paths[0] for hdri_path in hdri_paths), "expected same hdri but found inconsistent ones"
            hdri_path = hdri_paths[0]
            hdri, hdri_sh = self.hdri_cache.get(hdri_path, o2w)
        except:
            hdri_path = "not available"
            hdri, hdri_sh = "not available", "not available"
        
        out =  {
            
            'camera_embeddings': camera_embeddings,
            'elevations_cond': camera_embeddings[:,0],
            'elevations_cond_deg': torch.rad2deg(camera_embeddings[:,0]),
            'elevations': camera_embeddings[:,1],
            'elevations_deg': torch.rad2deg(camera_embeddings[:,1]),
            'azimuths': camera_embeddings[:,2],
            'azimuths_deg': torch.rad2deg(camera_embeddings[:,2]),
            
            'ref_rgba': ref_color,
            'ref_img': ref_color[:,:3],
            'ref_color': ref_color[:,:3],
            'ref_rgb': ref_color[:,:3],
            'ref_intrinsic': ref_intrinsic,
            
            'manifold_mesh_verts': manifold_mesh_verts,
            'manifold_mesh_faces': manifold_mesh_faces,
            
            'rgba': color,
            'imgs_in': color[:,:3],
            'color': color[:,:3],
            'rgb': color[:,:3],
    
            'alphas': mask,
            'mask': mask,
            
            'normal': normal[:,:3],
            'normals': normal[:,:3],
            'normal_rgba': normal,
            
            'albedo': albedo[:,:3],
            'albedo_rgba': albedo,
            
            'xyz': xyz[:,:3],
            'xyz_rgba': xyz,
            
            'depth': depth,
            'distance': distance,
            
            'view_id': torch.tensor(view_id),
            
            'filename': (dataset_name, obj_id),
            'dataset_name': dataset_name,
            'obj_id': obj_id,
            
            'obj2world': o2w,
            'cam2world': c2w,
            'normal2world': n2w,
            'extrinsic': c2w,
            'cam2object': torch.inverse(o2w) @ c2w,
            'intrinsic': intrinsic,
            'mvp': w2ndc_nv,
            'mvp_near': near,
            'mvp_far': far,
            
            'obj_idx': obj_idx,
            'camsys_idx': camsys_idx, 
            'refview_idx': refview_idx,
            
            "hdri": hdri,
            "hdri_sh": hdri_sh,
            "hdri_path": hdri_path,
            
            'augmented_images': 
            {
                'rgba': augmented_color,
                'imgs_in': augmented_color[:,:3],
                'color': augmented_color[:,:3],
                'rgb': augmented_color[:,:3],
        
                'alphas': augmented_mask,
                'mask': augmented_mask,
                
                'normal': augmented_normal[:,:3],
                'normals': augmented_normal[:,:3],
                'normal_rgba': augmented_normal,
                
                'albedo': augmented_albedo[:,:3],
                'albedo_rgba': augmented_albedo,
                
                'xyz': augmented_xyz[:,:3],
                'xyz_rgba': augmented_xyz,
                'augmentation_params': augmentation_params,
            }
                  
        }
        
        if not images_only:
            
            out.update({
                
            'surface_points': surface_pts,
            'surface_normals': surface_norms,
            
            'near_surface_points': near_surface_pts,
            'near_surface_visibility': near_surface_visibility,
            'near_surface_occupancy': 1 - near_surface_visibility,
            'near_surface_sdf': near_surface_sdf,
            
            'space_points': space_pts,
            'space_visibility': space_visibility,
            'space_occupancy': 1 - space_visibility,
            'space_sdf': space_sdf,
            
            })
            
          
        if 'custom_shading' in self.FLAGS:
            out.update({
                "shaded": shaded[:,:3],
                "shaded_rgba": shaded,
            })
        
        out.update(class_embeddings)
        
        if 'vae_latent_dir' in self.FLAGS:
            out.update(self._get_vae_cache(obj_idx, view_id))
        
        return out
    
    
    def __len__(self):
        return self.n_objs_all
    
    def __getitem__(self, obj_idx):
        
        try:
            camsys_idx, refview_idx = self._get_random_cam_id(obj_idx)
            return self._build_data(obj_idx, camsys_idx, refview_idx, images_only=self.FLAGS.get('images_only', False), online_sample_pcd=self.FLAGS.get('online_sample_pcd', False))
        except Exception as e:
            raise ValueError(f"Error when parsing {obj_idx}-th item, with id {self._get_obj_id(obj_idx)} from \"{self._get_raw_json_path(obj_idx)}\"") from e
            
    
    def visualize(self, obj_idx, camsys_idx=None, refview_idx=None, augmented=False):
        
        rand_camsys_idx, rand_refview_idx = self._get_random_cam_id(obj_idx)
        
        if camsys_idx is None:
            camsys_idx = rand_camsys_idx
        if refview_idx is None:
            refview_idx = rand_refview_idx
            
        import cv2
        
        sample = self._build_data(obj_idx, camsys_idx, refview_idx, images_only=self.FLAGS.get('images_only', False), online_sample_pcd=self.FLAGS.get('online_sample_pcd', False))
        
        if augmented:
            sample.update(sample['augmented_images'])
        
        def normalize(depth, near, min=0.0, max=1.0):
            depth = depth.squeeze(-1)
            fg_mask = (depth>=10.0)
            fg_d = np.where(fg_mask, depth, np.nan)
            min_d = np.nanmin(fg_d, axis=(1, 2), keepdims=True)
            near_d = np.reshape(near, (near.shape[0],1,1))
            min_d = np.maximum(min_d, near_d)
            max_d = np.nanmax(fg_d, axis=(1, 2), keepdims=True)
            
            norm_d = min + (fg_d - min_d) / (max_d - min_d) * max
            norm_d = np.where(fg_mask, norm_d, -1)
            return np.expand_dims(norm_d, -1)
        
        normal = torch.cat(tuple(sample['normals'].permute(0,2,3,1)), dim=1).cpu().numpy() * 255
        color = torch.cat(tuple(sample['imgs_in'].permute(0,2,3,1)), dim=1).cpu().numpy() * 255
        xyz = torch.cat(tuple(sample['xyz'].permute(0,2,3,1)), dim=1).cpu().numpy() * 255
        albedo = torch.cat(tuple(sample['albedo'].permute(0,2,3,1)), dim=1).cpu().numpy() * 255
        distance = sample['distance'].permute(0,2,3,1).cpu().numpy() 
        depth = sample['depth'].permute(0,2,3,1).cpu().numpy() 
        
        distance = np.concatenate(normalize(distance, sample['mvp_near'].cpu().numpy(), 0.0, 1.0), axis=1)
        depth = np.concatenate(normalize(depth, sample['mvp_near'].cpu().numpy(), 0.0, 1.0), axis=1)
        distance_mask = (distance > 0).squeeze(-1)
        depth_mask = (depth > 0).squeeze(-1)
        distance = cv2.applyColorMap((distance*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        depth = cv2.applyColorMap((depth*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        distance[distance_mask==False] = 255
        depth[depth_mask==False] = 255
        
        if 'shaded' in sample:
            shaded = torch.cat(tuple(sample['shaded'].permute(0,2,3,1)), dim=1).cpu().numpy() * 255

        if not self.FLAGS.get('images_only', False):
            
                
            render_image = np.ones_like(color) 
            N_views = sample['cam2world'].shape[0]
            mvp = sample['intrinsic'] @ torch.linalg.inv(sample['cam2world'])
            
            # render surface pts
            projected_pcd = torch.cat((sample['surface_points'], torch.ones_like(sample['surface_points'][...,:1])), dim=-1)  @ mvp.transpose(1,2)
            d = projected_pcd[...,-1:].numpy().reshape(-1)
            projected_pcd = projected_pcd[...,:-1] / projected_pcd[...,-1:] # [batch, uv]
            projected_pcd[..., 0] += torch.arange(len(projected_pcd), dtype=projected_pcd.dtype).reshape(-1,1) * sample['albedo'].shape[-1]
            u,v = projected_pcd.split([1,1], dim=-1)
            u = u.clip(0, render_image.shape[1]-1).int().numpy().reshape(-1)
            v = v.clip(0, render_image.shape[0]-1).int().numpy().reshape(-1)

            surface_nrm = render_pcd(render_image, u,v,d, (sample['surface_normals'].expand(N_views,-1,-1).reshape(-1,3).numpy()[...,[1,2,0]]+1)/2)*255
            surface_xyz = render_pcd(render_image, u,v,d, (sample['surface_points'].expand(N_views,-1,-1).reshape(-1,3).numpy()+1)/2)*255
            
            # render sdf inside points
            projected_pcd = torch.cat((sample['space_points'], torch.ones_like(sample['space_points'][...,:1])), dim=-1)  @ mvp.transpose(1,2)
            projected_pcd = projected_pcd[:,sample['space_sdf'].squeeze(-1) >= 0]
            d = projected_pcd[...,-1:].numpy().reshape(-1)
            projected_pcd = projected_pcd[...,:-1] / projected_pcd[...,-1:] # [batch, uv]
            projected_pcd[..., 0] += torch.arange(len(projected_pcd), dtype=projected_pcd.dtype).reshape(-1,1) * sample['albedo'].shape[-1]
            u,v = projected_pcd.split([1,1], dim=-1)
            u = u.clip(0, render_image.shape[1]-1).int().numpy().reshape(-1)
            v = v.clip(0, render_image.shape[0]-1).int().numpy().reshape(-1)

            space_sdf_in = render_pcd(render_image, u,v,d, np.zeros((u.shape[0],3))+[1,0,0])*255
            
            # render occupancy inside points
            projected_pcd = torch.cat((sample['space_points'], torch.ones_like(sample['space_points'][...,:1])), dim=-1)  @ mvp.transpose(1,2)
            projected_pcd = projected_pcd[:,sample['space_occupancy'].squeeze(-1) >= 0.9]
            d = projected_pcd[...,-1:].numpy().reshape(-1)
            projected_pcd = projected_pcd[...,:-1] / projected_pcd[...,-1:] # [batch, uv]
            projected_pcd[..., 0] += torch.arange(len(projected_pcd), dtype=projected_pcd.dtype).reshape(-1,1) * sample['albedo'].shape[-1]
            u,v = projected_pcd.split([1,1], dim=-1)
            u = u.clip(0, render_image.shape[1]-1).int().numpy().reshape(-1)
            v = v.clip(0, render_image.shape[0]-1).int().numpy().reshape(-1)

            space_occ_in = render_pcd(render_image, u,v,d, np.zeros((u.shape[0],3))+[1,0,0])*255
                
        if 'vae_latent_dir' in self.FLAGS:
            vae = self.get_vae().to(torch.float16).cuda()

            vae_xyz = np.concatenate(vae_decode(vae, sample['xyz_mean'].half()), axis=1).clip(0,1)*255
            vae_color = np.concatenate(vae_decode(vae, sample['color_mean'].half()), axis=1).clip(0,1)*255
            vae_albedo = np.concatenate(vae_decode(vae, sample['albedo_mean'].half()), axis=1).clip(0,1)*255
            vae_normal = np.concatenate(vae_decode(vae, sample['normal_mean'].half()), axis=1).clip(0,1)*255

            if not self.FLAGS.get('images_only', False): 
                img_all = [
                    cv2.putText(color.astype(np.uint8), 'rgb', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(vae_color.astype(np.uint8).copy(), 'vae.rgb', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(normal.astype(np.uint8), 'nrm', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2),
                    cv2.putText(vae_normal.astype(np.uint8).copy(), 'vae.nrm', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(surface_nrm.astype(np.uint8), 'surface nrm', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(albedo.astype(np.uint8), 'alb', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(vae_albedo.astype(np.uint8).copy(), 'vae.alb', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(xyz.astype(np.uint8), 'xyz', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(vae_xyz.astype(np.uint8).copy(), 'vae.xyz', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(surface_xyz.astype(np.uint8), 'surface xyz', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(space_sdf_in.astype(np.uint8), 'space sdf>0', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(space_occ_in.astype(np.uint8), 'space occu>0.9', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(depth.astype(np.uint8), 'depth (normalized)', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(distance.astype(np.uint8), 'distance (normalized)', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    ]
            else:
                img_all = [
                    cv2.putText(color.astype(np.uint8), 'rgb', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(vae_color.astype(np.uint8).copy(), 'vae.rgb', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(normal.astype(np.uint8), 'nrm', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2),
                    cv2.putText(vae_normal.astype(np.uint8).copy(), 'vae.nrm', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(albedo.astype(np.uint8), 'alb', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(vae_albedo.astype(np.uint8).copy(), 'vae.alb', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(xyz.astype(np.uint8), 'xyz', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(vae_xyz.astype(np.uint8).copy(), 'vae.xyz', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(depth.astype(np.uint8), 'depth (normalized)', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(distance.astype(np.uint8), 'distance (normalized)', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    ]
        else:
            if not self.FLAGS.get('images_only', False): 
                img_all = [
                    cv2.putText(color.astype(np.uint8), 'rgb', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(normal.astype(np.uint8), 'nrm', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2),
                    cv2.putText(surface_nrm.astype(np.uint8), 'surface nrm', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(albedo.astype(np.uint8), 'alb', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(xyz.astype(np.uint8), 'xyz', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(surface_xyz.astype(np.uint8), 'surface xyz', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(space_sdf_in.astype(np.uint8), 'space sdf>0', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(space_occ_in.astype(np.uint8), 'space occu>0.9', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(depth.astype(np.uint8), 'depth (normalized)', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(distance.astype(np.uint8), 'distance (normalized)', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    ]
            else:
                img_all = [
                    cv2.putText(color.astype(np.uint8), 'rgb', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(normal.astype(np.uint8), 'nrm', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2),
                    cv2.putText(albedo.astype(np.uint8), 'alb', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(xyz.astype(np.uint8), 'xyz', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(depth.astype(np.uint8), 'depth (normalized)', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    cv2.putText(distance.astype(np.uint8), 'distance (normalized)', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2), 
                    ]
        
        if 'shaded' in sample:
            img_all.insert(1, cv2.putText(shaded.astype(np.uint8), 'shaded', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2))
        
        return np.concatenate(img_all, axis=0)
    
    @staticmethod
    def collate_fn(batch):
        collated_batch = {}
        for key in batch[0].keys():
            if key in ["manifold_mesh_verts", "manifold_mesh_faces"]:
                collated_batch[key] = [item[key] for item in batch]
            else:
                collated_batch[key] = torch.utils.data.default_collate([item[key] for item in batch])
        return collated_batch
        
if __name__ == "__main__":
    
    # senbo's 120 views rendering
    # 5 x 24 views systems, where the 0-th system is aligned to objects coordinates and next 4 systmes are randomly rotated.
    # for each system the first 12 views are captured at azimuth (0,90,180,270, 0,90,180,270 0,90,180,270) and 
    # elevations (0,0,0,0, 20,20,20,20, -20,-20,-20,-20). the next 12 views are augmented 0-th view, each at random elevation 
    # in [-30, 45], and 15-17 views are augmented for fov, 18-20 for lighting and 12-23 for fov and lighting.
    
    import_paths = {
        "cfs": 
        [
            "/aigc_cfs_11/Asset/active_list/3d_diffusion/new_manifold/results/part1.json",
            "/aigc_cfs_11/Asset/active_list/3d_diffusion/new_manifold/results/part2.json",
            "/aigc_cfs_11/Asset/active_list/3d_diffusion/new_manifold/results/part3_and_part4.json",
            "/aigc_cfs_11/Asset/active_list/3d_diffusion/new_manifold/results/part5_30k.json",
            "/aigc_cfs_11/Asset/active_list/3d_diffusion/new_manifold/results/part6.json",
            "/aigc_cfs_11/Asset/active_list/3d_diffusion/new_manifold/results/part7_and_part8.json",
            "/aigc_cfs_11/Asset/active_list/3d_diffusion/new_manifold/results/part9.json",
            "/aigc_cfs_11/Asset/active_list/3d_diffusion/new_manifold/results/part10.json",
            "/aigc_cfs_11/Asset/active_list/3d_diffusion/new_manifold/results/avatar_color_only.json",
            "/aigc_cfs_11/Asset/active_list/3d_diffusion/new_manifold/results/part12.json",
            "/aigc_cfs_11/Asset/active_list/3d_diffusion/new_manifold/results/part13.json",
            "/aigc_cfs_11/Asset/active_list/3d_diffusion/new_manifold/results/part14.json",
            "/aigc_cfs_11/Asset/active_list/3d_diffusion/new_manifold/results/part15.json"
        ],
        # "910b":
        # [
        #     # "/aigc_cfs_11/Asset/active_list/3d_diffusion/objaverse/part1_56k_with_normal_with_point_cloud.json",
        #     # "/aigc_cfs_11/Asset/active_list/3d_diffusion/objaverse/part2_58k_with_normal_with_point_cloud.json",
        #     # "/aigc_cfs_11/Asset/active_list/3d_diffusion/objaverse/part3_55k_with_normal_with_point_cloud.json",
        #     # "/aigc_cfs_11/Asset/active_list/3d_diffusion/objaverse/part4_58k_with_normal_with_point_cloud.json",
        #     # "/aigc_cfs_11/Asset/active_list/3d_diffusion/avatar/avatar_52k_with_normal_with_point_cloud.json",
        #     "/aigc_cfs_11/Asset/active_list/3d_diffusion/avatar/new_avatar_28k_with_normal_with_point_cloud.json"
        # ],
    }
    
    
    expected_camera_embeddings = [
        # (azimuths, elevations)
        (0, 0),
        (90, 0),
        (180, 0),
        (270, 0),
        
        (0, 20),
        (90, 20),
        (180, 20),
        (270, 20),
        
        (0, -20),
        (90, -20),
        (180, -20),
        (270, -20),
        
        (0,0),
        (0,0),
        (0,0),
        (0,0),
        (0,0),
        (0,0),
        (0,0),
        (0,0),
        (0,0),
        (0,0),
        (0,0),
        (0,0),
        
    ]
    
    # # 1. most basic set up, 0-th system and w/o reference view augmentation
    # config_idx = 1
    # systems = [0] # out of 5
    # ref_cameras = [0] # out of 24
    # src_cameras = [1,2,3] # out of 24
    
    # 2. with all five systems and w/o reference view augmentation
    config_idx = 2
    systems = [0, 1, 2, 3, 4] # out of 5
    ref_cameras = [0] # out of 24
    src_cameras = [1,2,3] # out of 24
    
    cameras_config = [
        {"ref": [(24*system + camera) for camera in ref_cameras], "src": [(24*system+camera) for camera in src_cameras]} for system in systems
    ]
    camera_embeddings = {
        "ref": [expected_camera_embeddings[camera][1] for camera in ref_cameras],
        "src": np.deg2rad(np.array([expected_camera_embeddings[src_camera] for src_camera in src_cameras]).reshape(-1,2)).tolist()
    }
    output_path = f"/aigc_cfs_2/zacheng/new_datajsons/24Sep20_objaverse_{config_idx:02}.json"
    
    filter_json = "/aigc_cfs_2/zacheng/new_datajsons/filtered_data_20240919.json"
    excludes = []
    with open(filter_json) as f:
        data = json.load(f)['data']
        for dset in data:
            for obj in data[dset]:
                filter = data[dset][obj]['hunyuan_filter']
                if sum(filter['building?']) >= 4 or sum(filter['hole, low_quality?']) >= 4:
                    excludes.append((dset, obj, {}))
        
    
    MVLRMDataset.make_data_json(import_paths, cameras_config, camera_embeddings, "ALL", excludes, output_path)
    print(f"json writte to {output_path}")
       
