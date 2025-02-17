import numpy as np
import torch
import imageio
from collections import OrderedDict, abc

def apply_to_tensors(nested, func):
    if isinstance(nested, torch.Tensor):
        return func(nested)
    elif isinstance(nested, (dict, OrderedDict)):
        # Use the constructor of the specific dict type (OrderedDict or dict)
        return type(nested)((key, apply_to_tensors(value, func)) for key, value in nested.items())
    elif isinstance(nested, tuple) and hasattr(nested, '_fields'): # named tuple
        as_dict = nested._asdict()
        processed_dict = {key: apply_to_tensors(value, func) for key, value in as_dict.items()}
        return type(nested)(**processed_dict)
    elif isinstance(nested, (list, tuple)):
        return type(nested)([apply_to_tensors(item, func) for item in nested])
    elif isinstance(nested, abc.Iterable):
        return type(nested)([apply_to_tensors(item, func) for item in nested])
    else:
        return nested

def prepare_data_multiview_from_fields(fields):
    '''Set up the dictionary to return
    '''
    data = {}
    for field in fields:
        data[field] = []

    return data


def get_intrinsic_focal(focal, cx, cy):
    """Generate intrinsic matrix`
    
    Args:
        focal: focal length
        cx: principle x
        cy: principle y
    """
    skew = 0.0
    ret = torch.tensor([
        [focal, skew, cx],
        [0, focal, cy],
        [0, 0, 1],
    ],
    dtype=torch.float32).reshape(1, 3, 3)
    return ret


def get_intrinsic(fov, resolution, fov_in_degree=True):
    """Generate intrinsic matrix`
    
    Args:
        fov: the fov of the camera
        img_size: the squar border length of the image
        fov_in_degree: is `fov` in degree or radian
    """
    cx = (resolution[0] - 1) / 2
    cy = (resolution[1] - 1) / 2
    if fov_in_degree:
        fov = np.deg2rad(fov)
    focal = (resolution[0] - 1) / (2 * np.tan(fov / 2))
    return get_intrinsic_focal(focal, cx, cy)


def intrinsic(fovy, width, height, device=None):
    focal = height / 2 / np.tan(fovy / 2)
    return torch.tensor([[
            focal, 0, (width - 1) / 2, 0], 
            [0, focal, (height - 1) / 2, 0],
            [0, 0, 1, 0]],
        dtype=torch.float32,
        device=device)


def cartesian_to_spherical(xyz):
    '''T to theta, phi, radius
    
    Args:
        xyz: T in RT matrix
        
    Returns:
        Tensor, each line is [theta, azimuth, radius]
    '''

    # ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    z = torch.sqrt(xy + xyz[:, 2]**2)
    theta = torch.arctan2(torch.sqrt(xy), xyz[:, 2])
    azimuth = torch.arctan2(xyz[:, 1], xyz[:, 0])
    return torch.stack([theta, azimuth, z]).T.unsqueeze(dim=0)


def focal_length_to_fovy(focal_length, sensor_height):
    return 2 * np.arctan(0.5 * sensor_height / focal_length)


def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f <= 0.04045, f / 12.92,
        torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))

def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f <= 0.0031308, 
        f * 12.92, 
        1.055 * torch.pow(f, 1/2.4) - 0.055
    )


def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat(
        (_srgb_to_rgb(f[..., 0:3]),
         f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _srgb_to_rgb(f)
    assert out.shape[:3] == f.shape[:3] 
    return out

def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat(
        (_rgb_to_srgb(f[..., 0:3]),
         f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _rgb_to_srgb(f)
    assert out.shape[:3] == f.shape[:3] 
    return out

# Reworked so this matches gluPerspective / glm::perspective, using fovy
def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)
    return torch.tensor([[1 / (y * aspect), 0, 0, 0], [0, 1 / -y, 0, 0],
                         [0, 0, -(f + n) / (f - n), -(2 * f * n) /
                          (f - n)], [0, 0, -1, 0]],
                        dtype=torch.float32,
                        device=device)


def translate(x, y, z, device=None):
    return torch.tensor(
        [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device)


def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor(
        [[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device)


def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)
    return torch.tensor([[1 / (y * aspect), 0, 0, 0], [0, 1 / -y, 0, 0],
                         [0, 0, -(f + n) / (f - n), -(2 * f * n) /
                          (f - n)], [0, 0, -1, 0]],
                        dtype=torch.float32,
                        device=device)


def load_image_raw(fn) -> np.ndarray:
    return imageio.imread(fn)

def normalize_reference_cam(extrinsic, ref_index, cam2world=True, random_rot=False):
    '''
    rotate world coordinate system such that the reference camera is on 
    the positive half of world x-axis and looks at (0,0,0)

    this function does not scale world units

    if random_rot is set, a random rotation is applied about the world x-axis;
    otherwise the new world z-axis will always points upwards in reference view

    inputs: 
        - extrinsic: a torch tensor of shape [..., n_views, 4, 4]
        - ref_index: integer in [0, n_views) indicating which view is reference
        - cam2world: a boolean, if set to false, extrinsics are considered world2cam matrices
        - random_rot: whether to apply random rotation about world x-axis
    
    returns:
        - extrinsic: transformed extrinsics matrices of shape [..., n_views, 4, 4]
        - obj2world: transformation from old world frame (object) to new world frame of shape [..., 4, 4]
    '''

    # print(f'normalize_reference_cam ref_index={ref_index} cam2world={cam2world} random_rot={random_rot}')
    ref_c2w = torch.zeros_like(extrinsic[...,:1,:,:])

    # rotation
    ref_c2w[...,0,2] = -1
    ref_c2w[...,1,0] = 1
    ref_c2w[...,2,1] = -1

    if random_rot:

        assert False, "this feature should not be used in training, are you sure you know what you're doing?"
        angle = torch.rand(extrinsic.shape[:-3], device=extrinsic.device, dtype=extrinsic.dtype) * 1e3
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        random_R = torch.zeros_like(extrinsic[...,:1,:3,:3])
        random_R[...,0,0] = 1
        random_R[...,0,1,1] = cos_angle
        random_R[...,0,1,2] = -sin_angle
        random_R[...,0,2,1] = sin_angle
        random_R[...,0,2,2] = cos_angle

        ref_c2w[...,:3,:3] = random_R @ ref_c2w[...,:3,:3]


    # translation
    ref_c2w[...,0,0,3] = torch.linalg.vector_norm(extrinsic[..., ref_index, :3, -1], dim=-1)

    ref_c2w[...,3,3] = 1

    if cam2world:
        obj2world = ref_c2w @ torch.linalg.inv(extrinsic[..., ref_index:ref_index+1,:,:])
        return obj2world @ extrinsic, obj2world.squeeze(dim=-3)
    else:
        obj2world = ref_c2w @ extrinsic[..., ref_index:ref_index+1,:,:]
        return extrinsic @ torch.linalg.inv(obj2world), obj2world.squeeze(dim=-3)

def safe_normalize(x, eps=1e-20):
    return (x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))).float()

def generate_pose(r, theta, phi, cam2world=True, debug=False):
    '''
    generate cam2world poses in Front-Rright-Up world system
    
    Arguments:
        theta: zenith
        phi: angle to x-axis
    '''

    # theta = torch.Tensor([(theta - 90) * np.pi / 180])
    # phi = torch.Tensor([phi * np.pi / 180])

    theta = torch.Tensor([theta])
    phi = torch.Tensor([phi])
    dx = torch.sin(theta) * torch.cos(phi)
    dy = torch.sin(theta) * torch.sin(phi)
    dz = torch.cos(theta)
    
    # print(theta, phi)
    # ipdb.set_trace()

    T = torch.stack((dx*r,dy*r,dz*r), dim=-1) # [N,3]
    targets = torch.zeros_like(T)

    v_forward = safe_normalize(targets - T)
    up = torch.tensor([0,0,1], dtype=torch.float32)
    if len(up.shape) == 1: 
        up = up.unsqueeze(0)
    v_right = torch.cross(v_forward, up, dim=-1) # [N, 3]
    v_down = torch.cross(v_forward, v_right, dim=-1) # [N, 3]

    R = torch.stack((v_right, v_down, v_forward), dim=-1)
    R = torch.nn.functional.normalize(R, p=2, dim=-2, eps=1e-10) # [N,3,3]
    
    ret = torch.eye(4) * 1.0
    ret[:3,:3] = R
    ret[:3,3] = T

    if cam2world == False:
        # Return world2cam pose
        ret = torch.linalg.inv(ret)

    return ret

def get_spherical(RT):
    R, T = RT[..., :3, :3], RT[0, :, :-1, -1]
    return cartesian_to_spherical(T.cpu()).to(T.device) # this needs to stay on CPU side otherwise cuda could trigger error due to old driver of taiji


def depth_to_world(depth_map, intrinsic_matrix, cam2world=None):
    '''
    intrinsic_matrix is 3x4, can be either pinhole or orthographic
    '''
    N, _, H, W = depth_map.shape

    y, x = torch.meshgrid(torch.arange(0, H, dtype=torch.float32, device=depth_map.device),
                          torch.arange(0, W, dtype=torch.float32, device=depth_map.device), indexing='ij')
    z = torch.ones_like(x)

    # Unproject to camera coordinates
    pixel_coords = torch.stack((x, y, z), dim=0)
    pixel_coords = pixel_coords.unsqueeze(0).repeat(N, 1, 1, 1)
    intrinsic_inv = torch.linalg.pinv(intrinsic_matrix) # [N,4,3]
    cam_coords = torch.einsum('nci,nihw->nchw', intrinsic_inv, pixel_coords) # [N,4,H,W]
    d = depth_map + 1e-4
    xt = cam_coords[:,0:1]
    yt = cam_coords[:,1:2]
    nt = cam_coords[:,2:3] + cam_coords[:,3:4] * d
    cam_coords_homo = torch.cat((xt*d, yt*d, nt*d, nt), dim=1)
    
    # Transform to world coordinates
    if cam2world is not None:
        world_coords_homo = torch.einsum('nxc,nchw->nxhw', cam2world, cam_coords_homo)
    else:
        world_coords_homo = cam_coords_homo
    world_coords = world_coords_homo[:,:3] / world_coords_homo[:,3:]
    return world_coords


def depth_to_distance(depth_map, intrinsic_matrix):
    '''
    intrinsic_matrix is 3x4, can be either pinhole or orthographic
    returns a ray image of shape 
    '''
    N, _, H, W = depth_map.shape

    y, x = torch.meshgrid(torch.arange(0, H, dtype=torch.float32, device=depth_map.device),
                          torch.arange(0, W, dtype=torch.float32, device=depth_map.device), indexing='ij')
    z = torch.ones_like(x)

    # Unproject to camera coordinates
    pixel_coords = torch.stack((x, y, z), dim=0)
    pixel_coords = pixel_coords.unsqueeze(0).repeat(N, 1, 1, 1)
    intrinsic_inv = torch.linalg.pinv(intrinsic_matrix) # [N,4,3]
    cam_coords = torch.einsum('nci,nihw->nchw', intrinsic_inv, pixel_coords) # [N,4,H,W]
    d = depth_map + 1e-4
    xt = cam_coords[:,0:1]
    yt = cam_coords[:,1:2]
    nt = cam_coords[:,2:3] + cam_coords[:,3:4] * d
    cam_coords_homo = torch.cat((xt*d, yt*d, nt*d, nt), dim=1)
    
    d = 1e-4
    xt = cam_coords[:,0:1]
    yt = cam_coords[:,1:2]
    nt = cam_coords[:,2:3] + cam_coords[:,3:4] * d
    cam_coords_ori = torch.cat((xt*d, yt*d, nt*d, nt), dim=1)
    
    cam_coords_homo = cam_coords_homo[:,:3] / cam_coords_homo[:,3:]
    cam_coords_ori = cam_coords_ori[:,:3] / cam_coords_ori[:,3:]
    
    distantce = torch.linalg.vector_norm(cam_coords_homo - cam_coords_ori, dim=1, keepdim=True, ord=2)
    
    return distantce

def image_16bitc1_to_8bitc2(image_16bit, channel_dim=-1):
    assert image_16bit.shape[channel_dim] == 1 and image_16bit.dtype == np.uint16

    high_bits = np.right_shift(image_16bit, 8).astype(np.uint8)  # Extracting the higher 8 bits
    low_bits = np.bitwise_and(image_16bit, 0xFF).astype(np.uint8)  

    return np.concatenate([high_bits, low_bits], axis=channel_dim)


def image_8bitc2_to_16bitc1(image_8bit, channel_dim=-1):

    high_bits, low_bits = np.split(image_8bit, [1], axis=channel_dim)
    image_16bit = (high_bits.astype(np.uint16) << 8) | low_bits.astype(np.uint16)

    return image_16bit

def exclude_from(dic, excluded_dic):
    ret_dict = {}
    
    if not isinstance(dic, dict):
        assert excluded_dic is None or len(excluded_dic) == 0, "excluded_dic is expected to be empty"
        return dic
    
    ## excluded_dic can be a collection or dict 
    if isinstance(excluded_dic, dict):
        for key in dic:
            if key in excluded_dic:
                if excluded_dic[key] == "ALL":
                    continue
                else:
                    ret_dict[key] = exclude_from(dic[key], excluded_dic[key])
            else:
               ret_dict[key] = dic[key]
               
    else:
        for key in dic:
            if key in excluded_dic:
                continue
            else:
                ret_dict[key] = dic[key]
            
    return ret_dict
            