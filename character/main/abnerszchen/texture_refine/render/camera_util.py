import os
import torch
import torch.nn.functional as F
import numpy as np


def pad_camera_extrinsics_4x4(extrinsics):
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    extrinsics = torch.cat([extrinsics, padding], dim=-2)
    return extrinsics


def center_looking_at_camera_pose(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    """
    Create OpenGL camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4) or (3, 4)
    """
    # by default, looking at the origin and world up is z-axis
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # OpenGL camera: z-backward, x-right, y-up
    z_axis = camera_position - look_at
    z_axis = F.normalize(z_axis, dim=-1).float()
    x_axis = torch.linalg.cross(up_world, z_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1).float()
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    y_axis = F.normalize(y_axis, dim=-1).float()

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics


def spherical_camera_pose(azimuths: np.ndarray, elevations: np.ndarray, radius=2.5, up_world=None):
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    xs = radius * np.cos(elevations) * np.cos(azimuths)
    ys = radius * np.cos(elevations) * np.sin(azimuths)
    zs = radius * np.sin(elevations)

    cam_locations = np.stack([xs, ys, zs], axis=-1)
    cam_locations = torch.from_numpy(cam_locations).float()

    c2ws = center_looking_at_camera_pose(cam_locations, up_world=up_world)
    return c2ws


def get_circular_camera_poses(M=120, radius=2.5, elevation=30.0):
    # M: number of circular views
    # radius: camera dist to center
    # elevation: elevation degrees of the camera
    # return: (M, 4, 4)
    assert M > 0 and radius > 0

    elevation = np.deg2rad(elevation)

    camera_positions = []
    for i in range(M):
        azimuth = 2 * np.pi * i / M
        x = radius * np.cos(elevation) * np.cos(azimuth)
        y = radius * np.cos(elevation) * np.sin(azimuth)
        z = radius * np.sin(elevation)
        camera_positions.append([x, y, z])
    camera_positions = np.array(camera_positions)
    camera_positions = torch.from_numpy(camera_positions).float()
    extrinsics = center_looking_at_camera_pose(camera_positions)
    return extrinsics


def FOV_to_intrinsics(fov, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """
    focal_length = 0.5 / np.tan(np.deg2rad(fov) * 0.5)
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics

# TODO(csz) need checking
def make_pose_npy_as_instant_from_cameras(cameras, pose_npy):
    out_poses = []
    
    for fov, Rwc, T in zip(cameras.fov, cameras.R, cameras.T):
        fov_rad = np.deg2rad(fov.cpu().numpy())
        fx = 0.5 / np.tan(0.5 * fov_rad)
        fy = fx
        cx, cy = 0.5, 0.5
        intrinsic = np.array([fx, fy, cx, cy])
        
        
        tcw = T
        twc = -torch.matmul(Rwc, T)
        twc = twc.view(3, 1)
        Twc = torch.cat([Rwc, twc], dim=-1)
        c2w = torch.eye(4, device=Twc.device)
        c2w[:3, :] = Twc
    
        # T_wins_wpy3d = torch.tensor([[0, 1, 0, 0],
        #                             [0, 0, 1, 0],
        #                             [1, 0, 0, 0],
        #                             [0, 0, 0, 1]]).to(Twc.device).float()
        # c2w = torch.matmul(T_wins_wpy3d, c2w)
                
        extrinsic = c2w[:3, :].view(-1).cpu().numpy()
        out_pose = np.concatenate([extrinsic, intrinsic]).reshape(1, -1)
        out_poses.append(out_pose)
    pose_np = np.concatenate(out_poses, axis=0)
    print('pose_np', pose_np.shape)
    np.save(pose_npy, pose_np)
    return pose_np

def make_pose_npy_as_instant(elevations, azimuths, dists, fov, pose_npy):
    """instant use z-up world and OpenGL coord with y-up cam

    Args:
        elevations: torch N
        azimuths: torch N
        dists: torch N
        fov: float
        pose_npy: _description_
    """
    elevations, azimuths, radius = [data.cpu().numpy() for data in [elevations, azimuths, dists]]
    c2ws = spherical_camera_pose(azimuths, elevations, radius, 
                                 up_world = torch.tensor([0, 0, 1], dtype=torch.float32), # instant mesh use Z-up
                                 )
    # T world py3d with z up to world opengl
    T_wins_wpy3d = torch.tensor([[0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 1]]).to(c2ws.device).float()    
    T_wins_wpy3d = T_wins_wpy3d.unsqueeze(0).repeat(c2ws.shape[0], 1, 1)
    c2ws = torch.bmm(T_wins_wpy3d, c2ws)
    print('c2ws', c2ws)
    
    c2ws = c2ws.float().flatten(-2)

    Ks = FOV_to_intrinsics(fov).unsqueeze(0).repeat(elevations.shape[0], 1, 1).float().flatten(-2)    
    extrinsics = c2ws[:, :12]
    intrinsics = torch.stack([Ks[:, 0], Ks[:, 4], Ks[:, 2], Ks[:, 5]], dim=-1)
    cameras = torch.cat([extrinsics, intrinsics], dim=-1)
    
    cameras_np = cameras.cpu().numpy()
    
    os.makedirs(os.path.dirname(pose_npy), exist_ok=True)
    np.save(pose_npy, cameras_np)
    print('cameras ', cameras_np.shape)

##### nvdiffrast
def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)
    return torch.tensor([[1/(y*aspect),    0,            0,              0], 
                         [           0, 1/-y,            0,              0], 
                         [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                         [           0,    0,           -1,              0]], dtype=torch.float32, device=device)



##### blender
def opencv_to_blender(T):
    """opencv format->opengl
        T: ndarray 4x4
       usecase: cam.matrix_world =  world_to_blender( np.array(cam.matrix_world))
    """
    origin = np.array(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
    return np.matmul(T, origin)  #T * origin