import torch
import numpy as np
# import kiui
import nvdiffrast.torch as dr
import torch.nn.functional as F
# from kiui.mesh import Mesh
import argparse
import os
from glob import glob
import cv2
import itertools
from ipdb import set_trace as st

def np2th(*tensors, device="cuda"):
    return [torch.from_numpy(t).to(dtype=(torch.float32 if t.dtype.kind in {'f'} else torch.int32),device=device) for t in tensors]

def th2np(*tensors):
    return [t.detach().cpu().numpy() for t in tensors]

def render_mesh_verts_tex(img_size, ndc_mat, c2w_mat, v, f, vf, ssaa=1, bg_color=(1,1,1)):
    
    '''
    th 2 np
    - img_size: int
    - ndc_mat: [(nviews), 4, 4]
    - w2c_mat: [nviews, 4, 4]
    - v: [nv, 3]
    - f: [nf, 3]
    - vf: [nv, C] or [nviews, nv, C]
    - ssaa: int
    returns: [N, img_size, img_size,C]
    '''
    
    N = c2w_mat.shape[0]
    
    v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(c2w_mat).transpose(-1,-2)).float()
    v_clip = v_cam @ ndc_mat.transpose(-1,-2)

    rast, rast_db = dr.rasterize(dr.RasterizeCudaContext(device=ndc_mat.device), v_clip.contiguous(), f.contiguous(), (img_size*ssaa, img_size*ssaa))

    alpha = (rast[..., 3:] > 0).float()
    
    color, texc_db = dr.interpolate(vf.reshape(-1,*vf.shape[-2:]).expand((N,)+vf.shape[-2:]).contiguous(), rast, f.contiguous(), rast_db=rast_db, diff_attrs='all')

    color = dr.antialias(color, rast, v_clip, f.contiguous())
    
    # color, alpha = th2np(color, alpha)
    # color = alpha * color + (1 - alpha) * np.array(bg_color)
    
    # return color.reshape(-1, img_size, ssaa, img_size, ssaa, vf.shape[-1]).mean(4).mean(2)
    return color, alpha

def render_mesh(img_size, ndc_mat, c2w_mat, v, f, vt, ft, tex, ssaa=1, bg_color=(1,1,1), max_mip_level=3):
    '''
    th 2 np
    - img_size: int
    - ndc_mat: [(nviews), 4, 4]
    - w2c_mat: [nviews, 4, 4]
    - v: [nv, 3]
    - f: [nf, 3]
    - vt: [nvt, 2]
    - ft: [nf, 3]
    - tex: [H,W,C]
    - ssaa: int
    returns: [N, img_size, img_size,C]
    '''
    
    N = c2w_mat.shape[0]
    
    v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(c2w_mat).transpose(-1,-2)).float()
    v_clip = v_cam @ ndc_mat.transpose(-1,-2)

    rast, rast_db = dr.rasterize(dr.RasterizeCudaContext(device=ndc_mat.device), v_clip.contiguous(), f.contiguous(), (img_size*ssaa, img_size*ssaa))

    alpha = (rast[..., 3:] > 0).float()
    
    texc, texc_db = dr.interpolate(vt.unsqueeze(0).expand((N,)+vt.shape).contiguous(), rast, ft.contiguous(), rast_db=rast_db, diff_attrs='all')
    color = dr.texture(tex.unsqueeze(0).expand((N,)+tex.shape).contiguous(), texc, uv_da=texc_db, filter_mode='auto', max_mip_level=max_mip_level) # [N, H, W, 3]
    # color = dr.texture(tex.unsqueeze(0).expand((N,)+tex.shape).contiguous(), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear', max_mip_level=1) # [N, H, W, 3]
    if max_mip_level > 0:
        color = dr.antialias(color, rast, v_clip, f.contiguous())
    return color, alpha
    # color, alpha = th2np(color, alpha)
    # color = alpha * color + (1 - alpha) * np.array(bg_color)
    
    # return color.reshape(-1, img_size, ssaa, img_size, ssaa, tex.shape[-1]).mean(4).mean(2)

def make_cameras(azimuths, elevations, radii):
    '''
    np to th
    '''
    
    n_views = len(azimuths)
    azimuths = np.radians(azimuths)
    elevations = np.radians(elevations)
    
    up = np.array([0,0,1])
    cam_forward = -np.stack((np.cos(elevations)*np.cos(azimuths), np.cos(elevations)*np.sin(azimuths), np.sin(elevations)), axis=-1) # [n_views, 3]
    cam_right = np.cross(cam_forward, up)
    cam_down = np.cross(cam_forward, cam_right)
    
    cam_c = -cam_forward * radii.reshape(-1,1)
    
    c2w = np.zeros((n_views, 4, 4))
    c2w[...,-1,-1] = 1
    c2w[...,:3,0] = cam_right / np.linalg.norm(cam_right, axis=-1, ord=2, keepdims=True)
    c2w[...,:3,1] = cam_down / np.linalg.norm(cam_down, axis=-1, ord=2, keepdims=True)
    c2w[...,:3,2] = cam_forward / np.linalg.norm(cam_forward, axis=-1, ord=2, keepdims=True)
    c2w[...,:3,3] = cam_c
    
    c2w, = np2th(c2w, device="cuda")
    return c2w

def make_ndc(zoom=None, near=0.1, far=100.0, camera_type="pinhole", intrinsics=None, w=None, h=None):
    '''
    np 2 th
    '''
    
    P = np.zeros((4, 4))
    
    if intrinsics is not None:
        fx = intrinsics[0,0]
        fy = intrinsics[1,1]
        cx = intrinsics[0,2]
        cy = intrinsics[1,2]
        if w is None:
            w = cx * 2
        if h is None:
            h = cy * 2
        P[0,0] = 2 * fx / w
        P[1, 1] = 2 * fy / h
        P[0, 2] = (2 * cx / w) - 1
        P[1, 2] = (2 * cy / h) - 1 
        P[2, 2] = (far + near) / (far - near)
        P[2, 3] = -2 * far * near / (far - near)   
        P[3, 2] = 1 

        return P

    P[0, 0] = zoom
    P[1, 1] = zoom
    
    if camera_type == "pinhole":
        P[2, 2] = (far + near) / (far - near)
        P[2, 3] = -2 * far * near / (far - near)
        P[3, 2] = 1
        
    elif camera_type == "ortho":
        P[2, 2] = 2 / (far - near)
        P[2, 3] = -(far + near) / (far - near)
        P[3, 3] = 1
    
    # P, = np2th(P, device="cuda")
    
    return P

# @torch.no_grad
# def render(vert, faces, verttex, c2w, intrinsics=None, width=None, height=None, camera_type="pinhole", resolution=256, azimith_step=5, distance=5, zoom=1.0, bg_color=(0,1,0), ssaa=1, front_dir="+z"):
    
#     # if azimith_step == 0:
#     #     azimuths = np.array([0])
#     # else:
#     #     azimuths = np.arange(-180, 180+azimith_step, azimith_step)
#     # elevations = azimuths * 0
#     # radii = distance + elevations
    
#     # if camera_type == "pinhole":
#     #     distance = min(distance, 1.4)
#     #     zoom *= (distance**2-1)**(0.5)
    
#     ndc = make_ndc(intrinsics=intrinsics, w=width, h=height)

#     # c2w = make_cameras(azimuths, elevations, radii)
#     # mesh = Mesh.load(mesh_file, resize=True, front_dir=front_dir, bound=0.99)
    
#     # yup2zup = torch.Tensor([
#     #         [1,0,0,0],
#     #         [0,0,-1,0],
#     #         [0,1,0,0],
#     #         [0,0,0,1]
#     #     ], device=mesh.v.device, dtype=mesh.v.dtype)
#     # mesh.v = mesh.v @ yup2zup.transpose(-1,-2)
    
#     # images = render_mesh(resolution, ndc, c2w, vert, faces, vertuv, faceuv, texturemap, ssaa=1, bg_color=bg_color)
    
#     images = render_mesh_verts_tex(resolution, ndc, c2w, vert, faces, verttex, ssaa=1, bg_color=bg_color)


#     return images

def render_normal_xyz(mesh_file, camera_type, resolution, azimuths, elevations, radius, zoom, bg_color=(1,1,1), ssaa=1, front_dir="+z"):
    
    radii = azimuths * 0 + radius
    
    ndc = make_ndc(zoom, camera_type=camera_type)
    c2w = make_cameras(azimuths, elevations, radii)
    
    mesh = Mesh.load(mesh_file, front_dir=front_dir)
    mesh.auto_normal()
    normal_yup = mesh.vn
    
    yup2zup = torch.Tensor([
            [1,0,0,0],
            [0,0,-1,0],
            [0,1,0,0],
            [0,0,0,1]
        ], device=mesh.v.device, dtype=mesh.v.dtype)
    xyz_zup = mesh.v @ yup2zup.transpose(-1,-2)
    
    normal_images = render_mesh_verts_tex(resolution, ndc, c2w, xyz_zup, mesh.f, normal_yup, ssaa, bg_color)
    xyz_images = render_mesh_verts_tex(resolution, ndc, c2w, xyz_zup, mesh.f, xyz_zup, ssaa, bg_color)
    
    return normal_images, xyz_images
    

def write_video(frames, video_path, duration):
    F, H, W, C = frames.shape

    fps = F / duration
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
    
    for i in range(F):
        frame = frames[i]
        video_writer.write(frame[...,::-1])
        
    video_writer.release()

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, default=".", help="path to obj or glb file, or directory that contains meshes")
    parser.add_argument("--camera_type", choices=["pinhole", "ortho"], default="ortho", help="camera type")
    parser.add_argument("--resolution", type=int, default=512, help="video resolution")
    parser.add_argument("--azimith_step", type=int, default=10, help="difference between azimuth angle in consecutive frames, in degrees")
    parser.add_argument("--distance", type=float, default=5.0, help="viewing distance relative to object size, only applicable to pinhole camera")
    parser.add_argument("--zoom", type=float, default=0.8, help="zoom-in factor, defaults to 1.0")
    parser.add_argument("--bg_color", type=float, nargs=3, default=(0,1,0), help="RGB background color, from 0 to 1.")
    parser.add_argument("--ssaa", type=int, default=1, help="SSAA factor")
    parser.add_argument("--duration", type=float, default=2.0, help="video duration in seconds.")
    parser.add_argument("--front_dir", type=str, default="+y1", help="defines input mesh rotation: e.g. +z means front is +z direction, -y3 means -y is front and mesh is further rotated 3x90 degrees around that direction")
    parser.add_argument("--mode", choices=["video", "front"], default="video", help="outputs video, or enumerate all possible front dirs and render front images only (to help figure out how to set front_dir).")
    
    args = parser.parse_args()
    
    assert os.path.exists(args.mesh), "mesh does not exist"
    
    if os.path.isdir(args.mesh):
        meshes = sorted(glob(os.path.join(args.mesh, "*.obj")) + glob(os.path.join(args.mesh, "*.glb")))
    else:
        meshes = [args.mesh]
    
    for mesh in meshes:
        mesh_name = os.path.splitext(os.path.basename(mesh))[0]
        
        if args.mode == "front":
            for front_dir in itertools.product(['+', '-'], ['x','y','z'], ['','1','2','3']):
                front_dir = ''.join(front_dir)
                output_img = os.path.join(os.path.dirname(mesh), f"{mesh_name}.{front_dir}.png")
                front = render(mesh, args.camera_type, args.resolution, 0, args.distance, args.zoom, args.bg_color, args.ssaa, front_dir)[0]
                cv2.imwrite(output_img, np.clip(front*255, 0, 255).astype(np.uint8)[...,::-1])
        elif args.mode == "video":
            output_video = os.path.join(os.path.dirname(mesh), f"{mesh_name}.mp4")
            frames = render(mesh, args.camera_type, args.resolution, args.azimith_step, args.distance, args.zoom, args.bg_color, args.ssaa, args.front_dir)
            frames = np.clip(frames*255, 0, 255).astype(np.uint8)
            write_video(frames, output_video, args.duration)
        
    
    
    
    
    
    
    
    

