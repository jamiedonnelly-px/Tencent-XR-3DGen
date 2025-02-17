
import cupy as cp
import nvdiffrast
import nvdiffrast.torch as dr
import torch
import numpy as np
from pdb import set_trace as st
import cv2
import os

###### voronoi.py ######
""" 
Program to compute Voronoi diagram using JFA.

@author yisiox
@version September 2022
"""

from random import sample

# global variables
x_dim = 512
y_dim = 512
noSeeds = 1024

# diagram is represented as a 2d array where each element is
# x coord of source * y_dim + y coord of source
ping = cp.full((x_dim, y_dim), -1, dtype = int)
pong = None

def construct_mesh_laplacian(faces, diagonal_val=-1):
    '''
    L[i, j] =  diagonal_val , if i == j
    L[i, j] =  1 / deg(i)   , if (i, j) is an edge
    L[i, j] =    0          , otherwise
    '''
    n_verts = faces.max() + 1  
    device = faces.device

    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    edges = torch.cat([edges, edges[:, [1, 0]]], dim=0) 

    row, col = edges[:, 0], edges[:, 1]
    values = torch.ones(row.size(0), dtype=torch.float32, device=device)
    adjacency_matrix = torch.sparse_coo_tensor(torch.stack([row, col]), values, (n_verts, n_verts)) # [n_verts,n_verts]

    degrees = torch.sparse.sum(adjacency_matrix, dim=1).to_dense() #[n_verts]
    non_diagonal = torch.sparse_coo_tensor(torch.stack([row, col]), values / degrees[row], (n_verts, n_verts)) # [n_verts,n_verts]
    
    diagonal = torch.sparse_coo_tensor(torch.stack(
        [torch.arange(n_verts,dtype=torch.long, device=device), torch.arange(n_verts,dtype=torch.long, device=device)]), 
            torch.ones(n_verts, dtype=torch.float32, device=device)*diagonal_val, (n_verts, n_verts)) # [n_verts,n_verts]

    laplacian_matrix = non_diagonal + diagonal
    return laplacian_matrix

def laplace_solve(faces, verts_feat, verts_mask):
    '''
    - faces: [n_faces, 3]
    - verts_feat: [n_verts, channel]
    - verts_mask: [n_verts]
    '''
    
    V = verts_feat.shape[0]
    
    invalid_index = torch.ones_like(verts_feat[:, 0]).bool()    # [V]
    invalid_index[verts_mask] = False
    invalid_index = torch.arange(V).to(verts_feat.device)[invalid_index]
    
    L = construct_mesh_laplacian(faces, diagonal_val=0).coalesce()
    L = torch.sparse_coo_tensor(L.indices(), L.values(), (V,V)).coalesce()
    colored_count = torch.ones_like(verts_feat[:, 0])   # [V]
    colored_count[invalid_index] = 0
    L_invalid = torch.index_select(L, 0, invalid_index)    # sparse [IV, V]
    
    total_colored = colored_count.sum()
    coloring_round = 0
    colors = verts_feat.clone()
    colors[invalid_index] = 0
    stage = "uncolored"
    while stage == "uncolored" or coloring_round > 0:
        new_color = torch.matmul(L_invalid, colors * colored_count[:, None])    # [IV, 3]
        new_count = torch.matmul(L_invalid, colored_count)[:, None]             # [IV, 1]
        colors[invalid_index] = torch.where(new_count > 0, new_color / new_count, colors[invalid_index])
        colored_count[invalid_index] = (new_count[:, 0] > 0).float()
        
        new_total_colored = colored_count.sum()
        if new_total_colored > total_colored:
            total_colored = new_total_colored
            coloring_round += 1
        else:
            stage = "colored"
            coloring_round -= 1
        if coloring_round > 10000:
            print("coloring_round > 10000, break")
            break
    return colors
    

@torch.no_grad()
def nn_solve(query, key, value, k, hardness, chunksize=128):
    '''
    args:
        query: (Nq, Cq)
        key: (Nk, Cq)
        value: (Nk, Cv)
    returns:
        (Nq, Cv)
    '''
    ret = []
    for q in torch.split(query, chunksize):
        dist = torch.linalg.vector_norm(q.unsqueeze(1) - key, dim=-1) # [Nq, Nk]
        if k > 0:
            k_dist, k_idx = torch.topk(dist, k=k, dim=1, largest=False)
            weights = (-k_dist * hardness).softmax(dim=-1) # [Nq, k]
            ret.append(torch.einsum("qk,qkc->qc", weights, value[k_idx]))
        else:  
            weights = (-dist * hardness).softmax(dim=-1) # [Nq, Nk]
            ret.append(weights @ value)
        
    return torch.cat(ret)
    

def voronoi_solve(texture, mask):
    '''
        This is a warpper of the original cupy voronoi implementation
        The texture color where mask value is 1 will propagate to its
        neighbors.
        args:
            texture - A multi-channel tensor, (H, W, C)
            mask - A single-channel tensor, (H, W)
        return:
            texture - Propagated tensor
    '''
    h, w, c = texture.shape
    # hwc_texture = texture.permute(1,2,0)
    valid_pix_coord = torch.where(mask>0)

    indices = torch.arange(0, h*w).cuda().reshape(h, w)
    idx_map = -1 * torch.ones((h,w), dtype=torch.int64).cuda()
    idx_map[valid_pix_coord] = indices[valid_pix_coord]

    ping = cp.asarray(idx_map)
    pong = cp.copy(ping)
    ping = JFAVoronoiDiagram(ping, pong)

    voronoi_map = torch.as_tensor(ping, device="cuda")
    nc_voronoi_texture = torch.index_select(texture.reshape(h*w, c), 0, voronoi_map.reshape(h*w))
    voronoi_texture = nc_voronoi_texture.reshape(h, w, c)

    return voronoi_texture

def generateRandomSeeds(n):
    """
    Function to generate n random seeds.

    @param n The number of seeds to generate.
    """
    global ping, pong

    if n > x_dim * y_dim:
        print("Error: Number of seeds greater than number of pixels.")
        return

    # take sample of cartesian product
    coords = [(x, y) for x in range(x_dim) for y in range(y_dim)]
    seeds = sample(coords, n)
    for i in range(n):
        x, y = seeds[i]
        ping[x, y] = x * y_dim + y
    pong = cp.copy(ping)

displayKernel = cp.ElementwiseKernel(
        "int64 x",
        "int64 y",
        f"y = (x < 0) ? x : x % 103",
        "displayTransform")



voronoiKernel = cp.RawKernel(r"""
    extern "C" __global__
    void voronoiPass(const long long step, const long long xDim, const long long yDim, const long long *ping, long long *pong) {
        long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        long long stp = blockDim.x * gridDim.x;

        for (long long k = idx; k < xDim * yDim; k += stp) {
            long long dydx[] = {-1, 0, 1};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    long long dx = (step * dydx[i]) * yDim;
                    long long dy = step * dydx[j];
                    long long src = k + dx + dy;
                    if (src < 0 || src >= xDim * yDim) 
                        continue;
                    if (ping[src] == -1)
                        continue;
                    if (pong[k] == -1) {
                        pong[k] = ping[src];
                        continue;
                    }
                    long long x1 = k / yDim;
                    long long y1 = k % yDim;
                    long long x2 = pong[k] / yDim;
                    long long y2 = pong[k] % yDim;
                    long long x3 = ping[src] / yDim;
                    long long y3 = ping[src] % yDim;
                    long long curr_dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
                    long long jump_dist = (x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3);
                    if (jump_dist < curr_dist)
                        pong[k] = ping[src];
                }
            }
        }
    }
    """, "voronoiPass")


'''

    y and x is actually w and h? (according to experiment result)

'''
def JFAVoronoiDiagram(ping, pong):
    # global ping, pong
    # compute initial step size
    x_dim, y_dim = ping.shape
    step = max(x_dim, y_dim) // 2
    # initalise frame number and display original state
    frame = 0
    # iterate while step size is greater than 0
    while step:
        voronoiKernel((min(x_dim, 512),), (min(y_dim, 512),), (step, x_dim, y_dim, ping, pong))
        # Ajusted the upper bound of the kernel dimension from 1024 to 512 to avoid CUDA OUT OF RESOURCE problem
        ping, pong = pong, ping
        frame += 1
        step //= 2
        # displayDiagram(frame, ping)
    return ping

######## utils.py


import torchvision.transforms as transforms

def resize(images, size):
    '''
    Inputs: 
        - images: torch tensor of size [batch, height, width, channels]
        - size: tuple of (new_height, new_width), or int if new_height=new_width
    Returns:
        - resized_images: torch tensor of size  [batch, new_height, new_width, channels]
    '''
    
    # Check if size is a single integer or a tuple
    if isinstance(size, int):
        size = (size, size)  # Convert to a tuple (new_height, new_width)
    
    # Create a transform for resizing
    resize_transform = transforms.Resize(size)
    
    # Reshape the input tensor to [batch * height * width, channels]
    batch_size, height, width, channels = images.shape
    images_reshaped = images.permute(0, 3, 1, 2)  # Change to [batch, channels, height, width]
    
    # Resize images
    resized_images = torch.stack([resize_transform(image) for image in images_reshaped])
    
    # Change back to [batch, new_height, new_width, channels]
    resized_images = resized_images.permute(0, 2, 3, 1)  # Change back to [batch, new_height, new_width, channels]
    
    return resized_images

def cvt_torch(*tensors, device="cuda"):
    ret = []
    for t in tensors:
        if isinstance(t, np.ndarray):
            ret.append(torch.from_numpy(t).to(dtype={'f':torch.float32,'u':torch.int32,'i':torch.int32,'b': torch.bool}[t.dtype.kind],device=device).contiguous())
        elif isinstance(t, torch.Tensor):
            ret.append(t.to(device=device, dtype=(torch.float32 if t.is_floating_point() else torch.int32 if t.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8] else torch.bool if t.dtype == torch.bool else None)).contiguous())
        elif isinstance(t, (list, tuple)):
            ret.append(cvt_torch(np.array(t), device=device))
        elif isinstance(t, (int, float, np.generic)):
            ret.append(cvt_torch(np.array([t]), device=device))
        elif t is None:
            ret.append(None)
        else:
            raise TypeError("Input must be torch or numpy tensors")
    return ret if len(ret) > 1 else ret[0]

def cvt_numpy(*tensors):
    ret = []
    for t in tensors:
        if isinstance(t, np.ndarray):
            ret.append(t)
        elif isinstance(t, torch.Tensor):
            ret.append(t.detach().cpu().numpy())
        elif isinstance(t, (list, tuple)):
            ret.append(np.array(t))
        elif isinstance(t, (int, float, np.generic)):
            ret.append(np.array([t]))
        elif t is None:
            ret.append(None)
        else:
            raise TypeError("Input must be torch or numpy tensors")
    return ret if len(ret) > 1 else ret[0]

def erode_masks(*masks, iterations=1):
    '''
    masks: [H, W] or [H,W,C]
    '''
    kernel = np.ones((3, 3), np.uint8)
    ret = []
    for m in masks:
        if iterations > 0:
            m = cv2.erode(np.clip(m*255,0,255).astype(np.uint8), kernel, iterations=iterations) / 255.
        ret.append(m)
    return np.stack(ret, axis=0) if len(ret) > 1 else ret[0]

def dilate_masks(*masks, iterations=1):
    '''
    masks: [H, W] or [H,W,C]
    '''
    kernel = np.ones((3, 3), np.uint8)
    ret = []
    for m in masks:
        if iterations > 0:
            m = cv2.dilate(np.clip(m*255,0,255).astype(np.uint8), kernel, iterations=iterations) / 255.
        ret.append(m)
    return np.stack(ret, axis=0) if len(ret) > 1 else ret[0]  


def transform_homogeneous(coords, transform):
    '''
    inputs:
        - coords: tensor of shape [..., ndim]
        - transform: tensor of shape [.., ndim+1, ndim+1]
    
    returns:
        - coords: tensor of shape [..., ndim]
    '''
    coords = torch.nn.functional.pad(coords, (0,1), value=1)
    coords = coords @ transform.transpose(-1,-2)
    coords = coords[...,:-1] / coords[...,-1:]
    return coords.contiguous()

def poisson_blend(foreground_imgs, background_imgs, masks, device="cuda"):
    '''
    foreground_imgs: torch tensor of shape [b,h,w,3]
    background_imgs: torch tensor of shape [b,h,w,3]
    masks: torch bool tensor of shape [b,h,w]
    
    return: blended image, torch tensor of shape [b,h,w,3]
    '''
    import cv2
    foreground_imgs, background_imgs, masks = cvt_numpy(foreground_imgs.clip(0,1), background_imgs.clip(0,1), masks)
    blended_img = []
    
    for fimg, bimg, mask in zip(foreground_imgs, background_imgs, masks):
        
        fimg = (fimg*255).astype(np.uint8)
        bimg = (bimg*255).astype(np.uint8)
        mask = (mask*255).astype(np.uint8)
        br = cv2.boundingRect(mask) # bounding rect (x,y,width,height)
        centerOfBR = (br[0] + br[2] // 2, br[1] + br[3] // 2)
        
        img = cv2.seamlessClone(fimg, bimg, mask, centerOfBR, cv2.NORMAL_CLONE)
        blended_img.append(img.astype(np.float32) / 255.)
        
    return cvt_torch(blended_img, device=device)
    

######### mesh.py

def compute_face_normals(verts, faces):
    '''
    verts: torch [n_verts, 3], float
    faces: torch [n_faces, 3], int in [0, n_verts)
    returns: torch [n_faces, 3], float face normals
    '''
    face_verts = verts[faces.long()] # [n_faces, tri=3, xyz=3]
    v1, v2, v3 = torch.split(face_verts, 1, dim=1)
    norms = torch.cross(v2-v1, v3-v1).squeeze(1) # [n_faces, xyz=3]
    return torch.nn.functional.normalize(norms, dim=-1)


def compute_vert_normals(verts, faces, weighting='area'):
    '''
    verts: torch [n_verts, 3], float
    faces: torch [n_faces, 3], int in [0, n_verts)
    weighting: 'area' or 'angle', defines the weighting strategy
    returns: torch [n_verts, 3], float vert normals
    
    interpolation is weighted either by surrounding face area or by angle
    '''
    
    assert weighting in ['area', 'angle']
    
    i0, i1, i2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()
    v0, v1, v2 = verts[i0, :], verts[i1, :], verts[i2, :]

    # Compute face normals (cross product of edges)
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    
    # Compute per-face areas (used in both strategies)
    face_areas = torch.norm(face_normals, dim=-1, keepdim=True) * 0.5
    
    if weighting == 'area':
        # Area-weighted: Simply splat face normals weighted by face area
        weights = face_areas.expand(-1,3)
    elif weighting == 'angle':
        # Angle-weighted: Compute angles at each vertex of the triangle
        def compute_angles(a, b, c):
            ab = b - a
            ac = c - a
            cos_angle = torch.sum(ab * ac, dim=-1) / (torch.norm(ab, dim=-1) * torch.norm(ac, dim=-1))
            return torch.acos(torch.clamp(cos_angle, -1.0, 1.0))

        angle0 = compute_angles(v1, v2, v0)  # Angle at vertex 0
        angle1 = compute_angles(v2, v0, v1)  # Angle at vertex 1
        angle2 = compute_angles(v0, v1, v2)  # Angle at vertex 2
        
        # Stack angles for weighting each corresponding vertex
        weights = torch.stack([angle0, angle1, angle2], dim=1)

    # Splat face normals to vertices, weighted accordingly
    vn = torch.zeros_like(verts)

    vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals * weights[:, 0:1])
    vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals * weights[:, 1:2])
    vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals * weights[:, 2:3])

    # Normalize the vertex normals
    vn = torch.where(
        torch.sum(vn * vn, dim=1, keepdim=True) > 1e-20,
        torch.nn.functional.normalize(vn, dim=-1),
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device),
    )
    
    return vn

class Mesh():
    
    def __init__(self, verts, faces, verts_uv, faces_uv, orientation, device="cuda"):
        
        self.v, self.f, self.vt, self.ft = cvt_torch(verts, faces, verts_uv, faces_uv, device=device)
        self.fn = compute_face_normals(self.v, self.f).unsqueeze(0) # [1, F, 3]
        self.vn = compute_vert_normals(self.v, self.f, 'area').unsqueeze(0) # [1, V, 3]
        self.v = self.v.unsqueeze(0)
        if self.vt is not None:
            self.vt = self.vt.unsqueeze(0)
        self.orientation = orientation
        
        if faces_uv is None:
            self.ft = self.f
    
    @staticmethod
    def load_mesh(mesh_file, orientation, bound=None, device="cuda"):
        from kiui.mesh import Mesh as _M
        m = _M.load(mesh_file, resize=(bound is not None), bound=bound)
        return Mesh(m.v, m.f, m.vt, m.ft, orientation, device)
    
    def has_uv(self):
        return (self.vt is not None)

    def unwrap_uv(self, uv_res, padding=2, max_stretch=0.15, parallel_regions=10):

        # import open3d as o3d
        # mesh = o3d.t.geometry.TriangleMesh() # if exception, pip install open3d-cpu
        
        # mesh.vertex.positions = o3d.core.Tensor(self.v[0].cpu().numpy(), o3d.core.float32)
        # mesh.triangle.indices = o3d.core.Tensor(self.f.cpu().numpy(), o3d.core.int64)
        # mesh.compute_uvatlas(size=uv_res, gutter=padding, max_stretch=max_stretch, parallel_partitions=max(1,parallel_regions))
        # tri_uvs = mesh.triangle.texture_uvs.numpy()
        # vt, ft = np.unique(tri_uvs.reshape(-1,2), axis=0, return_inverse=True)
        # ft = ft.reshape(self.f.shape)

        # self.vt = torch.from_numpy(vt).to(dtype=self.v.dtype, device=self.v.device).unsqueeze(0)
        # self.ft = torch.from_numpy(ft).to(dtype=self.f.dtype, device=self.f.device)

        import multiprocessing
        import time

        import open3d as o3d
        mesh = o3d.t.geometry.TriangleMesh() # if exception, pip install open3d-cpu
        
        mesh.vertex.positions = o3d.core.Tensor(self.v[0].cpu().numpy(), o3d.core.float32)
        mesh.triangle.indices = o3d.core.Tensor(self.f.cpu().numpy(), o3d.core.int64)
        mesh.compute_uvatlas(size=uv_res, gutter=padding, max_stretch=max_stretch, parallel_partitions=max(1,parallel_regions))
        # timeout=60
        # print("[RENDERER] Start compute_uvatlas, timeout: ",timeout)
        # p = multiprocessing.Process(target=mesh.compute_uvatlas, args=(uv_res, padding, max_stretch, max(1,parallel_regions)))
        # p.start()
        # # Wait for 30 seconds or until process finishes
        # p.join(timeout)
        # # If thread is still active
        # if p.is_alive():
        #     print("[RENDERER] compute_uvatlas takes too long to finish, kill it...")
        #     # Terminate - may not work if process is stuck for good
        #     # p.terminate()
        #     # OR Kill - will work for sure, no chance for process to finish nicely however
        #     p.kill()
        #     p.join()
        #     return False
        tri_uvs = mesh.triangle.texture_uvs.numpy()
        vt, ft = np.unique(tri_uvs.reshape(-1,2), axis=0, return_inverse=True)
        ft = ft.reshape(self.f.shape)

        self.vt = torch.from_numpy(vt).to(dtype=self.v.dtype, device=self.v.device).unsqueeze(0)
        self.ft = torch.from_numpy(ft).to(dtype=self.f.dtype, device=self.f.device)

        return True

        # import stopit
        # timeout = 30
        # with stopit.ThreadingTimeout(timeout) as to_ctx_mgr: # Timeout 30s for uv unwrapping
        #     print("[RENDERER] Start unwrap_uv, timeout: ", timeout)

        #     import open3d as o3d
        #     mesh = o3d.t.geometry.TriangleMesh() # if exception, pip install open3d-cpu
            
        #     mesh.vertex.positions = o3d.core.Tensor(self.v[0].cpu().numpy(), o3d.core.float32)
        #     mesh.triangle.indices = o3d.core.Tensor(self.f.cpu().numpy(), o3d.core.int64)
        #     mesh.compute_uvatlas(size=uv_res, gutter=padding, max_stretch=max_stretch, parallel_partitions=max(1,parallel_regions))
        #     tri_uvs = mesh.triangle.texture_uvs.numpy()
        #     vt, ft = np.unique(tri_uvs.reshape(-1,2), axis=0, return_inverse=True)
        #     ft = ft.reshape(self.f.shape)
    
        # if to_ctx_mgr.state == to_ctx_mgr.EXECUTED: # uv unwrapping executed within 30s
        #     self.vt = torch.from_numpy(vt).to(dtype=self.v.dtype, device=self.v.device).unsqueeze(0)
        #     self.ft = torch.from_numpy(ft).to(dtype=self.f.dtype, device=self.f.device)
        # else:
        #     print("[RENDERER] Unable to finish unwrap_uv in 30s! Killed.")
        #     return False

######## camera.py

def make_cameras(azimuths, elevations, radii, device="cuda"):
    '''
    returns camera to world where camera system follows nvdiffrast convention (+x right, +y down and +z from camera)
    and world is +z up, with +x having 0 azimuth and +y 90 deg azimuth
    '''
    
    azimuths, elevations, radii = cvt_numpy(azimuths, elevations, radii)
    
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
    
    c2w = cvt_torch(c2w, device=device)
    return c2w

def make_ndc(zoom, near=0.1, far=100.0, camera_type="pinhole", device="cuda"):
    
    P = np.zeros((4, 4))
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
    
    P = cvt_torch(P, device=device)
    
    return P


def get_flu_convention(flu_convention="xyz", homogeneous=False, device="cuda"):
    cols = []
    sign = 1
    for c in flu_convention:
        if c == "-":
            sign *= -1
            continue
        elif c == "+":
            continue
        elif c == "x" or c == "X":
            cols.append([sign, 0, 0])
            sign = 1
        elif c == "y" or c == "Y":
            cols.append([0, sign, 0])
            sign = 1
        elif c == "z" or c == "Z":
            cols.append([0, 0, sign])
            sign = 1
        else:
            raise ValueError
    
    if len(cols) != 3:
        raise ValueError
    
    transform = torch.tensor(cols, dtype=torch.float32, device=device).transpose(0,1)
    assert (transform @ torch.inverse(transform) - torch.eye(3, device=device)).abs().max() == 0
    
    if homogeneous:
        transform_ = torch.eye(4, dtype=torch.float32, device=device)
        transform_[:3,:3] = transform
        transform = transform_
    
    return transform

def get_convention(convention_str, homogeneous=False, device="cuda"):
    if convention_str == "y-up":
        convention_str = "zxy"
    elif convention_str == "z-up":
        convention_str = "-yxz"
    return get_flu_convention(convention_str, homogeneous, device)


######## raytrace


def ray_trace(verts, faces, ori, dir):
    '''
    verts: numpy [n_verts, 3], float
    faces: numpy [n_faces, 3], int in [0, n_verts)
    scene: EmbreeScene, if None a new scene gets created from verts and faces, otherwise an existing will be used and verts and faces are ignored
    ori: numpy [...,3] float, origin of rays
    dir: numpy [...,3] float, direction of rays
    returns: face_id, bary, dist of shape [...], [...,3], [...,1], respectively
    '''
    
    import pyembree
    from pyembree import rtcore_scene

    scene = rtcore_scene.EmbreeScene()
    pyembree.mesh_construction.TriangleMesh(scene, verts[faces].astype(np.float32))
    ori = ori.astype(np.float32) + np.zeros_like(dir).astype(np.float32)
    dir = np.zeros_like(ori).astype(np.float32) + dir.astype(np.float32)
    rtres = scene.run(ori.reshape(-1,3), dir.reshape(-1,3), query='INTERSECT', output=True)
    shape = ori.shape[:-1]
    bary = np.stack((rtres['u'], rtres['v'], 1-rtres['u']-rtres['v']), axis=-1).reshape(*shape, 3)
    dist = rtres['tfar'].reshape(*shape, 1)
    face_id = rtres['primID'].reshape(*shape)
    return face_id, bary, dist


def ray_occluded(verts, faces, ori, dir, scene=None, return_scene=False):
    '''
    verts: numpy [n_verts, 3], float
    faces: numpy [n_faces, 3], int in [0, n_verts)
    scene: EmbreeScene, if None a new scene gets created from verts and faces, otherwise an existing will be used and verts and faces are ignored
    ori: numpy [...,3] float, origin of rays
    dir: numpy [...,3] float, direction of rays
    returns: mask of [...], whether or not rays are occluded
    '''
    
    import pyembree
    from pyembree import rtcore_scene

    if scene is None:
        scene = rtcore_scene.EmbreeScene()
        pyembree.mesh_construction.TriangleMesh(scene, verts[faces].astype(np.float32))
    ori = ori.astype(np.float32) + np.zeros_like(dir).astype(np.float32)
    dir = np.zeros_like(ori).astype(np.float32) + dir.astype(np.float32)
    face_id = scene.run(ori.reshape(-1,3), dir.reshape(-1,3), query='OCCLUDED')
    shape = ori.shape[:-1]
    mask =  (face_id >= 0).reshape(shape)
    if return_scene:
        return mask, scene
    else:
        return mask

def intersect_triangles(ray_origin, ray_direction, face_verts, epsilon=1e-10):
    """
    compte barycentric coordinates (u, v) and their derivatives with respect to
    ray origin and ray direction

    Inputs:
    - ray_origin of shape [b, 3]
    - ray_direction of shape [b, 3]
    - face_verts, triangle verts of shape [b, 3(verts), 3(xyz)]

    Returns:
    - u, v, t: barycentric and distance of intersection, of shape (b,)
    - du_dO, dv_dO: Tensors of shape (b, 3), derivatives w.r.t ray origin
    - du_dD, dv_dD: Tensors of shape (b, 3), derivatives w.r.t ray direction
    """
    
    v0 = face_verts[:, 0, :]  # (b, 3)
    v1 = face_verts[:, 1, :]  # (b, 3)
    v2 = face_verts[:, 2, :]  # (b, 3)
    e1 = v0 - v2  # (b, 3)
    e2 = v1 - v2  # (b, 3)

    n = torch.cross(e1, e2, dim=1)  # (b, 3)
    s = v2 - ray_origin  # (b, 3)
    delta = torch.sum(ray_direction * n, dim=1)  # (b,)

    if not torch.all(torch.abs(delta) >= epsilon):
        raise ValueError("some rays are parallel to triangles")

    delta_t = torch.sum(s * n, dim=1)  # (b,)
    t = delta_t / delta  # (b,)

    p = torch.cross(ray_direction, e2, dim=1)  # (b, 3)
    delta_u = torch.sum(p * s, dim=1)  # (b,)
    u = delta_u / delta  # (b,)

    q = torch.cross(e1, ray_direction, dim=1)  # (b, 3)
    delta_v = torch.sum(q * s, dim=1)  # (b,)
    v = delta_v / delta  # (b,)

    du_dO = -p / delta.unsqueeze(1)  # (b, 3)
    dv_dO = -q / delta.unsqueeze(1)  # (b, 3)

    delta_sq = delta.unsqueeze(1) ** 2  # (b, 1)
    d_delta_dD = n  # (b, 3)

    d_delta_u_dD = torch.cross(e2, s, dim=1)  # (b, 3)
    d_delta_v_dD = -torch.cross(e1, s, dim=1)  # (b, 3)  

    numerator_u = delta.unsqueeze(1) * d_delta_u_dD - delta_u.unsqueeze(1) * d_delta_dD  # (b, 3)
    numerator_v = delta.unsqueeze(1) * d_delta_v_dD - delta_v.unsqueeze(1) * d_delta_dD  # (b, 3)

    du_dD = numerator_u / delta_sq  # (b, 3)
    dv_dD = numerator_v / delta_sq  # (b, 3)

    return u, v, t, du_dO, dv_dO, du_dD, dv_dD

    
def rasterise_by_raytrace(verts_ndc, faces, ray_o, ray_d, d_rayo_dxy=None, d_rayd_dxy=None):
    '''
    verts_ndc: [b/1, v, 4]
    faces: [f, 3]
    ray_o: [b/1, h, w, 3] in ndc
    ray_d: [b/1, h, w, 3] in ndc
    d_rayo_dxy: [b/1, h, w, 3, 2] in ndc
    d_rayd_dxy: [b/1, h, w, 3, 2] in ndc
    
    returns: 
    '''
    returns_db = (d_rayo_dxy is not None) and (d_rayd_dxy is not None)
    
    b = max(verts_ndc.shape[0], ray_o.shape[0], ray_d.shape[0])
    h,w = ray_o.shape[1:3]
    verts_ndc = verts_ndc.expand(b,-1,-1)
    ray_o = ray_o.expand(b,-1,-1,-1)
    ray_d = ray_d.expand(b,-1,-1,-1)
    if returns_db:
        d_rayo_dxy = d_rayo_dxy.expand(b, -1,-1,-1,-1)
        d_rayd_dxy = d_rayd_dxy.expand(b, -1,-1,-1,-1)
    
    rasts, rast_dbs = [], []
    
    for i in range(b):
        verts = verts_ndc[i]
        verts = verts[...,:3] / verts[...,3:] # [v,3]
        ro = ray_o[i] # [h,w,3]
        rd = ray_d[i]
        
        face_id, bary, dist = cvt_torch(*ray_trace(*cvt_numpy(verts, faces, ro, rd)), device=verts_ndc.device) # [h,w]
        hit_mask = (face_id >= 0)
        
        u,v,d, du_dO, dv_dO, du_dD, dv_dD = intersect_triangles(ro[hit_mask], rd[hit_mask], verts[faces[face_id[hit_mask].long()]]) # [n_hits,], [n_hist,3]
        
        
        rast = torch.zeros(h,w,4, dtype=torch.float32, device=verts_ndc.device)

        rast[hit_mask] = torch.cat([
                u.unsqueeze(-1).float(), v.unsqueeze(-1).float(), 
                (d.unsqueeze(-1)).float(), 
                (face_id[hit_mask].unsqueeze(-1)+1).float()
            ], dim=-1)
        
        if returns_db:
            rast_db = torch.zeros(h,w,4, dtype=torch.float32, device=verts_ndc.device)
            dO_dxy = d_rayo_dxy[i][hit_mask] # [n_hits, 3, 2]
            dD_dxy = d_rayd_dxy[i][hit_mask] # [n_hits, 3, 2]
            
            du_dxy = (du_dO.unsqueeze(2) * dO_dxy + du_dD.unsqueeze(2) * dD_dxy).sum(1) # [n_hits, 2]
            dv_dxy = (dv_dO.unsqueeze(2) * dO_dxy + dv_dD.unsqueeze(2) * dD_dxy).sum(1) # [n_hits, 2]
            
            rast_db[hit_mask] = torch.cat([du_dxy, dv_dxy], dim=-1)
        else:
            rast_db = torch.zeros(h,w,0, dtype=torch.float32, device=verts_ndc.device)
        
        rasts.append(rast)
        rast_dbs.append(rast_db)
    
    return torch.stack(rasts), torch.stack(rast_dbs)
        
########

class Cameras():
    
    def __init__(self, azimuths, elevations, dists, zooms, near=0.1, far=100.0, camera_type="pinhole", world_orientation="z-up", device="cuda"):
        
        '''
        zooms: controls the view frustum size of camera, same object appears smaller with greater zoom
            - for camera_type="pinhole" this is tangent(fov/2)
            - for camera_type="ortho" this is the half sensor size in world unit
        '''
        
        self.c2w = get_convention(world_orientation, homogeneous=True, device=device) @ make_cameras(azimuths, elevations, dists, device=device)
        self.ndc = torch.stack([make_ndc(float(zoom), near, far, camera_type=camera_type, device=device) for zoom in zooms])
        self.mvp = self.ndc @ torch.inverse(self.c2w)
        self.camera_type = camera_type
        self.world_orientation = world_orientation 
        self.device = device
        
        self.support_aa = True
        
    def transform_verts_to_ndc(self, verts_world):
        '''
        Inputs: 
            - verts_world of shape [b, nv, 3/4] last dimension can be 3 or 4 as homogeneous coordinates
        Returns:
            - verts_ndc of shape [n, nv, 4]
        '''
        if verts_world.shape[-1] == 3:
            verts_world = torch.nn.functional.pad(verts_world, (0,1), value=1)
        
        verts_ndc = verts_world @ self.mvp.transpose(-1,-2) # [b, nv, 4]
        return verts_ndc
    
    def generate_ndc_rays(self, image_size):
        '''
        generate rays from screen to object in ndc
        
        Inputs:
            - image_size: tuple of (h,w)
        Returns:
            - rays_o of shape [h,w, 3], in NDC
            - rays_d of shape [h,w, 3], in NDC 
            - d_rayo_dxy: [h, w, 3, 2] derivative of rays_o w.r.t. pixel coordinates (x,y) in [0, img_size)
            - d_rayd_dxy: [h, w, 3, 2] derivative of rays_d w.r.t. pixel coordinates (x,y) in [0, img_size)
        '''
        h,w = image_size
        device= self.device
        horizontal = (torch.arange(w, device=device, dtype=torch.float32).reshape(1,w).expand(h,w) + 0.5) * (2 / w) - 1 # values in (-1,1)
        vertical = (torch.arange(h, device=device, dtype=torch.float32).reshape(h,1).expand(h,w) + 0.5) * (2 / h) - 1 # values in (-1,1)
        zeros = torch.zeros_like(horizontal)
        ones = torch.ones_like(horizontal)
        dev_scale = torch.tensor([2.0 / w, 2.0 / h], device=device, dtype=torch.float32) # [2]
        
        
        xo = horizontal
        zo = -ones
        yo = vertical
        xd = zeros
        zd = ones
        yd = zeros
        
        dxo_dh = ones
        dyo_dh = zeros
        dzo_dh = zeros
        dxd_dh = zeros
        dyd_dh = zeros
        dzd_dh = zeros
        dxo_dv = zeros
        dyo_dv = ones
        dzo_dv = zeros
        dxd_dv = zeros
        dyd_dv = zeros
        dzd_dv = zeros
        
        rays_o = torch.stack((xo, yo, zo), dim=-1) # [h,w,3]
        rays_d = torch.stack((xd, yd, zd), dim=-1) # [h,w,3]
        
        d_rayo_dxy = torch.stack((dxo_dh, dxo_dv, dyo_dh, dyo_dv, dzo_dh, dzo_dv), dim=-1).reshape(h,w,3,2) * dev_scale
        d_rayd_dxy = torch.stack((dxd_dh, dxd_dv, dyd_dh, dyd_dv, dzd_dh, dzd_dv), dim=-1).reshape(h,w,3,2) * dev_scale
        
        return rays_o, rays_d, d_rayo_dxy, d_rayd_dxy
     
        
    def rasterize(self, verts_world, faces, image_size, nvdffrast_ctx=None, tile_size=None):
        '''
        Inputs: 
        - verts_world of shape [b, nv, 3/4] last dimension can be 3 or 4 as homogeneous coordinates
        - faces of shape [nf, 3]
        - image_size: tuple of (h,w)
        
        Returns:
        two tensors as defined by nvdiffrast's rasterize function https://nvlabs.github.io/nvdiffrast/#pytorch-api-reference
        '''
        
        if nvdffrast_ctx is None:
            nvdffrast_ctx = dr.RasterizeCudaContext()

        rast, rast_db = rasterize(nvdffrast_ctx, self.transform_verts_to_ndc(verts_world), faces.contiguous(), image_size, tile_size=tile_size)
        return rast, rast_db
    
    def depth_ndc2abs(self, ndc_depth, mask=None):
        '''
        maps ndc space depth in [-1,1] to [near, far]
        
        Inputs: 
            - ndc_depth of shape [B,H,W,1]
            - mask foreground mask of shape [B,H,W,1], optional
        Returns:
            - abs_depth of shape [B,H,W,1], background values are set to 0 if mask is provided
        '''
        b,h,w,_ = ndc_depth.shape
        transform_mat = self.ndc[...,2:4,2:4]
        ndc_depth = torch.nn.functional.pad(ndc_depth, (0,1), value=1) # [B,H,W,2]
        abs_depth = ndc_depth.reshape(b,h*w,2) @ torch.inverse(transform_mat).transpose(-1,-2) # [B,H,W,2]
        abs_depth = abs_depth.reshape(b,h,w,2)
        abs_depth = abs_depth[...,:1] / abs_depth[...,-1:] # [B,H,W,1]
        if mask is not None:
            abs_depth = torch.where(mask, abs_depth, torch.zeros_like(abs_depth))
        return abs_depth
    
    def project_onto_screen(self, points, is_ndc=False):
        '''
        project world points onto image plane, return screen coordinates in [-1,1]^2
        
        inputs of shape [n_views, n_points, 3/4]
        outputs of shape [n_views, n_points, 2]
        '''
        if is_ndc:
            ndc_points = points
        else:
            ndc_points = self.transform_verts_to_ndc(points)
        
        if ndc_points.shape[-1] == 3:
            return ndc_points[...,:2]
        elif ndc_points.shape[-1] == 4:
            return ndc_points[...,:2] / ndc_points[...,-1:]
    
    def get_ndc_ray_dirs(self, ndc_points):
        '''
        given ndc points, get ray directions that project these point onto screen. returned dirs are in ndc as well
        
        inputs of shape [..., n_points, 3/4]
        output of shape [..., n_points, 3]
        '''
        ones = torch.ones_like(ndc_points[...,:1])
        zeros = torch.zeros_like(ndc_points[...,:1])
        dirs = torch.cat((zeros, zeros, -ones), dim=-1)
        return dirs
    
    def generate_world_rays(self, image_size):
        '''
        generate rays from screen to object in world
        
        Inputs:
            - image_size: tuple of (h,w)
        Returns:
            - rays_o of shape [views, h,w, 3], in world space
            - rays_d of shape [views, h,w, 3], in world space 
        '''
        
        ro_ndc, rd_ndc, *_ = self.generate_ndc_rays(image_size) # [h,w, 3]
        rt_ndc = ro_ndc + rd_ndc
        ro_ndc = torch.nn.functional.pad(ro_ndc, (0,1), value=1)
        rt_ndc = torch.nn.functional.pad(rt_ndc, (0,1), value=1)
        ndc2world = torch.inverse(self.mvp)
        
        rays_t = torch.einsum('hwx,vyx->vhwy', rt_ndc, ndc2world)
        rays_t = rays_t[...,:3] / rays_t[...,-1:] 
        
        if self.camera_type == "ortho":
            rays_o = torch.einsum('hwx,vyx->vhwy', ro_ndc, ndc2world)
            rays_o = rays_o[...,:3] / rays_o[...,-1:] 
        elif self.camera_type == "pinhole":
            rays_o = self.c2w[:,:3,3].reshape(-1,1,1,3).expand(-1, *image_size, 3)
            
        rays_d = torch.nn.functional.normalize(rays_t - rays_o, dim=-1)
        return rays_o, rays_d

class PanoramicCameras(Cameras):

    def __init__(self, azimuths, elevations, dists=0, radii=1, zooms=1, camera_type="spherical_equirectangular", world_orientation="z-up", device="cuda"):
        assert camera_type in ["cylindrical", "spherical", "spherical_equirectangular"]
        
        self.c2w = get_convention(world_orientation, homogeneous=True, device=device) @ make_cameras(azimuths, elevations, dists, device=device)
        self.radii = cvt_torch(radii, device=device).reshape(-1,1,1).float()
        self.zooms = cvt_torch(zooms, device=device).reshape(-1,1,1).float()
        self.mvp = torch.inverse(self.c2w)
        if camera_type == "cylindrical":
            self.mvp[:,0:1] = self.mvp[:,0:1] / self.radii
            self.mvp[:,2:3] = self.mvp[:,2:3] / self.radii
            self.mvp[:,1:2] = self.mvp[:,1:2] * self.zooms
        else:
            self.mvp[:,:3] = self.mvp[:,:3] / self.radii
        self.camera_type = camera_type
        self.world_orientation = world_orientation 
        self.device = device
        
        self.support_aa = False
        
    def generate_ndc_rays(self, image_size):
        '''
        generate rays from screen to object in ndc
        
        Inputs:
            - image_size: tuple of (h,w)
        Returns:
            - rays_o of shape [h,w, 3], in NDC
            - rays_d of shape [h,w, 3], in NDC 
            - d_rayo_dxy: [h, w, 3, 2] derivative of rays_o w.r.t. pixel coordinates (x,y) in [0, img_size)
            - d_rayd_dxy: [h, w, 3, 2] derivative of rays_d w.r.t. pixel coordinates (x,y) in [0, img_size)
        '''
        h,w = image_size
        device= self.device
        horizontal = (torch.arange(w, device=device, dtype=torch.float32).reshape(1,w).expand(h,w) + 0.5) * (2 / w) - 1 # values in (-1,1)
        vertical = (torch.arange(h, device=device, dtype=torch.float32).reshape(h,1).expand(h,w) + 0.5) * (2 / h) - 1 # values in (-1,1)
        azimuths = horizontal * np.pi + np.pi # values in (0, 2pi)
        zeros = torch.zeros_like(azimuths)
        ones = torch.ones_like(azimuths)
        dev_scale = torch.tensor([2.0 / w, 2.0 / h], device=device, dtype=torch.float32) # [2]
        
        if self.camera_type == "cylindrical":
            xo = -torch.sin(azimuths)
            zo = torch.cos(azimuths)
            yo = vertical
            xd = -xo
            zd = -zo
            yd = torch.zeros_like(vertical)
            
            dxo_dh = -zo * np.pi
            dyo_dh = zeros
            dzo_dh = xo * np.pi
            dxd_dh = -dxo_dh
            dyd_dh = zeros
            dzd_dh = -dzo_dh
            dxo_dv = zeros
            dyo_dv = ones
            dzo_dv = zeros
            dxd_dv = zeros
            dyd_dv = zeros
            dzd_dv = zeros
            
        elif self.camera_type == "spherical":
            elevations = -vertical * (np.pi / 2)
            radius = torch.cos(elevations)
            xo = -torch.sin(azimuths) * radius
            zo = torch.cos(azimuths) * radius
            yo = -torch.sin(elevations)
            xd = -xo
            yd = -yo
            zd = -zo
            
            dxo_dh = -zo * np.pi
            dyo_dh = zeros
            dzo_dh = xo * np.pi
            dxd_dh = -dxo_dh  
            dyd_dh = zeros
            dzd_dh = -dzo_dh 
            dxo_dv = torch.sin(azimuths) * yo * (np.pi / 2)
            dyo_dv = radius * (np.pi / 2)
            dzo_dv = -torch.cos(azimuths) * yo * (np.pi / 2)
            dxd_dv = -dxo_dv 
            dyd_dv = -dyo_dv  
            dzd_dv = -dzo_dv
                        
        elif self.camera_type == "spherical_equirectangular":
            radius = torch.sqrt(1 - vertical**2)
            xo = -torch.sin(azimuths) * radius
            zo = torch.cos(azimuths) * radius
            yo = vertical
            xd = -xo
            yd = -yo
            zd = -zo
            
            dxo_dh = -zo * np.pi
            dyo_dh = zeros
            dzo_dh = xo * np.pi 
            dxd_dh = -dxo_dh 
            dyd_dh = zeros
            dzd_dh = -dzo_dh
            dxo_dv = torch.sin(azimuths) * vertical / radius
            dyo_dv = ones
            dzo_dv = -torch.cos(azimuths) * vertical / radius
            dxd_dv = -dxo_dv 
            dyd_dv = -dyo_dv  
            dzd_dv = -dzo_dv 
            
        rays_o = torch.stack((xo, yo, zo), dim=-1) # [h,w,3]
        rays_d = torch.stack((xd, yd, zd), dim=-1) # [h,w,3]
        
        d_rayo_dxy = torch.stack((dxo_dh, dxo_dv, dyo_dh, dyo_dv, dzo_dh, dzo_dv), dim=-1).reshape(h,w,3,2) * dev_scale
        d_rayd_dxy = torch.stack((dxd_dh, dxd_dv, dyd_dh, dyd_dv, dzd_dh, dzd_dv), dim=-1).reshape(h,w,3,2) * dev_scale
        
        return rays_o, rays_d, d_rayo_dxy, d_rayd_dxy
    
    def rasterize(self, verts_world, faces, image_size, nvdffrast_ctx=None, tile_size=None):
        '''
        Inputs: 
        - verts_world of shape [b, nv, 3/4] last dimension can be 3 or 4 as homogeneous coordinates
        - faces of shape [nf, 3]
        - image_size: tuple of (h,w)
        
        Returns:
        two tensors as defined by nvdiffrast's rasterize function https://nvlabs.github.io/nvdiffrast/#pytorch-api-reference
        '''
        if tile_size is not None:
            import warnings
            warnings.warn("Tiled rasterisation is not supported by panoramic cameras.", category=UserWarning)
        
        rays_o, rays_d, d_rayo_dxy, d_rayd_dxy = self.generate_ndc_rays(image_size)
        verts_ndc = self.transform_verts_to_ndc(verts_world) # [b, nv, 4]
        
        rast, rast_db = rasterise_by_raytrace(
                verts_ndc, faces, rays_o.unsqueeze(0), rays_d.unsqueeze(0), 
                d_rayo_dxy=d_rayo_dxy.unsqueeze(0), d_rayd_dxy=d_rayd_dxy.unsqueeze(0)
            )
        
        u,v,d,f = torch.split(rast, 1, dim=-1)
        d = torch.where(f>0, d-1, d) # convert ray distance to ndc depth
        
        rast = torch.cat((u,v,d,f), dim=-1)
        
        return rast, rast_db
    
    def depth_ndc2abs(self, ndc_depth, mask=None):
        '''
        maps ndc space depth to absolute depth
        
        Inputs: 
            - ndc_depth of shape [B,H,W,1]
            - mask foreground mask of shape [B,H,W,1], optional
        Returns:
            - abs_depth of shape [B,H,W,1], background values are set to 0 if mask is provided
        '''
        b,h,w,_ = ndc_depth.shape
        
        abs_depth = (ndc_depth + 1) * self.radii.unsqueeze(-1)
        
        if mask is not None:
            abs_depth = torch.where(mask, abs_depth, torch.zeros_like(abs_depth))
        return abs_depth
    
    def project_onto_screen(self, points, is_ndc=False):
        '''
        project world points onto image plane, return screen coordinates in [-1,1]^2
        '''
        if is_ndc:
            ndc_points = points
        else:
            ndc_points = self.transform_verts_to_ndc(points)
        if ndc_points.shape[-1] == 4:
            ndc_points = ndc_points[...,:3] / ndc_points[...,-1:]

        x,y,z = torch.split(ndc_points, 1, dim=-1) # [b,n,1]
        u = torch.atan2(x.contiguous(), -z.contiguous()) / np.pi # [b,n,1] in (-1,1)
        if self.camera_type == "cylindrical":
            v = y
        elif self.camera_type == "spherical" or "spherical_equirectancular":
            elevation = torch.atan2(-y, (x**2+z**2)**0.5)
            if self.camera_type == "spherical":
                v = -elevation * (2 / np.pi)
            else:
                v = torch.sin(-elevation)
        return torch.cat((u,v), dim=-1)
    
    def get_ndc_ray_dirs(self, ndc_points):
        '''
        given ndc points, get ray directions that project these point onto screen. returned dirs are in ndc as well
        
        inputs of shape [..., n_points, 3/4]
        output of shape [..., n_points, 3]
        '''
        if ndc_points.shape[-1] == 4:
            ndc_points = ndc_points[...,:3] / ndc_points[...,-1:]
        
        x,y,z = torch.split(ndc_points.contiguous(), 1, dim=-1) # [b,n,1]
        
        if self.camera_type == "cylindrical":
            dirs = torch.cat((x,torch.zeros_like(y),z), dim=-1)
        elif self.camera_type == "spherical" or "spherical_equirectancular":
            dirs = torch.cat((x,y,z), dim=-1)
        dirs = torch.nn.functional.normalize(dirs, dim=-1)
        return dirs

    def generate_world_rays(self, image_size):
        '''
        generate rays from screen to object in world
        
        Inputs:
            - image_size: tuple of (h,w)
        Returns:
            - rays_o of shape [views, h,w, 3], in world space
            - rays_d of shape [views, h,w, 3], in world space 
        '''
        
        ro_ndc, rd_ndc, *_ = self.generate_ndc_rays(image_size) # [h,w, 3]
        rt_ndc = ro_ndc + rd_ndc
        ro_ndc = torch.nn.functional.pad(ro_ndc, (0,1), value=1)
        rt_ndc = torch.nn.functional.pad(rt_ndc, (0,1), value=1)
        ndc2world = torch.inverse(self.mvp)
        rays_o = torch.einsum('hwx,vyx->vhwy', ro_ndc, ndc2world)
        rays_t = torch.einsum('hwx,vyx->vhwy', rt_ndc, ndc2world)
        rays_o = rays_o[...,:3] / rays_o[...,-1:] 
        rays_t = rays_t[...,:3] / rays_t[...,-1:] 
        rays_d = torch.nn.functional.normalize(rays_t - rays_o, dim=-1)
        return rays_o, rays_d

def rasterize(ctx, pos, tri, resolution, grad_db=True, tile_size=None):
    
    created_context = 0
    
    if ctx == "cuda":
        ctx = dr.RasterizeCudaContext()
        created_context = 1
    elif ctx == "gl":
        ctx = dr.RasterizeGLContext()
        created_context = 2
    elif ctx == "embree":
        ctx = None
    
    h,w = resolution
        
    if tile_size is not None:
        rast = torch.zeros((pos.shape[0], h,w,4), dtype=torch.float32, device=pos.device)
        rast_db = torch.zeros((pos.shape[0], h,w, 4 if grad_db else 0), dtype=torch.float32, device=pos.device)
        for tile_i in range(0, h, tile_size):
            for tile_j in range(0, w, tile_size):
                tile_i_end = min(h, tile_i+tile_size)
                tile_j_end = min(w, tile_j+tile_size)
                
                l = tile_j * 2.0 / w - 1
                r = tile_j_end * 2.0 / w - 1
                u = tile_i * 2.0 / h - 1
                d = tile_i_end * 2.0 / h - 1
                
                pos_transformed = pos @ torch.tensor([
                    [2/(r-l), 0, 0, -(r+l)/(r-l)],
                    [0, 2/(d-u), 0, -(d+u)/(d-u)],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]).transpose(0,1).to(pos)
                
                rast_tile, rast_db_tile = rasterize(ctx, pos_transformed, tri, (tile_i_end-tile_i, tile_j_end-tile_j), grad_db)
                
                rast[:,tile_i:tile_i_end, tile_j:tile_j_end] = rast_tile
                rast_db[:,tile_i:tile_i_end, tile_j:tile_j_end] = rast_db_tile
    else:
        if ctx is not None:
            rast, rast_db = dr.rasterize(ctx, pos, tri, resolution, grad_db=grad_db)
        else:
            device= pos.device
            horizontal = (torch.arange(w, device=device, dtype=torch.float32).reshape(1,w).expand(h,w) + 0.5) * (2 / w) - 1 # values in (-1,1)
            vertical = (torch.arange(h, device=device, dtype=torch.float32).reshape(h,1).expand(h,w) + 0.5) * (2 / h) - 1 # values in (-1,1)
            zeros = torch.zeros_like(horizontal)
            ones = torch.ones_like(horizontal)
            dev_scale = torch.tensor([2.0 / w, 2.0 / h], device=device, dtype=torch.float32) # [2]
            
            
            xo = horizontal
            zo = -ones
            yo = vertical
            xd = zeros
            zd = ones
            yd = zeros
            
            rays_o = torch.stack((xo, yo, zo), dim=-1) # [h,w,3]
            rays_d = torch.stack((xd, yd, zd), dim=-1) # [h,w,3]
            
            if grad_db:
                dxo_dh = ones
                dyo_dh = zeros
                dzo_dh = zeros
                dxd_dh = zeros
                dyd_dh = zeros
                dzd_dh = zeros
                dxo_dv = zeros
                dyo_dv = ones
                dzo_dv = zeros
                dxd_dv = zeros
                dyd_dv = zeros
                dzd_dv = zeros
                
                d_rayo_dxy = torch.stack((dxo_dh, dxo_dv, dyo_dh, dyo_dv, dzo_dh, dzo_dv), dim=-1).reshape(h,w,3,2) * dev_scale
                d_rayd_dxy = torch.stack((dxd_dh, dxd_dv, dyd_dh, dyd_dv, dzd_dh, dzd_dv), dim=-1).reshape(h,w,3,2) * dev_scale
                    
                rast, rast_db = rasterise_by_raytrace(pos, tri, rays_o.unsqueeze(0), rays_d.unsqueeze(0), 
                            d_rayo_dxy=d_rayo_dxy.unsqueeze(0), d_rayd_dxy=d_rayd_dxy.unsqueeze(0))
            else:
                rast, rast_db = rasterise_by_raytrace(pos, tri, rays_o.unsqueeze(0), rays_d.unsqueeze(0))
                
            u,v,d,f = torch.split(rast, 1, dim=-1)
            d = torch.where(f>0, d-1, d) # convert ray distance to ndc depth
            
            rast = torch.cat((u,v,d,f), dim=-1)
    
    if created_context:
        if created_context == 2:
            ctx.release_context()
        del ctx
        
    return rast, rast_db

class Renderer:
    def __init__(self, render_res=1024, tex_res=1024, ssaa=1, world_orientation="y-up", ctx="cuda", rast_tile_size=None):
        '''
        init the renderer
        
        Inputs: 
            - render_res, integer for image resolution
            - tex_res, integer for texture map resolution
            - ssaa, integer for anti-aliasing super sampling ratio, defaults to 1 (no super sampling)
            - world_orientation, a string that represents world coordinate system convention, can be one of following
                - "z-up" front direction is -y, up is +z
                - "y-up" front is +z, up is +y
                - a string that's in "front-right-up" format, e.g. "-yxz" means front=-y, right=+x and up=+z i.e. "z-up"
                
                Note the world's front is always pointing toward any camera at azimuth=0 and elevation=0, and world's right to 
                camera at azimuth=90 and elevation=0
            - rast_ctx: context for rasterizer, can be either a string or a nvdiffrast context. if string, must be one of
                - "cuda": RasterizeCudaContext
                - "gl": RasterizeGLContext, graphic driver must support GL
                - "embree": use Embree to compute rasterization fragment on CPU
            - rast_tile_size: int or None. if integer, rasterization is splitted into multiple tiles of this dimension. 
              tiles are rasterised independently and stitched together at the end. this saves memory on extremely large image/texture
        '''
        
        if (render_res > 2048 or tex_res > 2048) and nvdiffrast.__version__ < '0.3.3':
            raise ValueError(f"Resolutions exceeding 2048 are only supported by nvdiffrast 0.3.3 or above.")
        
        assert tex_res % 32 == 0, "tex res should be multiples of 32 for mip map filtering"
        
        if ctx == "cuda":
            self.ctx = dr.RasterizeCudaContext()
        elif ctx == "gl":
            self.ctx = dr.RasterizeGLContext()
        else:
            self.ctx = ctx
        
        self.tile_size = rast_tile_size
            
        self.render_res = render_res
        self.ssaa = ssaa
        self.tex_res = tex_res
        
        self.mesh = None
        self.mvp = None
        self.world_orientation = world_orientation
        
    def modify_res(self, render_res=None, tex_res=None, ssaa=None):
        new_render_res = self.render_res if render_res is None else render_res
        new_tex_res = self.tex_res if tex_res is None else tex_res
        new_ssaa = self.ssaa if ssaa is None else ssaa
        
        if new_render_res * new_ssaa != self.render_res * self.ssaa:
            self._rast = None
            self._vpos = None
        if new_tex_res != self.tex_res:
            print("[RENDERER] Warning: Texture resolution changed, clearing texture cache.")
            self._rast_tex = None
        
        self.render_res = new_render_res
        self.tex_res = new_tex_res
        self.ssaa = new_ssaa
            
        
    def set_object(self, mesh_file=None, verts=None, faces=None, verts_uv=None, faces_uv=None, bound=None, orientation=None, uv_origin="bottom-left", merge_verts=False):
        '''
        load the object of interest either from file or from tensors
        
        Inputs: 
            - mesh_file [optional]: str path to object file (*.obj, *fbx or *.glb), or None if passing mesh as verts and faces tensors
            - verts, faces, verts_uv, faces_uv [optional]: numpy or pytorch tensors representing mesh
                - verts: float of shape (n_verts, 3) 
                - faces: integer of shape (n_faces, 3)  in range [0, n_verts)
                - verts_uv: float of shape (n_uv_verts, 2) in range [0,1], 
                - faces_uv: integer of shape (n_faces, 3) in range [0, n_uv_verts)
            - bound [optional]: if not None, mesh will be normalized to [-bound, bound]^3
            - orientation [optional]: object coordinate convention, can be "y-up" or "z-up" or FLU convention string, defaults to world convention
            - uv_origin [optional]: "bottom-left" or "top-left", only applicable when supplying verts_uv
              whether uv origin (0,0) is bottom-left conor or top-left of texture map, defaults to "bottom-left"
            - merge_verts: merge non-unique verts, this would change vertex count but not face numbers. 
        '''
        assert uv_origin in ["bottom-left", "top-left"]
        
        if orientation is None:
            orientation = self.world_orientation
        
        if mesh_file is not None:
            self.mesh = Mesh.load_mesh(mesh_file, orientation, bound)
        else:
            if bound is not None:
                verts = cvt_torch(verts)
                bbox = verts.min(0)[0], verts.max(0)[0]
                c = sum(bbox) * 0.5
                s = (bbox[1] - bbox[0]).max()
                verts = (verts - c) / s * bound * 2
            if uv_origin == "bottom-left":
                verts_uv = cvt_torch(verts_uv)
                verts_uv = torch.stack((verts_uv[:,0], 1-verts_uv[:,1]), dim=-1)
            self.mesh = Mesh(verts, faces, verts_uv, faces_uv, orientation)  
              
        if merge_verts:
            verts, faces = cvt_torch(self.mesh.v[0], self.mesh.f)
            verts_unique, indices = torch.unique(verts, dim=0, return_inverse=True, sorted=False)
            verts = verts_unique
            faces = indices[faces.long()].to(faces.dtype)
            self.mesh = Mesh(verts, faces, self.mesh.vt[0], self.mesh.ft, orientation)
        
        if orientation is not None:
            self.o2w = get_convention(orientation, homogeneous=True) @ torch.inverse(get_convention(orientation, homogeneous=True))  
        else:
            self.o2w = torch.eye(4, device="cuda")
            
        self._o2w = self.o2w
        
        self._clear_cache(clear_uv=True)
    
    def unwrap_uv(self, padding=2, max_stretch=0.15, parallel_regions=4):
        success = self.mesh.unwrap_uv(self.tex_res, padding, max_stretch, parallel_regions)
        self._clear_cache(clear_uv=True)
        return success
    
    def reset_obj_transform(self):
        if hasattr(self, "_o2w"):
            self.o2w = self._o2w
            self._clear_cache(clear_uv=False)
        
    def transform_obj(self, transform):
        '''
        transform: 4x4 numpy or torch, object space transformation
        '''
        self.o2w = self.o2w @ cvt_torch(transform, device=self.o2w.device)
        self._clear_cache(clear_uv=False)
    
    def set_cameras(self, azimuths, elevations, dists, camera_type="ortho", radii=1.0, zooms=0.9, near=0.1, far=100.0, **args):
        '''
        set camera poses, including intrinsic and extrinsics
        
        Inputs: 
            - azimuths: a list or 1d array/tensor of cameras' azimuth angles in degrees, the world's front is pointing toward azimuth=0 and world's right to 
                azimuth=90
            - elevations: cameras elevation angles in degrees
            - dists: viewing distance of cameras from world origin point
            - radii: radius for cylindrical and spherical cameras, ignored for pinhole/orthographic cams
            - camera_type: either "ortho" for orthographical cameras or "pinhole" for perspective ones
            - zooms: a scalar or list of scalars for camera zooming ratios that control the size of viewing frustum, same fixed object appears bigger with greater zoom
                - for camera_type="pinhole" this is cot(fov/2), e.g. zoom=1 translates to 90 degrees fov
                - for camera_type="ortho" this is the half sensor size in world unit, e.g. zoom=0.9 means a cube spanning [-1,1]^3 will take up 90% of image
            - near and far: depths of near and far planes, anything not in between will be invisible
        '''
        
        if isinstance(zooms, (int, float)):
            zooms = [zooms] * len(cvt_numpy(azimuths))
        
        if camera_type in ["ortho", "pinhole"]: 
            self.camera = Cameras(azimuths, elevations, dists, zooms, near, far, camera_type, self.world_orientation)
        elif camera_type in ["cylindrical", "spherical", "spherical_equirectangular"]:
            self.camera = PanoramicCameras(azimuths, elevations, dists, radii, zooms, camera_type, self.world_orientation)
        else:
            raise ValueError(f"unknown camera type {camera_type}")
        self.n_views = len(cvt_numpy(azimuths))
        
        self._clear_cache(clear_uv=False)
    
    def _clear_cache(self, clear_uv=True):
        if clear_uv:
            self._rast_tex = None
        self._rast = None
        self._vpos = None
        self._texlu = None
        
    def _rasterize_uv(self):
        if self._rast_tex is None:
            verts_u, verts_v = torch.split(self.mesh.vt, 1, dim=-1) # [1, verts, 1]
            verts_uv_ndc = torch.cat([verts_u*2-1, verts_v*2-1, torch.zeros_like(verts_u), torch.ones_like(verts_u)], dim=-1) # [1, verts, 4]
            self._rast_tex = rasterize(self.ctx, verts_uv_ndc.contiguous(), self.mesh.ft.contiguous(), (self.tex_res, self.tex_res,), grad_db=False, tile_size=self.tile_size) 
            # pad rast
            uv_mask = (self._rast_tex[0][...,-1:] > 0).cpu().numpy()
            uv_mask_padded = dilate_masks(uv_mask[0]).reshape(uv_mask.shape)
            rast_padded = self._voronoi_inpaint(self._rast_tex[0], uv_mask)
            self._rast_tex = torch.where(cvt_torch(uv_mask_padded)>0, rast_padded, self._rast_tex[0]), self._rast_tex[1]
        return self._rast_tex
    
    def _camera_list(self, cameras=None):
        if cameras is None:
            cameras = list(range(self.n_views))
        elif isinstance(cameras, int):
            cameras = [cameras]
        return cvt_torch(cameras).long() % self.n_views
    
    def _rasterize(self, cameras):
        if self._rast is None: # FIXME: rasterisation of all cameras is not necessary
            verts_world = torch.nn.functional.pad(self.mesh.v, (0,1), value=1) @ self.o2w.transpose(-1,-2)
            self._vpos = self.camera.transform_verts_to_ndc(verts_world)
            self._rast = self.camera.rasterize(verts_world, self.mesh.f, (self.render_res*self.ssaa, self.render_res*self.ssaa,), self.ctx, tile_size=self.tile_size)
        return self._rast[0][cameras].contiguous(), self._rast[1][cameras].contiguous()
    
    def _texlookup(self, cameras):
        if self._texlu is None: # FIXME: interpolation of all cameras is not necessary
            rast, rast_db = self._rasterize(self._camera_list(None))
            n_views = self.n_views
            self._texlu = dr.interpolate(self.mesh.vt.expand(n_views,-1,-1).contiguous(), rast, self.mesh.ft.contiguous(), rast_db=rast_db, diff_attrs='all')
        return self._texlu[0][cameras].contiguous(), self._texlu[1][cameras].contiguous()
    
    def _aa(self, color, cameras):
        if self.camera.support_aa:
            rast, rast_db = self._rasterize(cameras)
            return dr.antialias(color, rast, self._vpos[cameras], self.mesh.f.contiguous())
        else:
            return color
    
    def _ssaa(self, color):
        return color.reshape(len(color), self.render_res, self.ssaa, self.render_res, self.ssaa, -1).mean((2,4))
            
    def sample_texture(self, textures, max_mip_level=4, antialias=True, ssaa=True, cameras=None):
        '''
        render images from given textures
        
        Inputs:
            - textures: shape [(1 or views,) tex_res, tex_res, tex_channels], numpy or torch
        Returns: 
            - color: [views, render_res, render_res, tex_channels], torch
            - alpha: [views, render_res, render_res, 1], torch float values in [0,1]
        '''
        
        cameras = self._camera_list(cameras)
        rast, rast_db = self._rasterize(cameras)
        texc, texc_da = self._texlookup(cameras)
        
        n_views = len(cameras)
        tex_data = cvt_torch(textures).expand(n_views,-1,-1,-1).contiguous()
        
        if max_mip_level is None or max_mip_level > 0:
            color = dr.texture(tex_data, texc, texc_da, filter_mode='auto', max_mip_level=max_mip_level)
        else:
            color = dr.texture(tex_data, texc, filter_mode='linear')
            
        color = torch.where(rast[...,-1:]>0, color, torch.zeros_like(color)) # zero background
        
        if antialias:
            color = self._aa(color, cameras)
            
        alpha = (rast[...,-1:] > 0).float()
        
        if ssaa:
            color = self._ssaa(color)
            alpha = self._ssaa(alpha)

        return color, alpha
    
    def sample_verts(self, textures, faces=None, antialias=True, ssaa=True, cameras=None):
        '''
        render images from per-vertex textures
        
        Inputs:
            - textures: shape [(1 or views,) n_verts, tex_channels], numpy or torch
            - faces [optional]: if provided, returned colors will be interpolated from faces as defined, must
              be an integer tensor of shape [n_faces, 3] whose values are in range [0, n_verts)
        Returns: 
            - color: [views, render_res, render_res, tex_channels], torch
            - alpha: [views, render_res, render_res, 1], torch float values in [0,1]
        '''
        
        cameras = self._camera_list(cameras)
        rast, rast_db = self._rasterize(cameras)
        
        if faces is None:
            faces = self.mesh.f.contiguous()
        else:
            faces = cvt_torch(faces).contiguous()
        
        n_views = len(cameras)
        tex_data = cvt_torch(textures).expand(n_views,-1,-1)
        
        color, _ = dr.interpolate(tex_data.contiguous(), rast, faces, rast_db=rast_db, diff_attrs='all')
        
        if antialias:
            color = self._aa(color, cameras)
        
        alpha = (rast[...,-1:] > 0).float()
        
        if ssaa:
            color = self._ssaa(color)
            alpha = self._ssaa(alpha)

        return color, alpha
               
    def sample_faces(self, textures, antialias=True, ssaa=True, cameras=None):
        '''
        render images from per-face textures
        
        Inputs:
            - textures: shape [(1 or views,) n_faces, tex_channels], numpy or torch
        Returns: 
            - color: [views, render_res, render_res, tex_channels], torch
            - alpha: [views, render_res, render_res, 1], torch float values in [0,1]
        '''
        
        cameras = self._camera_list(cameras)
        rast, rast_db = self._rasterize(cameras)
        
        n_views = len(cameras)
        n_channels = textures.shape[-1]
        tex_data = cvt_torch(textures).expand(n_views,-1,-1) # [n_views, n_faces, n_channels]
        
        faces_id = torch.round(rast[...,-1]).long().reshape(n_views, -1, 1).expand(-1,-1,n_channels) # [n_views, res*res, n_channels]
        color = torch.gather(torch.nn.functional.pad(tex_data, (0,0,1,0), value=0), dim=1, index=faces_id).reshape(n_views, self.render_res*self.ssaa, self.render_res*self.ssaa, n_channels) # [n_views, res*res, n_channels]
        
        if antialias:
            color = self._aa(color, cameras)
        
        alpha = (rast[...,-1:] > 0).float()

        if ssaa:
            color = self._ssaa(color)
            alpha = self._ssaa(alpha)

        return color, alpha
    
    @staticmethod
    def _voronoi_inpaint(texture, mask):
        texture = cvt_torch(texture)
        mask = cvt_torch(mask)
        
        inpainted_texture = []
        
        for tex, msk in zip(texture, mask):
            assert msk.sum(), "keep region cannot be empty"
            if not torch.all(msk):
                tex = voronoi_solve(tex, msk.squeeze(-1))
            inpainted_texture.append(tex)
        
        return torch.stack(inpainted_texture, dim=0)
    
    @staticmethod
    def _nn_inpaint(texture, index, keep_mask, inpaint_mask, k=0, hardness=1e5, max_n_keys=None):
        texture = cvt_torch(texture)
        index = cvt_torch(index)
        keep_mask = cvt_torch(keep_mask).squeeze(-1)
        inpaint_mask = cvt_torch(inpaint_mask).squeeze(-1)
        
        inpainted_texture = []
        
        for tex, idx, kep_msk, inp_msk in zip(texture, index, keep_mask, inpaint_mask):
            tex = tex.clone()
            assert kep_msk.sum(), "keep region cannot be empty"
            if inp_msk.sum():
                if max_n_keys and kep_msk.sum() > max_n_keys:
                    kep_msk = kep_msk.clone()
                    indices = torch.randperm(kep_msk.sum())[:max_n_keys].to(dtype=torch.long, device=kep_msk.device)
                    kep_msk_ = torch.zeros_like(kep_msk[kep_msk])
                    kep_msk_[indices] = True
                    kep_msk[kep_msk.clone()] = kep_msk_
                tex[inp_msk] = nn_solve(idx[inp_msk], idx[kep_msk],tex[kep_msk], k=k, hardness=hardness)
            inpainted_texture.append(tex)
        
        return torch.stack(inpainted_texture, dim=0)
    
    def uv_tex_to_verts(self, uv_tex):
        '''
        convert uv textures to vertex texture
        
        Inputs:
            - uv_tex: tensor of shape [batch, tex_res, tex_res, channel]
        
        Returns:
            - vert_tex: tensor of shape [batch, n_verts, channel]
        '''
        batch, *_, channels = uv_tex.shape
        n_verts = self.mesh.v.shape[1]
        uv_verts_tex = torch.nn.functional.grid_sample(uv_tex.permute(0,3,1,2), self.mesh.vt.unsqueeze(dim=1)*2-1, align_corners=False).squeeze(dim=2).permute(0,2,1) # [batch, n_uv_verts, channel]
        
        face_verts_tex = uv_verts_tex[:,self.mesh.ft.flatten()] # [batch, n_faces*3, channels]
        verts_tex = torch.zeros((batch, n_verts, channels), dtype=uv_tex.dtype, device=uv_tex.device)
        verts_tex_count = torch.zeros((batch, n_verts, 1), dtype=uv_tex.dtype, device=uv_tex.device)
        
        verts_tex.index_add_(dim=1, index=self.mesh.f.flatten(), source=face_verts_tex)
        verts_tex_count.index_add_(dim=1, index=self.mesh.f.flatten(), source=torch.ones_like(face_verts_tex[...,:1]))
        
        return verts_tex / verts_tex_count
        
    def verts_tex_to_uv(self, verts_tex):
        '''
        convert vertex textures to uv texture
        
        Inputs:
            - verts_tex: tensor of shape [batch, n_verts, channel]
        
        Returns:
            - uv_tex: tensor of shape [batch, tex_res, tex_res, channel]
        '''
        rast_uv, _ = self._rasterize_uv()
        uv_tex, _ = dr.interpolate(verts_tex.contiguous(), rast_uv, self.mesh.f.contiguous()) # [1, tex_res, tex_res, 3]
        uv_tex = self._voronoi_inpaint(uv_tex, rast_uv[...,-1:]>0)
        return uv_tex
        
        
    def _vertex_inpaint(self, textures, keep_mask, inpaint_mask ):
        '''
        inpaint vertex colors and convert to uv texture
        - verts_tex: [B, n_verts, c]
        - verts_mask: [B, n_verts], which verts are known (rest get inpainted)
        
        returns uv textures of shape [B, tex_res, tex_res, C]
        '''
        
        textures_rgba = torch.cat([textures, keep_mask], dim=-1) # [B, tex_res, tex_res, C+1]
        verts_tex_rgba = self.uv_tex_to_verts(textures_rgba) # [B, n_verts, C+1]
        verts_tex_alpha = verts_tex_rgba[...,-1] # [B, n_verts]
        verts_tex = verts_tex_rgba[...,:-1] / verts_tex_rgba[...,-1:] #  [B, n_verts, C]
        valid_verts_mask = (verts_tex_alpha > 0.99)
        
        # import trimesh
        # vertex_color = torch.where(valid_verts_mask.unsqueeze(-1), verts_tex, torch.ones_like(verts_tex))
        # _ = trimesh.Trimesh(self.mesh.v[0].cpu().numpy(), self.mesh.f.cpu().numpy(), vertex_colors=vertex_color[0].cpu().clip(0,1).numpy()*255).export("tmp.obj")
        
        
        verts_tex_inpainted = torch.stack([laplace_solve(self.mesh.f, vt, vm) for vt,vm in zip(verts_tex,valid_verts_mask)], dim=0)
        
        # _ = trimesh.Trimesh(self.mesh.v[0].cpu().numpy(), self.mesh.f.cpu().numpy(), vertex_colors=verts_tex_inpainted[0].cpu().clip(0,1).numpy()*255).export("tmp.obj")
        
        inpaint_tex = self.verts_tex_to_uv(verts_tex_inpainted)
        textures = textures.clone()
        textures[inpaint_mask.squeeze(-1)] = inpaint_tex[inpaint_mask.squeeze(-1)]
        
        return textures
        
    
    def inpaint_textures(self, textures, inpaint_mask, k=0, hardness=1e5, inpaint_method="knn"):
        '''
        inpaint missing textels by spatial interpolation and pad texture islands, where inpain_mask is True
        
        inputs:
            - textures: numpy or torch tensor of shape [views, tex_res, tex_res, tex_channels]
            - inpaint_mask: numpy or torch tensor of shape [views, tex_res, tex_res, 1], boolean typed, regions to inpaint
            - k : k nearest neightbor used for inpaint, if k<=0, all points are used
            - hardness: scalar that controls how hard missing values are interpolated, e.g. with sufficiently large hardness this becomes nearest neighbor inpaint, if hardness=0 all k neighbors have equal contribution
            - inpaint_method: "knn" or "laplace"
        returns
            - textures: tensor of shape [views, tex_res, tex_res, tex_channels]
        '''
        
        assert inpaint_method in ("knn", "laplace"), f"unknown inpaint method '{inpaint_method}', expected 'knn' or 'laplace'"
        
        rast_uv, _ = self._rasterize_uv()
        xyz_uv, _ = dr.interpolate(self.mesh.v.contiguous(), rast_uv, self.mesh.f.contiguous()) # [1, tex_res, tex_res, 3]
        
        textel_mask = (rast_uv[...,-1:] > 0) # [1, tex_res, tex_res, 1]
        textel_inpaint_mask = torch.logical_and(inpaint_mask, textel_mask)
        textel_keep_mask = torch.logical_and(torch.logical_not(inpaint_mask), textel_mask)
        non_textel_inpaint_mask = torch.logical_and(inpaint_mask, torch.logical_not(textel_mask))
        
        if inpaint_method == "knn":
            textures = self._nn_inpaint(textures, xyz_uv, textel_keep_mask, textel_inpaint_mask, k, hardness, max_n_keys=500000)
        elif inpaint_method == "laplace":
            textures = self._vertex_inpaint(textures, textel_keep_mask, textel_inpaint_mask)
        
        textures[non_textel_inpaint_mask.squeeze(-1)] = self._voronoi_inpaint(textures, textel_mask)[non_textel_inpaint_mask.squeeze(-1)]
        
        return textures
        
    
    def bake_textures(self, in_images, max_mip_level=4, antialias=True, ssaa=True, epsilon=1e-8, cameras=None, return_contribution=False, inpaint=False):
        '''
        in_images: shape [views, render_res, render_res, tex_channels]
        
        return 
            - textures shape [views, tex_res, tex_res, tex_channels]
            - if return_contribution: weights shape [views. tex_res, tex_res, 1] total contribution of each textel to all pixels
        '''
        cameras = self._camera_list(cameras)
        n_views = len(cameras)
        
        n_channels = in_images.shape[-1]
        in_images = cvt_torch(in_images)
        
        with torch.enable_grad():
            textures = torch.zeros((n_views, self.tex_res, self.tex_res, n_channels+1), dtype=torch.float32, device="cuda", requires_grad=True)
            images, mask = self.sample_texture(textures, max_mip_level, antialias, ssaa, cameras)
            (-(images - torch.nn.functional.pad(in_images, (0,1), value=1))**2).sum().backward()
            textures, weights = textures.grad[...,:-1], textures.grad[...,-1:]
        
        textures = textures.detach()
        weights = weights.detach()
        
        textures = textures / (weights + epsilon)
        
        if inpaint:
            textures = self._voronoi_inpaint(textures, weights)
        
        if return_contribution:
            return textures, weights
        else:
            return textures
        
    
    def bake_textures_raycast(self, in_images, epsilon=1e-4, cameras=None, interpolation="bilinear", return_contribution=False, inpaint=False):
        '''
        in_images: shape [views, render_res, render_res, tex_channels]
        
        interpolation can be "nearest" or "bilinear" or "bicubic"
        
        return 
            - textures shape [views, tex_res, tex_res, tex_channels]
            - if return_contribution: weights shape [views. tex_res, tex_res, 1] whether or not textel is visible in view, values are either 0 or 1 floating point
        '''
        cameras = self._camera_list(cameras)
        n_views = len(cameras)
        
        n_channels = in_images.shape[-1]
        in_images = cvt_torch(in_images)
        
        rast_uv, _ = self._rasterize_uv()
        verts_obj = torch.nn.functional.pad(self.mesh.v, (0,1), value=1)
        verts_world = verts_obj @ (self.o2w).transpose(-1,-2)
        verts_world = verts_world[...,:3] / verts_world[...,-1:]
        xyz_uv, _ = dr.interpolate(verts_world.contiguous(), rast_uv, self.mesh.f.contiguous()) # [1, tex_res, tex_res, 3]
        
        textel_mask = (rast_uv[...,-1] > 0) # [1, tex_res, tex_res]
        textel_xyz = xyz_uv[textel_mask] # [n_textel, 3]
        verts_ndc = self.camera.transform_verts_to_ndc(verts_world)[cameras] # [n_views, n_verts, 4]
        verts_ndc = verts_ndc[...,:3] / verts_ndc[...,-1:] # [n_views, v_verts, 3]
        
        textel_ndc = self.camera.transform_verts_to_ndc(textel_xyz) # [all_views, n_textels, 4]
        textel_screen = self.camera.project_onto_screen(textel_ndc, is_ndc=True) # [all_views, n_textels, 2]
        ray_ndc = self.camera.get_ndc_ray_dirs(textel_ndc)[cameras]  # [n_views, n_textels, 4]
        textel_screen = textel_screen[cameras] # [n_views, n_textels, 2]
        textel_ndc = textel_ndc[cameras] # [n_views, n_textels, 4]
        textel_ndc = textel_ndc[...,:3] / textel_ndc[...,-1:] # [n_views, n_textels, 3]
        textel_rgb = torch.nn.functional.grid_sample( \
                in_images.permute(0,3,1,2),
                textel_screen.unsqueeze(dim=1),
                mode=interpolation, padding_mode="border", align_corners=False).squeeze(2).transpose(1,2) # [n_views, n_textels, n_channels]
        
        textel_occluded = []
        
        for i in range(n_views):
            v_ndc = verts_ndc[i]
            rd_ndc = ray_ndc[i] 
            ro_ndc = textel_ndc[i] + rd_ndc * epsilon
            
            occ = torch.from_numpy(ray_occluded(*cvt_numpy(v_ndc, self.mesh.f, ro_ndc, rd_ndc))).to(device=in_images.device) # [n_textels]
            textel_occluded.append(occ)
        
        textel_occluded = torch.stack(textel_occluded) # [n_views, n_textels]
        
        textures = torch.zeros((n_views, self.tex_res, self.tex_res, n_channels), dtype=torch.float32, device="cuda")
        visible = torch.zeros((n_views, self.tex_res, self.tex_res, 1), dtype=torch.float32, device="cuda")
        
        textures[:,textel_mask[0]] = textel_rgb
        visible[:,textel_mask[0]] = (textel_occluded==False).float().unsqueeze(-1)
        textures = textures * visible
        
        if inpaint:
            textures = self._voronoi_inpaint(textures, visible)
        
        if return_contribution:
            return textures, visible
        else:
            return textures 
        
    
    def render_xyz(self, system="world", antialias=True, ssaa=True, cameras=None):
        '''
        renders xyz map
        
        Inputs:
            - system: a string for coordinate system, can be "object", "world", "camera" or "ndc"
        
        Returns:
            - xyz: torch tensor of shape [views, render_res, render_res, 3]
            - mask: torch tensor of shape [views, render_res, render_res, 1]
        '''
        
        assert system in ["object", "world", "camera", "ndc"]
        
        cameras = self._camera_list(cameras)
        
        verts = torch.nn.functional.pad(self.mesh.v, (0,1), value=1)
        
        if system == "world":
            verts = verts @ (self.o2w).transpose(-1,-2)
        elif system == "camera":
            verts = verts @ (torch.inverse(self.camera.c2w[cameras]) @ self.o2w).transpose(-1,-2)
        elif system == "ndc":
            verts = self._vpos[cameras]
            
        verts = verts[...,:3] / verts[...,3:]
        
        return self.sample_verts(verts, antialias=antialias, ssaa=ssaa, cameras=cameras)
    
    def render_depth(self, mode="absolute", normalize=None, bg=0, antialias=True, ssaa=True, cameras=None):
        '''
        renders depth map
        
        Inputs:
            - mode: a string for coordinate system, can be "absolute" for linear depth or "ndc" for clip space depth
            - normalize [optional]: None or a tuple for normalisation range, e.g. (255, 50) will map min and max foreground depth values linearly to 255 and 50, respectively
            - bg [option]: a scalar to set background value to, defaults to 0
            
        Returns:
            - depth: torch tensor of shape [views, render_res, render_res, 1]
            - mask: torch tensor of shape [views, render_res, render_res, 1]
        '''
        
        assert mode in ["absolute", "ndc"]
        
        cameras = self._camera_list(cameras)
        n_views = len(cameras)
        
        rast, _ = self._rasterize(self._camera_list(None)) # FIXME: rasterisation of all cameras is not necessary
        ndc_depth = rast[...,2:3]
        mask = (rast[...,3:4] > 0).float()
        
        if mode == "absolute":
            depth = self.camera.depth_ndc2abs(ndc_depth, mask>0.5)[cameras]
            mask = mask[cameras]
        elif mode == "ndc":
            depth = ndc_depth[cameras]
            mask = mask[cameras]
        else:
            raise ValueError(f"unknown depth mode \"{mode}\"")
        
        if normalize is not None:
            depth[mask < 0.5] = torch.inf # [n_views, res, res, 1]
            min_d = depth.reshape(n_views, -1).min(1)[0].reshape(-1,1,1,1) # [n_views,1,1,1]
            depth[mask < 0.5] = -torch.inf # [n_views, res, res, 1]
            max_d = depth.reshape(n_views, -1).max(1)[0].reshape(-1,1,1,1) # [n_views,1,1,1]
            depth[mask < 0.5] = torch.nan
            
            depth = (normalize[1]-normalize[0]) / (max_d-min_d) * (depth - min_d) + normalize[0]
        
        depth[mask < 0.5] = bg
        
        if antialias:
            depth = self._aa(depth, cameras)
        
        if ssaa:
            depth = self._ssaa(depth)
            mask = self._ssaa(mask)
            
        return depth, mask
    
    def render_normal(self, system="world", mode="vertex", antialias=True, ssaa=True, cameras=None):
        '''
        renders normal map
        
        Inputs:
            - system: a string for coordinate system, can be "object", "world", "camera", "camera_gl"
            - mode: "face" or "vertex", if "face" faces appear flat, if "vertex" normals are smoothed out through interpolation
        
        Returns:
            - normal_img: torch tensor of shape [views, render_res, render_res, 3]
            - mask: torch tensor of shape [views, render_res, render_res, 1]
        '''
        
        assert system in ["object", "world", "camera", "camera_gl"]
        assert mode in ["face", "vertex"]
        
        cameras = self._camera_list(cameras)
        
        if mode == "face":
            normals = self.mesh.fn
        elif mode == "vertex":
            normals = self.mesh.vn
        
        if system == "world":
            normals = normals @ (self.o2w[:3,:3]).transpose(-1,-2)
        elif system == "camera" or system == "camera_gl":
            normals = normals @ (torch.inverse(self.camera.c2w[cameras,:3,:3]) @ self.o2w[:3,:3]).transpose(-1,-2)
            if system == "camera_gl":
                normals = normals * torch.tensor([1,-1,-1], dtype=normals.dtype, device=normals.device) # GL system has +x right, +y up and +z backward
        elif system == "object":
            normals = normals
            
        normals = torch.nn.functional.normalize(normals, dim=-1)
        
        if mode == "face":
            normal_img, mask = self.sample_faces(normals, antialias=antialias, ssaa=ssaa, cameras=cameras)
        elif mode == "vertex":
            normal_img, mask = self.sample_verts(normals, antialias=antialias, ssaa=ssaa, cameras=cameras)
        
        return normal_img, mask
    
            
    def render_view_cos(self, mode="vertex", antialias=True, ssaa=True, cameras=None, backface="cull"):
        '''
        render angle cosine between viewing direction and mesh normal
        return view cos as 1-channel image and mask
        '''
        assert backface in ["cull", "flip", "keep"]
        normal_world, mask = self.render_normal("world", mode, antialias, ssaa, cameras)
        cameras = self._camera_list(cameras)
        view_origin, view_dir = self.camera.generate_world_rays((self.render_res, self.render_res))
        view_cos = -(normal_world * view_dir[cameras]).sum(-1, keepdim=True).clip(-1,1)
        

        if backface == "cull":
            return view_cos.clip(0,1), mask
        elif backface == "flip":
            return view_cos.abs(), mask
        elif backface == "keep":
            return view_cos, mask
            
    
    def render_texture_area(self, antialias=True, ssaa=True, cameras=None, inverse=False, return_singulars=False):
        '''
        render images of shape [n_views, image_res, image_res, 1] whose pixel values
        indicate the corresponding uv area relative to the size of entire uv.
        E.g. a value of 1.0 means this pixel spans roughly 1.0 squared pixel on texture map
        
        if inverse=True, return the inverse ratio that is (pixel area) / (uv area)
        
        this function computes jacobian matrix of uv wrt screen coordinate (x,y)
          [[du/dx, du/dy]
           [dv/dx, dv/dy]]
        and return its absolute determinant  
        
        return area image and mask of same shape. 
        if return_singulars=True, return a 3rd tensor of shape [n_views, image_res, image_res, 2]
        where the last dimension contains the absolute singular values of (J^T J) in ascending order, that
        correspond to the min and max lengths that a unit length in screen space maps to in texture space.
        if inverse=True, return inverse of these singular values
        '''
        
        cameras = self._camera_list(cameras)
        rast, _ = self._rasterize(cameras)
        mask = (rast[...,-1:]>0).float()
        _, texc_da = self._texlookup(cameras) # [n_views, image_res, image_res, 4]
        
        jacobian_uv_xy = texc_da.reshape(*texc_da.shape[:-1],2,2) * self.tex_res
        area_img = torch.linalg.det(jacobian_uv_xy).unsqueeze(-1).abs()
        
        if inverse:
            area_img = torch.where(mask > 0, 1/(area_img+1e-10), torch.zeros_like(area_img))   
        else:
            area_img = torch.where(mask > 0, area_img, torch.zeros_like(area_img))
        
        if antialias:
            area_img = self._aa(area_img, cameras)
        
        if ssaa:
            area_img = self._ssaa(area_img)
            mask = self._ssaa(mask)
        
        if return_singulars:
            eigen_img = torch.linalg.eigvalsh(jacobian_uv_xy.transpose(-2, -1) @ jacobian_uv_xy)
            if inverse:
                singular_img = torch.where(mask > 0, eigen_img**-0.5, torch.zeros_like(eigen_img))
            else:
                singular_img = torch.where(mask > 0, eigen_img**0.5, torch.zeros_like(eigen_img))
            
            if antialias:
                singular_img = self._aa(singular_img, cameras)
            if ssaa:
                singular_img = self._ssaa(singular_img)
    
            return area_img, mask, singular_img

        else:
            return area_img, mask
        
        
    def export_mesh(self, path, texture, val_range=(0,1), orientation="object"):
        '''
        exports mesh to obj/glb files, requires pip install kiui
        
        Inputs:
            - path: path to export to
            - texture: numpy or pytorch tensor of shape [H,W,1/3/5] or [1,H,W,1/3/5] or [H,W] or [1,H,W], with H>1 and W>1
                - if 1-channeled, gray scale albedo
                - if 3-channeled, RGB albedo
                - if 5-channeled, RGB + roughness + metallic in that order
            - val_range: valid range of texture values, e.g. (0,1) or (0,255), or (-1,1) for normal
            - orientation: a string representing coordinate system orientation of exported mesh, 
              e.g. "camera", "world", "y-up", "z-up" or FLU convention string
        '''
        
        from kiui.mesh import Mesh as _M
        
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        texture = cvt_torch(texture).squeeze() # [H,W] or [H,W,3/5]
        
        if texture.ndim == 2:
            texture = texture.unqueeze(-1).expand(-1,-1,3)
            
        assert texture.ndim == 3, f"unsupported texture shape {list(texture.shape)}, expected [H,W,3/1] or [1,H,W,3/1] or [H,W] or [1,H,W]"
        
        texture = ((texture - val_range[0]) / (val_range[1] - val_range[0])).clip(0,1)
        
        if texture.shape[-1] == 5:
            albedo, roughness, metallic = torch.split(texture, (3,1,1), dim=-1)
            rm = torch.cat((torch.ones_like(roughness), roughness, metallic), dim=-1)
        else:
            albedo = texture
            rm = None
            
        if orientation == "object":
            transform = torch.eye(4, device=self.o2w.device, dtype=self.o2w.dtype)
        elif orientation == "world":
            transform = self.o2w
        else:
            transform = get_convention(orientation, device=self.o2w.device, homogeneous=True) @ torch.inverse(get_convention(self.mesh.orientation, device=self.o2w.device, homogeneous=True))
        
        v = torch.nn.functional.pad(self.mesh.v[0], (0,1), value=1) @ transform.transpose(0,1)
        v = v[...,:3] / v[...,-1:]
        
        mesh = _M(v=v, f=self.mesh.f, vt=self.mesh.vt[0], ft=self.mesh.ft, albedo=albedo, metallicRoughness=rm)
        mesh.write(path)
    
        
    @staticmethod
    def overwrite_mtl(path, texture, val_range=(0,1), texture_prefix="texture", default_material_name="raw_mesh"):
        '''
        overwrites an existing mtl file with input textures
        
        Inputs:
            - path: path to existing mtl file
            - texture: numpy or pytorch tensor of shape [H,W,1/3/5] or [1,H,W,1/3/5] or [H,W] or [1,H,W], with H>1 and W>1
                - if 1-channeled, gray scale albedo
                - if 3-channeled, RGB albedo
                - if 5-channeled, RGB + roughness + metallic in that order
            - val_range: valid range of texture values, e.g. (0,1) or (0,255), or (-1,1) for normal
            - texture_prefix: prefix of texture map paths, e.g. if "path/to/XXX", then albedo will be saved to "path/to/XXX_Kd.png"
        '''
        
        texture = cvt_torch(texture).squeeze()  # [H, W] or [H, W, 3/5]
        
        if texture.ndim == 2:
            texture = texture.unsqueeze(-1).expand(-1, -1, 3) 
        
        assert texture.ndim == 3, f"unsupported texture shape {list(texture.shape)}, expected [H,W,3/1] or [1,H,W,3/1] or [H,W] or [1,H,W]"
        
        texture = ((texture - val_range[0]) / (val_range[1] - val_range[0])).clip(0, 1)
        
        albedo_path = f"{texture_prefix}_Kd.png"
        roughness_path = f"{texture_prefix}_Pr.png"
        metallic_path = f"{texture_prefix}_Pm.png"
        
        if texture.shape[2] == 1 or texture.shape[2] == 3:  # Albedo only
            albedo_texture = cvt_numpy(texture[..., :3] * 255).astype(np.uint8)
            cv2.imwrite(albedo_path, cv2.cvtColor(albedo_texture, cv2.COLOR_RGB2BGR))
            has_rm = False
        elif texture.shape[2] == 5:  # Albedo, roughness, metallic
            albedo_texture = cvt_numpy(texture[..., :3] * 255).astype(np.uint8)
            roughness_texture = cvt_numpy(texture[..., 3] * 255).astype(np.uint8)
            metallic_texture = cvt_numpy(texture[..., 4] * 255).astype(np.uint8)
            
            cv2.imwrite(albedo_path, cv2.cvtColor(albedo_texture, cv2.COLOR_RGB2BGR))
            cv2.imwrite(roughness_path, roughness_texture)
            cv2.imwrite(metallic_path, metallic_texture)
            has_rm = True
        else:
            raise ValueError(f"Unsupported number of channels: {texture.shape[2]}. Expected 3 or 5.")

        try:
            material_name = default_material_name
            
            with open(path, 'r') as mtl_file:
                lines = mtl_file.readlines()
                
            for line in lines:
                if line.startswith("newmtl"):
                    material_name = line.split()[1]
                    break
        except:
            lines = None
            
        # Write updated .mtl file
        new_mtl_content = (
            f"# Mesh & texture generated by Pandora-X" + "\n" +
            f"newmtl {material_name}" + "\n" + 
            f"map_Kd {os.path.basename(albedo_path)}" + "\n" +
            (f"map_Pr {os.path.basename(roughness_path)}""\n"f"map_Pm {os.path.basename(metallic_path)}""\n" if has_rm else "Ns 250.000000\nKs 0.500000 0.500000 0.500000\nNi 1.450000\n") +
            ""# "\n#==== original file below ====\n"+("  #".join(l for l in lines)) if lines else ""
        )
        with open(path, 'w+') as mtl_file:
            mtl_file.write(new_mtl_content)
    
    def select_k_views_for_inpaint(self, inpaint_tex_mask, k, cameras=None, return_coverage=False): 
        '''
        
        select k views that maximise view coverage of textels to be inpainted. 
        this translates to a max k-cover problem that is known NP-hard, so a greedy strategy is applied that
        iteratively selects the next view that sees most remaining textels not covered by previous views
        
        Inputs: 
            - inpaint_tex_mask: boolean tensor of shape [n_views, tex_res, tex_res] or [1, tex_res, tex_res] or [tex_res, tex_res]
            - k: integer, 0 < k < n_views, number of views to return
            - cameras: list of n_views selecting which cameras the input mask corresponds to, default to all cameras
            - return_coverage: if true, return a mask of shape [n_views, tex_res, tex_res] indicating which textels are
              visible from which views
              
        Returns:
            - cams: a list of length k with integer values in [0, n_total_cameras)
            - if return_coverage: return another torch boolean tensor of shape [n_views, tex_res, tex_res] indicating 
              which textels are visible from which views
        '''
        
        cameras = self._camera_list(cameras)
        n_views = len(cameras)
        
        rast_uv, _ = self._rasterize_uv()
        vtx_world = transform_homogeneous(self.mesh.v, self.o2w)
        xyz_uv, _ = dr.interpolate(vtx_world, rast_uv, self.mesh.f.contiguous()) # [1, tex_res, tex_res, 3]
        xyz_uv = self._voronoi_inpaint(xyz_uv, rast_uv[...,-1:]>0).squeeze(0) # [tex_res, tex_res, 3]
        
        mesh_verts_world_np = vtx_world[0].cpu().numpy() # [v_verts, 3]
        mesh_faces_world_np = self.mesh.f.cpu().numpy() # [v_verts, 3]
        
        if inpaint_tex_mask.ndim == 2:
            inpaint_tex_mask = inpaint_tex_mask.unsqueeze(0)
        inpaint_tex_mask = inpaint_tex_mask.expand(n_views, -1,-1) # [n_views, tex_res, tex_res]
        
        visible_inpaint_mask = torch.zeros_like(inpaint_tex_mask) # [n_views, tex_res, tex_res]
        
        embree_scene = None
        
        for i in range(n_views):
            i_view = cameras[i]
            inpaint_mask = inpaint_tex_mask[i] # [tex_res, tex_res]
            xyz_world = xyz_uv[inpaint_mask] # [n_pts, 3]
            xyz_ndc = transform_homogeneous(xyz_world, self.camera.mvp[i_view]) # [n_pts, 3]
            dir_ndc = self.camera.get_ndc_ray_dirs(xyz_ndc) # [n_pts, 3]
            rayt_ndc = xyz_ndc + dir_ndc # [n_pts, 3]
            rayt_world = transform_homogeneous(rayt_ndc, torch.inverse(self.camera.mvp[i_view])) # [n_pts, 3]
            rayd_world = torch.nn.functional.normalize(rayt_world - xyz_world, dim=1) # [n_pts, 3]
            
            occ, embree_scene = ray_occluded(mesh_verts_world_np, mesh_faces_world_np, xyz_world.cpu().numpy(), rayd_world.cpu().numpy(), embree_scene, return_scene=True)
            vis = torch.from_numpy(occ==False).to(device=inpaint_tex_mask.device) # [n_pts]
            
            vis_mask = visible_inpaint_mask[i]
            vis_mask[inpaint_mask] = vis
            visible_inpaint_mask[i] = vis_mask
        
        ret_k_views = []
        curr_visible_mask = torch.zeros_like(inpaint_tex_mask[0]) # [tex_res, tex_res]
        
        # greedily solves max k-cover
        for _ in range(k):
            
            new_visible = torch.logical_and(visible_inpaint_mask, torch.logical_not(curr_visible_mask)) # [n_views, tex_res, tex_res]
            new_visible_cnt = new_visible.sum(dim=(1,2)) # [n_views]
            i = torch.argmax(new_visible_cnt)
            i_view = cameras[i]
            
            ret_k_views.append(int(i_view))
            curr_visible_mask = torch.logical_or(curr_visible_mask, visible_inpaint_mask[i])
        
        return ret_k_views
            
    def get_uv_mask(self):
        '''
        returns a bool mask of shape [tex_res, tex_res] for texture islands pixels
        '''    
        return (self._rasterize_uv()[0][0,...,-1] > 0)
            
           