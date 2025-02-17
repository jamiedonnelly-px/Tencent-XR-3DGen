import sys
sys.path.insert(0,'/aigc_cfs_2/weimao/avatar_face_generation/')
import torch
import trimesh
from PIL import Image
from ipdb import set_trace as st
import numpy as np
import nvdiffrast
import nvdiffrast.torch as dr
from render_mesh import make_ndc, render_mesh_verts_tex, render_mesh
from utils import *
from torch import nn

def cross_prod(a, b):
    a = torch.cat([a,torch.zeros_like(a[...,:1])],dim=-1)
    b = torch.cat([b,torch.zeros_like(b[...,:1])],dim=-1)
    return torch.cross(a, b, dim=-1)

def barycentric_coordinates(points, triangles):
    # Points: (N, 2)
    # Triangles: (M, 3, 2)
    # sarea pbc/abc, apc/abc, abp/abc
    # get triangle area
    st()
    abc = cross_prod(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])[...,-1]/2 # m
    pbc = []
    apc = []
    for i in range(points.shape[0]):
        pbc.append(cross_prod(triangles[:,1]-points[i:i+1], triangles[:,2]-points[i:i+1])[...,-1][None]/2) # n x m
        apc.append(cross_prod(points[i:i+1]-triangles[:,0], triangles[:,2]-triangles[:,0])[...,-1][None]/2) # n x m
        # abp = cross_prod(triangles[:,1]-triangles[:,0], points[:,None]-triangles[None,:][:,:,0])[...,-1]/2 # n x m
    pbc = torch.cat(pbc,dim=0)
    apc = torch.cat(apc, dim=0)
    w1 = pbc / (abc[None] + 1e-10)
    w2 = apc / (abc[None] + 1e-10)
    w3 = 1 - w2 - w1

    return torch.cat([w1[...,None],w2[...,None],w3[...,None]],dim=-1)  # Shape: (N, M, 3)

quest_tex_dir = '/aigc_cfs_2/weimao/avatar_face_generation/data/quest_head_model/quest_head.png'
quest_head_dir = "/aigc_cfs_2/weimao/avatar_face_generation/data/quest_head_model/quest_head.obj"

head_mesh = trimesh.load(quest_head_dir)
verts = head_mesh.vertices
faces = head_mesh.faces
uvs = head_mesh.visual.uv
face_uvs = torch.from_numpy(np.concatenate([uvs[faces[:,0]][:,None],
                                            uvs[faces[:,1]][:,None], 
                                            uvs[faces[:,2]][:,None]],axis=1))

tex_img = Image.open(quest_tex_dir)
tex_img = torch.from_numpy(np.array(tex_img)/255).float().cuda()
tex_res = tex_img.shape[0]

zoom = 7.0
Pndc = make_ndc(zoom=zoom, camera_type="ortho")
Pndc = torch.from_numpy(Pndc).float().cuda()
cam2world = torch.eye(4)[None].float().cuda()
rot = angaxe2rot(np.pi, np.array([1,0,0.]))
cam2world[:,:3,:3] = torch.from_numpy(rot[None]).float().cuda()
cam2world[:,1,3] = (verts.max(axis=0)[1] + verts.min(axis=0)[1])/2
cam2world[:,2,3] = 2.0

sz = 1024
# render rgb
quest_tex = tex_img[np.arange(-1, -1-tex_img.shape[0],-1)]
quest_tv = torch.from_numpy(verts).float().cuda()
quest_faces_torch = torch.from_numpy(faces.astype(np.int32)).cuda()
quest_uvs_torch = torch.from_numpy(uvs).float().cuda()
img_quest, alpha_quest = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=quest_tv, 
                                        f=quest_faces_torch, vt=quest_uvs_torch, ft=quest_faces_torch, tex=quest_tex)
img_quest = img_quest[0]
alpha_quest = alpha_quest[0,:,:,0]
img = Image.fromarray(np.uint8(img_quest.cpu().data.numpy()*255))
img.save(f'../output/render_quest.png')

# grid_v, grid_u = torch.where(tex_img[:,:,3] > 128)
# grid_uv = torch.cat([grid_u[...,None],grid_v[...,None]],dim=-1).reshape([-1,2])
# grid_uv = grid_uv / (tex_img.shape[0]-1)
# bary_centric_coord = barycentric_coordinates(grid_uv, face_uvs)

# sz_y, sz_x = tex_img.shape
# grid_y, grid_x = torch.meshgrid(torch.arange(sz_y), torch.arange(sz_x), indexing='ij')
# uv_grid = torch.cat([grid_x[...,None],grid_y[...,None]], dim=-1).reshape([-1,2])

verts = torch.from_numpy(verts).float().cuda()
uvs = torch.from_numpy(uvs).float().cuda()
faces = torch.from_numpy(faces).to(dtype=torch.int32).cuda()
ctx = dr.RasterizeCudaContext(device=verts.device)

verts_u, verts_v = uvs[:,0][None,:,None], 1-uvs[:,1][None,:,None] #torch.split(uvs, 1, dim=-1) # [1, verts, 1]
verts_uv_ndc = torch.cat([verts_u*2-1, verts_v*2-1, torch.zeros_like(verts_u), torch.ones_like(verts_u)], dim=-1)
rast_tex, _ = dr.rasterize(ctx, verts_uv_ndc.contiguous(), faces, (tex_res, tex_res,))
mask = rast_tex[0,:,:,-1] > 0
img = Image.fromarray(np.uint8(mask.cpu().data.numpy()*255))
img.save('../output/img.png')
mask = mask.float().cpu().data.numpy()
uv_tex, _ = dr.interpolate(verts.contiguous()[None], rast_tex, faces) # [1, tex_res, tex_res, 3]


v_cam = torch.matmul(F.pad(uv_tex.reshape(-1,3), pad=(0, 1), mode='constant', value=1.0), torch.inverse(cam2world).transpose(-1,-2)).float()
v_clip = v_cam @ Pndc.transpose(-1,-2)
v_clip = v_clip.reshape([uv_tex.shape[1],uv_tex.shape[1],-1])[:,:,:2]
color_from_face = nn.functional.grid_sample(img_quest.permute([2,0,1])[None], v_clip[None])[0].permute(1,2,0) # [nv_mp, 3]
tex_pts_img = Image.fromarray(np.uint8(color_from_face.cpu().data.numpy()*255))
tex_pts_img.save('../output/reconstruction.png')
st()
tex_pts = uv_tex[0].cpu().data.numpy()
np.savez_compressed('/aigc_cfs_2/weimao/avatar_face_generation/data/quest_head_model/xyz_texmap.npz',xyz_tex=tex_pts, mask=mask)
# vmin = verts.cpu().data.numpy().min(axis=0)
# vmax = verts.cpu().data.numpy().max(axis=0)
# tex_pts_img = np.uint8((tex_pts - vmin) / (vmax - vmin).max() * 255)
# tex_pts_img = Image.fromarray(tex_pts_img)
# tex_pts_img.save('../output/pts_tex.png')

print('done')