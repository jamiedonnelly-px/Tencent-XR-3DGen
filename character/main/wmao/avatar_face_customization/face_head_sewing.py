import trimesh
import numpy as np
from ipdb import set_trace as st
import os
import open3d as o3d
import argparse
import os
import sys
from PIL import Image, ImageFilter, ImageOps
import torch
from torch.nn import functional as F
import torch.utils.checkpoint
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import cv2
import matplotlib.pyplot as plt
import time
import shutil
from render_mesh import make_ndc, render_mesh_verts_tex, render_mesh
import xatlas
import glob
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import torchvision.transforms as T
from tqdm.auto import tqdm

def get_graph_from_faces(face):
    nv = np.max(face) + 1
    graph = np.zeros([nv,nv])
    for f in face:
        v1, v2, v3 = f[0], f[1], f[2]
        graph[v1,v2] = 1
        graph[v2,v1] = 1
        graph[v2,v3] = 1
        graph[v3,v2] = 1
        graph[v1,v3] = 1
        graph[v3,v1] = 1
    return graph

def cross_prod(a, b):
    a = torch.cat([a,torch.zeros_like(a[...,:1])],dim=-1)
    b = torch.cat([b,torch.zeros_like(b[...,:1])],dim=-1)
    return torch.cross(a, b, dim=-1)

def barycentric_coordinates(points, triangles):
    # Points: (N, 2)
    # Triangles: (M, 3, 2)
    # sarea pbc/abc, apc/abc, abp/abc
    # get triangle area
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

def get_edges_with_sobel(mask):
    # Define Sobel filters for horizontal and vertical edges
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0).to(device=mask.device,dtype=mask.dtype)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0).to(device=mask.device,dtype=mask.dtype)
    
    # Ensure mask is a 4D tensor (batch_size, channels, height, width)
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0).unsqueeze(0).float()

    # Convolve the mask with Sobel filters
    edges_x = F.conv2d(mask, sobel_x, padding=1)
    edges_y = F.conv2d(mask, sobel_y, padding=1)

    # Compute edge magnitude
    edges = torch.sqrt(edges_x**2 + edges_y**2)

    return edges

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--face_dir', type=str, default="/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/cute_you11_uv_aligned.obj")
    parser.add_argument('--face_texture_dir', type=str, default="/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/cute_you11_texture_map.png")
    parser.add_argument('--head_dir', type=str, default="/aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_temp_uv_edited.obj")
    parser.add_argument('--head_texture_dir', type=str, default="/aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_temp_uv_edited2.png")
    parser.add_argument('--out_dir', type=str, default="/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/cute_you11_uv_aligned_combined.obj")
    args = parser.parse_args()

    face_dir = args.face_dir
    face_texture_dir = args.face_texture_dir
    head_dir = args.head_dir
    head_texture_dir = args.head_texture_dir
    head_dir_wouv = "/aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_temp_edited_wouv.obj"
    out_dir = args.out_dir
    out_fn = os.path.basename(out_dir).split('.')[0]
    out_path = os.path.dirname(out_dir)
    start_time = time.time()
    max_iter = 500
    
    # load pipeline
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "/aigc_cfs_2/weimao/pretrained_model_cache/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
    )
    pipeline.enable_model_cpu_offload()
    generator = torch.Generator("cuda").manual_seed(92)
    prompt = "bald cartoon head, pure color background, no hair, skin color, no shadow, no shading, no grey color, bright color"
    negative_prompt = "hair, black, shading, shadow, highlight"
    
    
    face_idx = np.array([137, 101, 64, 97, 63, 66, 128, 32, 32, 209, 126, 159, 159, 197, 127, 125, 153, 154, 129, 129, 155, 158, 363, 340, 340, 362, 361, 336, 338, 400, 366, 366, 337, 412, 244, 339, 278, 278, 275, 308, 276, 312])
    head_idx = np.array([82, 79, 78, 13, 84, 11, 10, 1, 0, 9, 16, 18, 17, 15, 64, 65, 67, 66, 70, 68, 85, 92, 262, 251, 249, 247, 248, 246, 245, 198, 200, 201, 199, 192, 183, 184, 193, 194, 261, 196, 257, 260])
    
    face_mesh = trimesh.load(face_dir)
    vert_face = face_mesh.vertices
    face_fa = face_mesh.faces
    face_uv = face_mesh.visual.uv

    head_mesh_wouv = trimesh.load(head_dir_wouv)
    vert_head_wouv = head_mesh_wouv.vertices
    face_head_wouv = head_mesh_wouv.faces # N x 3
    # vert_head_wouv[head_idx] = vert_face[face_idx]

    head_mesh = trimesh.load(head_dir)
    vert_head = head_mesh.vertices
    face_head = head_mesh.faces
    head_uv = head_mesh.visual.uv
    
    cdist = np.linalg.norm(vert_head_wouv[head_idx,None] - vert_head[None],axis=-1)
    idx0, idx1 = np.where(cdist <= 1e-10)
    vert_head[idx1] = vert_face[face_idx[idx0]]
    
    # combine two mesh
    verts = np.concatenate([vert_face, vert_head], axis=0)
    faces = np.concatenate([face_fa, face_head + vert_face.shape[0]], axis=0)
    uvs = np.concatenate([face_uv,head_uv], axis=0)
    xatlas.export(out_dir, verts, faces, uvs)
    # assert False
    # mesh = trimesh.Trimesh(vertices=verts,faces=faces)
    # mesh.export(out_dir)

    # render face image
    zoom = 5.0
    Pndc = make_ndc(zoom=zoom, camera_type="ortho")
    Pndc = torch.from_numpy(Pndc).float().cuda()
    cam2world = torch.eye(4)[None].float().cuda()
    # rotate along x axis for 180 degree
    cam2world[:,1,1] = -1.0
    cam2world[:,2,2] = -1.0
    # cam2world[:,0,3] = vert_face[:,0].mean()
    # cam2world[:,1,3] = vert_face[:,1].mean()
    cam2world[:,0,3] = 0
    cam2world[:,1,3] = 1.4303
    cam2world[:,2,3] = 2.0

    vert_face = torch.from_numpy(vert_face).float().cuda()
    face_fa = torch.from_numpy(face_fa).to(dtype=torch.int32).cuda()
    uv_face = torch.from_numpy(face_uv).float().cuda()

    vert_head = torch.from_numpy(vert_head).float().cuda()
    face_head = torch.from_numpy(face_head).to(dtype=torch.int32).cuda()
    head_uv = torch.from_numpy(head_uv).float().cuda()
    
    # vis render head background image
    head_mesh = trimesh.load('/aigc_cfs_2/weimao/avatar_face_generation/data/head_render.obj')
    vert_head = head_mesh.vertices
    face_head = head_mesh.faces
    head_uv = head_mesh.visual.uv
    vert_head = torch.from_numpy(vert_head).float().cuda()
    face_head = torch.from_numpy(face_head).to(dtype=torch.int32).cuda()
    head_uv = torch.from_numpy(head_uv).float().cuda()
    head_texture = Image.open('/aigc_cfs_2/weimao/avatar_face_generation/data/Material_007.010_T_Head_F_D_yellow.PNG')
    head_texture = torch.from_numpy(np.array(head_texture)/255).to(device='cuda', dtype=torch.float32)
    idx = np.arange(head_texture.shape[0]-1,-1,-1)
    head_texture = head_texture[idx]
    img_head, alpha_head = render_mesh(img_size=1024, ndc_mat=Pndc, c2w_mat=cam2world, v=vert_head, f=face_head, vt=head_uv, ft=face_head, tex=head_texture)
    bg_uv = (200, head_texture.shape[1]-480)
    alpha_head = (alpha_head > 0.5).float()
    # img_head = alpha_head * img_head + (1-alpha_head) * head_texture[bg_uv[1],bg_uv[0]][None, None, None]
    # edge = get_edges_with_sobel(alpha_head.permute(0,3,1,2)).permute(0,2,3,1)
    # img_head[edge[...,0]>0] = head_texture[bg_uv[1],bg_uv[0]]
    
    img_head_log = img_head.cpu().numpy()[0]
    img_head_log = Image.fromarray(np.uint8(img_head_log*255))
    img_head_log.save('/aigc_cfs_2/weimao/avatar_face_generation/data/' + 'head_img.png')
    assert False
    face_texture = Image.open(face_texture_dir)
    face_texture = torch.from_numpy(np.array(face_texture)/255).to(device='cuda', dtype=torch.float32)
    idx = np.arange(face_texture.shape[0]-1,-1,-1)
    face_texture = face_texture[idx]

    sz = 1024
    img_face, alpha_face = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=vert_face, f=face_fa, vt=uv_face, ft=face_fa, tex=face_texture)
    alpha_face = (alpha_face > 0.9).float()
    img_face_log = img_face.cpu().numpy()[0]
    img_face_log = Image.fromarray(np.uint8(img_face_log*255))
    img_face_log.save(out_path +'/' + out_fn + '_face_img.png')
    mask = alpha_face[0].repeat([1,1,3]).cpu().data.numpy()
    mask = Image.fromarray(np.uint8(mask*255))
    mask.save(out_path +'/' + out_fn + '_face_mask.png')
    
    mask_cv = cv2.imread(out_path +'/' + out_fn + '_face_mask.png')
    face_cv = cv2.imread(out_path +'/' + out_fn + '_face_img.png')
    head_cv = cv2.imread('/aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_img.png')
    center = (512,512)
    # Clone seamlessly.
    output = cv2.seamlessClone(face_cv, head_cv, mask_cv[:,:,0], center, cv2.NORMAL_CLONE)
    # Save result
    cv2.imwrite(out_path +'/' + out_fn + '_clone.png', output)
    
    
    head_texture = Image.open(head_texture_dir)
    head_texture = torch.from_numpy(np.array(head_texture)/255).to(device='cuda', dtype=torch.float32)
    idx = np.arange(head_texture.shape[0]-1,-1,-1)
    head_texture = head_texture[idx]
    img_head, alpha_head = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=vert_head, f=face_head, vt=head_uv, ft=face_head, tex=head_texture)
    alpha_head = (alpha_head > 0.5).float()
    img_head_log = img_head.cpu().numpy()[0]
    img_head_log = Image.fromarray(np.uint8(img_head_log*255))
    img_head_log.save(out_path +'/' + out_fn + '_head_img.png')
    alpha_head = torch.logical_and(alpha_face < 0.9, alpha_head>0.5).float()
    mask = alpha_head[0].repeat([1,1,3]).cpu().data.numpy()
    mask = Image.fromarray(np.uint8(mask*255))
    mask.save(out_path +'/' + out_fn + '_head_mask.png')

    """interpolation """
    # edge = get_edges_with_sobel(alpha_face.permute(0,3,1,2)).permute(0,2,3,1)
    # face_edge = torch.logical_and(alpha_face>0.9, edge>0).float()
    # head_edge = torch.logical_and(alpha_head>0.9, edge>0)
    # mask = face_edge[0].repeat([1,1,3]).cpu().data.numpy()
    # mask = Image.fromarray(np.uint8(mask*255))
    # mask.save(out_path +'/' + out_fn + '_face_edge.png')
    # face_edge = face_edge.squeeze()
    # head_edge = head_edge.squeeze()
    # edge_v, edge_u = torch.where(face_edge)
    # face_edge_uv = torch.cat([edge_u[:,None], edge_v[:,None]],dim=-1) # M x 2
    # edge_v, edge_u = torch.where(head_edge)
    # head_edge_uv = torch.cat([edge_u[:,None], edge_v[:,None]],dim=-1)
    # head_v, head_u = torch.where(alpha_head.squeeze())
    # head_uv = torch.cat([head_u[:,None], head_v[:,None]],dim=-1) # N x 2
    # uvdist = torch.cdist(head_uv.float(), face_edge_uv.float(), p=2) # N x M
    # min_uv_dist, face_uv_idx = torch.min(uvdist,dim=-1)
    # face_uv_closest = face_edge_uv[face_uv_idx] # N x 2
    # region_sz = 50
    # min_uv_dist = torch.clamp(min_uv_dist, min=0, max=region_sz)
    # head_color_interpolate = (min_uv_dist[:,None]/region_sz) * img_head[0,head_uv[:,1], head_uv[:,0],:3] + (region_sz - min_uv_dist[:,None])/region_sz *  img_face[0,face_uv_closest[:,1], face_uv_closest[:,0],:3]
    # img_head[0,head_uv[:,1], head_uv[:,0],:3] = head_color_interpolate

    
    img_face_head = (img_head[...,:3] * alpha_head + img_face * alpha_face) #/(alpha_head_blur + alpha_face_blur + 1e-10)
    # kernel = T.GaussianBlur(kernel_size=51, sigma=5.0).cuda()
    # img_face_head_blur = kernel(img_face_head.permute(0,3,1,2).clone()).permute(0,2,3,1)
    kernel = torch.ones([1, 1, 51,51], device='cuda',dtype=torch.float32)/11**2
    alpha_face_blur = F.conv2d(alpha_face.permute(0,3,1,2),kernel,padding=25,groups=1).permute(0,2,3,1)
    alpha_head_blur = F.conv2d(alpha_head.permute(0,3,1,2),kernel,padding=25,groups=1).permute(0,2,3,1)
    mask_inpaint = torch.logical_and(alpha_head_blur > 0, alpha_face_blur > 0)
    mask = mask_inpaint[0].repeat([1,1,3]).cpu().data.numpy()
    mask1 = Image.fromarray(np.uint8(mask*255))
    mask1.save(out_path +'/' + out_fn + '_inpaint_mask.png')
    
    kernel = torch.ones([1, 1, 101,101], device='cuda',dtype=torch.float32)/11**2
    alpha_face2 = torch.logical_or(alpha_face>0, mask_inpaint).float()
    alpha_head2 = torch.logical_and(alpha_head>0, mask_inpaint.float()==0).float()
    alpha_face_blur = F.conv2d(alpha_face2.permute(0,3,1,2),kernel,padding=50,groups=1).permute(0,2,3,1)
    alpha_head_blur = F.conv2d(alpha_head2.permute(0,3,1,2),kernel,padding=50,groups=1).permute(0,2,3,1)
    mask_inpaint = torch.logical_and(alpha_head_blur > 0, alpha_face_blur > 0.5)
    mask = mask_inpaint[0].repeat([1,1,3]).cpu().data.numpy()
    mask2 = Image.fromarray(np.uint8(mask*255))
    mask2.save(out_path +'/' + out_fn + '_inpaint_mask2.png')
    
    img_log = img_face_head.cpu().numpy()[0]
    img_log = Image.fromarray(np.uint8(img_log*255))
    img_log.save(out_path +'/' + out_fn + '_img.png')

    init_image = img_log
    mask_image = mask1
    # mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius = 20))
    # mask_image.save("/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/cute_you8_uv_aligned_whead_head_mask_blur.png")
    
    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image, generator=generator,
                    guidance_scale=10.0, strength=0.9).images[0]
    image.save(out_path +'/' + out_fn + '_inpaint_img1.png')
    
    init_image = image
    mask_image = mask2
    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image, generator=generator,
                    guidance_scale=10.0, strength=0.7).images[0]
    image.save(out_path +'/' + out_fn + '_inpaint_img2.png')

    image_label = torch.from_numpy(np.array(image)[:,:,:3]/255).float().cuda()
    verts = torch.from_numpy(verts).float().cuda()
    faces = torch.from_numpy(faces.astype(np.int32)).cuda()
    uvs = torch.from_numpy(uvs).float().cuda()
    para = nn.Parameter(face_texture) 
    optimizer = optim.Adam([para],lr=0.5,betas=(0.9,0.999))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma =0.99)
    # optimizer= optim.SGD([para], lr=1.0, momentum=0.9)
    pb = tqdm(total=max_iter)
    for i in range(max_iter):
        # images: [nv, h, w, 4], alpha: [nv, h, w, 1]
        img_reco, alpha = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=verts, f=faces, vt=uvs, ft=faces, tex=para)
        img_reco = img_reco[0]
        alpha = alpha[0,:,:,0]
        
        reco = (img_reco[alpha>0.5] - image_label[alpha>0.5]).pow(2).sum(dim=-1).mean()
        loss = reco
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        scheduler.step()
        if i % 100 == 0:
            with torch.no_grad():
                img_reco[alpha<0.5] = 0
                img = Image.fromarray(np.uint8(img_reco.cpu().data.numpy()*255))
                img.save(os.path.join(out_path,f"{out_fn}_combine_{i:03d}.png"))
        # break
        pb.set_description(f'img_reo {reco.item():.5f}')
        pb.update(1)

    tsz = para.shape[0]
    img = Image.fromarray(np.uint8(torch.clamp(para,min=0,max=1).cpu().data.numpy()*255)[np.arange(tsz-1,-1,-1)])
    img.save(os.path.join(out_path,f"{out_fn}_texture_map.png"))
    
    with open(out_dir, 'r') as file:
        content = file.readlines()
    # usemtl Material.001
    # mtllib cute_you5.mtl
    # o Face.001
    new_line = f"o Face.001\n"
    content.insert(0, new_line)
    new_line = f"mtllib {out_fn}.mtl\n"
    content.insert(0, new_line)
    new_line = f"usemtl Material.001\n"
    content.insert(0, new_line)
    with open(out_dir, 'w') as file:
        file.writelines(content)
    
    out_file = os.path.join(out_path,f"{out_fn}.mtl")
    # Open the file in write mode ('w')
    with open(out_file, 'w') as file:
        # Write a line to the files 1
        # newmtl Material.001
        # map_Kd cute_you5_texture_map.png
        file.write(f"newmtl Material.001\n")
        file.write(f"map_Kd {out_fn}_texture_map.png\n")
        

    
    # pipeline = AutoPipelineForInpainting.from_pretrained(
    # "/aigc_cfs_2/weimao/pretrained_model_cache/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
    # )
    # pipeline.enable_model_cpu_offload()
    # generator = torch.Generator("cuda").manual_seed(92)
    # prompt = "bald cartoon head, pure color background, no hair, skin color, no shadow, no shading, no grey color, bright color"
    # negative_prompt = "hair, black, shading, shadow, highlight"
    # image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image, generator=generator,
    #                 guidance_scale=15.0, strength=0.9).images[0]
    # # image = image.resize(init_image.size,Image.Resampling.BILINEAR)
    # image.save('/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/cute_you8_uv_aligned_whead_inpaint_0.9.png')


    
    # alpha_face_blur = kernel(alpha_face.permute(0,3,1,2)).permute(0,2,3,1)
    # kernel = torch.ones([1, 1, 11,11], device='cuda',dtype=torch.float32)/11**2
    # alpha_face_blur = F.conv2d(alpha_face.permute(0,3,1,2),kernel,padding=50,groups=1).permute(0,2,3,1)
    # mask = alpha_face_blur[0].repeat([1,1,3]).cpu().data.numpy()
    # mask = Image.fromarray(np.uint8(mask*255))
    # mask.save('./test.png')
    # alpha_head_blur = F.conv2d(alpha_head.permute(0,3,1,2),kernel,padding=5,groups=1).permute(0,2,3,1)
    # # alpha_head_blur = kernel(alpha_head.permute(0,3,1,2)).permute(0,2,3,1)
    # img_face_head = (img_head[...,:3] * (1-alpha_face_blur) + img_face * alpha_face_blur) #/(alpha_head_blur + alpha_face_blur + 1e-10)
    # img_log = img_face_head.cpu().numpy()[0]
    # img_log = Image.fromarray(np.uint8(img_log*255))
    # img_log.save(out_dir.replace('.obj','rendered.png'))
    # st()
    # # get mask
    # mask = torch.logical_and(alpha_face < 0.9, alpha_head>0.5)[0]
    # mask = mask.repeat([1,1,3]).cpu().data.numpy()
    # mask = Image.fromarray(np.uint8(mask*255))
    # # mask = mask.filter(ImageFilter.GaussianBlur(radius = 20))
    # mask.save(out_dir.replace('.obj','_whead_inpaint_mask.png'))

    # img_face = load_image("/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/cute_you8_uv_aligned_whead_rendered.png")
    # mask = load_image("/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/cute_you8_uv_aligned_whead_whead_inpaint_mask.png")
    # generator = torch.Generator("cuda").manual_seed(92)
    # for gs in np.linspace(10, 20, 10):
    #     # for streng in np.linspace(0.1, 1.0, 10):
    #     image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=img_face, mask_image=mask, generator=generator,
    #                     guidance_scale=gs, strength=1.0).images[0]
    #     image = np.array(image)
    #     mask2 = np.logical_or(alpha_face[0].cpu().data.numpy() >= 0.5, np.array(mask)[:,:,:1]>0.5)
    #     mask2 = np.uint8(mask2*255)
    #     image = np.concatenate([image, mask2],axis=-1)
    #     image = Image.fromarray(image)
    #     image.convert("RGBA")
    #     image.save(out_dir.replace('.obj',f'_whead_inpainted_{int(gs):02d}.png'))

    print(f'time used {time.time()-start_time:.3f}seconds')

