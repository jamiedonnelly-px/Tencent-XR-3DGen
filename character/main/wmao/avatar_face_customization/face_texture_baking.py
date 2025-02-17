import trimesh
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm.auto import tqdm
from pdb import set_trace as st
from matplotlib import pyplot as plt
import xatlas
from render_mesh import make_ndc, render_mesh_verts_tex, render_mesh
import os
import glob
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

img_dirs = glob.glob('/aigc_cfs_2/weimao/avatar_face_generation/test_data/*.png')
mask_dirs = '/aigc_cfs_2/weimao/avatar_face_generation/output_uv_edited/'
# img_dirs = glob.glob('/aigc_cfs_2/weimao/avatar_face_generation/output_uv_edited/*_inpainted.png')
obj_dir = '/aigc_cfs_2/weimao/avatar_face_generation/output_uv_edited/'
max_iter = 500
out_dir = "/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/"
os.makedirs(out_dir, exist_ok=True)


pipeline = AutoPipelineForInpainting.from_pretrained(
    "/aigc_cfs_2/weimao/pretrained_model_cache/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()
generator = torch.Generator("cuda").manual_seed(92)
prompt = "bald cartoon head, no hair, skin color, no shadow, no shading, no grey color, no ear"
negative_prompt = "hair, hair, hair, black, shading, shading, shading, shadow, shadow, shadow, highlight, ear, boundary, edges"

for img_dir in img_dirs:
    # if 'cute_you11' not in img_dir:
    #     continue
    # mesh_dir = img_dir.replace('.png','.obj')
    # mesh_dir = "/aigc_cfs_2/weimao/texture_generation/test_data/cute_you8.obj"
    # img_dir = "/aigc_cfs_2/weimao/texture_generation/test_data/cute_you8.png"
    file_name = os.path.basename(img_dir).split('.')[0]
    mesh_dir = obj_dir + '/' + file_name + '.obj'

    mesh = trimesh.load(mesh_dir)
    verts = mesh.vertices - 0.5
    faces = mesh.faces
    uvs = None
    try:
        uvs = mesh.visual.uv
    except:
        print('mesh does not have uv')
    img = Image.open(img_dir)
    img= np.array(img)[:,:,:3]/255.
    sz = img.shape[0]
    color_label = torch.from_numpy(img).float().cuda()
    mask = None
    try:
        mask = Image.open(mask_dirs + os.path.basename(img_dir).split('.')[0] + '_mask.png')
        # get outlayer mask
        # mask_tmp = mask.filter(ImageFilter.GaussianBlur(radius = 20))
        # mask_tmp = np.array(mask_tmp)/255.
        
        mask = mask.filter(ImageFilter.GaussianBlur(radius = 10)) 
        mask = np.array(mask)/255.
        # mask_outer = torch.from_numpy(mask[:,:,0]).float().cuda() < 0.01
        mask = torch.from_numpy(mask[:,:,0]).float().cuda() > 0.9
    except:
        print('no mask')

    zoom = 2.0
    Pndc = make_ndc(zoom=zoom, camera_type="ortho")
    Pndc = torch.from_numpy(Pndc).float().cuda()
    cam2world = torch.eye(4)[None].float().cuda()
    # cam2world[:,0,2] = 0.5
    # cam2world[:,1,2] = 0.1
    cam2world[:,2,3] = -2.0
    if uvs is None:
        atlas = xatlas.Atlas()
        # use original mesh not the manifold_mesh
        atlas.add_mesh(verts, faces)
        chart_options = xatlas.ChartOptions()
        chart_options.max_iterations = 10
        pack_options = xatlas.PackOptions()
        pack_options.create_image = False
        pack_options.padding = 5
        pack_options.bruteForce = True
        # pack_options.max_chart_size = 1024
        atlas.generate(pack_options=pack_options,chart_options=chart_options)
        vmapping, faces, uvs = atlas[0]
        verts = verts[vmapping.tolist()]
    out_file = os.path.join(out_dir,f"{file_name}_uv.obj")
    xatlas.export(out_file, verts, faces, uvs)
    # Specify the filename
    # The line you want to append at the beginning
    with open(out_file, 'r') as file:
        content = file.readlines()
    # usemtl Material.001
    # mtllib cute_you5.mtl
    # o Face.001
    new_line = f"o Face.001\n"
    content.insert(0, new_line)
    new_line = f"mtllib {file_name}.mtl\n"
    content.insert(0, new_line)
    new_line = f"usemtl Material.001\n"
    content.insert(0, new_line)
    with open(out_file, 'w') as file:
        file.writelines(content)
    
    out_file = os.path.join(out_dir,f"{file_name}.mtl")
    # Open the file in write mode ('w')
    with open(out_file, 'w') as file:
        # Write a line to the files 1
        # newmtl Material.001
        # map_Kd cute_you5_texture_map.png
        file.write(f"newmtl Material.001\n")
        file.write(f"map_Kd {file_name}_texture_map.png\n")
        
    tex = torch.zeros([1024, 1024, 3]).float().cuda()
    # #default skin color
    # skin_col = torch.from_numpy(np.array([251,216,197])/255).float().cuda()
    # skin_col = torch.logit(skin_col)
    # tex[:,:] = skin_col
    para = nn.Parameter(tex.detach().clone())

    # remove the outerline of the face
    idx_to_remove = [127,234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454,356,389,251,284,332,297,338,10,109,67,103,54,21,162]
    idx_old = np.setdiff1d(np.arange(verts.shape[0]),np.array(idx_to_remove))
    idx_new = np.arange(len(idx_old))
    verts = verts[idx_old]
    uvs = uvs[idx_old]
    face_new = np.zeros_like(faces) - 1
    for i in idx_new:
        face_new[faces == idx_old[i]] = i
    idx_tmp = np.setdiff1d(np.arange(face_new.shape[0]),np.where(face_new < 0)[0])
    face_new = face_new[idx_tmp]
    faces = face_new

    tv = torch.from_numpy(verts).float().cuda()
    faces = torch.from_numpy(faces.astype(np.int32)).cuda()
    uvs = torch.from_numpy(uvs).float().cuda()

    optimizer = optim.Adam([para],lr=0.5,betas=(0.9,0.999))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma =0.99)
    # optimizer= optim.SGD([para], lr=1.0, momentum=0.9)
    pb = tqdm(total=max_iter)
    for i in range(max_iter):
        # images: [nv, h, w, 4], alpha: [nv, h, w, 1]
        img_reco, alpha = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=tv, f=faces, vt=uvs, ft=faces, tex=torch.sigmoid(para))
        img_reco = img_reco[0]
        alpha = alpha[0,:,:,0]
        
        if i == 0:
            # get outer mask
            kernel = torch.ones([1, 1, 21, 21],dtype=alpha.dtype,device=alpha.device)/100
            mask_outer = alpha.unsqueeze(0).unsqueeze(0)
            for _ in range(3):
                mask_outer = F.conv2d(mask_outer, kernel, padding=10, groups=1)
            mask_outer = mask_outer[0,0] <= 0

            assert mask is not None
            mask = alpha.clone().detach() * mask
            # inpaint the image
            mask_inpaint = mask[:,:,]
            mask_inpaint = torch.logical_or(mask[:,:,None] > 0.5, mask_outer[:,:,None])
            mask_inpaint = torch.cat([mask_inpaint,mask_inpaint,mask_inpaint],dim=-1) * 255
            mask_inpaint = np.uint8(mask_inpaint.cpu().data.numpy())
            mask_inpaint = Image.fromarray(mask_inpaint)
            mask_inpaint.save(f'{mask_dirs}/{file_name}_mask_mesh.png')
            img_inpaint = color_label.clone().detach()
            img_inpaint[mask<=0.5] = torch.from_numpy(np.array([251,216,197])/255).to(device=color_label.device, dtype=color_label.dtype)
            # img_inpaint[mask_outer] = torch.from_numpy(np.array([251,216,197])/255).to(device=color_label.device, dtype=color_label.dtype)
            # img_inpaint = torch.cat([img_inpaint, (alpha[:,:,None]>0.5) * 1.0], dim=-1)
            img_inpaint = Image.fromarray(np.uint8(255*img_inpaint.cpu().data.numpy()))
            img_inpaint.save(f'{mask_dirs}/{file_name}_masked.png')
            mask_inpaint_invert = ImageOps.invert(mask_inpaint)

            generator = torch.Generator("cuda").manual_seed(92)
            image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=img_inpaint, mask_image=mask_inpaint_invert, generator=generator,
                            guidance_scale=10.0).images[0]
            image.save(f'{mask_dirs}/{file_name}_inpainted.png')
            color_label = torch.from_numpy(np.array(image) / 255.0).to(device=color_label.device, dtype=color_label.dtype)

        reco = (img_reco[alpha>0.5] - (color_label[alpha>0.5])).pow(2).sum(dim=-1).mean()
        loss = reco
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            with torch.no_grad():
                img_reco[alpha<0.5] = 0
                img = Image.fromarray(np.uint8(img_reco.cpu().data.numpy()*255))
                img.save(os.path.join(out_dir,f"{file_name}_{i:03d}.png"))
        # break
        pb.set_description(f'img_reo {reco.item():.5f}')
        pb.update(1)

    tsz = para.shape[0]
    img = Image.fromarray(np.uint8(torch.sigmoid(para).cpu().data.numpy()*255)[np.arange(tsz-1,-1,-1)])
    img.save(os.path.join(out_dir,f"{file_name}_texture_map.png"))
    

