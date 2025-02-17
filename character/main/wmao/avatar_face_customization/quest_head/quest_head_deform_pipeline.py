
import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace as st
import argparse
import cv2
import trimesh
import open3d as o3d
import copy
import torch
import glob
import os
import sys
codedir = os.path.dirname(os.path.abspath(__file__))
print('code dir is:', codedir)
sys.path.append(codedir)
# sys.path.insert(0,'/aigc_cfs_2/weimao/avatar_face_generation/continuous_remeshing')
sys.path.insert(0,'/aigc_cfs_2/weimao/avatar_face_generation/')
from PIL import Image as Image
from segment_anything import sam_model_registry, SamPredictor
import time
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from itertools import combinations

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from PIL import Image, ImageFilter, ImageOps
from tqdm.auto import tqdm
from pdb import set_trace as st
from matplotlib import pyplot as plt
import xatlas
from render_mesh import make_ndc, render_mesh_verts_tex, render_mesh
import os
import glob
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import subprocess

import nvdiffrast
import nvdiffrast.torch as dr

from utils import *
# from continuous_remeshing.core.remesh import calc_vertex_normals, calc_face_normals
# from continuous_remeshing.core.opt import MeshOptimizer

import zipfile

def extract_mesh(rgb_image, detection_result, out_dir = './', canonical_face_model=None, face_data=None):
    """
    extract mesh from mp face
    rbg_image: face image
    detection_result: results from mp face detector
    out_file: output file 
    canonical_face_model: 
    
    """
    assert face_data is not None
    # face data keys ['idxs', 'face_edge', 'head_edge', 'face_id_to_connect', 'head_vert_wouv', 'head_face_wouv', 'face_face', 'uv', 'face_uv']
    idxs_remain = face_data['idxs']
    uv_tex = face_data['face_uv']
    faces = face_data['face_face']
    face_landmarks = detection_result.face_landmarks[0]
    verts = []
    image_rows, image_cols, _ = rgb_image.shape
    uvs = []
    for landmark in face_landmarks:
        verts.append([landmark.x,landmark.y,landmark.z])
        u = landmark.x * image_cols
        v = landmark.y * image_rows
        uvs.append([u,v])
    verts = np.array(verts)
    uvs = np.array(uvs)
    verts = verts[idxs_remain]
    uvs = uvs[idxs_remain]
    out_file = os.path.join(out_dir,f"face_mesh.obj")
    xatlas.export(out_file, verts, faces, uv_tex)
    return verts, faces, uvs, uv_tex

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
        
        solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
        
        # solutions.drawing_utils.draw_landmarks(
        #		 image=annotated_image,
        #		 landmark_list=face_landmarks_proto,
        #		 connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        #		 landmark_drawing_spec=None,
        #		 connection_drawing_spec=mp.solutions.drawing_styles
        #		 .get_default_face_mesh_contours_style())
        
        # solutions.drawing_utils.draw_landmarks(
        #		 image=annotated_image,
        #		 landmark_list=face_landmarks_proto,
        #		 connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        #			 landmark_drawing_spec=None,
        #			 connection_drawing_spec=mp.solutions.drawing_styles
        #			 .get_default_face_mesh_iris_connections_style())

    return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()

def mp_face_detector(file_path, out_dir, detector, canonical_face_dir=None, face_data=None, sz=1024):
    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(file_path)
    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(image)
    
    # crop image
    face_landmarks = detection_result.face_landmarks[0]
    rgb_img = image.numpy_view()[:,:,:3]
    image_rows, image_cols, _ = rgb_img.shape
    uvs = []
    for landmark in face_landmarks:
        u = landmark.x * image_cols
        v = landmark.y * image_rows
        uvs.append([u,v])
    uvs = np.array(uvs)
    uv_min = uvs.min(axis=0)
    uv_max = uvs.max(axis=0)
    wh = uv_max - uv_min
    l = wh.max()
    dwh = (l*1.4 - wh)/2
    
    uv_min[0] = max(uv_min[0] - dwh[0], 0)
    uv_min[1] = max(uv_min[1] - dwh[1]*1.5, 0)
    uv_max[0] = min(uv_max[0] + dwh[0], image_cols-1)
    uv_max[1] = min(uv_max[1] + dwh[1]/1.5, image_rows-1)
    rgb_img = rgb_img[int(uv_min[1]):int(uv_max[1])][:,int(uv_min[0]):int(uv_max[0])]
    img = Image.fromarray(rgb_img)
    img = img.resize((sz,sz))
    img.save(f'{out_dir}/face_crop.png')

    image = mp.Image.create_from_file(f'{out_dir}/face_crop.png')
    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(image)
    # from canonical face to detected face
    np.savez_compressed(f"{out_dir}/face_pose.npz", face_pose=detection_result.facial_transformation_matrixes[0])
    verts, faces, uvs_img, uvs_tex= extract_mesh(image.numpy_view()[:,:,:3], detection_result, 
                                                        out_dir=out_dir,
                                                        face_data=face_data)
    # # STEP 5: Process the detection result. In this case, visualize it.
    # annotated_image = draw_landmarks_on_image(image.numpy_view()[:,:,:3], detection_result)

    # img = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(f"{out_dir}/annotated_img.png", img)
    image = image.numpy_view()
    return verts, faces, uvs_img, uvs_tex, image

def remove_hair(image, uvs, out_dir, predictor):
    
    # STEP 5 segment the face region using SAM
    # using two eyes mouth and nose as prompt
    image = image[:,:,:3]
    predictor.set_image(image)
    
    # prompt_idx = [4, 468, 473, 101, 330, 0, 17, 40, 270, 52, 282]
    prompt_idx = [4, 47, 244, 17, 141, 8]
    input_point = uvs[prompt_idx]
    leye = (uvs[137] + uvs[127])/2
    reye = (uvs[332] + uvs[323])/2
    mouth = (uvs[13] + uvs[14])/2
    input_point = np.concatenate([input_point,mouth[None], leye[None], reye[None]],axis=0)
    input_label = np.array([1]*input_point.shape[0])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    mask_inv = 1 - masks
    
    masks = np.uint8(masks.transpose(1,2,0) * 255)
    masks_log = Image.fromarray(np.concatenate([masks,masks,masks],axis=-1))
    masks_log.save(f"{out_dir}/hair_mask.png")

def remove_vertices_by_id(verts, faces, uvs, idx_to_remove=[127,234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454,356,389,251,284,332,297,338,10,109,67,103,54,21,162]):
    # remove the outerline of the face
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
    return verts, faces, uvs	

def add_mtl(out_file):
    """
    out_file: the dir of obj file
    """
    out_dir = os.path.dirname(out_file)
    obj_name = os.path.basename(out_file).split('.')[0]
    # Specify the filename
    # The line you want to append at the beginning
    with open(out_file, 'r') as file:
        content = file.readlines()
    # usemtl Material.001
    # mtllib cute_you5.mtl
    # o Face.001
    new_line = f"o Face.001\n"
    content.insert(0, new_line)
    new_line = f"mtllib {obj_name}.mtl\n"
    content.insert(0, new_line)
    new_line = f"usemtl Material.001\n"
    content.insert(0, new_line)
    with open(out_file, 'w') as file:
        file.writelines(content)
    
    out_file = os.path.join(out_dir,f"{obj_name}.mtl")
    # Open the file in write mode ('w')
    with open(out_file, 'w') as file:
        # Write a line to the files 1
        # newmtl Material.001
        # map_Kd cute_you5_texture_map.png
        file.write(f"newmtl Material.001\n")
        file.write(f"map_Kd {obj_name}_tex.png\n")

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

def blend_face_to_head_img(head_mesh_dir='/aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_render.obj',
                         head_texture_dir='/aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_brighter.jpg',
                         Pndc=None, cam2world=None, face_mask=None,face_img=None, out_dir=None, bg_color=None):
    # vis render head background image
    head_mesh = trimesh.load(head_mesh_dir)
    vert_head = head_mesh.vertices
    face_head = head_mesh.faces
    head_uv = head_mesh.visual.uv
    vert_head = torch.from_numpy(vert_head).float().cuda()
    face_head = torch.from_numpy(face_head).to(dtype=torch.int32).cuda()
    head_uv = torch.from_numpy(head_uv).float().cuda()
    head_texture = Image.open(head_texture_dir)
    head_texture = torch.from_numpy(np.array(head_texture)/255).to(device='cuda', dtype=torch.float32)
    idx = np.arange(head_texture.shape[0]-1,-1,-1)
    head_texture = head_texture[idx]
    img_head, alpha_head = render_mesh(img_size=1024, ndc_mat=Pndc, c2w_mat=cam2world, v=vert_head, f=face_head, vt=head_uv, ft=face_head, tex=head_texture)
    bg_uv = (200, head_texture.shape[1]-480)
    alpha_head = (alpha_head > 0.5).float()
    img_head = alpha_head * img_head + (1-alpha_head) * head_texture[bg_uv[1],bg_uv[0]][None, None, None]
    edge = get_edges_with_sobel(alpha_head.permute(0,3,1,2)).permute(0,2,3,1)
    img_head[edge[...,0]>0] = head_texture[bg_uv[1],bg_uv[0]]
    img_head = img_head.cpu().data.numpy()
    head_mask = torch.logical_or(alpha_head>0.5, edge>0).float().repeat(1,1,1,3).cpu().data.numpy()
    img_head_new = []
    for i in range(img_head.shape[0]):
        head_mask_log = np.uint8(head_mask[i]*255)
        img_head_log = np.uint8(img_head[i]*255)

        # bg_img = np.uint8(np.zeros_like(img_head_log) + bg_color[None, None]*255)
        # v, u = np.where(head_mask_log[:,:,0] > 128)
        # center = (int((u.min() + u.max()) /2), int((v.min() + v.max()) /2))
        # img_head_log = cv2.seamlessClone(img_head_log, bg_img, head_mask_log, center, cv2.NORMAL_CLONE)

        head_mask_log = Image.fromarray(head_mask_log)
        head_mask_log.save(f'/aigc_cfs_2/weimao/avatar_face_generation/tmp/head_mask_{i:01d}.png')
        img_head_log = Image.fromarray(img_head_log)
        img_head_log.save(f'/aigc_cfs_2/weimao/avatar_face_generation/tmp/head_{i:01d}.png')
        img_head_new.append(np.float32(np.array(img_head_log)[None])/255)
    img_head_new = np.concatenate(img_head_new,axis=0)

    img_blend = []
    for i in range(face_img.shape[0]):
        face_cv = np.uint8(face_img[i]*255*0.8)
        mask_cv = np.uint8(face_mask[i]*255)
        head_cv = np.uint8(img_head_new*255*0.8)
        kernel = np.ones((3, 3), np.uint8)
        # Apply erosion
        mask_cv = cv2.erode(mask_cv, kernel, iterations=2)
        v, u = np.where(face_mask[i,:,:,0] > 0.5)
        center = (int((u.min() + u.max()) /2), int((v.min() + v.max()) /2))
        # Clone seamlessly.
        output = cv2.seamlessClone(face_cv, head_cv[i], mask_cv, center, cv2.NORMAL_CLONE)
        # Save result
        cv2.imwrite(f"{out_dir}/face_head_blend_{i:01d}_before_hist.png", output[:,:,[2,1,0]])
        
        mask_cv = cv2.erode(mask_cv, kernel, iterations=2)
        output = hist_matching(output, mask_cv, face_cv, mask_cv)
        cv2.imwrite(f"{out_dir}/face_head_blend_{i:01d}.png", output[:,:,[2,1,0]])
        img_blend.append(np.float32(output[None])/255)
    
    return np.concatenate(img_blend,axis=0)


def face_detection_and_texture_baking(file_path, out_dir=None, 
                                      face_detector=None,
                                      is_rm_hair=True, sam_predictor=None,
                                      inpainting_pipeline=None):
    st0 = time.time()
    img_name = os.path.basename(file_path).split('.')[0]
    # resize face image to 1024 the input image is whole body image
    sz=1024
    image = Image.open(file_path)
    # image = image.resize((sz,sz))
    image.save(out_dir + '/' + img_name + '.png')
    file_path = out_dir + '/' + img_name + '.png'

    face_data = np.load(f'/aigc_cfs_2/weimao/avatar_face_generation/data/timer_model_v2/face_data.npz') # reuse the face data in timer model
    """## step 1 face detector
    
    """
    detector = face_detector

    # image is a numpy array with shape (h,w,3or4)
    verts, faces, uvs_img, uvs_tex, image = mp_face_detector(file_path, out_dir, detector, face_data=face_data, sz=sz)

    out_file = os.path.join(out_dir,f"face_mesh.npz")
    np.savez_compressed(out_file, verts=verts, faces=faces, uvs=uvs_tex)
    file_path = out_dir + '/face_crop.png'
    print(f'>>> finish face mesh generation in {time.time()-st0:.3f} s')

    """## step2 optional remove hair
    
    """
    if is_rm_hair:
        st1 = time.time()
        predictor = sam_predictor
        remove_hair(image, uvs_img, out_dir, predictor)
        print(f'>>> finish hair mask generation in {time.time()-st1:.3f} s')
    
    """## step3 face texture baking
    
    """
    # pipeline = AutoPipelineForInpainting.from_pretrained(
    # 	"/aigc_cfs_2/weimao/pretrained_model_cache/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
    # 	)
    # pipeline.enable_model_cpu_offload()
    pipeline = inpainting_pipeline
    generator = torch.Generator("cuda").manual_seed(92)
    prompt = "bald cartoon head, no hair, skin color, no shadow, no shading, no grey color, no ear"
    negative_prompt = "hair, hair, hair, black, shading, shading, shading, shadow, shadow, shadow, highlight, ear, boundary, edges"

    st2 = time.time()
    # verts's xy is actually the uv where the top left corner is (0,0) and bottom right corner is (1,1)
    verts = verts - 0.5
    faces = faces
    uvs = uvs_tex
    # img = Image.open(file_path)
    # img= np.array(img)[:,:,:3]/255.
    img = image[:,:,:3]/255
    sz = img.shape[0]
    color_label = torch.from_numpy(img).float().cuda()
    mask = None
    try:
        mask = Image.open(f"{out_dir}/hair_mask.png")
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
    out_file = os.path.join(out_dir,f"face_mesh.obj")
    add_mtl(out_file)

    tex = torch.zeros([1024, 1024, 3]).float().cuda()
    para = nn.Parameter(tex.detach().clone())
    tv = torch.from_numpy(verts).float().cuda()
    faces_torch = torch.from_numpy(faces.astype(np.int32)).cuda()
    uvs_torch = torch.from_numpy(uvs).float().cuda()
    max_iter = 500
    optimizer = optim.Adam([para],lr=0.5,betas=(0.9,0.999))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma =0.99)
    # optimizer= optim.SGD([para], lr=1.0, momentum=0.9)
    pb = tqdm(total=max_iter)
    for i in range(max_iter):
        # images: [nv, h, w, 4], alpha: [nv, h, w, 1]
        img_reco, alpha = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=tv, f=faces_torch, vt=uvs_torch, ft=faces_torch, tex=torch.sigmoid(para))
        img_reco = img_reco[0]
        alpha = alpha[0,:,:,0]
        
        if i == 0 and mask is not None:
            # if there are hair in the face region, we do inpainting to remove hair
            if (alpha > 0.5).sum()> torch.logical_and(mask>0,alpha>0.5).sum():
                # get outer mask
                kernel = torch.ones([1, 1, 21, 21],dtype=alpha.dtype,device=alpha.device)/100
                mask_outer = alpha.unsqueeze(0).unsqueeze(0)
                for _ in range(3):
                    mask_outer = F.conv2d(mask_outer, kernel, padding=10, groups=1)
                mask_outer = mask_outer[0,0] <= 0
                mask = alpha.clone().detach() * mask
                mask_face = Image.fromarray(np.uint8((mask.cpu().data.numpy()>0.5)*255))
                mask_face.save(f'{out_dir}/face_mask.png')
                # inpaint the image
                mask_inpaint = torch.logical_or(mask[:,:,None] > 0.5, mask_outer[:,:,None])
                mask_inpaint = torch.cat([mask_inpaint,mask_inpaint,mask_inpaint],dim=-1) * 255
                mask_inpaint = np.uint8(mask_inpaint.cpu().data.numpy())
                mask_inpaint = Image.fromarray(mask_inpaint)
                mask_inpaint.save(f'{out_dir}/inpaint_mask.png')
                mask_inpaint_invert = ImageOps.invert(mask_inpaint)
                img_inpaint = color_label.clone().detach()
                img_inpaint[mask<=0.5] = torch.from_numpy(np.array([230, 215, 207])/255).to(device=color_label.device, dtype=color_label.dtype)
                # img_inpaint[mask_outer] = torch.from_numpy(np.array([251,216,197])/255).to(device=color_label.device, dtype=color_label.dtype)
                # img_inpaint = torch.cat([img_inpaint, (alpha[:,:,None]>0.5) * 1.0], dim=-1)
                img_inpaint = Image.fromarray(np.uint8(255*img_inpaint.cpu().data.numpy()))
                img_inpaint.save(f'{out_dir}/img_masked.png')

                
                generator = torch.Generator("cuda").manual_seed(92)
                image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=img_inpaint, mask_image=mask_inpaint_invert, generator=generator,
                                guidance_scale=10.0).images[0]
                
                image.save(f'{out_dir}/img_inpainted_before_hist_matching.png')
                # match the inpainted image to original image
                src_img = np.array(image)
                src_mask = np.array(mask_face)
                tgt_img = np.array(img_inpaint)
                image = hist_matching(src_img,src_mask,tgt_img,src_mask)
                image = Image.fromarray(image)
                
                image.save(f'{out_dir}/img_inpainted.png')
                color_label = torch.from_numpy(np.array(image) / 255.0).to(device=color_label.device, dtype=color_label.dtype)

        reco = (img_reco[alpha>0.5] - (color_label[alpha>0.5])).pow(2).sum(dim=-1).mean()
        loss = reco
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
                
        pb.set_description(f'img_reo {reco.item():.5f}')
        pb.update(1)

    tsz = para.shape[0]
    img = Image.fromarray(np.uint8(torch.sigmoid(para).cpu().data.numpy()*255)[np.arange(tsz-1,-1,-1)])
    img.save(os.path.join(out_dir,f"face_mesh_tex.png"))
    print(f'>>> finish face texture baking iin {time.time()-st2:.3f} s')

    return out_file, os.path.join(out_dir,f"face_mesh_tex.png")


def overall_pipeline(file_path=None, 
                     face_detector=None, 
                     sam_predictor=None, 
                     inpainting_pipeline=None, 
                     out_dir=None, 
                     is_rm_hair=True, 
                     model_dir='/aigc_cfs_2/weimao/avatar_face_generation/data/quest_head_model'):
    
    img_name = os.path.basename(file_path).split('.')[0]
    if out_dir is None:
        out_dir = os.path.dirname(file_path) + '/' + img_name
    os.makedirs(out_dir,exist_ok=True)
    start_time0=time.time()
    mp_face_dir, mp_texture_dir = face_detection_and_texture_baking(file_path, out_dir=out_dir, 
                                                                    face_detector=face_detector,
                                                                    is_rm_hair=is_rm_hair,
                                                                    sam_predictor=sam_predictor,
                                                                    inpainting_pipeline=inpainting_pipeline)
    print(f'generate mp face finished use {time.time()-start_time0:.3f}')
    # preprocess mesh
    start_time = time.time()
    command = [
        "/usr/blender-4.0.1-linux-x64/blender",
        "-b",
        "--python",
        "./blender_deform_quest_head.py",
        "--",
        "--quest_head_dir", f'{model_dir}/quest_head.obj',
        "--correspondance_npz", f'{model_dir}/correspondance_aligned.npz',
        "--mp_face_aligned_npz", f'{model_dir}/mp_face_aligned2quest_head.npz',
        "--mp_face_target_npz", f'{out_dir}/face_mesh.npz',
        "--out_file", f'{out_dir}/quest_deformed.obj'
        ]
    print(' '.join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(f'deformed mesh time: {time.time() - start_time:.3f} second')

    st0 = time.time()
    # # load quest_head
    # target_name='taylor'
    # quest_head_dir = f'/aigc_cfs_2/weimao/avatar_face_generation/output/quest_{target_name}.obj'
    # texture_dir = '/aigc_cfs_2/weimao/avatar_face_generation/output/quest_head.png'
    # mp_face_dir = f'/aigc_cfs_2/weimao/avatar_face_generation/output/web/{target_name}/face_mesh.obj'
    # mp_face_aligned_dir = f'/aigc_cfs_2/weimao/avatar_face_generation/output/aligned_{target_name}.obj'
    # mp_texture_dir = f'/aigc_cfs_2/weimao/avatar_face_generation/output/web/{target_name}/face_mesh_tex.png'
    quest_head_dir = f'{out_dir}/quest_deformed.obj'
    texture_dir = f'{model_dir}/quest_head.png'
    mp_face_aligned_npz_dir = f'{out_dir}/mp_face_target_aligned.npz'

    """step 1: render face and head image"""
    start_time = time.time()
    head_mesh = trimesh.load(quest_head_dir)
    verts = head_mesh.vertices
    faces = head_mesh.faces
    uvs = head_mesh.visual.uv
    head_tex = np.array(Image.open(texture_dir))[:,:,:3]/255.
    out_file = os.path.join(out_dir,f"quest_deformed.obj")
    xatlas.export(out_file, verts, faces, uvs)
    add_mtl(out_file)

    mp_mesh = np.load(mp_face_aligned_npz_dir)
    mp_verts = mp_mesh['verts']
    mp_faces = mp_mesh['faces']
    mp_uvs = mp_mesh['uvs']
    mp_tex = np.array(Image.open(mp_texture_dir))[:,:,:3]/255.

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
    quest_tex = torch.from_numpy(head_tex[np.arange(-1, -1-head_tex.shape[0],-1)]).float().cuda()
    quest_tv = torch.from_numpy(verts).float().cuda()
    quest_faces_torch = torch.from_numpy(faces.astype(np.int32)).cuda()
    quest_uvs_torch = torch.from_numpy(uvs).float().cuda()
    img_quest, alpha_quest = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=quest_tv, 
                                         f=quest_faces_torch, vt=quest_uvs_torch, ft=quest_faces_torch, tex=quest_tex)
    img_quest = img_quest[0]
    alpha_quest = alpha_quest[0,:,:,0]
    img = Image.fromarray(np.uint8(img_quest.cpu().data.numpy()*255))
    img.save(f'{out_dir}/render_quest.png')

    tex = torch.from_numpy(mp_tex[np.arange(-1, -1-mp_tex.shape[0],-1)]).float().cuda()
    tv = torch.from_numpy(mp_verts).float().cuda()
    faces_torch = torch.from_numpy(mp_faces.astype(np.int32)).cuda()
    uvs_torch = torch.from_numpy(mp_uvs).float().cuda()
    img_mp, alpha_mp = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=tv, f=faces_torch, 
                                   vt=uvs_torch, ft=faces_torch, tex=tex)
    img_mp = img_mp[0]
    alpha_mp = alpha_mp[0,:,:,0]
    img = Image.fromarray(np.uint8(img_mp.cpu().data.numpy()*255))
    img.save(f'{out_dir}/render_mp.png')

    """step2 blend face to head image"""
    face_cv = np.uint8(img_mp.cpu().data.numpy() * 0.8 * 255)
    mask_cv = np.uint8(alpha_mp.cpu().data.numpy() * 255)
    if len(mask_cv.shape) == 3:
        mask_cv = mask_cv[:,:,0]
    head_cv = np.uint8(img_quest.cpu().data.numpy() * 255)
    kernel = np.ones((3, 3), np.uint8)
    # Apply erosion
    mask_cv = cv2.erode(mask_cv, kernel, iterations=3)
    
    v, u = np.where(mask_cv > 128)
    center = (int((u.min() + u.max()) /2), int((v.min() + v.max()) /2))
    # Clone seamlessly.

    output = cv2.seamlessClone(face_cv, head_cv, mask_cv, center, cv2.NORMAL_CLONE)
    src_img = output
    mask = np.uint8(np.logical_and(alpha_quest.cpu().data.numpy() > 0.5, alpha_mp.cpu().data.numpy()<0.5)*255)
    tgt_img = np.uint8(img_quest.cpu().data.numpy()*255)
    output = hist_matching(src_img, mask, tgt_img=tgt_img, tgt_mask=mask)
    
    # Save result
    cv2.imwrite(f"{out_dir}/seamless_clone.png", output[:,:,[2,1,0]])
    print(f'clone face to head image time used {time.time()-start_time:.3f}')

    """step3 bake the front image back to head"""
    start_time = time.time()

    tv = quest_tv.detach().clone()
    faces_torch = quest_faces_torch
    uvs_torch = quest_uvs_torch
    color_label = torch.from_numpy(output/255).float().cuda()
    # get tex mask
    para = nn.Parameter(torch.ones_like(quest_tex))
    tv = quest_tv.detach().clone()
    faces_torch = quest_faces_torch
    uvs_torch = quest_uvs_torch
    img_reco, _ = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=tv, f=faces_torch, vt=uvs_torch, 
                                    ft=faces_torch, tex=para, max_mip_level=0)
    img_reco = img_reco[0]
    img_reco[alpha_mp>0.5].sum().backward()
    grad_mask = para.grad.abs().sum(dim=-1)>0
    
    para = nn.Parameter(quest_tex.detach().clone())
    # para = nn.Parameter(torch.zeros_like(quest_tex.detach().clone())-1)
    max_iter =200
    optimizer = optim.Adam([para],lr=0.5,betas=(0.9,0.999))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma =0.95)
    # optimizer= optim.SGD([para], lr=1.0, momentum=0.9)
    mask = torch.from_numpy(mask_cv/255.).float().cuda()[...,None]
    pb = tqdm(total=max_iter)
    for i in range(max_iter):
        # images: [nv, h, w, 4], alpha: [nv, h, w, 1]
        img_reco, _ = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=tv, f=faces_torch, vt=uvs_torch, 
                                      ft=faces_torch, tex=para, max_mip_level=0)
        img_reco = img_reco[0]
        
        reco = (img_reco[alpha_mp>0.5] - (color_label[alpha_mp>0.5])).pow(2).sum(dim=-1).mean()
        # reco = ((img_reco - color_label) * mask).pow(2).sum(dim=-1).mean()
        smoothness = (para[1:] - para[:-1])[torch.logical_or(grad_mask[1:],grad_mask[:-1])].pow(2).sum(dim=-1).mean() + \
                    (para[:, 1:] - para[:, :-1])[torch.logical_or(grad_mask[:,1:],grad_mask[:,:-1])].pow(2).sum(dim=-1).mean()
        loss = reco + 0.002 * smoothness
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
                
        pb.set_description(f'img_reo {reco.item():.5f}')
        pb.update(1)
    
    tsz = para.shape[0]
    img = Image.fromarray(np.uint8(np.clip(para.cpu().data.numpy(), 0, 1)*255)[np.arange(tsz-1,-1,-1)])
    img.save(os.path.join(out_dir,f"quest_deformed_tex.png"))

    # """step3 using back rendering to get the tex map correspondence"""
    # # get tex mask
    # para = nn.Parameter(torch.ones_like(quest_tex))
    # # para = nn.Parameter(torch.zeros_like(quest_tex.detach().clone())-1)
    # tv = quest_tv.detach().clone()
    # faces_torch = quest_faces_torch
    # uvs_torch = quest_uvs_torch
    # img_reco, _ = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=tv, f=faces_torch, vt=uvs_torch, 
    #                                 ft=faces_torch, tex=para, max_mip_level=0)
    # img_reco = img_reco[0]
    # img_reco[alpha_mp>0.5].sum().backward()
    # grad_mask = para.grad.abs().sum(dim=-1)>0
    # grad_mask = grad_mask[np.arange(-1, -1-grad_mask.shape[0],-1)]

    # # xyz_texmap_data = np.load(f'{model_dir}/xyz_texmap.npz',allow_pickle=True)
    # # xyz_tex = torch.from_numpy(xyz_texmap_data['xyz_tex']).float().cuda()
    # # xyz_mask = torch.from_numpy(xyz_texmap_data['mask']).float().cuda()
    
    # ctx = dr.RasterizeCudaContext(device=quest_uvs_torch.device)
    # verts_u, verts_v = quest_uvs_torch[:,0][None,:,None], 1-quest_uvs_torch[:,1][None,:,None] #torch.split(uvs, 1, dim=-1) # [1, verts, 1]
    # verts_uv_ndc = torch.cat([verts_u*2-1, verts_v*2-1, torch.zeros_like(verts_u), torch.ones_like(verts_u)], dim=-1)
    # rast_tex, _ = dr.rasterize(ctx, verts_uv_ndc.contiguous(), faces_torch, (quest_tex.shape[0], quest_tex.shape[0],))
    # mask = rast_tex[0,:,:,-1] > 0
    # # img = Image.fromarray(np.uint8(mask.cpu().data.numpy()*255))
    # # img.save('../output/img.png')
    # # mask = mask.float().cpu().data.numpy()
    # xyz_tex, _ = dr.interpolate(tv.contiguous()[None], rast_tex, faces_torch) # [1, tex_res, tex_res, 3]
    # xyz_tex = xyz_tex[0]
    
    # v_cam = torch.matmul(F.pad(xyz_tex.reshape(-1,3), pad=(0, 1), mode='constant', value=1.0), torch.inverse(cam2world).transpose(-1,-2)).float()
    # v_clip = v_cam @ Pndc.transpose(-1,-2)
    # v_clip = v_clip.reshape([xyz_tex.shape[0],xyz_tex.shape[0],-1])[:,:,:2]

    # # xyz_tex_homo = torch.cat([xyz_tex,torch.ones_like(xyz_tex[:,:,:1])],dim=-1)
    # # uv_xyz_tex = (torch.inverse(cam2world[0]) @ xyz_tex_homo.reshape([-1,4]).transpose(1,0)).transpose(1,0)[:,:2].reshape([xyz_tex.shape[0],xyz_tex.shape[1],2])
    # # uv_xyz_tex = uv_xyz_tex * zoom #(-1,1)

    # color_img = torch.from_numpy(output/255).float().cuda()
    # # mask_color =  torch.from_numpy(mask_cv > 128).float().cuda()

    # color_from_face = nn.functional.grid_sample(color_img.permute([2,0,1])[None], v_clip[None])[0].permute(1,2,0) # [nv_mp, 3]
    # # face_mask_xyz_tex = nn.functional.grid_sample(mask_color[None,None], uv_xyz_tex[None])[0,0] # [nv_mp, 3]
    # # blending_mask = np.uint8(torch.logical_and(torch.logical_and(xyz_mask>0, face_mask_xyz_tex>0.5), grad_mask).float().cpu().data.numpy()*255)
    # blending_mask = np.uint8(grad_mask.float().cpu().data.numpy()*255)
    # # Creating kernel 
    # # kernel = np.ones((3, 3), np.uint8)  
    # # blending_mask = cv2.erode(blending_mask, kernel) 
    # color_from_face = np.uint8(color_from_face.cpu().data.numpy() * 255)

    # img = Image.fromarray(blending_mask)
    # img.save(os.path.join(out_dir,f"blending_mask.png"))

    # img = Image.fromarray(color_from_face * (blending_mask[...,None]>128))
    # img.save(os.path.join(out_dir,f"color_from_face.png"))

    # quest_deformed_tex = copy.deepcopy(head_tex)
    # quest_deformed_tex[blending_mask>128] = color_from_face[blending_mask>128]/255.
    # img = Image.fromarray(np.uint8(quest_deformed_tex*255))
    # img.save(os.path.join(out_dir,f"quest_deformed_tex_before_blend.png"))
    # st()
    # v, u = np.where(blending_mask > 128)
    # center = (int((u.min() + u.max()) /2), int((v.min() + v.max()) /2))
    # output = cv2.seamlessClone(color_from_face, np.uint8(255*head_tex), blending_mask, center, cv2.NORMAL_CLONE)
    # img = Image.fromarray(output)
    # img.save(os.path.join(out_dir,f"quest_deformed_tex.png"))

    # # src_img = output
    # # mask = np.uint8(torch.logical_and(xyz_mask>0,face_mask_xyz_tex<=0.5).float().cpu().data.numpy()*255)
    # # tgt_img = np.uint8(tex*255)
    # # output = hist_matching(src_img, mask, tgt_img=tgt_img, tgt_mask=mask)
    # # img = Image.fromarray(output)
    # # img.save(os.path.join(out_dir,f"quest_deformed_tex.png"))


    print(f'finish face texture baking time used {time.time()-start_time:.3f}')
    print(f'all done time used in total {time.time()-start_time0:.3f}')
    def zip_files(file_list, zip_filename):
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in file_list:
                if os.path.isfile(file):
                    zipf.write(file, os.path.basename(file))
                    print(f"Added {file} to {zip_filename}")
                else:
                    print(f"File {file} not found.")
    
    zip_files([f"{out_dir}/quest_deformed.obj",f"{out_dir}/quest_deformed_tex.png",f"{out_dir}/quest_deformed.mtl"], f"{out_dir}/quest_head.zip")
 
    return f"{out_dir}/quest_deformed.obj", f"{out_dir}/quest_head.zip"


def hist_matching(src_img, src_mask, tgt_img=None, tgt_mask=None):
    """
    mask: h,w
    img: h,w,3
    all images' dtype is uint8
    """
    def calculate_cdf(histogram):
        """Calculate the cumulative distribution function (CDF) from a histogram."""
        cdf = histogram.cumsum()# Cumulative sum of the histogram
        cdf_normalized = cdf / cdf[-1]# Normalize to get CDF in range [0,1]
        return cdf_normalized
    
    src_pixel = src_img[src_mask>128][:,:3]
    tgt_pixel = tgt_img[tgt_mask>128][:,:3]

    img_final = []
    mappings = []
    for i in range(3):
        src_hist, _ = np.histogram(src_pixel[:,i], bins=256, range=(0, 256))
        # Calculate the CDFs
        source_cdf = calculate_cdf(src_hist)
        tgt_hist, _ = np.histogram(tgt_pixel[:,i], bins=256, range=(0, 256))
        reference_cdf = calculate_cdf(tgt_hist)

        # Create a mapping from source image pixel values to reference image pixel values
        mapping = np.zeros(256, dtype=np.uint8)
        for src_pixel_value in range(256):
            ref_pixel_value = np.argmin(np.abs(source_cdf[src_pixel_value] - reference_cdf))
            mapping[src_pixel_value] = ref_pixel_value
        mappings.append(mapping[None])

        # Apply the mapping to get the matched image
        matched_image = mapping[src_img[:,:,i:i+1]]
        img_final.append(matched_image)
    img_final = np.concatenate(img_final, axis=-1)
    return img_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default="/aigc_cfs_2/weimao/avatar_face_generation/output/web/taylor/cuteyou.png")
    parser.add_argument('--out_dir', type=str, default="/aigc_cfs_2/weimao/avatar_face_generation/output/quest/taylor_pipeline/")
    parser.add_argument('--is_rm_hair', type=bool, default=True)
    args = parser.parse_args()
    
    base_options = python.BaseOptions(model_asset_path='/aigc_cfs_2/weimao/avatar_face_generation/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,num_faces=1)
    face_detector = vision.FaceLandmarker.create_from_options(options)

    sam_checkpoint = "/aigc_cfs_2/weimao/pretrained_model_cache/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)


    inpainting_pipeline = AutoPipelineForInpainting.from_pretrained(
        "/aigc_cfs_2/weimao/pretrained_model_cache/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
        )
    inpainting_pipeline.enable_model_cpu_offload()
    
    overall_pipeline(args.file_path, face_detector, sam_predictor, inpainting_pipeline, out_dir=args.out_dir)



