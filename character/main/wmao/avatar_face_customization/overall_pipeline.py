
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

import zipfile

def extract_mesh(rgb_image, detection_result, out_dir = './', canonical_face_model=None):
	"""
	extract mesh from mp face
	rbg_image: face image
	detection_result: results from mp face detector
	out_file: output file 
	canonical_face_model: 
	
	"""
	if canonical_face_model is None:
		# connection to faces
		conns = mp.solutions.face_mesh.FACEMESH_TESSELATION
		faces = []
		sz = len(conns)
		conns = list(conns)
		edges = {}
		for con in conns:
			con = list(con)
			i,j	= con[0], con[1]
			if i not in edges.keys():
				edges[i] = [j]
			elif j not in edges[i]:
				edges[i].append(j)

		faces = []
		for i, v in edges.items():
			for j in v:
				for k in edges[j]:
					if k in v and {i,j,k} not in faces:
						faces.append({i,j,k})
		faces = [list(a) for a in faces]
		
		faces_left_eye = [[33,7,246],[246,7,161],[161,7,163],[161,163,160],[163,160,144],[160,144,145],[160,145,159],[145,159,153],[159,153,158],[153,158,154],[158,154,157],[154,157,155],[155,157,173],[155,173,133]]
		faces.extend(faces_left_eye)
		faces_right_eye = [[362,382,398],[382,398,381],[398,381,384],[381,384,380],[384,380,385],[380,385,374],[385,374,373],[385,373,386],[373,386,390],[386,390,387],[390,387,249],[387,249,388],[388,249,466],[249,466,263]]
		faces.extend(faces_right_eye)
		faces_mouth = [[78,95,191],[95,191,80],[80,95,88],[80,88,178],[80,178,81],[81,178,87],[81,87,82],[82,87,14],[82,14,13],[14,13,317],[13,317,312],[312,317,402],[312,402,311],[311,402,318],[318,311,310],[318,310,324],[324,310,415],[415,324,308]]
		faces.extend(faces_mouth)
		
		faces = np.array(faces)
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
		
		img_torch = torch.from_numpy(rgb_image).float() / 255.0
		uvs_torch = torch.from_numpy(uvs).float()
		uvs_torch[:, 0] = uvs_torch[:, 0] * 2 / (image_cols - 1) - 1
		uvs_torch[:, 1] = uvs_torch[:, 1] * 2 / (image_rows - 1) - 1
		colors = torch.nn.functional.grid_sample(img_torch.permute(2,0,1)[None],uvs_torch[None,None])
		colors = colors.data.numpy()[0,:,0].transpose(1,0)
		faces_new = []
		for f in faces:
			norm = np.cross(verts[f[1]]-verts[f[0]], verts[f[2]]-verts[f[0]])
			f_new = f.tolist()
			if norm[2] > 0: 
				f_new = [f[2],f[1],f[0]]
			faces_new.append(f_new)
		faces = np.array(faces_new)
		mesh = o3d.geometry.TriangleMesh()
		mesh.vertices = o3d.utility.Vector3dVector(verts)
		mesh.triangles = o3d.utility.Vector3iVector(faces)
		mesh.compute_vertex_normals()
		mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
		o3d.io.write_triangle_mesh(f'{out_dir}/face_mesh_vex.ply',mesh)
		
		out_file = os.path.join(out_dir,f"face_mesh.obj")
		xatlas.export(out_file, verts, faces)
		# mesh = trimesh.Trimesh(vertices=verts, faces=faces)
		# mesh.export(f'{out_file}.obj') 
	else:
		mesh_cano = trimesh.load(canonical_face_model)
		faces = mesh_cano.faces
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
		# mesh_cano.vertices = verts[:mesh_cano.vertices.shape[0]]
		# mesh_cano.export(out_file)
		out_file = os.path.join(out_dir,f"face_mesh.obj")
		xatlas.export(out_file, verts[:mesh_cano.vertices.shape[0]], mesh_cano.faces, mesh_cano.visual.uv)
	return verts, faces, uvs, mesh_cano.visual.uv

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

def mp_face_detector(file_path, out_dir, detector, canonical_face_dir = None):
	# STEP 3: Load the input image.
	image = mp.Image.create_from_file(file_path)
	# STEP 4: Detect face landmarks from the input image.
	detection_result = detector.detect(image)

	# from canonical face to detected face
	np.savez_compressed(f"{out_dir}/face_pose.npz", face_pose=detection_result.facial_transformation_matrixes[0])
	verts, faces, uvs_img, uvs_tex  = extract_mesh(image.numpy_view()[:,:,:3], detection_result, 
														out_dir=out_dir,
														canonical_face_model=canonical_face_dir)
	# STEP 5: Process the detection result. In this case, visualize it.
	annotated_image = draw_landmarks_on_image(image.numpy_view()[:,:,:3], detection_result)

	img = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
	cv2.imwrite(f"{out_dir}/annotated_img.png", img)
	image = image.numpy_view()
	return verts, faces, uvs_img, uvs_tex, image

def remove_hair(image, uvs, out_dir, predictor):
	
	# STEP 5 segment the face region using SAM
	# using two eyes mouth and nose as prompt
	image = image[:,:,:3]
	predictor.set_image(image)
	
	prompt_idx = [4, 468, 473, 101, 330, 0, 17, 40, 270, 52, 282]
	input_point = uvs[prompt_idx]
	mouth = (uvs[13] + uvs[14])/2
	input_point = np.concatenate([input_point,mouth[None]],axis=0)
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

def overall_pipeline(file_path, face_detector, sam_predictor, inpainting_pipeline, out_dir=None, is_rm_hair=True):
	st0 = time.time()
	img_name = os.path.basename(file_path).split('.')[0]
	if out_dir is None:
		out_dir = os.path.dirname(file_path) + '/' + img_name
	os.makedirs(out_dir,exist_ok=True)

	# resize face image to 1024
	sz=1024
	image = Image.open(file_path)
	image = image.resize((sz,sz))
	image.save(out_dir + '/' + img_name + '.png')
	file_path = out_dir + '/' + img_name + '.png'
	
	"""## step 1 face detector
	
	"""
	# base_options = python.BaseOptions(model_asset_path='./face_landmarker_v2_with_blendshapes.task')
	# options = vision.FaceLandmarkerOptions(base_options=base_options,output_face_blendshapes=True,
	# 										output_facial_transformation_matrixes=True,num_faces=1)
	# detector = vision.FaceLandmarker.create_from_options(options)
	detector = face_detector

	# image is a numpy array with shape (h,w,3/4)
	verts, faces, uvs_img, uvs_tex, image = mp_face_detector(file_path, out_dir, detector, canonical_face_dir="./canonical_face_model_uv_edited_v2.obj")
	print(f'>>> finish face mesh generation in {time.time()-st0:.3f} s')
	
	"""## step2 optional remove hair
	
	"""
	if is_rm_hair:
		st1 = time.time()
		# sam_checkpoint = "/aigc_cfs_2/weimao/pretrained_model_cache/sam_vit_h_4b8939.pth"
		# model_type = "vit_h"
		# device = "cuda"
		# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
		# sam.to(device=device)
		# predictor = SamPredictor(sam)
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
	# vertex ids to be removed, remove the outer boundary of the mesh for better texture mapping
	idx_to_remove = [127,234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454,356,389,251,284,332,297,338,10,109,67,103,54,21,162]
	# vertex ids to be connected to head model (with respect to the original mesh)
	idx_to_connect = [34,227,137,117,215,138,135,169,170,140,171,175,396,369,395,394,364,367,435,401,366,447,264,368,301,298,333,299,337,151,108,69,104,68,71,139]
	

	st2 = time.time()
	# the verts will have 10 more points than the actual mesh because of the eye center etc.
	verts = verts[:-10] - 0.5
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
	if uvs is None:
		# potential risk is that atlas will change the number of verts after uv
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
		out_file = os.path.join(out_dir,f"face_mesh.obj")
		xatlas.export(out_file, verts, faces, uvs)
	
	out_file = os.path.join(out_dir,f"face_mesh.obj")
	add_mtl(out_file)

	tex = torch.zeros([1024, 1024, 3]).float().cuda()
	para = nn.Parameter(tex.detach().clone())
	
	if False:
		verts, faces, uvs = remove_vertices_by_id(verts, faces, uvs, idx_to_remove=idx_to_remove)

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
		if i % 100 == 0 and False:
			with torch.no_grad():
				img_reco[alpha<0.5] = 0
				img = Image.fromarray(np.uint8(img_reco.cpu().data.numpy()*255))
				img.save(os.path.join(out_dir,f"face_texture_{i:03d}.png"))
				
		pb.set_description(f'img_reo {reco.item():.5f}')
		pb.update(1)

	tsz = para.shape[0]
	img = Image.fromarray(np.uint8(torch.sigmoid(para).cpu().data.numpy()*255)[np.arange(tsz-1,-1,-1)])
	img.save(os.path.join(out_dir,f"face_mesh_tex.png"))
	print(f'>>> finish face texture baking iin {time.time()-st2:.3f} s')
	
	"""# step 4 align face
	
	"""
	st4 = time.time()
	vert_fa = verts
	face_fa = faces
	uv_fa = uvs_tex

	vert_fa_new, face_new, uv_fa_new = remove_vertices_by_id(vert_fa, face_fa, uv_fa, idx_to_remove=[127,234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454,356,389,251,284,332,297,338,10,109,67,103,54,21,162])
	idx_old = np.setdiff1d(np.arange(vert_fa.shape[0]),np.array(idx_to_remove))
	idx_to_connect_new = []
	for i in idx_to_connect:
		idx_to_connect_new.append(np.where(idx_old==i)[0][0])
	# get scale:
	# this key point is for head model in /aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_temp_uv_edited.obj
	head_keypts = np.array([[0, 1.58396, 0.191507], # 82 forehead
						 	[0, 1.16356, 0.183996], # 92 chin
						 	[-0.232957, 1.34533, 0.046458], # 9 left ear
						 	[0.232957, 1.34533, 0.046458], # 192 right ear
						 	])
	
	face_idx = [137, 158, 209, 412]
	h_head = np.linalg.norm(head_keypts[1] - head_keypts[0])
	w_head = np.linalg.norm(head_keypts[2] - head_keypts[3])
	h_face = np.linalg.norm(vert_fa_new[face_idx[1]] - vert_fa_new[face_idx[0]])
	w_face = np.linalg.norm(vert_fa_new[face_idx[2]] - vert_fa_new[face_idx[3]])
	scale = (h_head / h_face + w_head / w_face)/2
	vert_fa_new = vert_fa_new * scale
	
	x_ax = (vert_fa_new[face_idx[3]] - vert_fa_new[face_idx[2]]) / (w_face * scale)
	y_ax = (vert_fa_new[face_idx[0]] - vert_fa_new[face_idx[1]]) / (h_face * scale)
	z_ax = np.cross(x_ax, y_ax)
	rot = np.vstack([x_ax,y_ax,z_ax])
	vert_fa_new = vert_fa_new @ rot.transpose(1, 0)
	t = head_keypts.mean(axis=0) - vert_fa_new[face_idx].mean(axis=0)
	vert_fa_new = vert_fa_new + t

	#  y = rot @ (s * x) + t
	np.savez_compressed(f"{out_dir}/face2head_transform.npz", scale=scale,rotation=rot,translation=t,comments='y=rot @ (s * x) + t')

	out_file = os.path.join(out_dir,f"face_mesh_aligned.obj")
	xatlas.export(out_file, vert_fa_new, face_new, uv_fa_new)
	
	# copy the first three line of the old obj file
	new_line = []
	with open(f"{out_dir}/face_mesh.obj", 'r') as file:
		new_line = file.readlines()[:3]
	lines = new_line
	with open(f"{out_dir}/face_mesh_aligned.obj", 'r') as file:
		for line in file:
			lines.append(line)

	with open(f"{out_dir}/face_mesh_aligned.obj", 'w') as file:
		file.writelines(lines)
	print(f'>>> face aligned in {time.time()-st4:.3f} s')

	"""# step 5 sewing face and head mesh and rebake the texture
	
	"""
	# cause head uv make the vertex number read using trimesh differ from that of the actual mesh so that we need to use the un-uved head mesh
	head_dir_wouv = "/aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_temp_edited_wouv.obj"
	# the idx to be connected refer to the face mesh with outer line of faces removed and the un-uved head mesh
	face_idx = np.array([137, 101, 64, 97, 63, 66, 128, 32, 32, 209, 126, 159, 159, 197, 127, 125, 153, 154, 129, 129, 155, 158, 363, 340, 340, 362, 361, 336, 338, 400, 366, 366, 337, 412, 244, 339, 278, 278, 275, 308, 276, 312])
	head_idx = np.array([82, 79, 78, 13, 84, 11, 10, 1, 0, 9, 16, 18, 17, 15, 64, 65, 67, 66, 70, 68, 85, 92, 262, 251, 249, 247, 248, 246, 245, 198, 200, 201, 199, 192, 183, 184, 193, 194, 261, 196, 257, 260])
	face_dir = out_dir + '/face_mesh_aligned.obj'
	face_tex_dir = out_dir + '/face_mesh_tex.png'
	head_dir = '/aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_temp_uv_edited.obj'
	head_tex_dir = "/aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_temp_uv_edited.png"
	head_tex_mask_dir = "/aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_temp_uv_edited_mask.png"

	st5 = time.time()
	max_iter = 500
	
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
	xatlas.export(out_dir+'/face_head.obj', verts, faces, uvs)

	# render face image
	zoom = 3.0
	Pndc = make_ndc(zoom=zoom, camera_type="ortho")
	Pndc = torch.from_numpy(Pndc).float().cuda()
	cam2world = torch.eye(4)[None].float().cuda().repeat(3,1,1)
	
	# front view rotate along x axis for 180 degree (front)
	cam2world[0,1,1] = -1.0
	cam2world[0,2,2] = -1.0
	cam2world[0,0,3] = vert_face[:,0].mean()
	cam2world[0,1,3] = vert_face[:,1].mean()
	cam2world[0,2,3] = 2.0

	# # right view
	cam2world[1,0,:] = torch.tensor([0,0,-1,0]).float().cuda()
	cam2world[1,1,:] = torch.tensor([0, -1,0,0]).float().cuda()
	cam2world[1,2,:] = torch.tensor([-1,0,0,0]).float().cuda()
	cam2world[1,0,3] = 2.0 
	cam2world[1,1,3] = vert_face[:,1].mean()
	cam2world[1,2,3] = -vert_face[:,0].mean()

	## left view
	cam2world[2,0,:] = torch.tensor([0,0,1,0]).float().cuda()
	cam2world[2,1,:] = torch.tensor([0, -1,0,0]).float().cuda()
	cam2world[2,2,:] = torch.tensor([1,0,0,0]).float().cuda()
	cam2world[2,0,3] = -2.0 
	cam2world[2,1,3] = vert_face[:,1].mean()
	cam2world[2,2,3] = -vert_face[:,0].mean()
	
	vert_face = torch.from_numpy(vert_face).float().cuda()
	face_fa = torch.from_numpy(face_fa).to(dtype=torch.int32).cuda()
	uv_face = torch.from_numpy(face_uv).float().cuda()

	vert_head = torch.from_numpy(vert_head).float().cuda()
	face_head = torch.from_numpy(face_head).to(dtype=torch.int32).cuda()
	head_uv = torch.from_numpy(head_uv).float().cuda()
	
	face_texture = Image.open(face_tex_dir)
	face_texture = torch.from_numpy(np.array(face_texture)/255).to(device='cuda', dtype=torch.float32)
	
	bg_color = face_texture[720:730, 610:620].mean(dim=(0,1)).cpu().data.numpy()
	idx = np.arange(face_texture.shape[0]-1,-1,-1)
	face_texture = face_texture[idx]
	

	sz = 1024
	img_face, alpha_face = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=vert_face, f=face_fa, vt=uv_face, ft=face_fa, tex=face_texture)
	alpha_face = (alpha_face > 0.5).float()
	for i in range(img_face.shape[0]):
		img_face_log = img_face.cpu().numpy()[i]
		img_face_log = Image.fromarray(np.uint8(img_face_log*255))
		img_face_log.save(f'{out_dir}/render_face_img_{i:01d}.png')
		mask = alpha_face[i].repeat([1,1,3]).cpu().data.numpy()
		mask = Image.fromarray(np.uint8(mask*255))
		mask.save(f'{out_dir}/render_face_mask_{i:01d}.png')

	head_mask = Image.open(head_tex_mask_dir)
	head_texture = Image.open(head_tex_dir)

	# change head texture according to face skin color
	head_texture = np.array(head_texture)
	head_mask = np.array(head_mask)
	bg_img = np.uint8(np.zeros_like(head_texture) + bg_color*255)

	v, u = np.where(head_mask[:,:,0] > 128)
	center = (int((u.min() + u.max()) /2), int((v.min() + v.max()) /2))
	head_texture = cv2.seamlessClone(head_texture, bg_img, head_mask, center, cv2.NORMAL_CLONE)

	head_texture = torch.from_numpy(head_texture/255).to(device='cuda', dtype=torch.float32)
	idx = np.arange(head_texture.shape[0]-1,-1,-1)
	head_texture = head_texture[idx]
	img_head, alpha_head = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=vert_head, f=face_head, vt=head_uv, ft=face_head, tex=head_texture)
	alpha_head = (alpha_head > 0.5).float()
	for i in range(img_head.shape[0]):
		img_head_log = img_head.cpu().numpy()[i]
		img_head_log = Image.fromarray(np.uint8(img_head_log*255))
		img_head_log.save(f'{out_dir}/render_head_img_{i:01d}.png')
		alpha_head = torch.logical_and(alpha_face < 0.9, alpha_head>0.5).float()
		mask = alpha_head[i].repeat([1,1,3]).cpu().data.numpy()
		mask = Image.fromarray(np.uint8(mask*255))
		mask.save(f'{out_dir}/render_head_mask_{i:01d}.png')
	
	img_comined= blend_face_to_head_img(head_mesh_dir='/aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_render.obj',
						   head_texture_dir='/aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_brighter.jpg',
						   Pndc=Pndc, cam2world=cam2world, face_mask=alpha_face.float().cpu().data.numpy(),
						   face_img=img_face.cpu().numpy(),out_dir=out_dir, bg_color=bg_color)
	
	image_label = torch.from_numpy(img_comined).float().cuda()
	verts = torch.from_numpy(verts).float().cuda()
	faces = torch.from_numpy(faces.astype(np.int32)).cuda()
	uvs = torch.from_numpy(uvs).float().cuda()
	
	head_texture = Image.open(head_tex_dir)
	head_texture = torch.from_numpy(np.array(head_texture)/255).to(device='cuda', dtype=torch.float32)
	idx = np.arange(head_texture.shape[0]-1,-1,-1)
	head_texture = head_texture[idx]

	head_texture_mask = Image.open(head_tex_mask_dir)
	head_texture_mask = torch.from_numpy(np.array(head_texture_mask)/255).to(device='cuda', dtype=torch.float32)
	idx = np.arange(head_texture.shape[0]-1,-1,-1)
	head_texture_mask = head_texture_mask[idx][:,:,0]
	
	# para = nn.Parameter(head_texture[head_texture_mask<0.9][:,:3].detach().clone())
	para = nn.Parameter(head_texture[:,:,:3].detach().clone())
	optimizer = optim.Adam([para],lr=0.25,betas=(0.9,0.999))
	scheduler = lr_scheduler.ExponentialLR(optimizer, gamma =0.99)
	# optimizer= optim.SGD([para], lr=1.0, momentum=0.9)
	pb = tqdm(total=max_iter)
	for i in range(max_iter):
		tex_tmp = para
		img_reco, alpha = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=verts, f=faces, vt=uvs, ft=faces, tex=tex_tmp, max_mip_level=3)
		img_reco = img_reco
		alpha = alpha[:,:,:,0]

		reco = (img_reco[alpha>0.5] - image_label[alpha>0.5]).pow(2).sum(dim=-1).mean()
		loss = reco
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()	
		scheduler.step()
		if i % 100 == 0 and False:
			with torch.no_grad():
				img_reco[alpha<0.5] = 0
				for j in range(img_reco.shape[0]):
					img = Image.fromarray(np.uint8(img_reco[j].cpu().data.numpy()*255))
					img.save(os.path.join(out_dir,f"render_combine_{i:03d}_{j:01d}.png"))
		# break
		pb.set_description(f'img_reo {reco.item():.5f}')
		pb.update(1)

	# tex_tmp = head_texture[:,:,:3].detach().clone()
	# tex_tmp[head_texture_mask<0.9] = para
	tex_tmp = para
	tsz = tex_tmp.shape[0]
	img = Image.fromarray(np.uint8(torch.clamp(tex_tmp,min=0,max=1).cpu().data.numpy()*255)[np.arange(tsz-1,-1,-1)])
	img.save(os.path.join(out_dir,f"face_head_tex.png"))
	add_mtl(f"{out_dir}/face_head.obj")

	print(f'>>> face head sewing in {time.time()-st5:.3f} s')
	print(f'>>> all finished in {time.time()-st0:.3f} s')
	def zip_files(file_list, zip_filename):
		with zipfile.ZipFile(zip_filename, 'w') as zipf:
			for file in file_list:
				if os.path.isfile(file):
					zipf.write(file, os.path.basename(file))
					print(f"Added {file} to {zip_filename}")
				else:
					print(f"File {file} not found.")
	
	zip_files([f"{out_dir}/face_head.obj",f"{out_dir}/face_head_tex.png",f"{out_dir}/face_head.mtl"], f"{out_dir}/head.zip")
	return f"{out_dir}/face_head.obj", f"{out_dir}/head.zip"

def hist_matching(src_img, src_mask, tgt_img, tgt_mask):
	"""
	mask: h,w
	img: h,w,3
	all images' dtype is uint8
	"""
	def calculate_cdf(histogram):
		"""Calculate the cumulative distribution function (CDF) from a histogram."""
		cdf = histogram.cumsum()  # Cumulative sum of the histogram
		cdf_normalized = cdf / cdf[-1]  # Normalize to get CDF in range [0,1]
		return cdf_normalized
	
	src_pixel = src_img[src_mask>128][:,:3]
	tgt_pixel = tgt_img[tgt_mask>128][:,:3]

	img_final = []
	mappings = []
	for i in range(3):
		src_hist, _ = np.histogram(src_pixel[:,i], bins=256, range=(0, 256))
		tgt_hist, _ = np.histogram(tgt_pixel[:,i], bins=256, range=(0, 256))
		# Calculate the CDFs
		source_cdf = calculate_cdf(src_hist)
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
	parser.add_argument('--file_path', type=str, default="/aigc_cfs_2/weimao/avatar_face_generation/test_data/cute_you1.png")
	parser.add_argument('--is_rm_hair', type=bool, default=True)
	args = parser.parse_args()
	
	base_options = python.BaseOptions(model_asset_path='./face_landmarker_v2_with_blendshapes.task')
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
	
	overall_pipeline(args.file_path, face_detector, sam_predictor, inpainting_pipeline)

	# st0 = time.time()
	# file_path = args.file_path
	# img_name = os.path.basename(file_path).split('.')[0]
	# out_dir = os.path.dirname(file_path) + '/' + img_name
	# os.makedirs(out_dir,exist_ok=True)
	
	# """## step 1 face detector
	
	# """
	# base_options = python.BaseOptions(model_asset_path='./face_landmarker_v2_with_blendshapes.task')
	# options = vision.FaceLandmarkerOptions(base_options=base_options,output_face_blendshapes=True,
	# 										output_facial_transformation_matrixes=True,num_faces=1)
	# detector = vision.FaceLandmarker.create_from_options(options)

	# # image is a numpy array with shape (h,w,3/4)
	# verts, faces, uvs_img, uvs_tex, image = mp_face_detector(file_path, out_dir, detector, canonical_face_dir="./canonical_face_model_uv_edited_v2.obj")
	# print(f'>>> finish face mesh generation in {time.time()-st0:.3f} s')
	
	# """## step2 optional remove hair
	
	# """
	# if args.is_rm_hair:
	# 	st1 = time.time()
	# 	sam_checkpoint = "/aigc_cfs_2/weimao/pretrained_model_cache/sam_vit_h_4b8939.pth"
	# 	model_type = "vit_h"
	# 	device = "cuda"
	# 	sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
	# 	sam.to(device=device)
	# 	predictor = SamPredictor(sam)
	# 	remove_hair(image, uvs_img, out_dir, predictor)
	# 	print(f'>>> finish hair mask generation in {time.time()-st1:.3f} s')
	
	# """## step3 face texture baking
	
	# """
	# pipeline = AutoPipelineForInpainting.from_pretrained(
	# 	"/aigc_cfs_2/weimao/pretrained_model_cache/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
	# 	)
	# pipeline.enable_model_cpu_offload()
	# generator = torch.Generator("cuda").manual_seed(92)
	# prompt = "bald cartoon head, no hair, skin color, no shadow, no shading, no grey color, no ear"
	# negative_prompt = "hair, hair, hair, black, shading, shading, shading, shadow, shadow, shadow, highlight, ear, boundary, edges"
	# # vertex ids to be removed, remove the outer boundary of the mesh for better texture mapping
	# idx_to_remove = [127,234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454,356,389,251,284,332,297,338,10,109,67,103,54,21,162]
	# # vertex ids to be connected to head model (with respect to the original mesh)
	# idx_to_connect = [34,227,137,117,215,138,135,169,170,140,171,175,396,369,395,394,364,367,435,401,366,447,264,368,301,298,333,299,337,151,108,69,104,68,71,139]
	

	# st2 = time.time()
	# # the verts will have 10 more points than the actual mesh because of the eye center etc.
	# verts = verts[:-10] - 0.5
	# faces = faces
	# uvs = uvs_tex
	# # img = Image.open(file_path)
	# # img= np.array(img)[:,:,:3]/255.
	# img = image[:,:,:3]/255
	# sz = img.shape[0]
	# color_label = torch.from_numpy(img).float().cuda()
	# mask = None
	# try:
	# 	mask = Image.open(f"{out_dir}/hair_mask.png")
	# 	mask = mask.filter(ImageFilter.GaussianBlur(radius = 10)) 
	# 	mask = np.array(mask)/255.
	# 	# mask_outer = torch.from_numpy(mask[:,:,0]).float().cuda() < 0.01
	# 	mask = torch.from_numpy(mask[:,:,0]).float().cuda() > 0.9
	# except:
	# 	print('no mask')

	# zoom = 2.0
	# Pndc = make_ndc(zoom=zoom, camera_type="ortho")
	# Pndc = torch.from_numpy(Pndc).float().cuda()
	# cam2world = torch.eye(4)[None].float().cuda()
	# # cam2world[:,0,2] = 0.5
	# # cam2world[:,1,2] = 0.1
	# cam2world[:,2,3] = -2.0
	# if uvs is None:
	# 	# potential risk is that atlas will change the number of verts after uv
	# 	atlas = xatlas.Atlas()
	# 	# use original mesh not the manifold_mesh
	# 	atlas.add_mesh(verts, faces)
	# 	chart_options = xatlas.ChartOptions()
	# 	chart_options.max_iterations = 10
	# 	pack_options = xatlas.PackOptions()
	# 	pack_options.create_image = False
	# 	pack_options.padding = 5
	# 	pack_options.bruteForce = True
	# 	# pack_options.max_chart_size = 1024
	# 	atlas.generate(pack_options=pack_options,chart_options=chart_options)
	# 	vmapping, faces, uvs = atlas[0]
	# 	verts = verts[vmapping.tolist()]
	# 	out_file = os.path.join(out_dir,f"face_mesh.obj")
	# 	xatlas.export(out_file, verts, faces, uvs)
	
	# out_file = os.path.join(out_dir,f"face_mesh.obj")
	# add_mtl(out_file)

	# tex = torch.zeros([1024, 1024, 3]).float().cuda()
	# para = nn.Parameter(tex.detach().clone())
	
	# if False:
	# 	verts, faces, uvs = remove_vertices_by_id(verts, faces, uvs, idx_to_remove=idx_to_remove)

	# tv = torch.from_numpy(verts).float().cuda()
	# faces_torch = torch.from_numpy(faces.astype(np.int32)).cuda()
	# uvs_torch = torch.from_numpy(uvs).float().cuda()
	# max_iter = 500
	# optimizer = optim.Adam([para],lr=0.5,betas=(0.9,0.999))
	# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma =0.99)
	# # optimizer= optim.SGD([para], lr=1.0, momentum=0.9)
	# pb = tqdm(total=max_iter)
	# for i in range(max_iter):
	# 	# images: [nv, h, w, 4], alpha: [nv, h, w, 1]
	# 	img_reco, alpha = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=tv, f=faces_torch, vt=uvs_torch, ft=faces_torch, tex=torch.sigmoid(para))
	# 	img_reco = img_reco[0]
	# 	alpha = alpha[0,:,:,0]
		
	# 	if i == 0 and mask is not None:
	# 		# get outer mask
	# 		kernel = torch.ones([1, 1, 21, 21],dtype=alpha.dtype,device=alpha.device)/100
	# 		mask_outer = alpha.unsqueeze(0).unsqueeze(0)
	# 		for _ in range(3):
	# 			mask_outer = F.conv2d(mask_outer, kernel, padding=10, groups=1)
	# 		mask_outer = mask_outer[0,0] <= 0

	# 		mask = alpha.clone().detach() * mask
	# 		# inpaint the image
	# 		mask_inpaint = torch.logical_or(mask[:,:,None] > 0.5, mask_outer[:,:,None])
	# 		mask_inpaint = torch.cat([mask_inpaint,mask_inpaint,mask_inpaint],dim=-1) * 255
	# 		mask_inpaint = np.uint8(mask_inpaint.cpu().data.numpy())
	# 		mask_inpaint = Image.fromarray(mask_inpaint)
	# 		mask_inpaint.save(f'{out_dir}/inpaint_mask.png')
	# 		img_inpaint = color_label.clone().detach()
	# 		img_inpaint[mask<=0.5] = torch.from_numpy(np.array([251,216,197])/255).to(device=color_label.device, dtype=color_label.dtype)
	# 		# img_inpaint[mask_outer] = torch.from_numpy(np.array([251,216,197])/255).to(device=color_label.device, dtype=color_label.dtype)
	# 		# img_inpaint = torch.cat([img_inpaint, (alpha[:,:,None]>0.5) * 1.0], dim=-1)
	# 		img_inpaint = Image.fromarray(np.uint8(255*img_inpaint.cpu().data.numpy()))
	# 		img_inpaint.save(f'{out_dir}/img_masked.png')
	# 		mask_inpaint_invert = ImageOps.invert(mask_inpaint)

	# 		generator = torch.Generator("cuda").manual_seed(92)
	# 		image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=img_inpaint, mask_image=mask_inpaint_invert, generator=generator,
	# 						guidance_scale=10.0).images[0]
	# 		image.save(f'{out_dir}/img_inpainted.png')
	# 		color_label = torch.from_numpy(np.array(image) / 255.0).to(device=color_label.device, dtype=color_label.dtype)

	# 	reco = (img_reco[alpha>0.5] - (color_label[alpha>0.5])).pow(2).sum(dim=-1).mean()
	# 	loss = reco
	# 	optimizer.zero_grad()
	# 	loss.backward()
	# 	optimizer.step()
	# 	scheduler.step()
	# 	if i % 100 == 0 and False:
	# 		with torch.no_grad():
	# 			img_reco[alpha<0.5] = 0
	# 			img = Image.fromarray(np.uint8(img_reco.cpu().data.numpy()*255))
	# 			img.save(os.path.join(out_dir,f"face_texture_{i:03d}.png"))
				
	# 	pb.set_description(f'img_reo {reco.item():.5f}')
	# 	pb.update(1)

	# tsz = para.shape[0]
	# img = Image.fromarray(np.uint8(torch.sigmoid(para).cpu().data.numpy()*255)[np.arange(tsz-1,-1,-1)])
	# img.save(os.path.join(out_dir,f"face_mesh_tex.png"))
	# print(f'>>> finish face texture baking iin {time.time()-st2:.3f} s')
	
	# """# step 4 align face
	
	# """
	# st4 = time.time()
	# vert_fa = verts
	# face_fa = faces
	# uv_fa = uvs_tex

	# vert_fa_new, face_new, uv_fa_new = remove_vertices_by_id(vert_fa, face_fa, uv_fa, idx_to_remove=[127,234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454,356,389,251,284,332,297,338,10,109,67,103,54,21,162])
	# idx_old = np.setdiff1d(np.arange(vert_fa.shape[0]),np.array(idx_to_remove))
	# idx_to_connect_new = []
	# for i in idx_to_connect:
	# 	idx_to_connect_new.append(np.where(idx_old==i)[0][0])
	# # get scale:
	# # this key point is for head model in /aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_temp_uv_edited.obj
	# head_keypts = np.array([[0, 1.58396, 0.191507], # 82 forehead
	# 					 	[0, 1.16356, 0.183996], # 92 chin
	# 					 	[-0.232957, 1.34533, 0.046458], # 9 left ear
	# 					 	[0.232957, 1.34533, 0.046458], # 192 right ear
	# 					 	])
	
	# face_idx = [137, 158, 209, 412]
	# h_head = np.linalg.norm(head_keypts[1] - head_keypts[0])
	# w_head = np.linalg.norm(head_keypts[2] - head_keypts[3])
	# h_face = np.linalg.norm(vert_fa_new[face_idx[1]] - vert_fa_new[face_idx[0]])
	# w_face = np.linalg.norm(vert_fa_new[face_idx[2]] - vert_fa_new[face_idx[3]])
	# scale = (h_head / h_face + w_head / w_face)/2
	# vert_fa_new = vert_fa_new * scale
	
	# x_ax = (vert_fa_new[face_idx[3]] - vert_fa_new[face_idx[2]]) / (w_face * scale)
	# y_ax = (vert_fa_new[face_idx[0]] - vert_fa_new[face_idx[1]]) / (h_face * scale)
	# z_ax = np.cross(x_ax, y_ax)
	# rot = np.vstack([x_ax,y_ax,z_ax])
	# vert_fa_new = vert_fa_new @ rot.transpose(1, 0)
	# t = head_keypts.mean(axis=0) - vert_fa_new[face_idx].mean(axis=0)
	# vert_fa_new = vert_fa_new + t

	# #  y = rot @ (s * x) + t
	# np.savez_compressed(f"{out_dir}/face2head_transform.npz", scale=scale,rotation=rot,translation=t,comments='y=rot @ (s * x) + t')

	# out_file = os.path.join(out_dir,f"face_mesh_aligned.obj")
	# xatlas.export(out_file, vert_fa_new, face_new, uv_fa_new)
	
	# # copy the first three line of the old obj file
	# new_line = []
	# with open(f"{out_dir}/face_mesh.obj", 'r') as file:
	# 	new_line = file.readlines()[:3]
	# lines = new_line
	# with open(f"{out_dir}/face_mesh_aligned.obj", 'r') as file:
	# 	for line in file:
	# 		lines.append(line)

	# with open(f"{out_dir}/face_mesh_aligned.obj", 'w') as file:
	# 	file.writelines(lines)
	# print(f'>>> face aligned in {time.time()-st4:.3f} s')

	# """# step 5 sewing face and head mesh and rebake the texture
	
	# """
	# # cause head uv make the vertex number read using trimesh differ from that of the actual mesh so that we need to use the un-uved head mesh
	# head_dir_wouv = "/aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_temp_edited_wouv.obj"
	# # the idx to be connected refer to the face mesh with outer line of faces removed and the un-uved head mesh
	# face_idx = np.array([137, 101, 64, 97, 63, 66, 128, 32, 32, 209, 126, 159, 159, 197, 127, 125, 153, 154, 129, 129, 155, 158, 363, 340, 340, 362, 361, 336, 338, 400, 366, 366, 337, 412, 244, 339, 278, 278, 275, 308, 276, 312])
	# head_idx = np.array([82, 79, 78, 13, 84, 11, 10, 1, 0, 9, 16, 18, 17, 15, 64, 65, 67, 66, 70, 68, 85, 92, 262, 251, 249, 247, 248, 246, 245, 198, 200, 201, 199, 192, 183, 184, 193, 194, 261, 196, 257, 260])
	# face_dir = out_dir + '/face_mesh_aligned.obj'
	# face_tex_dir = out_dir + '/face_mesh_tex.png'
	# head_dir = '/aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_temp_uv_edited.obj'
	# head_tex_dir = "/aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_temp_uv_edited.png"
	# head_tex_mask_dir = "/aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_temp_uv_edited_mask.png"

	# st5 = time.time()
	# max_iter = 500
	
	# face_mesh = trimesh.load(face_dir)
	# vert_face = face_mesh.vertices
	# face_fa = face_mesh.faces
	# face_uv = face_mesh.visual.uv

	# head_mesh_wouv = trimesh.load(head_dir_wouv)
	# vert_head_wouv = head_mesh_wouv.vertices
	# face_head_wouv = head_mesh_wouv.faces # N x 3
	# # vert_head_wouv[head_idx] = vert_face[face_idx]

	# head_mesh = trimesh.load(head_dir)
	# vert_head = head_mesh.vertices
	# face_head = head_mesh.faces
	# head_uv = head_mesh.visual.uv
	
	# cdist = np.linalg.norm(vert_head_wouv[head_idx,None] - vert_head[None],axis=-1)
	# idx0, idx1 = np.where(cdist <= 1e-10)
	# vert_head[idx1] = vert_face[face_idx[idx0]]
	
	# # combine two mesh
	# verts = np.concatenate([vert_face, vert_head], axis=0)
	# faces = np.concatenate([face_fa, face_head + vert_face.shape[0]], axis=0)
	# uvs = np.concatenate([face_uv,head_uv], axis=0)
	# xatlas.export(out_dir+'/face_head.obj', verts, faces, uvs)

	# # render face image
	# zoom = 3.0
	# Pndc = make_ndc(zoom=zoom, camera_type="ortho")
	# Pndc = torch.from_numpy(Pndc).float().cuda()
	# cam2world = torch.eye(4)[None].float().cuda().repeat(3,1,1)
	
	# # front view rotate along x axis for 180 degree (front)
	# cam2world[0,1,1] = -1.0
	# cam2world[0,2,2] = -1.0
	# cam2world[0,0,3] = vert_face[:,0].mean()
	# cam2world[0,1,3] = vert_face[:,1].mean()
	# cam2world[0,2,3] = 2.0

	# # # right view
	# cam2world[1,0,:] = torch.tensor([0,0,-1,0]).float().cuda()
	# cam2world[1,1,:] = torch.tensor([0, -1,0,0]).float().cuda()
	# cam2world[1,2,:] = torch.tensor([-1,0,0,0]).float().cuda()
	# cam2world[1,0,3] = 2.0 
	# cam2world[1,1,3] = vert_face[:,1].mean()
	# cam2world[1,2,3] = -vert_face[:,0].mean()

	# ## left view
	# cam2world[2,0,:] = torch.tensor([0,0,1,0]).float().cuda()
	# cam2world[2,1,:] = torch.tensor([0, -1,0,0]).float().cuda()
	# cam2world[2,2,:] = torch.tensor([1,0,0,0]).float().cuda()
	# cam2world[2,0,3] = -2.0 
	# cam2world[2,1,3] = vert_face[:,1].mean()
	# cam2world[2,2,3] = -vert_face[:,0].mean()
	
	# vert_face = torch.from_numpy(vert_face).float().cuda()
	# face_fa = torch.from_numpy(face_fa).to(dtype=torch.int32).cuda()
	# uv_face = torch.from_numpy(face_uv).float().cuda()

	# vert_head = torch.from_numpy(vert_head).float().cuda()
	# face_head = torch.from_numpy(face_head).to(dtype=torch.int32).cuda()
	# head_uv = torch.from_numpy(head_uv).float().cuda()
	
	# face_texture = Image.open(face_tex_dir)
	# face_texture = torch.from_numpy(np.array(face_texture)/255).to(device='cuda', dtype=torch.float32)
	
	# bg_color = face_texture[720:730, 610:620].mean(dim=(0,1)).cpu().data.numpy()
	# idx = np.arange(face_texture.shape[0]-1,-1,-1)
	# face_texture = face_texture[idx]
	

	# sz = 1024
	# img_face, alpha_face = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=vert_face, f=face_fa, vt=uv_face, ft=face_fa, tex=face_texture)
	# alpha_face = (alpha_face > 0.5).float()
	# for i in range(img_face.shape[0]):
	# 	img_face_log = img_face.cpu().numpy()[i]
	# 	img_face_log = Image.fromarray(np.uint8(img_face_log*255))
	# 	img_face_log.save(f'{out_dir}/render_face_img_{i:01d}.png')
	# 	mask = alpha_face[i].repeat([1,1,3]).cpu().data.numpy()
	# 	mask = Image.fromarray(np.uint8(mask*255))
	# 	mask.save(f'{out_dir}/render_face_mask_{i:01d}.png')

	# head_mask = Image.open(head_tex_mask_dir)
	# head_texture = Image.open(head_tex_dir)

	# # change head texture according to face skin color
	# head_texture = np.array(head_texture)
	# head_mask = np.array(head_mask)
	# bg_img = np.uint8(np.zeros_like(head_texture) + bg_color*255)

	# v, u = np.where(head_mask[:,:,0] > 128)
	# center = (int((u.min() + u.max()) /2), int((v.min() + v.max()) /2))
	# head_texture = cv2.seamlessClone(head_texture, bg_img, head_mask, center, cv2.NORMAL_CLONE)

	# head_texture = torch.from_numpy(head_texture/255).to(device='cuda', dtype=torch.float32)
	# idx = np.arange(head_texture.shape[0]-1,-1,-1)
	# head_texture = head_texture[idx]
	# img_head, alpha_head = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=vert_head, f=face_head, vt=head_uv, ft=face_head, tex=head_texture)
	# alpha_head = (alpha_head > 0.5).float()
	# for i in range(img_head.shape[0]):
	# 	img_head_log = img_head.cpu().numpy()[i]
	# 	img_head_log = Image.fromarray(np.uint8(img_head_log*255))
	# 	img_head_log.save(f'{out_dir}/render_head_img_{i:01d}.png')
	# 	alpha_head = torch.logical_and(alpha_face < 0.9, alpha_head>0.5).float()
	# 	mask = alpha_head[i].repeat([1,1,3]).cpu().data.numpy()
	# 	mask = Image.fromarray(np.uint8(mask*255))
	# 	mask.save(f'{out_dir}/render_head_mask_{i:01d}.png')
	
	# img_comined= blend_face_to_head_img(head_mesh_dir='/aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_render.obj',
	# 					   head_texture_dir='/aigc_cfs_2/weimao/avatar_face_generation/data/timer_female/head_brighter.jpg',
	# 					   Pndc=Pndc, cam2world=cam2world, face_mask=alpha_face.float().cpu().data.numpy(),
	# 					   face_img=img_face.cpu().numpy(),out_dir=out_dir, bg_color=bg_color)
	
	# image_label = torch.from_numpy(img_comined).float().cuda()
	# verts = torch.from_numpy(verts).float().cuda()
	# faces = torch.from_numpy(faces.astype(np.int32)).cuda()
	# uvs = torch.from_numpy(uvs).float().cuda()
	
	# head_texture = Image.open(head_tex_dir)
	# head_texture = torch.from_numpy(np.array(head_texture)/255).to(device='cuda', dtype=torch.float32)
	# idx = np.arange(head_texture.shape[0]-1,-1,-1)
	# head_texture = head_texture[idx]

	# head_texture_mask = Image.open(head_tex_mask_dir)
	# head_texture_mask = torch.from_numpy(np.array(head_texture_mask)/255).to(device='cuda', dtype=torch.float32)
	# idx = np.arange(head_texture.shape[0]-1,-1,-1)
	# head_texture_mask = head_texture_mask[idx][:,:,0]
	
	# # para = nn.Parameter(head_texture[head_texture_mask<0.9][:,:3].detach().clone())
	# para = nn.Parameter(head_texture[:,:,:3].detach().clone())
	# optimizer = optim.Adam([para],lr=0.25,betas=(0.9,0.999))
	# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma =0.99)
	# # optimizer= optim.SGD([para], lr=1.0, momentum=0.9)
	# pb = tqdm(total=max_iter)
	# for i in range(max_iter):
	# 	tex_tmp = para
	# 	img_reco, alpha = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=verts, f=faces, vt=uvs, ft=faces, tex=tex_tmp, max_mip_level=3)
	# 	img_reco = img_reco
	# 	alpha = alpha[:,:,:,0]

	# 	reco = (img_reco[alpha>0.5] - image_label[alpha>0.5]).pow(2).sum(dim=-1).mean()
	# 	loss = reco
	# 	optimizer.zero_grad()
	# 	loss.backward()
	# 	optimizer.step()	
	# 	scheduler.step()
	# 	if i % 100 == 0 and False:
	# 		with torch.no_grad():
	# 			img_reco[alpha<0.5] = 0
	# 			for j in range(img_reco.shape[0]):
	# 				img = Image.fromarray(np.uint8(img_reco[j].cpu().data.numpy()*255))
	# 				img.save(os.path.join(out_dir,f"render_combine_{i:03d}_{j:01d}.png"))
	# 	# break
	# 	pb.set_description(f'img_reo {reco.item():.5f}')
	# 	pb.update(1)

	# # tex_tmp = head_texture[:,:,:3].detach().clone()
	# # tex_tmp[head_texture_mask<0.9] = para
	# tex_tmp = para
	# tsz = tex_tmp.shape[0]
	# img = Image.fromarray(np.uint8(torch.clamp(tex_tmp,min=0,max=1).cpu().data.numpy()*255)[np.arange(tsz-1,-1,-1)])
	# img.save(os.path.join(out_dir,f"face_head_tex.png"))
	# add_mtl(f"{out_dir}/face_head.obj")

	# print(f'>>> face head sewing in {time.time()-st5:.3f} s')

	# print(f'>>> all finished in {time.time()-st0:.3f} s')
		
