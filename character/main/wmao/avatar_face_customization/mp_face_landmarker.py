from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace as st
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
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



def extract_mesh(rgb_image, detection_result, out_file = './', canonical_face_model=None):
  if canonical_face_model is None:
    # connection to faces
    conns = mp.solutions.face_mesh.FACEMESH_TESSELATION
    faces = []
    sz = len(conns)
    conns = list(conns)
    edges = {}
    for con in conns:
      con = list(con)
      i,j  = con[0], con[1]
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
    
    # add eye ball
    # left_eye_conn = list(mp.solutions.face_mesh.FACEMESH_LEFT_EYE)
    # left_eye_idx = set()
    # for conn in left_eye_conn:
    #   left_eye_idx = left_eye_idx + conn
    # left_eye_idx = list(left_eye_idx)
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
    o3d.io.write_triangle_mesh(f'{out_file}.ply',mesh)
    
    mesh = trimesh.Trimesh(vertices=verts,faces=faces)
    mesh.export(f'{out_file}.obj') 
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
    mesh_cano.vertices = verts[:mesh_cano.vertices.shape[0]]
    mesh_cano.export(f'{out_file}.obj')
    return verts, uvs
  
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
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp.solutions.drawing_styles
    #     .get_default_face_mesh_contours_style())
    
    # solutions.drawing_utils.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_IRISES,
    #       landmark_drawing_spec=None,
    #       connection_drawing_spec=mp.solutions.drawing_styles
    #       .get_default_face_mesh_iris_connections_style())

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


# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='./face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

img_dirs = glob.glob('./test_data/*.png')
sz = 1024
for file_path in img_dirs:
  image = Image.open(file_path)
  image = image.resize((sz,sz))
  image.save(file_path)

out_dir = './output_uv_edited'
os.makedirs(out_dir,exist_ok=True)

sam_checkpoint = "/aigc_cfs_2/weimao/pretrained_model_cache/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# pipeline = AutoPipelineForInpainting.from_pretrained(
#   "/aigc_cfs_2/weimao/pretrained_model_cache/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
# )
# pipeline.enable_model_cpu_offload()

# prompt = "bald cartoon head, pure color background, no hair, skin color, no shadow, no shading, no grey color, bright color"
# negative_prompt = "hair, black, shading, shadow, highlight"

for file_path in img_dirs:
  start_time = time.time()
  # file_path = './test_data/cute_you8.png'
  # STEP 3: Load the input image.
  image = mp.Image.create_from_file(file_path)

  # STEP 4: Detect face landmarks from the input image.
  detection_result = detector.detect(image)

  # from canonical face to detected face
  np.savez_compressed(f"{out_dir}/{os.path.basename(file_path).replace('.png','_pose.npz')}", face_pose=detection_result.facial_transformation_matrixes[0])
  verts, uvs = extract_mesh(image.numpy_view()[:,:,:3], detection_result, 
                            out_file=f"{out_dir}/{os.path.basename(file_path).replace('.png','')}",
                            canonical_face_model='./canonical_face_model_uv_edited_v2.obj')
  # STEP 5: Process the detection result. In this case, visualize it.
  annotated_image = draw_landmarks_on_image(image.numpy_view()[:,:,:3], detection_result)

  img = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
  cv2.imwrite(f"{out_dir}/{os.path.basename(file_path)}", img)

  # STEP 5 segment the face region using SAM
  # using two eyes mouth and nose as prompt
  image = image.numpy_view()[:,:,:3]
  predictor.set_image(image)
# if True:
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
  
  # # compose a image with skin color background
  # skin_color = image[int(uvs[4][1]),int(uvs[4][0])]
  # image = copy.deepcopy(image)
  # image[mask_inv[0]>0.2] = skin_color
  # image = Image.fromarray(image)
  # image.save(f"{out_dir}/{os.path.basename(file_path).split('.')[0]}_skincolor_fill.png")
  
  masks = np.uint8(masks.transpose(1,2,0) * 255)
  masks = Image.fromarray(np.concatenate([masks,masks,masks],axis=-1))
  masks.save(f"{out_dir}/{os.path.basename(file_path).split('.')[0]}_mask.png")

  # masks = np.uint8(mask_inv.transpose(1,2,0) * 255)
  # masks = Image.fromarray(np.concatenate([masks,masks,masks],axis=-1))
  # masks.save(f"{out_dir}/{os.path.basename(file_path).split('.')[0]}_mask_inverse.png")
  # #plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0]) 

  # # step 6 inpainting using SDXL
  # init_image = image
  # mask_image = masks
  # generator = torch.Generator("cuda").manual_seed(92)
  # image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image, generator=generator,
  #                 guidance_scale=10.0).images[0]
  # image.save(f"{out_dir}/{os.path.basename(file_path).split('.')[0]}_inpainted.png")
  # print(f'processing {os.path.basename(file_path)}, time used {time.time() - start_time:.5f}. ')