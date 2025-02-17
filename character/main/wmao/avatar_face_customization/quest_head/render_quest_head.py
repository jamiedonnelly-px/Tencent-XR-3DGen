
import os
import sys
sys.path.append(os.path.abspath('../'))
import trimesh
import numpy as np
import torch
from torch import nn
import xatlas
from render_mesh import make_ndc, render_mesh_verts_tex, render_mesh
from ipdb import set_trace as st
from PIL import Image
from utils import angaxe2rot
import open3d as o3d

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def align_point_clouds(P, Q):
    """
    Aligns two point clouds, `P` and `Q`, using given correspondences, by solving for the 
    optimal scale, rotation, and translation that minimizes the distance between corresponding 
    points.

    Parameters:
    -----------
    P : numpy.ndarray
        An (n, 3) array representing the first point cloud with `n` points in 3D.
    Q : numpy.ndarray
        An (n, 3) array representing the second point cloud with `n` points in 3D, where each 
        point in `Q` corresponds to a point in `P`.

    Returns:
    --------
    P_aligned : numpy.ndarray
        The transformed version of `P`, aligned to `Q`.
    scale : float
        The scaling factor applied to `P` for optimal alignment with `Q`.
    R : numpy.ndarray
        A (3, 3) rotation matrix that aligns `P` to `Q`.
    translation : numpy.ndarray
        A (3,) translation vector applied after scaling and rotation to align `P` to `Q`.

    Steps:
    ------
    1. Compute the centroids of `P` and `Q`.
    2. Center both point clouds by subtracting their respective centroids.
    3. Calculate the scaling factor that best aligns the centered `P` to the centered `Q`.
    4. Compute the optimal rotation matrix using Singular Value Decomposition (SVD) of the 
       covariance matrix of the centered points.
    5. Adjust for reflection if necessary to ensure a proper rotation.
    6. Apply scaling, rotation, and translation to obtain the aligned points.

    Example:
    --------
    >>> P = np.random.rand(10, 3)  # Replace with actual data
    >>> Q = np.random.rand(10, 3)  # Replace with actual data
    >>> P_aligned, scale, R, translation = align_point_clouds(P, Q)
    >>> print("Aligned Point Cloud:", P_aligned)
    >>> print("Scale:", scale)
    >>> print("Rotation Matrix:\n", R)
    >>> print("Translation Vector:", translation)
    """
    
    # Step 2: Calculate centroids
    P_centroid = np.mean(P, axis=0)
    Q_centroid = np.mean(Q, axis=0)
    
    # Step 3: Center the point clouds
    P_prime = P - P_centroid
    Q_prime = Q - Q_centroid
    
    # Step 4: Compute scale
    scale = np.sum(np.linalg.norm(Q_prime, axis=1) * np.linalg.norm(P_prime, axis=1)) / np.sum(np.linalg.norm(P_prime, axis=1)**2)
    
    # Step 5: Compute the optimal rotation matrix
    H = P_prime.T @ Q_prime
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure a proper rotation (reflection correction)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Step 6: Apply scaling and combine transformations
    P_aligned = scale * (R @ P_prime.T).T + Q_centroid
    
    return P_aligned, scale, R, Q_centroid - scale * (R @ P_centroid)


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


quest_head_dir = '/aigc_cfs_2/weimao/avatar_face_generation/data/quest_head_for_correspondance.obj'
texture_dir = '/aigc_cfs_2/weimao/avatar_face_generation/data/quest_head.png'
out_dir = '/aigc_cfs_2/weimao/avatar_face_generation/output/quest_head/'
model_dir = '/aigc_cfs_2/weimao/avatar_face_generation/data/timer_model_v2/'
mp_model_ckpt = '../face_landmarker_v2_with_blendshapes.task'
os.makedirs(out_dir, exist_ok=True)

head_mesh = trimesh.load(quest_head_dir)
verts = head_mesh.vertices
faces = head_mesh.faces
uvs = head_mesh.visual.uv
tex = np.array(Image.open(texture_dir))[:,:,:3]/255.

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
tex = torch.from_numpy(tex[np.arange(-1, -1-tex.shape[0],-1)]).float().cuda()
tv = torch.from_numpy(verts).float().cuda()
faces_torch = torch.from_numpy(faces.astype(np.int32)).cuda()
uvs_torch = torch.from_numpy(uvs).float().cuda()
img_reco, alpha = render_mesh(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=tv, f=faces_torch, vt=uvs_torch, ft=faces_torch, tex=tex)
img_reco = img_reco[0]
alpha = alpha[0,:,:,0]

img = Image.fromarray(np.uint8(img_reco.cpu().data.numpy()*255))
img.save(f'{out_dir}/rbg.png')


face_data = np.load(f'{model_dir}/face_data.npz')
base_options = python.BaseOptions(model_asset_path=mp_model_ckpt)
options = vision.FaceLandmarkerOptions(base_options=base_options,output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,num_faces=1)
face_detector = vision.FaceLandmarker.create_from_options(options)
verts_mp, faces_mp, uvs_img, uvs_tex, image = mp_face_detector(f'{out_dir}/rbg.png', out_dir, face_detector, face_data=face_data, sz=sz)

# render xyz
xyz, alpha_xyz = render_mesh_verts_tex(img_size=sz, ndc_mat=Pndc, c2w_mat=cam2world, v=tv, f=faces_torch, vf=tv, ssaa=1, bg_color=(1,1,1))
xyz = xyz[0]
alpha_xyz = alpha_xyz[0,:,:,0]
xyz_norm = (xyz - tv.min(dim=0)[0])/(tv.max(dim=0)[0]-tv.min(dim=0)[0]).max()
xyz_norm = xyz_norm * alpha_xyz[:,:,None]
img = Image.fromarray(np.uint8(xyz_norm.cpu().data.numpy()*255))
img.save(f'{out_dir}/xyz.png')

uvs_img = uvs_img / sz
uvs_img = uvs_img * 2 - 1 # [-1, 1], (nv, 2)
uvs_img = torch.from_numpy(uvs_img).float().cuda()
xyz_quest_of_mp = nn.functional.grid_sample(xyz.permute([2,0,1])[None], uvs_img[None, None])[0,:,0].transpose(1,0) # [nv_mp, 3]

_, s, r, t = align_point_clouds(verts_mp, xyz_quest_of_mp.cpu().data.numpy())
verts_aligned = s * (r@verts_mp.T).T + t
mesh_aligned = trimesh.Trimesh(vertices=verts_aligned, faces=faces_mp)
mesh_aligned.export(f'{out_dir}/mp_face_aligned.obj')


# get correspondance
cdist = torch.cdist(xyz_quest_of_mp, tv) #[nv_mp, nv_quest]
idx_mp2quest = cdist.min(dim=1)[1] #[nv_mp]
idx_quest2mp = cdist.min(dim=0)[1] # [nv_quest]

idxs = np.random.choice(np.arange(verts_mp.shape[0]), 10)
v_mp = verts_mp
faces_mp = faces_mp
v_quest = verts
faces_quest = faces

for idx in idxs:
    # save intermediate results for debugging
    mesh_mp = o3d.geometry.TriangleMesh()
    mesh_mp.vertices = o3d.utility.Vector3dVector(v_mp)
    mesh_mp.triangles = o3d.utility.Vector3iVector(faces_mp)
    vcolor = np.ones_like(v_mp)
    vcolor[idx] = np.array([[1.0,0,0]])
    mesh_mp.vertex_colors = o3d.utility.Vector3dVector(vcolor)
    o3d.io.write_triangle_mesh(out_dir+f'/{idx:03d}_mp.ply', mesh_mp)

    mesh_mp = o3d.geometry.TriangleMesh()
    mesh_mp.vertices = o3d.utility.Vector3dVector(v_quest)
    mesh_mp.triangles = o3d.utility.Vector3iVector(faces_quest)
    vcolor = np.ones_like(v_quest)
    vcolor[idx_mp2quest[idx]] = np.array([[1.0,0,0]])
    mesh_mp.vertex_colors = o3d.utility.Vector3dVector(vcolor)
    o3d.io.write_triangle_mesh(out_dir+f'/{idx:03d}_quest.ply', mesh_mp)


mesh_mp = o3d.geometry.TriangleMesh()
mesh_mp.vertices = o3d.utility.Vector3dVector(v_quest)
mesh_mp.triangles = o3d.utility.Vector3iVector(faces_quest)
vcolor = np.ones_like(v_quest)
vcolor[idx_mp2quest.cpu().data.numpy()] = np.array([[1.0,0,0]])
mesh_mp.vertex_colors = o3d.utility.Vector3dVector(vcolor)
o3d.io.write_triangle_mesh(out_dir+f'/quest_face_bymp.ply', mesh_mp)

idx_quest_face = torch.where(cdist.min(dim=0)[0] <  0.01)[0].cpu().data.numpy()
mesh_mp = o3d.geometry.TriangleMesh()
mesh_mp.vertices = o3d.utility.Vector3dVector(v_quest)
mesh_mp.triangles = o3d.utility.Vector3iVector(faces_quest)
vcolor = np.ones_like(v_quest)
vcolor[idx_quest_face] = np.array([[1.0,0,0]])
mesh_mp.vertex_colors = o3d.utility.Vector3dVector(vcolor)
o3d.io.write_triangle_mesh(out_dir+f'/quest_face_bydist.ply', mesh_mp)

np.savez_compressed(f'{out_dir}/correspondance.npz', idx_mp2quest=idx_mp2quest.cpu().data.numpy(), idx_quest_face=idx_quest_face, 
                    xyz_quest_of_mp=xyz_quest_of_mp.cpu().data.numpy(), scale=s, rotation=r, translation=t)

print('done')
