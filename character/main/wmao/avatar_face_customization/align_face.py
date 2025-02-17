import trimesh
import numpy as np
from ipdb import set_trace as st
import os
import open3d as o3d
import argparse
import os
import sys
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--face_dir', type=str, default="/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/cute_you11_uv.obj")
    parser.add_argument('--out_dir', type=str, default="/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/cute_you11_uv_aligned.obj")
    args = parser.parse_args()
    face_dir = args.face_dir
    out_dir = args.out_dir
    start_time = time.time()
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    # head_temp_dir = "/Users/weimao/Documents/avatar/timer_female/head_temp.obj"
    # face_dir = "/Users/weimao/Documents/avatar/output_face_baking_uv_edited/cute_you8_uv_aligned.obj"
    # fname = os.path.basename(face_dir).split('.')[0]
    # out_dir = os.path.dirname(face_dir)
    face_mesh = trimesh.load(face_dir)
    vert_fa = face_mesh.vertices
    face_fa = face_mesh.faces
    uv_fa = face_mesh.visual.uv

    idx_to_remove = [127,234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454,356,389,251,284,332,297,338,10,109,67,103,54,21,162]
    idx_to_connect = [34,227,137,117,215,138,135,169,170,140,171,175,396,369,395,394,364,367,435,401,366,447,264,368,301,298,333,299,337,151,108,69,104,68,71,139]
    idx_old = np.setdiff1d(np.arange(vert_fa.shape[0]),np.array(idx_to_remove))
    idx_new = np.arange(len(idx_old))
    vert_fa_new = vert_fa[idx_old]
    uv_fa_new = uv_fa[idx_old]
    face_new = np.zeros_like(face_fa) - 1
    for i in idx_new:
        face_new[face_fa == idx_old[i]] = i
    idx_tmp = np.setdiff1d(np.arange(face_new.shape[0]),np.where(face_new < 0)[0])
    face_new = face_new[idx_tmp]

    idx_to_connect_new = []
    for i in idx_to_connect:
        idx_to_connect_new.append(np.where(idx_old==i)[0][0])

    #align_face
    
    # get scale:
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
    np.savez_compressed(f"{out_dir.replace('.obj','.npz')}", scale=scale,rotation=rot,translation=t,comments='y=rot @ (s * x) + t')

    mesh = trimesh.Trimesh()
    mesh.vertices = vert_fa_new
    mesh.faces = face_new
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv_fa_new)
    mesh.export(out_dir)
    
    # copy the first three line of the old obj file
    new_line = []
    with open(face_dir, 'r') as file:
        new_line = file.readlines()[:3]
        # for line in file:
        #     if line.startswith('mtllib') or line.startswith('newmtl') or line.startswith('o '):
        #         new_line.append(line)

    lines = new_line
    with open(out_dir, 'r') as file:
        for line in file:
            if line.startswith('mtllib') or line.startswith('usemtl') or line.startswith('#') :
                continue
            lines.append(line)

    with open(out_dir, 'w') as file:
        file.writelines(lines)
    print(f'time used {time.time()-start_time:.3f}seconds')

