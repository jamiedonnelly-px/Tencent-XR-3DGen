import numpy as np
import torch
import os
import open3d as o3d
import pathlib
# 
# tgt_scan_dir = "/home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/smpl_mesh_1280/smpl/"
# smpl_faces = np.load(os.path.join(tgt_scan_dir, "smpl_faces.npy"))
# smpl_verts = np.load(os.path.join(tgt_scan_dir, "smpl_verts.npy"))
# 
# a = 0
# 
# # mesh = "/home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/yuanmeng/naked/body.obj"
# dump_array = os.path.join ( tgt_scan_dir, "mesh_array.npy" )
# 
# 
# with open(dump_array, 'wb') as f:
#     np.save(f, smpl_verts)
#     np.save(f, smpl_faces)

if __name__ == '__main__':
    
    fname = "examples/face_src/face_src.ply"

    mname = fname.split("/")[-1][:-4]

    dump_array = os.path.join( pathlib.Path(fname).parent , mname + ".npy" )

    m = o3d.io.read_triangle_mesh( fname )

    with open(dump_array, 'wb') as f:
        np.save(f, np.asarray(m.vertices ))
        np.save(f, np.asarray(m.triangles ))
    