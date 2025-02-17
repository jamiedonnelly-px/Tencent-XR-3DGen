import  numpy as np
# import open3d
import  open3d as o3d
import  pathlib
import os

mesh = "/home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/yuanmeng/naked/body.obj"
dump_array = os.path.join ( pathlib.Path(mesh).parent, "mesh_array.npy" )



mesh = o3d.io.read_triangle_mesh( mesh )

print( "len(mesh.vertices)", len(mesh.vertices))
print( "len(mesh.triangles)", len(mesh.triangles))

mesh.merge_close_vertices(eps=0.0001)

print( "len(mesh.vertices)", len(mesh.vertices))
print( "len(mesh.triangles)", len(mesh.triangles))

verts = np.asarray( mesh.vertices )
faces = np.asarray( mesh.triangles )

with open(dump_array, 'wb') as f:
    np.save(f, verts)
    np.save(f, faces)