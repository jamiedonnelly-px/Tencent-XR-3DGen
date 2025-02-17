import glob
import numpy as np
import open3d.visualization
from scipy.spatial.transform import Rotation as R
import  open3d as o3d
import trimesh
import os.path
import json

from matplotlib import colors
import matplotlib.colors as mcolors
template_colors = [ colors.to_rgba(key)[:-1] for key in mcolors.TABLEAU_COLORS ]
clst = []
for  i in range (30): # assume max 300 points
    clst = clst + template_colors


def smplify_mesh ( m, clr ):
    m1 = o3d.geometry.TriangleMesh()
    m1.vertices = o3d.utility.Vector3dVector(np.array(m.vertices))
    m1.triangles = o3d.utility.Vector3iVector(np.array(m.triangles))
    m1.paint_uniform_color(clr)
    m1.compute_vertex_normals()
    return m1


root = "/aigc_cfs_2/xiaqiangdai/project/objaverse_retrieve/data/generate"
lst = root + "/lst.txt"



def load_json(j):
    with open( j) as f:
        data = json.load(f)
    return data

base_body_map = load_json("./base_body_map.json")

with open(lst, "rb") as l:
    lst = l.readlines()
    lst = [e.decode("utf-8").strip() for e in lst]
    # print(lst)


cnt = 0

for e in lst:

    cnt +=1
    print( cnt,"/", len(lst ))

    body = load_json( os.path.join( root, e, "object_lst.txt" ) ) [ "body_attr"]

    body_path = base_body_map[body[0]][body[1]]["path"]




    body_path = os.path.join( body_path,  "naked/head_body.obj")
    body_mesh = o3d.io.read_triangle_mesh( body_path)
    body_mesh = smplify_mesh(body_mesh, (0.7, 0.7, 0.7))

    G_trns = np.eye(3)
    G_trns[:3, :3] = R.from_euler('x', 90, degrees=True).as_matrix()
    # body_mesh.apply_transform(G_trns )
    # save_path_body = os.path.join(root, e, "body_mesh.obj")
    # body_mesh.export(save_path_body)
    body_mesh.rotate(G_trns, center=(0, 0, 0))




    objs = glob.glob( os.path.join( root, e, "part_*/*obj" ) )

    meshes = []
    for idx, obj in enumerate( objs):

        m = o3d.io.read_triangle_mesh(obj)
        m = smplify_mesh(m, clst[idx])
        body_mesh = body_mesh + m


    save_path = os.path.join(root, e, "combined_mesh_" + e +".ply")

    o3d.io.write_triangle_mesh( save_path , body_mesh)






    # break