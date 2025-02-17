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


root = "/aigc_cfs_2/xiaqiangdai/project/objaverse_retrieve/data/generate"
lst =  "../combine_example/lst.txt"



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



    combine_path = os.path.join(root, e, "combined_mesh_" + e +".ply")

    # o3d.io.write_triangle_mesh( save_path , body_mesh)


    cmd = " ".join(
        [
            "scp",
            "devcld:"+combine_path,
            "../combine_example/"
        ]
    )

    print( cmd )

    os.system(cmd)

