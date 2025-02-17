import trimesh
import numpy as np
from ipdb import set_trace as st
import open3d as o3d
import json

def evenly_divide_triangle(n):
    """
    Generate barycentric coordinates for subdividing a triangle evenly into n levels.

    Parameters:
        n (int): The number of divisions along each edge of the triangle.

    Returns:
        list: A list of barycentric coordinates (u, v, w).
    """
    bary_coords = []
    for i in range(n + 1):
        for j in range(n - i + 1):
            k = n - i - j
            bary_coords.append((i / n, j / n, k / n))
    
    # Function to calculate the index of a point in the 1D list
    def index(row, col):
        return sum(n + 1 - r for r in range(row)) + col
    
    # Identify all small triangles and compute their centroids
    centroids = []
    for i in range(n):
        for j in range(n - i):
            # Vertices of the current small triangle
            v1 = bary_coords[index(i, j)]
            v2 = bary_coords[index(i, j+1)]
            v3 = bary_coords[index(i+1, j)]
            
            # Centroid of the triangle
            centroid = (
                (v1[0] + v2[0] + v3[0]) / 3,
                (v1[1] + v2[1] + v3[1]) / 3,
                (v1[2] + v2[2] + v3[2]) / 3
            )
            centroids.append(centroid)

            # Second triangle for split (only if not at the edge)
            if j < n - i - 1:
                v4 = bary_coords[index(i+1,j+1)]
                centroid = (
                    (v2[0] + v3[0] + v4[0]) / 3,
                    (v2[1] + v3[1] + v4[1]) / 3,
                    (v2[2] + v3[2] + v4[2]) / 3
                )
                centroids.append(centroid)

    return centroids

mp_face_dir = "/aigc_cfs_2/weimao/avatar_face_generation/data/quest_head_model/mp_face_aligned2quest_head.obj"

face_mesh = trimesh.load(mp_face_dir)
verts = face_mesh.vertices
faces = face_mesh.faces

# uniformly sample barycentric coordinates according to triangle area
areas = []
for f in faces:
    area = np.linalg.norm(np.cross((verts[f[1]] - verts[f[0]]),(verts[f[2]]-verts[f[0]])))/2
    areas.append(area)

areas = np.array(areas)
bary_coords_all = {}
for fi, area in enumerate(areas):
    if area > np.mean(areas):
        bary_coords_all[fi]= [1/3,1/3,1/3]
    
    # num = int(np.round(np.sqrt(area / areas.median())))
    # if True:
    #     num = 3
    #     bary_coords = evenly_divide_triangle(num)
    #     bary_coords_all.append(bary_coords)

    # # vis
    # if True:
    #     bary_coords = np.array(bary_coords)
    #     vert_fi = verts[faces[487]]
    #     pts = bary_coords @ vert_fi

    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(pts)
    #     o3d.io.write_point_cloud('/aigc_cfs_2/weimao/avatar_face_generation/data/quest_head_model/test_barycentric.ply', pcd)
# np.savez_compressed('/aigc_cfs_2/weimao/avatar_face_generation/data/quest_head_model/mp_face_barycoords.npz', bary_coords_all=bary_coords_all)
with open('/aigc_cfs_2/weimao/avatar_face_generation/data/quest_head_model/mp_face_bary.json', 'w') as f:
    json.dump(bary_coords_all, f, indent=4)