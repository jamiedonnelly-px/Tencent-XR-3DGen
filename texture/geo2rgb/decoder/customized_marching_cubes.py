import torch
import numba
import numpy as np
import trimesh
from einops import repeat, rearrange
import os

os.environ['CUDA_LAUNCH_BLOCKING']='1'

# // Axes are:
# //
# //      y
# //      |     z
# //      |   /
# //      | /
# //      +----- x
# //
# // Vertex and edge layout:
# //
# //            6             7
# //            +-------------+               +-----6-------+   
# //          / |           / |             / |            /|   
# //        /   |         /   |          11   7         10   5
# //    2 +-----+-------+  3  |         +-----+2------+     |   
# //      |   4 +-------+-----+ 5       |     +-----4-+-----+   
# //      |   /         |   /           3   8         1   9
# //      | /           | /             | /           | /       
# //    0 +-------------+ 1             +------0------+         
# //
# // Triangulation cases are generated prioritising rotations over inversions, which can introduce non-manifold geometry.


def cubes_to_edges(cubes):
    '''
    cubes: [n_cubes, 8] int, in range [0, n_verts)
    returns:
        edges: [n_edges, 2] int, verts per edge are ordered by verts_id
        cubes_to_edges: [n_cubes, 12] int in range [0, n_edges)
    '''

    edges = torch.cat([
        cubes[:,[0,1]], # 0
        cubes[:,[1,3]], # 1
        cubes[:,[2,3]], # 2
        cubes[:,[0,2]], # 3
        cubes[:,[4,5]], # 4
        cubes[:,[5,7]], # 5
        cubes[:,[6,7]], # 6
        cubes[:,[4,6]], # 7
        cubes[:,[0,4]], # 8
        cubes[:,[1,5]], # 9
        cubes[:,[3,7]], # 10
        cubes[:,[2,6]], # 11
    ], dim=0) #[12*ncubes, 2]

    edges_sort = torch.sort(edges, dim=1)

    edges_unique, unique_idx = torch.unique(edges_sort.values, return_inverse=True, dim=0)

    return edges_unique, unique_idx.reshape(12,-1).transpose(0,1)

def cubes_to_faces(cubes):
    '''
    cubes: [n_cubes, 8] int, in range [0, n_verts)
    returns:
        faces: [n_faces, 4] int in range [0, n_verts), verts per face are ordered by verts_id
        cubes_to_faces: [n_cubes, 6] int in range [0, n_faces)
    '''
    
    faces = torch.cat([
        cubes[:,[0,1,2,3]], # front
        cubes[:,[0,1,4,5]], # down
        cubes[:,[0,2,4,6]], # left
        cubes[:,[1,3,5,7]], # right
        cubes[:,[2,3,6,7]], # up
        cubes[:,[4,5,6,7]], # back
    ], dim=0) #[6*ncubes, 4]

    faces_sort = torch.sort(faces, dim=1)

    faces_unique, unique_idx = torch.unique(faces_sort.values, return_inverse=True, dim=0)

    return faces_unique, unique_idx.reshape(6,-1).transpose(0,1)
    

def interpolate_edge_verts(verts, vals, edges):
    '''
    verts: [n_verts, 3]
    vals: [n_verts]
    edges: [n_edges, 2] with values in [0, n_verts)

    returns: [n_edges, 3]

    edges must have vals of different signs, otherwise behaviour is not determined
    '''
    edge_vals = vals[edges] # [n_edges, 2]
    edge_verts = verts[edges] # [n_edges, 2, 3]
    weights = edge_vals[:,0:1] / ( edge_vals[:,0:1] - edge_vals[:,1:2] ) # [n_edges, 1]
    weights = torch.cat((1-weights, weights), dim=-1)  # [n_edges, 2]
    return ( weights.unsqueeze(dim=-1) * edge_verts ).sum(dim=1) # [n_edges, 3]

@numba.njit
def _get_faces_nb(cube_types, cubes_to_edges):
    '''
    cube_types: int numpy array of shape [n_cubes], with values in 0-255
    cubes_to_edges: int numpy of shape [n_cubes, 12], with values in [0, n_new_verts)
    '''
    triangle_table = [
    [ 0, 3, 8 ],
    [ 0, 9, 1 ],
    [ 3, 8, 1, 1, 8, 9 ],
    [ 2, 11, 3 ],
    [ 8, 0, 11, 11, 0, 2 ],
    [ 3, 2, 11, 1, 0, 9 ],
    [ 11, 1, 2, 11, 9, 1, 11, 8, 9 ],
    [ 1, 10, 2 ],
    [ 0, 3, 8, 2, 1, 10 ],
    [ 10, 2, 9, 9, 2, 0 ],
    [ 8, 2, 3, 8, 10, 2, 8, 9, 10 ],
    [ 11, 3, 10, 10, 3, 1 ],
    [ 10, 0, 1, 10, 8, 0, 10, 11, 8 ],
    [ 9, 3, 0, 9, 11, 3, 9, 10, 11 ],
    [ 8, 9, 11, 11, 9, 10 ],
    [ 4, 8, 7 ],
    [ 7, 4, 3, 3, 4, 0 ],
    [ 4, 8, 7, 0, 9, 1 ],
    [ 1, 4, 9, 1, 7, 4, 1, 3, 7 ],
    [ 8, 7, 4, 11, 3, 2 ],
    [ 4, 11, 7, 4, 2, 11, 4, 0, 2 ],
    [ 0, 9, 1, 8, 7, 4, 11, 3, 2 ],
    [ 7, 4, 11, 11, 4, 2, 2, 4, 9, 2, 9, 1 ],
    [ 4, 8, 7, 2, 1, 10 ],
    [ 7, 4, 3, 3, 4, 0, 10, 2, 1 ],
    [ 10, 2, 9, 9, 2, 0, 7, 4, 8 ],
    [ 10, 2, 3, 10, 3, 4, 3, 7, 4, 9, 10, 4 ],
    [ 1, 10, 3, 3, 10, 11, 4, 8, 7 ],
    [ 10, 11, 1, 11, 7, 4, 1, 11, 4, 1, 4, 0 ],
    [ 7, 4, 8, 9, 3, 0, 9, 11, 3, 9, 10, 11 ],
    [ 7, 4, 11, 4, 9, 11, 9, 10, 11 ],
    [ 9, 4, 5 ],
    [ 9, 4, 5, 8, 0, 3 ],
    [ 4, 5, 0, 0, 5, 1 ],
    [ 5, 8, 4, 5, 3, 8, 5, 1, 3 ],
    [ 9, 4, 5, 11, 3, 2 ],
    [ 2, 11, 0, 0, 11, 8, 5, 9, 4 ],
    [ 4, 5, 0, 0, 5, 1, 11, 3, 2 ],
    [ 5, 1, 4, 1, 2, 11, 4, 1, 11, 4, 11, 8 ],
    [ 1, 10, 2, 5, 9, 4 ],
    [ 9, 4, 5, 0, 3, 8, 2, 1, 10 ],
    [ 2, 5, 10, 2, 4, 5, 2, 0, 4 ],
    [ 10, 2, 5, 5, 2, 4, 4, 2, 3, 4, 3, 8 ],
    [ 11, 3, 10, 10, 3, 1, 4, 5, 9 ],
    [ 4, 5, 9, 10, 0, 1, 10, 8, 0, 10, 11, 8 ],
    [ 11, 3, 0, 11, 0, 5, 0, 4, 5, 10, 11, 5 ],
    [ 4, 5, 8, 5, 10, 8, 10, 11, 8 ],
    [ 8, 7, 9, 9, 7, 5 ],
    [ 3, 9, 0, 3, 5, 9, 3, 7, 5 ],
    [ 7, 0, 8, 7, 1, 0, 7, 5, 1 ],
    [ 7, 5, 3, 3, 5, 1 ],
    [ 5, 9, 7, 7, 9, 8, 2, 11, 3 ],
    [ 2, 11, 7, 2, 7, 9, 7, 5, 9, 0, 2, 9 ],
    [ 2, 11, 3, 7, 0, 8, 7, 1, 0, 7, 5, 1 ],
    [ 2, 11, 1, 11, 7, 1, 7, 5, 1 ],
    [ 8, 7, 9, 9, 7, 5, 2, 1, 10 ],
    [ 10, 2, 1, 3, 9, 0, 3, 5, 9, 3, 7, 5 ],
    [ 7, 5, 8, 5, 10, 2, 8, 5, 2, 8, 2, 0 ],
    [ 10, 2, 5, 2, 3, 5, 3, 7, 5 ],
    [ 8, 7, 5, 8, 5, 9, 11, 3, 10, 3, 1, 10 ],
    [ 5, 11, 7, 10, 11, 5, 1, 9, 0 ],
    [ 11, 5, 10, 7, 5, 11, 8, 3, 0 ],
    [ 5, 11, 7, 10, 11, 5 ],
    [ 6, 7, 11 ],
    [ 7, 11, 6, 3, 8, 0 ],
    [ 6, 7, 11, 0, 9, 1 ],
    [ 9, 1, 8, 8, 1, 3, 6, 7, 11 ],
    [ 3, 2, 7, 7, 2, 6 ],
    [ 0, 7, 8, 0, 6, 7, 0, 2, 6 ],
    [ 6, 7, 2, 2, 7, 3, 9, 1, 0 ],
    [ 6, 7, 8, 6, 8, 1, 8, 9, 1, 2, 6, 1 ],
    [ 11, 6, 7, 10, 2, 1 ],
    [ 3, 8, 0, 11, 6, 7, 10, 2, 1 ],
    [ 0, 9, 2, 2, 9, 10, 7, 11, 6 ],
    [ 6, 7, 11, 8, 2, 3, 8, 10, 2, 8, 9, 10 ],
    [ 7, 10, 6, 7, 1, 10, 7, 3, 1 ],
    [ 8, 0, 7, 7, 0, 6, 6, 0, 1, 6, 1, 10 ],
    [ 7, 3, 6, 3, 0, 9, 6, 3, 9, 6, 9, 10 ],
    [ 6, 7, 10, 7, 8, 10, 8, 9, 10 ],
    [ 11, 6, 8, 8, 6, 4 ],
    [ 6, 3, 11, 6, 0, 3, 6, 4, 0 ],
    [ 11, 6, 8, 8, 6, 4, 1, 0, 9 ],
    [ 1, 3, 9, 3, 11, 6, 9, 3, 6, 9, 6, 4 ],
    [ 2, 8, 3, 2, 4, 8, 2, 6, 4 ],
    [ 4, 0, 6, 6, 0, 2 ],
    [ 9, 1, 0, 2, 8, 3, 2, 4, 8, 2, 6, 4 ],
    [ 9, 1, 4, 1, 2, 4, 2, 6, 4 ],
    [ 4, 8, 6, 6, 8, 11, 1, 10, 2 ],
    [ 1, 10, 2, 6, 3, 11, 6, 0, 3, 6, 4, 0 ],
    [ 11, 6, 4, 11, 4, 8, 10, 2, 9, 2, 0, 9 ],
    [ 10, 4, 9, 6, 4, 10, 11, 2, 3 ],
    [ 4, 8, 3, 4, 3, 10, 3, 1, 10, 6, 4, 10 ],
    [ 1, 10, 0, 10, 6, 0, 6, 4, 0 ],
    [ 4, 10, 6, 9, 10, 4, 0, 8, 3 ],
    [ 4, 10, 6, 9, 10, 4 ],
    [ 6, 7, 11, 4, 5, 9 ],
    [ 4, 5, 9, 7, 11, 6, 3, 8, 0 ],
    [ 1, 0, 5, 5, 0, 4, 11, 6, 7 ],
    [ 11, 6, 7, 5, 8, 4, 5, 3, 8, 5, 1, 3 ],
    [ 3, 2, 7, 7, 2, 6, 9, 4, 5 ],
    [ 5, 9, 4, 0, 7, 8, 0, 6, 7, 0, 2, 6 ],
    [ 3, 2, 6, 3, 6, 7, 1, 0, 5, 0, 4, 5 ],
    [ 6, 1, 2, 5, 1, 6, 4, 7, 8 ],
    [ 10, 2, 1, 6, 7, 11, 4, 5, 9 ],
    [ 0, 3, 8, 4, 5, 9, 11, 6, 7, 10, 2, 1 ],
    [ 7, 11, 6, 2, 5, 10, 2, 4, 5, 2, 0, 4 ],
    [ 8, 4, 7, 5, 10, 6, 3, 11, 2 ],
    [ 9, 4, 5, 7, 10, 6, 7, 1, 10, 7, 3, 1 ],
    [ 10, 6, 5, 7, 8, 4, 1, 9, 0 ],
    [ 4, 3, 0, 7, 3, 4, 6, 5, 10 ],
    [ 10, 6, 5, 8, 4, 7 ],
    [ 9, 6, 5, 9, 11, 6, 9, 8, 11 ],
    [ 11, 6, 3, 3, 6, 0, 0, 6, 5, 0, 5, 9 ],
    [ 11, 6, 5, 11, 5, 0, 5, 1, 0, 8, 11, 0 ],
    [ 11, 6, 3, 6, 5, 3, 5, 1, 3 ],
    [ 9, 8, 5, 8, 3, 2, 5, 8, 2, 5, 2, 6 ],
    [ 5, 9, 6, 9, 0, 6, 0, 2, 6 ],
    [ 1, 6, 5, 2, 6, 1, 3, 0, 8 ],
    [ 1, 6, 5, 2, 6, 1 ],
    [ 2, 1, 10, 9, 6, 5, 9, 11, 6, 9, 8, 11 ],
    [ 9, 0, 1, 3, 11, 2, 5, 10, 6 ],
    [ 11, 0, 8, 2, 0, 11, 10, 6, 5 ],
    [ 3, 11, 2, 5, 10, 6 ],
    [ 1, 8, 3, 9, 8, 1, 5, 10, 6 ],
    [ 6, 5, 10, 0, 1, 9 ],
    [ 8, 3, 0, 5, 10, 6 ],
    [ 6, 5, 10 ],
    [ 10, 5, 6 ],
    [ 0, 3, 8, 6, 10, 5 ],
    [ 10, 5, 6, 9, 1, 0 ],
    [ 3, 8, 1, 1, 8, 9, 6, 10, 5 ],
    [ 2, 11, 3, 6, 10, 5 ],
    [ 8, 0, 11, 11, 0, 2, 5, 6, 10 ],
    [ 1, 0, 9, 2, 11, 3, 6, 10, 5 ],
    [ 5, 6, 10, 11, 1, 2, 11, 9, 1, 11, 8, 9 ],
    [ 5, 6, 1, 1, 6, 2 ],
    [ 5, 6, 1, 1, 6, 2, 8, 0, 3 ],
    [ 6, 9, 5, 6, 0, 9, 6, 2, 0 ],
    [ 6, 2, 5, 2, 3, 8, 5, 2, 8, 5, 8, 9 ],
    [ 3, 6, 11, 3, 5, 6, 3, 1, 5 ],
    [ 8, 0, 1, 8, 1, 6, 1, 5, 6, 11, 8, 6 ],
    [ 11, 3, 6, 6, 3, 5, 5, 3, 0, 5, 0, 9 ],
    [ 5, 6, 9, 6, 11, 9, 11, 8, 9 ],
    [ 5, 6, 10, 7, 4, 8 ],
    [ 0, 3, 4, 4, 3, 7, 10, 5, 6 ],
    [ 5, 6, 10, 4, 8, 7, 0, 9, 1 ],
    [ 6, 10, 5, 1, 4, 9, 1, 7, 4, 1, 3, 7 ],
    [ 7, 4, 8, 6, 10, 5, 2, 11, 3 ],
    [ 10, 5, 6, 4, 11, 7, 4, 2, 11, 4, 0, 2 ],
    [ 4, 8, 7, 6, 10, 5, 3, 2, 11, 1, 0, 9 ],
    [ 1, 2, 10, 11, 7, 6, 9, 5, 4 ],
    [ 2, 1, 6, 6, 1, 5, 8, 7, 4 ],
    [ 0, 3, 7, 0, 7, 4, 2, 1, 6, 1, 5, 6 ],
    [ 8, 7, 4, 6, 9, 5, 6, 0, 9, 6, 2, 0 ],
    [ 7, 2, 3, 6, 2, 7, 5, 4, 9 ],
    [ 4, 8, 7, 3, 6, 11, 3, 5, 6, 3, 1, 5 ],
    [ 5, 0, 1, 4, 0, 5, 7, 6, 11 ],
    [ 9, 5, 4, 6, 11, 7, 0, 8, 3 ],
    [ 11, 7, 6, 9, 5, 4 ],
    [ 6, 10, 4, 4, 10, 9 ],
    [ 6, 10, 4, 4, 10, 9, 3, 8, 0 ],
    [ 0, 10, 1, 0, 6, 10, 0, 4, 6 ],
    [ 6, 10, 1, 6, 1, 8, 1, 3, 8, 4, 6, 8 ],
    [ 9, 4, 10, 10, 4, 6, 3, 2, 11 ],
    [ 2, 11, 8, 2, 8, 0, 6, 10, 4, 10, 9, 4 ],
    [ 11, 3, 2, 0, 10, 1, 0, 6, 10, 0, 4, 6 ],
    [ 6, 8, 4, 11, 8, 6, 2, 10, 1 ],
    [ 4, 1, 9, 4, 2, 1, 4, 6, 2 ],
    [ 3, 8, 0, 4, 1, 9, 4, 2, 1, 4, 6, 2 ],
    [ 6, 2, 4, 4, 2, 0 ],
    [ 3, 8, 2, 8, 4, 2, 4, 6, 2 ],
    [ 4, 6, 9, 6, 11, 3, 9, 6, 3, 9, 3, 1 ],
    [ 8, 6, 11, 4, 6, 8, 9, 0, 1 ],
    [ 11, 3, 6, 3, 0, 6, 0, 4, 6 ],
    [ 8, 6, 11, 4, 6, 8 ],
    [ 10, 7, 6, 10, 8, 7, 10, 9, 8 ],
    [ 3, 7, 0, 7, 6, 10, 0, 7, 10, 0, 10, 9 ],
    [ 6, 10, 7, 7, 10, 8, 8, 10, 1, 8, 1, 0 ],
    [ 6, 10, 7, 10, 1, 7, 1, 3, 7 ],
    [ 3, 2, 11, 10, 7, 6, 10, 8, 7, 10, 9, 8 ],
    [ 2, 9, 0, 10, 9, 2, 6, 11, 7 ],
    [ 0, 8, 3, 7, 6, 11, 1, 2, 10 ],
    [ 7, 6, 11, 1, 2, 10 ],
    [ 2, 1, 9, 2, 9, 7, 9, 8, 7, 6, 2, 7 ],
    [ 2, 7, 6, 3, 7, 2, 0, 1, 9 ],
    [ 8, 7, 0, 7, 6, 0, 6, 2, 0 ],
    [ 7, 2, 3, 6, 2, 7 ],
    [ 8, 1, 9, 3, 1, 8, 11, 7, 6 ],
    [ 11, 7, 6, 1, 9, 0 ],
    [ 6, 11, 7, 0, 8, 3 ],
    [ 11, 7, 6 ],
    [ 7, 11, 5, 5, 11, 10 ],
    [ 10, 5, 11, 11, 5, 7, 0, 3, 8 ],
    [ 7, 11, 5, 5, 11, 10, 0, 9, 1 ],
    [ 7, 11, 10, 7, 10, 5, 3, 8, 1, 8, 9, 1 ],
    [ 5, 2, 10, 5, 3, 2, 5, 7, 3 ],
    [ 5, 7, 10, 7, 8, 0, 10, 7, 0, 10, 0, 2 ],
    [ 0, 9, 1, 5, 2, 10, 5, 3, 2, 5, 7, 3 ],
    [ 9, 7, 8, 5, 7, 9, 10, 1, 2 ],
    [ 1, 11, 2, 1, 7, 11, 1, 5, 7 ],
    [ 8, 0, 3, 1, 11, 2, 1, 7, 11, 1, 5, 7 ],
    [ 7, 11, 2, 7, 2, 9, 2, 0, 9, 5, 7, 9 ],
    [ 7, 9, 5, 8, 9, 7, 3, 11, 2 ],
    [ 3, 1, 7, 7, 1, 5 ],
    [ 8, 0, 7, 0, 1, 7, 1, 5, 7 ],
    [ 0, 9, 3, 9, 5, 3, 5, 7, 3 ],
    [ 9, 7, 8, 5, 7, 9 ],
    [ 8, 5, 4, 8, 10, 5, 8, 11, 10 ],
    [ 0, 3, 11, 0, 11, 5, 11, 10, 5, 4, 0, 5 ],
    [ 1, 0, 9, 8, 5, 4, 8, 10, 5, 8, 11, 10 ],
    [ 10, 3, 11, 1, 3, 10, 9, 5, 4 ],
    [ 3, 2, 8, 8, 2, 4, 4, 2, 10, 4, 10, 5 ],
    [ 10, 5, 2, 5, 4, 2, 4, 0, 2 ],
    [ 5, 4, 9, 8, 3, 0, 10, 1, 2 ],
    [ 2, 10, 1, 4, 9, 5 ],
    [ 8, 11, 4, 11, 2, 1, 4, 11, 1, 4, 1, 5 ],
    [ 0, 5, 4, 1, 5, 0, 2, 3, 11 ],
    [ 0, 11, 2, 8, 11, 0, 4, 9, 5 ],
    [ 5, 4, 9, 2, 3, 11 ],
    [ 4, 8, 5, 8, 3, 5, 3, 1, 5 ],
    [ 0, 5, 4, 1, 5, 0 ],
    [ 5, 4, 9, 3, 0, 8 ],
    [ 5, 4, 9 ],
    [ 11, 4, 7, 11, 9, 4, 11, 10, 9 ],
    [ 0, 3, 8, 11, 4, 7, 11, 9, 4, 11, 10, 9 ],
    [ 11, 10, 7, 10, 1, 0, 7, 10, 0, 7, 0, 4 ],
    [ 3, 10, 1, 11, 10, 3, 7, 8, 4 ],
    [ 3, 2, 10, 3, 10, 4, 10, 9, 4, 7, 3, 4 ],
    [ 9, 2, 10, 0, 2, 9, 8, 4, 7 ],
    [ 3, 4, 7, 0, 4, 3, 1, 2, 10 ],
    [ 7, 8, 4, 10, 1, 2 ],
    [ 7, 11, 4, 4, 11, 9, 9, 11, 2, 9, 2, 1 ],
    [ 1, 9, 0, 4, 7, 8, 2, 3, 11 ],
    [ 7, 11, 4, 11, 2, 4, 2, 0, 4 ],
    [ 4, 7, 8, 2, 3, 11 ],
    [ 9, 4, 1, 4, 7, 1, 7, 3, 1 ],
    [ 7, 8, 4, 1, 9, 0 ],
    [ 3, 4, 7, 0, 4, 3 ],
    [ 7, 8, 4 ],
    [ 11, 10, 8, 8, 10, 9 ],
    [ 0, 3, 9, 3, 11, 9, 11, 10, 9 ],
    [ 1, 0, 10, 0, 8, 10, 8, 11, 10 ],
    [ 10, 3, 11, 1, 3, 10 ],
    [ 3, 2, 8, 2, 10, 8, 10, 9, 8 ],
    [ 9, 2, 10, 0, 2, 9 ],
    [ 8, 3, 0, 10, 1, 2 ],
    [ 2, 10, 1 ],
    [ 2, 1, 11, 1, 9, 11, 9, 8, 11 ],
    [ 11, 2, 3, 9, 0, 1 ],
    [ 11, 0, 8, 2, 0, 11 ],
    [ 3, 11, 2 ],
    [ 1, 8, 3, 9, 8, 1 ],
    [ 1, 9, 0 ],
    [ 8, 3, 0 ],
    ]

    faces = []
    for i in range(len(cube_types)):
        edges = cubes_to_edges[i] # [12]
        cube_type = cube_types[i] # int
        if cube_type == 0 or cube_type == 255:
            continue # does not contain face
        tris = triangle_table[cube_type - 1] # [n_tri*3], vales in [0,12)
        verts = [edges[e] for e in tris]
        faces.extend(verts)
    return np.array(faces).reshape(-1,3)


@torch.no_grad()
def get_faces(cube_types, cubes_to_edges):
    device = cube_types.device
    cube_types = cube_types.byte().cpu().numpy()
    cubes_to_edges = cubes_to_edges.cpu().numpy()
    return torch.from_numpy(_get_faces_nb(cube_types, cubes_to_edges)).to(device)

@torch.no_grad()
def get_boundary_cubes(cubes, verts_sdf):
    '''
    boundary_cubes: [n_cubes, 8], with values in [0, n_verts)
    verts_sdf: [n_verts]
    '''
    cube_sign = (verts_sdf[cubes] >= 0) # [n_cubes, 8]
    mask = torch.any(cube_sign, dim=1) & torch.any(torch.logical_not(cube_sign), dim=1)
    
    cubes = cubes[mask]
    cube_sign = cube_sign[mask]

    cube_type = (cube_sign.int() * (2 ** torch.arange(8, device=cube_sign.device)).int()).sum(1)
    return cubes, cube_type

@torch.no_grad()
def remove_unused_verts(cubes, verts, verts_attr=None):
    verts_mask = torch.zeros(verts.shape[0], dtype=torch.bool, device=verts.device)
    verts_id = torch.zeros(verts.shape[0], dtype=torch.long, device=verts.device) - 1
    # breakpoint()
    verts_mask[torch.unique(cubes.reshape(-1))] = True
    verts_id[verts_mask] = torch.arange(verts_mask.sum(), dtype=torch.long, device=verts.device)
    cubes = verts_id[cubes].contiguous()
    if not torch.all(cubes >= 0):
        pass
        # raise ValueError("expected non-negative ids")
    verts = verts[verts_mask].contiguous()
    if verts_attr is not None:
        verts_attr = verts_attr[verts_mask].contiguous()
    return cubes, verts, verts_attr

@torch.no_grad()
def get_roi_cubes(cubes, verts_sdf, under_thresh, over_thresh):
    '''
    returns cubes that are either on boundary or inside surface, remove the ones that are outside
    '''
    # breakpoint()    
    # cubes = cubes.int()
    inside = (verts_sdf[cubes] > over_thresh) # [n_cubes, 8]
    outside = (verts_sdf[cubes] < under_thresh) # [n_cubes, 8]
    roi_mask = torch.logical_not(torch.logical_or(torch.all(inside, dim=1), torch.all(outside, dim=1)))
    return cubes[roi_mask]

@torch.no_grad()
def split_cubes(cubes, verts):
    '''
    split the cubes to 8 smaller ones of half edge length and append new verts to list of verts
    receives and returns:
        cubes: [n_cubes, 8]
        verts: [n_verts, 3]
    returned n_cubes and n_verts will be greater than recevied
    '''
    edges, cube2edge = cubes_to_edges(cubes) # [n_edges, 2], [n_cubes, 12]
    faces, cube2face = cubes_to_faces(cubes) # [n_faces, 4], [n_cubes, 6]
    
    edge_verts = verts[edges].mean(1) # [n_edges, 3]
    face_verts = verts[faces].mean(1) # [n_faces, 3]
    cent_verts = verts[cubes].mean(1) # [n_cubes, 3]
    
    edge_verts_id = torch.arange(verts.shape[0], edge_verts.shape[0]+verts.shape[0], dtype=cubes.dtype, device=cubes.device)
    face_verts_id = torch.arange(edge_verts.shape[0]+verts.shape[0], edge_verts.shape[0]+verts.shape[0]+face_verts.shape[0], dtype=cubes.dtype, device=cubes.device)
    cent_verts_id = torch.arange(edge_verts.shape[0]+verts.shape[0]+face_verts.shape[0], edge_verts.shape[0]+verts.shape[0]+face_verts.shape[0]+cent_verts.shape[0],dtype=cubes.dtype, device=cubes.device)
    
    v = cubes.t() # [8, n_cubes]
    e = edge_verts_id[cube2edge].t() # [12, n_cubes]
    f = face_verts_id[cube2face].t() # [6, n_cubes]
    c = cent_verts_id # [n_cubes]
    
    new_cubes = torch.stack([
        v[0], e[0], e[3], f[0], e[8], f[1], f[2], c,
        e[0], v[1], f[0], e[1], f[1], e[9], c, f[3],
        e[3], f[0], v[2], e[2], f[2], c, e[11], f[4],
        f[0], e[1], e[2], v[3], c, f[3], f[4], e[10],
        e[8], f[1], f[2], c, v[4], e[4], e[7], f[5],
        f[1], e[9], c, f[3], e[4], v[5], f[5], e[5],
        f[2], c, e[11], f[4], e[7], f[5], v[6], e[6],
        c, f[3], f[4], e[10], f[5], e[5], e[6], v[7],
    ], dim=-1) # [n_cubes, 8 sub-cubes x 8verts]
    
    new_verts = torch.cat([verts, edge_verts, face_verts, cent_verts])
    
    return new_cubes.reshape(-1, 8), new_verts

def _marching_cubes(cubes, verts, verts_sdf):
    '''
    boundary_cubes: [n_cubes, 8], with values in [0, n_verts)
    verts: [n_verts, 3]
    verts_sdf: [n_verts, 1]
    '''
    boundary_cubes, cube_types = get_boundary_cubes(cubes, verts_sdf.squeeze(dim=-1))
    edges, edge_id = cubes_to_edges(boundary_cubes)
    mesh_verts = interpolate_edge_verts(verts, verts_sdf.squeeze(dim=-1), edges)
    mesh_faces = get_faces(cube_types, edge_id)
    return mesh_verts, mesh_faces

def sparse_marching_cubes(query_func, latents, init_depth, final_depth, thresh, under_thresh, over_thresh, thresh_attenuation_factor=0.5, bounds=(-1,1), verbose=False, device="cuda"):
    '''
    performs a more efficient version of coarse to fine sparse marching cubes
    inputs:
        - query_func: a function that takes in a torch tensor of shape (N,3) and returns occupancy/sdf of shape (N)
                     smaller values indicate inside and greater values outside
        - init_depth: an integer that defines the initial coarse grid resolution for marching cubes, resolution = 2**depth
        - final_depth: an integer that defines the final resolution = 2**final_depth
        - thresh: marching cubes threshold for extracting final surface
        - under_thresh and over_thresh: underestimation and overestimation of thresh to define the region of interest.
          anythine outside this region are discarded during course to fine, to save computation
        - thresh_attenuation_factor: whether or not to move under_thresh and over_thresh to be closer to thresh
          as depth increases, if 0 they are fixed. must be in range [0,1) and recommended to be no greater than 0.5. 
          larger values improve efficiency at the risk of creating holes
        - bounds: bounding box to perform marching cubes on
        - verbose: whether to print logs
        - device: device of input/output tensors to query_func
    '''
    assert (under_thresh <= thresh <= over_thresh) and (init_depth <= final_depth)
    
    init_gridres = 2**init_depth
    bounds_min, bounds_max = bounds
    unit_cube = torch.tensor([
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [1,1,0],
        [0,0,1],
        [1,0,1],
        [0,1,1],
        [1,1,1],
    ]).to(device)  # [8,3]
    init_grid = torch.stack(torch.meshgrid(torch.arange(init_gridres, device=device), torch.arange(init_gridres, device=device), torch.arange(init_gridres, device=device), indexing="ij"), dim=-1).reshape(-1,3)
    init_grid = init_grid.unsqueeze(dim=1) + unit_cube # [-1, 8, 3]
    init_grid = (bounds_max - bounds_min) / init_gridres * init_grid + bounds_min # [-1, 8, 3]
    verts, cubes = torch.unique(init_grid.reshape(-1, 3), dim=0, return_inverse=True)
    cubes = cubes.reshape(-1,8)
    
    current_depth = init_depth
    
    while True:
        
        if verbose:
            print(f"[info] depth {current_depth}")

        if current_depth > init_depth:
            n_cubes_before = cubes.shape[0]
            n_verts_before = verts.shape[0]
            cubes, verts = split_cubes(cubes, verts)
            n_cubes_after = cubes.shape[0]
            n_verts_after = verts.shape[0]
            
            if verbose:
                print(f"[info]     splitting {n_cubes_before} cubes {n_verts_before} vertrs -> {n_cubes_after} cubes {n_verts_after} verts")
        
        if verbose:
            print(f"[info]     sending {verts.shape[0]} queries")

        # breakpoint()
        verts_sdf = query_func(verts, latents)
        
        # breakpoint()
        # verts = repeat(verts, "p c -> b p c", b=1).to(latents)
        
        # verts_sdf = query_func(verts, latents)
        
        # breakpoint()
        # verts = verts.squeeze(0)
        # verts_sdf = verts_sdf.squeeze(0)
        # breakpoint()
        
        verts_sdf = verts_sdf.to(device=verts.device)
        cubes = cubes.to(device=verts.device)
        
        # breakpoint()
        n_cubes_before = cubes.shape[0]
        n_verts_before = verts.shape[0]
        
        # assert torch.all(cubes >= 0)
        cubes = get_roi_cubes(cubes, verts_sdf, under_thresh, over_thresh)
        # assert torch.all(cubes >= 0)
        cubes, verts, verts_sdf = remove_unused_verts(cubes, verts, verts_sdf)
        # assert torch.all(cubes >= 0)
        
        n_cubes_after = cubes.shape[0]
        n_verts_after = verts.shape[0]
        if verbose:
            print(f"[info]     reducing {n_cubes_before} cubes {n_verts_before} vertrs -> {n_cubes_after} cubes {n_verts_after} verts")
        
        print(current_depth)
        if current_depth == final_depth:
            break
        else:
            # TODO: adjust under_thresh and over_thresh to be closer to thresh as depth increases
            under_thresh += (thresh - under_thresh) * thresh_attenuation_factor
            over_thresh += (thresh - over_thresh) * thresh_attenuation_factor
            current_depth += 1
    
    if verbose:
        print(f"[info] extracing surface from {n_cubes_after} cubes and {n_verts_after} verts")
        
    mesh_verts, mesh_faces = _marching_cubes(cubes, verts, verts_sdf - thresh)
    return mesh_verts, mesh_faces
    
def dense_marching_cubes(query_func, depth, thresh, bounds=(-1,1), verbose=False, device="cuda"):
    return sparse_marching_cubes(query_func, depth, depth, thresh, thresh, thresh, 0, bounds, verbose, device)
    

if __name__ == "__main__":

    def sdf_func(xyz):
        return -(torch.linalg.norm(xyz, dim=1, ord=2) - 0.5)
    
    init_depth = 5
    final_depth = 9
    
    surface_in = -0.1
    surface_out = 0.1
    surface_range_decay = 0.5
    
    print(f"performing marching cubes @ resolution {2**final_depth}^3")
    
    mesh_verts, mesh_faces = sparse_marching_cubes(sdf_func, init_depth, final_depth, 0.0, surface_in, surface_out, surface_range_decay, bounds=(-1,1), verbose=True)
    trimesh.Trimesh(mesh_verts.detach().cpu().numpy(), mesh_faces.cpu().numpy()).export('sparse_mc_sphere.obj')
    
    try:
        mesh_verts, mesh_faces = dense_marching_cubes(sdf_func, final_depth, 0, bounds=(-1,1), verbose=True)
        trimesh.Trimesh(mesh_verts.detach().cpu().numpy(), mesh_faces.cpu().numpy()).export('dense_mc_sphere.obj')
    except Exception as e:
        print(f"dense marching cubes: {e}")





