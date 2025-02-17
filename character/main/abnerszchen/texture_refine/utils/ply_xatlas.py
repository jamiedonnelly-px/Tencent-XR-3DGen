import os
import argparse
import json
import torch
import numpy  as np

import trimesh
import xatlas
import nvdiffrast.torch as dr
import torch.nn as nn
import open3d as o3d

def color_ply_xatlas_to_obj(in_ply, out_dir):
    """ convert colored ply to obj, mtl and texture map.

    Args:
        in_ply: with vertices xyz [N, 3], face [Nf, 3] and vertex_colors [N, 4] rgba
        out_dir: dir with .obj, .mtl, .png
    """
    
    try:
        mesh = trimesh.load_mesh(in_ply)
        # Extract vertices
        vertices = mesh.vertices
        faces = mesh.faces
        # Extract vertex colors
        vertex_colors = mesh.visual.vertex_colors 
        
        assert vertices.shape[0] == vertex_colors.shape[0], print(f'v shape != color shape {vertices.shape[0]} != {vertex_colors.shape[0]}')
    except:
        print(f'Can not lod trimesh ply {in_ply}')
        return
    
    os.makedirs(out_dir, exist_ok=True)
    
    
    return

    # Export mesh with UV atlas to OBJ file
    # trimesh_mesh.export(os.path.join(out_dir, 'output.obj'))    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='triplane')
    parser.add_argument('in_ply', type=str)
    parser.add_argument('out_dir', type=str)
    args = parser.parse_args()
    
    color_ply_xatlas_to_obj(args.in_ply, args.out_dir)
    