import torch
import numpy as np
import trimesh
from pdb import set_trace as st
from mc33 import *

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


@torch.no_grad()
def remove_unused_verts(cubes, verts, verts_attr=None):
    verts_mask = torch.zeros(verts.shape[0], dtype=torch.bool, device=verts.device)
    verts_id = torch.zeros(verts.shape[0], dtype=torch.long, device=verts.device) - 1
    verts_mask[torch.unique(cubes.reshape(-1))] = True
    verts_id[verts_mask] = torch.arange(verts_mask.sum(), dtype=torch.long, device=verts.device)
    cubes = verts_id[cubes].contiguous()
    verts = verts[verts_mask].contiguous()
    if verts_attr is not None:
        verts_attr = verts_attr[verts_mask].contiguous()
    return cubes, verts, verts_attr

@torch.no_grad()
def get_roi_cubes(cubes, verts_sdf, under_thresh, over_thresh):
    '''
    returns cubes that are either on boundary or inside surface, remove the ones that are outside
    '''
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

# @numba.jit(nopython=True)
# def _find_connected_components(n_verts, edges):

#     adj_list = [[] for _ in range(n_verts)]
#     for edge in edges:
#         u, v = edge
#         adj_list[u].append(v)
#         adj_list[v].append(u)
#     visited = np.zeros(n_verts, dtype=np.bool_)
#     components = []
#     def dfs(vert, component):
#         stack = [vert]
#         visited[vert] = True
#         while stack:
#             node = stack.pop()
#             component.append(node)
#             for neighbor in adj_list[node]:
#                 if not visited[neighbor]:
#                     visited[neighbor] = True
#                     stack.append(neighbor)
#     for vert in range(n_verts):
#         if not visited[vert]:
#             component = []
#             dfs(vert, component)
#             components.append(np.array(component))

#     return components

# def _find_borders(cubes, edges, verts):
#     '''
#     boundary_cubes: [n_cubes, 8], with values in [0, n_verts)
#     verts: [n_verts, 3]
#     '''
#     n_verts = verts.shape[0]
#     device = verts.device
#     n_cubes_per_vert = torch.zeros(n_verts, dtype=torch.int32, device=device)
#     n_cubes_per_vert.index_add_(dim=0, index=cubes.flatten(), source=torch.ones_like(cubes.flatten(), dtype=torch.int32))
    
#     border_verts = torch.where(torch.logical_and(n_cubes_per_vert > 0, n_cubes_per_vert < 8)) # [n_border_verts]
#     border_verts_id = torch.zeros(n_verts, device=device, dtype=torch.long) - 1
#     border_verts_id[border_verts] = torch.arange(len(border_verts), device=device, dtype=torch.long)
    
#     border_edges = border_verts_id[edges]
#     border_edges = border_edges[edges.min(dim=1).values >= 0]
#     borders = _find_connected_components(len(border_verts), border_edges.cpu().numpy())
#     return [border_verts[torch.from_numpy(b).to(device=device, dtype=torch.long)] for b in borders]


def _pad_cubes(cubes, verts, pad_verts_mask, verts_sdf, query_func):
    
    pad_verts = verts[pad_verts_mask] # [n, 3]
    n = pad_verts.shape[0]
    pad_cube_verts = torch.stack(torch.meshgrid(
        torch.arange(-1,2, dtype=verts.dtype, device=verts.device),
        torch.arange(-1,2, dtype=verts.dtype, device=verts.device),
        torch.arange(-1,2, dtype=verts.dtype, device=verts.device),
        indexing="ij",
        ), dim=-1).flip(-1).reshape(27,3) # [27,3]
    pad_cube = torch.tensor([
        0,1,3,4,9,10,12,13
    ], dtype=cubes.dtype, device=cubes.device)
    pad_cube = pad_cube + pad_cube.reshape(8,1) # [8,8]
    
    new_verts = pad_verts.unsqueeze(1) + pad_cube_verts # [n, 27, 3]
    new_cubes = pad_cube + 27 * torch.arange(n, dtype=cubes.dtype, device=cubes.device).reshape(-1,1,1) # [n,8,8]
    
    new_verts = new_verts.reshape(-1,3)
    new_cubes = new_cubes.reshape(-1,8)
    
    new_verts, new_verts_map = torch.unique(new_verts, dim=0, sorted=True, return_inverse=True)
    new_cubes = torch.unique(new_verts_map[new_cubes], dim=0)
    
    all_verts = torch.cat((verts, new_verts))
    all_cubes = torch.cat((cubes, new_cubes + len(verts)))
    
    # get unique verts and preserve order of occurence
    unique_verts, idx_map, count = torch.unique(all_verts, dim=0, sorted=True, return_inverse=True, return_counts=True)
    unique_occ = idx_map.sort(stable=True)[1][torch.nn.functional.pad(count[:-1], (1,0), value=0).cumsum(0)]
    
    unique_sdf = torch.zeros(unique_occ.shape[0], dtype=verts_sdf.dtype, device=verts_sdf.device)
    existing_occ = unique_occ < len(verts)
    unique_sdf[existing_occ] = verts_sdf[unique_occ[existing_occ]]
    
    if (existing_occ==False).sum() > 0:
        unique_sdf[existing_occ==False] = query_func(unique_verts[existing_occ==False])
    
    unique_cubes = torch.unique(idx_map[all_cubes], dim=0)
    return unique_verts, unique_cubes, unique_sdf
    
def _pad_borders(cubes, verts, border_verts, verts_sdf, verts_min, cube_size, query_func):
    verts_i = torch.round((verts - verts_min) / cube_size).to(torch.int32)

    if len(border_verts) > 0:
        n_cubes_before = cubes.shape[0]
        verts_i, cubes, verts_sdf = _pad_cubes(cubes, verts_i, border_verts, verts_sdf, lambda vi: query_func(vi.to(verts.dtype) * cube_size + verts_min))
        verts = verts_i.to(verts.dtype) * cube_size + verts_min
        padded = (cubes.shape[0] > n_cubes_before)
    else:
        padded = False
        
    return verts, cubes, verts_sdf, padded
    
def sparse_marching_cubes(query_func, init_depth, final_depth, thresh, under_thresh, over_thresh, thresh_attenuation_factor=0.5, bounds=(-1,1), verbose=False, device="cuda", method=None, flip_faces=False):
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
        - method: a string, "classic" or "mc33" or None
    '''
    assert (under_thresh <= thresh <= over_thresh) and (init_depth <= final_depth)
    assert method in ["classic", "mc33"] or method is None
    
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
            
        verts_sdf = query_func(verts)
        
        n_cubes_before = cubes.shape[0]
        n_verts_before = verts.shape[0]
        cubes = get_roi_cubes(cubes, verts_sdf, under_thresh, over_thresh)
        cubes, verts, verts_sdf = remove_unused_verts(cubes, verts, verts_sdf)
        n_cubes_after = cubes.shape[0]
        n_verts_after = verts.shape[0]
        if verbose:
            print(f"[info]     reducing {n_cubes_before} cubes {n_verts_before} vertrs -> {n_cubes_after} cubes {n_verts_after} verts")
        
        if current_depth == final_depth:
            break
        else:
            # TODO: adjust under_thresh and over_thresh to be closer to thresh as depth increases
            under_thresh += (thresh - under_thresh) * thresh_attenuation_factor
            over_thresh += (thresh - over_thresh) * thresh_attenuation_factor
            current_depth += 1
    
        
    while True:
        # pad boundary cubes to fill holes
        
        n_cubes_before = cubes.shape[0]
        n_verts_before = verts.shape[0]
        
        # find verts to pad, whose degree < 8 and is connected to other verts of different sign
        edges, _ = cubes_to_edges(cubes)
        edges_sign = torch.sign(verts_sdf[edges] - thresh).reshape(-1,2)
        int_edges = edges[edges_sign.sum(1).abs() < 2]
        v_deg = torch.zeros(n_verts_before, dtype=torch.float, device=edges.device)
        v_deg.index_add_(0, cubes.flatten(), torch.ones_like(cubes.flatten(), dtype=torch.float))
        v_intdeg = torch.zeros(n_verts_before, dtype=torch.float, device=edges.device)
        v_intdeg.index_add_(0, int_edges.flatten(), torch.ones_like(int_edges.flatten(), dtype=torch.float))
        
        border_verts = torch.logical_and(v_deg < 8, v_intdeg > 0)
        
        verts, cubes, verts_sdf, padded = _pad_borders(cubes, verts, border_verts, verts_sdf, bounds_min, (bounds_max-bounds_min)/(2**final_depth), query_func)
        
        n_cubes_after = cubes.shape[0]
        n_verts_after = verts.shape[0]
        if verbose:
            print(f"[info] padded border {n_cubes_before} cubes {n_verts_before} vertrs -> {n_cubes_after} cubes {n_verts_after} verts")
        
        if not padded:
            break
    
    if verbose:
        print(f"[info] extracing surface from {n_cubes_after} cubes and {n_verts_after} verts")
    
    if method == "classic":
        mesh_verts, mesh_faces = marching_cubes_classic(cubes, verts, verts_sdf - thresh)
    elif method == "mc33":
        mesh_verts, mesh_faces = marching_cubes_33(cubes, verts, verts_sdf - thresh)
    else:
        try:
            mesh_verts, mesh_faces = marching_cubes_33(cubes, verts, verts_sdf - thresh)
        except:
            mesh_verts, mesh_faces = marching_cubes_classic(cubes, verts, verts_sdf - thresh)
    
    if flip_faces:
        mesh_faces = torch.fliplr(mesh_faces)
        
    return mesh_verts.contiguous(), mesh_faces.contiguous()
    
def dense_marching_cubes(query_func, depth, thresh, bounds=(-1,1), verbose=False, device="cuda", method=None, flip_faces=False):
    return sparse_marching_cubes(query_func, depth, depth, thresh, thresh, thresh, 0, bounds, verbose, device, method, flip_faces)
    

if __name__ == "__main__":
    
    from pdb import set_trace as st

    def sdf_func(xyz):
        sdf = -(torch.linalg.norm(xyz, dim=1, ord=2) - 0.5)
        sdf[sdf.abs() < 1e-8] = 1e-8
        return sdf
    
    
    init_depth = 3
    final_depth = 9
    
    print(f"performing marching cubes @ resolution {2**final_depth}^3")
    
    mesh_verts, mesh_faces = sparse_marching_cubes(sdf_func, init_depth, final_depth, 0.0, -0.1, 0.1, 0.5, bounds=(-1,1), verbose=True)

    print(trimesh.Trimesh(mesh_verts.detach().cpu().numpy(), mesh_faces.cpu().numpy(), process=False).is_watertight)
    print(trimesh.Trimesh(mesh_verts.detach().cpu().numpy(), mesh_faces.cpu().numpy()).is_watertight)
    _ = trimesh.Trimesh(mesh_verts.cpu().numpy(), mesh_faces.cpu().numpy(), process=False).export('sparse_mc_sphere.obj')

    
    try:
        mesh_verts, mesh_faces = dense_marching_cubes(sdf_func, final_depth, 0, bounds=(-1,1), verbose=True)
        trimesh.Trimesh(mesh_verts.detach().cpu().numpy(), mesh_faces.cpu().numpy()).export('dense_mc_sphere.obj')
    except Exception as e:
        print(f"dense marching cubes: {e}")





