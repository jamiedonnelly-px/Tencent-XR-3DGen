import torch
import numpy as np

# // Cubes:
# // axis
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


def tris_to_edges(tris):
    '''
    cubes: [tris, 8] int, in range [0, n_verts)
    returns:
        edges: [n_edges, 2] int, verts per edge are ordered by verts_id
        cubes_to_edges: [tris, 3] int in range [0, n_edges)
    '''

    edges = torch.cat([
        tris[:,[0,1]], # 0
        tris[:,[1,2]], # 1
        tris[:,[0,2]], # 2
    ], dim=0) #[3*ncubes, 2]

    edges_sort = torch.sort(edges, dim=1)

    edges_unique, unique_idx = torch.unique(edges_sort.values, return_inverse=True, dim=0)

    return edges_unique, unique_idx.reshape(3,-1).transpose(0,1)


def cubes_to_edges(cubes):
    '''
    cubes: [n_cubes, 8] int
    returns:
        edges: [n_edges, 8] int
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
    ], dim=0) #[n_tets*6, 2]

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
    
    
# Cube
#         ^ +z
#         |  6 -------- 7
#         | /|        / |
#         4 -------- 5  |
#         |  | / +y  |  |
#         |  2 ------|- 3
#         | /        |/
# -x <--- 0 -------- 1 --->+x

#  * Cube33 description:
#  *         7 ________ 6           _____6__             ________
#  *         /|       /|         7/|       /|          /|       /|
#  *       /  |     /  |        /  |     /5 |        /  6     /  |
#  *   4 /_______ /    |      /__4____ /    10     /_______3/    |
#  *    |     |  |5    |     |    11  |     |     |     |  |   2 |
#  *    |    3|__|_____|2    |     |__|__2__|     | 4   |__|_____|
#  *    |    /   |    /      8   3/   9    /      |    /   |    /
#  *    |  /     |  /        |  /     |  /1       |  /     5  /
#  *    |/_______|/          |/___0___|/          |/_1_____|/
#  *   0          1                  
#  */
# //-----------------------------------------------------------------------------

def cubes_to_cubes33(cubes):
    return cubes[:,[0,1,3,2,4,5,7,6]]

def cubes33_to_edges33(cubes):
    '''
    cubes: [n_cubes, 8] int
    returns:
        edges: [n_edges, 8] int
        cubes_to_edges: [n_cubes, 12] int in range [0, n_edges)
    '''

    edges = torch.cat([
        cubes[:,[0,1]], # 0
        cubes[:,[1,2]], # 1
        cubes[:,[2,3]], # 2
        cubes[:,[0,3]], # 3
        cubes[:,[4,5]], # 4
        cubes[:,[5,6]], # 5
        cubes[:,[6,7]], # 6
        cubes[:,[4,7]], # 7
        cubes[:,[0,4]], # 8
        cubes[:,[1,5]], # 9
        cubes[:,[2,6]], # 10
        cubes[:,[3,7]], # 11
    ], dim=0) #[n_tets*6, 2]

    edges_sort = torch.sort(edges, dim=1)

    edges_unique, unique_idx = torch.unique(edges_sort.values, return_inverse=True, dim=0)
    unique_idx = unique_idx.reshape(12,-1).transpose(0,1)
    return edges_unique, unique_idx

@torch.no_grad()
def get_boundary_edges(verts_sdf, edges, cube_edge_id):
    '''
    determine if edges have different sign sdf; discard edge if not
    '''
    verts_sdf = verts_sdf.squeeze(dim=-1)
    cube_sign = (verts_sdf[edges] >= 0) # [n_edges, 2]
    mask = torch.any(cube_sign, dim=1) & torch.any(torch.logical_not(cube_sign), dim=1)

    edge_id_map = torch.zeros_like(mask).long() - 1
    edge_id_map[mask] = torch.arange(mask.sum(), dtype=edge_id_map.dtype, device=edge_id_map.device)

    edges = edges[mask]
    cube_edge_id = edge_id_map[cube_edge_id] # -1 if no edge

    return edges, cube_edge_id

def interpolate_cube_verts(cube_edge_id, edge_verts, edge_features=None, index_select=True):
    '''
    cube_edge_id: [n_cubes, 8] with int values in [-1, n_edges)
    edge_verts: [n_edges, 3]
    edge_features: [n_batch, n_edges, n_channels]

    returns: 
    '''

    edge_verts = torch.cat([edge_verts, torch.zeros_like(edge_verts[:1])], dim=0) # append 0
    if index_select:
        cube_edge_id = cube_edge_id + 0
        cube_edge_id[cube_edge_id < 0] = len(edge_verts) + cube_edge_id[cube_edge_id < 0]    
        cube_verts_sum = torch.index_select(edge_verts, 0, cube_edge_id.flatten()).reshape(cube_edge_id.shape + (3,))
    else:
        cube_verts_sum = edge_verts[cube_edge_id] # [n_cubes, 8, 3]

    cube_verts_sum = cube_verts_sum.sum(dim=1) # [n_cubes, 3]

    cube_verts_cnt = ((cube_edge_id>=0) & (cube_edge_id < len(edge_verts) - 1)).sum(-1,keepdim=True).to(edge_verts.dtype) # [n_cubes, 1]

    cube_verts = cube_verts_sum / cube_verts_cnt

    if edge_features is not None:
        n_batch, n_edges, n_channels = edge_features.shape
        edge_features = torch.cat([edge_features, torch.zeros_like(edge_features[:,:1])], dim=1) # append 0 
        # cube_features_sum = edge_features[:,cube_edge_id]
        cube_features_sum = torch.index_select(edge_features, 1, cube_edge_id.flatten()).reshape(n_batch, -1, 12, n_channels)
        cube_features_sum = cube_features_sum.sum(dim=2) # [n_batch, n_cubes, n_channels]
        cube_features = cube_features_sum / cube_verts_cnt

        return cube_verts, cube_features
    else:   
        return cube_verts, None


def interpolate_edge_verts(verts, vals, edges, verts_features=None, index_select=True):
    '''
    verts: [n_verts, n_dim]
    vals: [n_verts]
    edges: [n_edges, 2] with values in [0, n_verts)
    verts_features: [n_batch, n_verts, n_channels]

    returns: [n_edges, n_dim]

    edges must have vals of different signs, otherwise behaviour is not determined
    '''
    if index_select:
        edge_vals = torch.index_select(vals, 0, edges.flatten()).reshape(edges.shape)
    else:
        edge_vals = vals[edges] # [n_edges, 2]
        
    diff = edge_vals[:,0:1] - edge_vals[:,1:2]
    diff = torch.relu(diff) - torch.relu(-diff) + 1e-10 # make non-zero
    weights = edge_vals[:,0:1] / diff # [n_edges, 1]
    weights = torch.cat((1-weights, weights), dim=-1)  # [n_edges, 2]

    if index_select:
        edge_verts = torch.index_select(verts, 0, edges.flatten()).reshape(edges.shape + (-1,))
    else:
        edge_verts = verts[edges] # [n_edges, 2, n_dim]
    edge_verts = ( weights.unsqueeze(dim=-1) * edge_verts ).sum(dim=1) # [n_edges, n_dim]

    if verts_features is not None:
        n_batch, _, n_channels = verts_features.shape
        # edge_features = verts_features[:, edges.reshape(-1)].reshape(n_batch, -1, 2, n_channels)  # [n_batch, n_edges, 2, n_channels]
        edge_features = torch.index_select(verts_features, 1, edges.flatten()).reshape(n_batch, -1, 2, n_channels)  # [n_batch, n_edges, 2, n_channels]
        edge_features = (weights.unsqueeze(dim=-1) * edge_features).sum(dim=2) # # [n_batch, n_edges, n_channels]

        return edge_verts, edge_features
    
    return edge_verts, None


try:
    import numba
    @numba.jit(nopython=True, parallel=True, cache=True)
    def _get_faces_nb(cube_types, cubes_to_edges):
        '''
        cube_types: int numpy array of shape [n_cubes], with values in 0-255
        cubes_to_edges: int numpy of shape [n_cubes, 12], with values in [0, n_new_verts)
        '''
        triangle_table = [
        [],
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
        [],
        ]
        
        face_table_len = np.array([0, 3, 3, 6, 3, 6, 6, 9, 3, 6, 6, 9, 6, 9, 9, 6, 3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9, 3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9, 6, 9, 9, 6, 9, 12, 12, 9, 9, 12, 12, 9, 12, 9, 9, 6, 3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9, 6, 9, 9, 12, 9, 6, 12, 9, 9, 12, 12, 9, 12, 9, 9, 6, 6, 9, 9, 12, 9, 12, 12, 9, 9, 12, 12, 9, 12, 9, 9, 6, 9, 12, 12, 9, 12, 9, 9, 6, 12, 9, 9, 6, 9, 6, 6, 3, 3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9, 6, 9, 9, 12, 9, 12, 12, 9, 9, 12, 12, 9, 12, 9, 9, 6, 6, 9, 9, 12, 9, 12, 12, 9, 9, 12, 6, 9, 12, 9, 9, 6, 9, 12, 12, 9, 12, 9, 9, 6, 12, 9, 9, 6, 9, 6, 6, 3, 6, 9, 9, 12, 9, 12, 12, 9, 9, 12, 12, 9, 6, 9, 9, 6, 9, 12, 12, 9, 12, 9, 9, 6, 12, 9, 9, 6, 9, 6, 6, 3, 9, 12, 12, 9, 12, 9, 9, 6, 12, 9, 9, 6, 9, 6, 6, 3, 6, 9, 9, 6, 9, 6, 6, 3, 9, 6, 6, 3, 6, 3, 3, 0])
        start_i = np.cumsum(face_table_len[cube_types])
        start_i = np.concatenate((np.zeros_like(start_i[:1]), start_i))
        total_size = start_i[-1]

        faces = np.zeros(total_size, dtype=cubes_to_edges.dtype)
        
        for i in range(len(cube_types)):
            edges = cubes_to_edges[i] # [12]
            cube_type = cube_types[i] # int
            if cube_type == 0 or cube_type == 255:
                continue # does not contain face
            tris = triangle_table[cube_type] # [n_tri*3], vales in [0,12)
            verts = [edges[e] for e in tris]
            for j in range(len(verts)):
                faces[start_i[i]+j] = verts[j]
                
        return np.ascontiguousarray(np.fliplr(faces.reshape(-1,3)))
except:
    print('skip numba')
    def _get_faces_nb(cube_types, cubes_to_edges):
        return

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



def marching_cubes_classic(cubes, verts, verts_sdf, verts_features=None):
    '''
    boundary_cubes: [n_cubes, 8], with values in [0, n_verts)
    verts: [n_verts, 3]
    verts_sdf: [n_verts, 1]
    verts_features: [[n_batch], n_verts, n_channels]
    '''
    boundary_cubes, cube_types = get_boundary_cubes(cubes, verts_sdf.squeeze(dim=-1))
    edges, edge_id = cubes_to_edges(boundary_cubes)
    edges, edge_id = get_boundary_edges(verts_sdf, edges, edge_id)
    
    mesh_verts, mesh_verts_features = interpolate_edge_verts(verts, verts_sdf.squeeze(dim=-1), edges, verts_features)
    mesh_faces = get_faces(cube_types, edge_id)

    if verts_features is None:
        return mesh_verts, mesh_faces 
    else:
        return mesh_verts, mesh_faces, mesh_verts_features

def marching_cubes_33(cubes, verts, verts_sdf, verts_features=None):
    '''
    boundary_cubes: [n_cubes, 8], with values in [0, n_verts)
    verts: [n_verts, 3]
    verts_sdf: [n_verts, 1]
    verts_features: [[n_batch], n_verts, n_channels]
    '''
    
    from ._mc33 import marching_cubes_33 as mc33
    
    cubes = cubes_to_cubes33(cubes)
    cubes, cube_types = get_boundary_cubes(cubes, verts_sdf.squeeze(dim=-1))
    edges, edge_id = cubes33_to_edges33(cubes)
    # edges, edge_id = cubes_to_edges(cubes)
    edges, edge_id = get_boundary_edges(verts_sdf, edges, edge_id)

    edge_verts, edge_verts_features = interpolate_edge_verts(verts, verts_sdf.squeeze(dim=-1), edges, verts_features)
    internal_verts, internal_verts_features = interpolate_cube_verts(edge_id, edge_verts, edge_verts_features)
    
    mesh_verts = torch.cat([edge_verts, internal_verts], dim=0)

    internal_id = torch.arange(len(cubes), device=edge_id.device, dtype=edge_id.dtype).reshape(-1,1) + len(edges)
    edge_internal_id = torch.cat((edge_id,internal_id), dim=-1) # [n_cubes, 13]

    cubes_sdf = verts_sdf[cubes]
    triangles = torch.zeros(12*len(cubes), 3, dtype=torch.int, device=cubes.device) # maxium 12 faces per triangle

    n_triangles = mc33(cubes_sdf.contiguous().float(), cube_types.contiguous().to(torch.uint8), edge_internal_id.contiguous().int(), triangles)
    triangles = triangles[:n_triangles].contiguous()

    # remove degenerate faces with duplicate vertex indices
    tri_sorted = torch.sort(triangles, dim=-1).values
    degen_face_mask = (tri_sorted[:,0] == tri_sorted[:,1]) | (tri_sorted[:,1] == tri_sorted[:,2])
    triangles = triangles[torch.logical_not(degen_face_mask)].contiguous()


    if verts_features is None:
        return mesh_verts, triangles 
    else:
        return mesh_verts, triangles, torch.cat([edge_verts_features, internal_verts_features], dim=1)