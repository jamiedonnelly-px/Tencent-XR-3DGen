import torch
import numba
import numpy as np
import open3d as o3d
import trimesh as tm
import pytorch3d as p3d
import pickle as pkl

def edge_to_face_mapping(faces):
    edge_face_map = {}
    for face_idx, face in enumerate(faces):
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            if edge in edge_face_map:
                edge_face_map[edge].append(face_idx)
            else:
                edge_face_map[edge] = [face_idx]
    return edge_face_map


def construct_vertex_adjacent_faces_mask(vertex_adjacent_faces, vertex_num, face_num):
    # vertex_adjacent_faces (torch tensor): [vertex_num, max_adjacent_face_num]
    vertices_adj_faces_mask = torch.zeros((vertex_num, face_num))
    for idx in range(vertex_num):
        vertices_adj_faces_mask[idx, vertex_adjacent_faces[idx]] = 1
    print(f"vertices adj face mask: {vertices_adj_faces_mask} shape: {vertices_adj_faces_mask.shape}")
    return vertices_adj_faces_mask

def compute_per_vertex_jacobian(jacbobian_per_face, vertices_adj_faces_mask, device="cuda"):
    # jacbobian_per_face (torch tensor): [1, face_num, 3, 3]
    # vertex_adjacent_faces (torch tensor): [vertex_num, max_adjacent_face_num]
    vertex_num = vertices_adj_faces_mask.shape[0]
    # jacbobian_per_vertex: [vertex_num, 3, 3]
    jacbobian_per_vertex = torch.eye(3, 3).unsqueeze(0).repeat(vertex_num, 1, 1)
    
    # Get actual adjacent faces num
    adjacent_faces_num = vertices_adj_faces_mask.sum(dim=1)
    # print(f"vertice adjacent face: {vertices_adj_faces_mask.shape}")
    adjacent_faces_num_repeat = adjacent_faces_num.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, 3).to(device)
    # print(f"adjacent_faces_num {adjacent_faces_num} shape: {adjacent_faces_num.shape}, adjacent_faces_num_repeat: {adjacent_faces_num_repeat.shape}")
    
    jacbobian_per_face = jacbobian_per_face.repeat(vertex_num, 1, 1, 1)
    vertices_adj_faces_mask = vertices_adj_faces_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3).to(device)
    # print(jacbobian_per_face.shape, vertices_adj_faces_mask.shape, vertices_adj_faces_mask.shape)
    jacbobian_per_vertex = (jacbobian_per_face * vertices_adj_faces_mask).sum(dim=1) / adjacent_faces_num_repeat
    # print(f"jacbobian_per_face: {jacbobian_per_face.shape}, jacbobian_per_vertex: {jacbobian_per_vertex.shape}")

    return jacbobian_per_vertex
    

@numba.jit
def find_adjacent_faces(vertex_index, face_number, face_matrix):
    adjacent_faces = []
    for face_index in range(face_number):
        face = face_matrix[face_index]
        if vertex_index in face:
            adjacent_faces.append(face_index)
    return adjacent_faces


@numba.jit
def _find_all_vertices_adjacent_faces(vertex_number, face_number, face_matrix):
    # vertices_adjacent_faces_arr: [vertex_number, max_num_adjacent_face]
    vertices_adjacent_faces = []
    for vertex_index in range(vertex_number):
        adjacent_faces = find_adjacent_faces(vertex_index, face_number, face_matrix)
        vertices_adjacent_faces.append([vertex_index, *adjacent_faces]) # [vertice_index, adjacent_faces_index]
    # Get the max number of adjacent faces
    max_num_adjacent_face = 0
    for faces in vertices_adjacent_faces:
        max_num_adjacent_face = max(max_num_adjacent_face, len(faces)-1)
    return vertices_adjacent_faces, max_num_adjacent_face


# @numba.jit
def _pad_to_max_num_adjacent_faces(vertices_adjacent_faces, vertices_adjacent_faces_arr):
    # Pad the adjacent faces to max number of adjacent faces
    for faces in vertices_adjacent_faces:
        vertex_idx = faces[0]
        for af_idx in range(len(faces)-1):
            vertices_adjacent_faces_arr[vertex_idx, af_idx] = faces[af_idx+1]
    return vertices_adjacent_faces_arr


def find_all_vertices_adjacent_faces(vertex_number, face_number, face_matrix):
    vertices_adjacent_faces, max_num_adjacent_face = _find_all_vertices_adjacent_faces(vertex_number, face_number, face_matrix)
    # vertices_adjacent_faces_arr = -np.ones((vertex_number, max_num_adjacent_face), dtype=np.int32)
    # vertices_adjacent_faces_arr = _pad_to_max_num_adjacent_faces(vertices_adjacent_faces, vertices_adjacent_faces_arr)
    # return vertices_adjacent_faces_arr
    return vertices_adjacent_faces

vertices_path = "meshes/vertices.pkl"
faces_path = "meshes/faces.pkl"
with open(vertices_path, "rb") as f:
	vertices_raw = pkl.load(f)
with open(faces_path, "rb") as f:
	faces_raw = pkl.load(f)
print(f"vertices raw shape: {vertices_raw.shape}, faces raw shape: {faces_raw.shape}")


def write_obj_o3d(vertices_raw, faces_raw, output_path):
    output_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    output_mesh.vertices = o3d.utility.Vector3dVector(vertices_raw)
    output_mesh.triangles = o3d.utility.Vector3iVector(faces_raw)
    o3d.io.write_triangle_mesh( f"{output_path}", output_mesh)

def write_obj_pytorch3d(vertices_raw, faces_raw, output_path):
    p3d.save_obj(output_path, vertices_raw, faces_raw)

# def write_obj_trimesh(vertices_raw, faces_raw, output_path):
#     p3d.save_obj(output_path, vertices_raw, faces_raw)
o3d_output =  "meshes/o3d_write_from_raw.obj"
p3d_output =  "meshes/p3d_write_from_raw.obj"
write_obj_o3d(vertices_raw, faces_raw, o3d_output)
write_obj_pytorch3d(vertices_raw, faces_raw, p3d_output)


# garment_mesh_path = "meshes/mesh_garment_deform.obj"
# garment_mesh_path = "meshes/mesh_pytorch3d.obj"
# garment_mesh_path = "meshes/mesh_o3d.obj"

garment_mesh_path = p3d_output
print(f"mesh loaded: {garment_mesh_path}")

# IO_TYPE = "trimesh"
IO_TYPE = "open3d"

if IO_TYPE == "trimesh":
    garment_mesh = tm.load(garment_mesh_path, force="mesh")
    vertices = garment_mesh.vertices
    faces = garment_mesh.faces
elif IO_TYPE == "open3d":
    garment_mesh = o3d.io.read_triangle_mesh(garment_mesh_path)
    vertices = np.asarray(garment_mesh.vertices)
    faces = np.asarray(garment_mesh.triangles)  


# Construct a bi-direction graph
# Initialize an empty set to store unique edges
edges_set = set()
# Iterate through faces and extract edges
for face in faces:
    # Get vertex indices for the current face
    v1, v2, v3 = face

    # Create edges (as tuples of sorted vertex indices)
    edge1 = tuple(sorted([v1, v2]))
    edge2 = tuple(sorted([v2, v3]))
    edge3 = tuple(sorted([v3, v1]))

    # Add edges to the set (duplicates will be ignored)
    edges_set.add(edge1)
    edges_set.add(edge2)
    edges_set.add(edge3)
edge_list_ij = [[e[0], e[1]] for e in edges_set]
edge_list_ji = [[e[1], e[0]] for e in edges_set]
edge_list = []
edge_list.extend(edge_list_ij)
edge_list.extend(edge_list_ji)
edge_info = torch.tensor(edge_list).transpose(0, 1)
print(f"edge info { edge_info.shape}, face info: {faces.shape}")

# Find adjancent
vertice_number = vertices.shape[0]
face_number = faces.shape[0]

# # [vertice_index, adjacent_faces_index]
# vertex_adjacent_faces = find_all_vertices_adjacent_faces(vertice_number, face_number, face_matrix=faces)
# print(f"vertex adjacent faces: {vertex_adjacent_faces}")
# # print(f"adjacent face shape: {vertex_adjacent_faces.shape}")
# vertex_adjacent_faces_mask = construct_vertex_adjacent_faces_mask(vertex_adjacent_faces,  vertice_number, face_number)
# print(f"adjacent face mask sshape: {vertex_adjacent_faces_mask.shape}")


# Get all boundary faces
ret = edge_to_face_mapping(faces)
all_boundary_faces = []
all_boundary_edges = []
for k, v in ret.items():
    if len(v) == 1:
        all_boundary_faces.append(v[0])
        all_boundary_edges.append(k)
print(f"all boundary edges {all_boundary_edges}, shape {np.array(all_boundary_edges).shape}")