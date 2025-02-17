
import torch
from torch.nn import functional as F
import numpy as np
from torch import Tensor
from ipdb import set_trace as st
import copy
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix


def angaxe2rot(ang, axe):
    K = np.array([
        [0, -axe[2], axe[1]],
        [axe[2], 0 , -axe[0]],
        [-axe[1], axe[0], 0 ]
    ])
    return np.eye(3) + np.sin(ang) * K + (1-np.cos(ang)) * K @ K

def get_neighbour_from_faces(faces):
    neighbors = {}
    for i in range(faces.shape[0]):
        for j in faces[i]:
            if j in neighbors:
                neighbors[j] = neighbors[j].union(set(faces[i].tolist())-{j})
            else:
                neighbors[j]= set(faces[i].tolist())-{j}
    return neighbors

def get_laplacian_matrix_from_faces(vertices, faces, type='naive'):
    """
    vertices
    """
    nv = vertices.shape[0]
    laplacian_mat = np.zeros([nv,nv])
    if type == 'naive':
        for vi in np.arange(nv):
            ni = np.unique(faces[np.unique(np.where(faces == vi)[0])].reshape(-1))
            nn = len(ni) - 1
            if nn > 0:
                laplacian_mat[vi, ni] = 1/nn
                laplacian_mat[vi, vi] = -1
    elif type == 'laplace_matrix_wodiag_area':
        # compute face areas
        v1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
        v2 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        face_areas = 0.5 * np.linalg.norm(np.cross(v1, v2),axis=1)
        rows = faces.reshape(-1)
        cols = np.zeros_like(rows)
        values = face_areas[np.arange(rows.shape[0])//3]
        vert_areas = coo_matrix((values,(rows,cols)),shape=(nv,1)).toarray()[:,0]
        
        edge_cot = np.zeros([faces.shape[0], 3],dtype=np.float32)
        rows = []
        cols = []
        values = []
        for i in range(3):
            v1 = vertices[faces[:,(i+1)%3]] - vertices[faces[:, i]]
            v2 = vertices[faces[:,(i+2)%3]] - vertices[faces[:, i]]
            edge_cot[:,i] = np.sum(v1*v2, axis=1) / (1e-6 + np.linalg.norm(np.cross(v1, v2), axis=1))
            laplacian_mat[faces[:,(i+1)%3], faces[:,(i+2)%3]] += 0.5 * edge_cot[:, i]
            laplacian_mat[faces[:,(i+2)%3], faces[:,(i+1)%3]] += 0.5 * edge_cot[:, i]
    else:
        # compute face areas
        v1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
        v2 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        face_areas = 0.5 * np.linalg.norm(np.cross(v1, v2),axis=1)
        rows = faces.reshape(-1)
        cols = np.zeros_like(rows)
        values = face_areas[np.arange(rows.shape[0])//3]
        vert_areas = coo_matrix((values,(rows,cols)),shape=(nv,1)).toarray()[:,0]
        
        edge_cot = np.zeros([faces.shape[0], 3],dtype=np.float32)
        rows = []
        cols = []
        values = []
        for i in range(3):
            v1 = vertices[faces[:,(i+1)%3]] - vertices[faces[:, i]]
            v2 = vertices[faces[:,(i+2)%3]] - vertices[faces[:, i]]
            edge_cot[:,i] = np.sum(v1*v2, axis=1) / (1e-6 + np.linalg.norm(np.cross(v1, v2), axis=1))
            laplacian_mat[faces[:,(i+1)%3], faces[:,(i+2)%3]] += 0.5 * edge_cot[:, i]
            laplacian_mat[faces[:,(i+2)%3], faces[:,(i+1)%3]] += 0.5 * edge_cot[:, i]
            
        laplacian_mat = laplacian_mat - np.diag(laplacian_mat.sum(axis=1))
        laplacian_mat = 3 * laplacian_mat / (vert_areas[:,None] + 1e-6)
    return laplacian_mat


def get_n_order_neighbour(neighbors_dict,idx,order=1):
    # st()
    tmp_idx = [idx]
    his_idx = []
    for i in range(order):
        idx_new = []
        for id in tmp_idx:
            if id in his_idx:
                continue
            try:
                neig = neighbors_dict[id]
            except:
                st()
                print(id)
                continue
            idx_new += neig
            his_idx.append(id)
        idx_new = set(idx_new)
        tmp_idx = idx_new
    his_idx += list(tmp_idx)
    his_idx = list(set(his_idx) - {idx})
    return his_idx
