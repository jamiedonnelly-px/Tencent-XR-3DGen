import copy

import torch
import json
import open3d as o3d
import numpy as np
# from
import sys
import os

codedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append( os.path.join(codedir, "../") )

# sys.path.append("../")
from layer_avatar.body_config.smpl_config import hair_index #as head_index


def expand_ring( V,F, ind, iter = 1):

    # import pdb; pdb.set_trace()

    index = np.asarray( copy.deepcopy(ind) )
    all_index = np.arange(V.shape[0])
    v_mask = np.zeros(V.shape[0]).astype(bool)

    for i in range(iter):

        head_mask = np.zeros( V.shape[0]).astype(bool)
        head_mask[index] = True
        valid_faces_mask = head_mask[ F.reshape(-1) ].reshape(-1, 3).sum(-1) > 0
        valid_faces = F [valid_faces_mask]
        v_mask [ valid_faces ] = True
        index = all_index [ v_mask]

    return index



def extract_sub_mesh( V, F, index   ):

    head_verts = V[index]
    head_mask = np.zeros( V.shape[0]).astype(bool)
    head_mask[index] = True
    valid_faces_mask = head_mask[ F.reshape(-1) ].reshape(-1, 3).sum(-1) == 3
    valid_faces = F [valid_faces_mask]

    # old to new index map
    imap = np.ones((V.shape[0])).astype(np.int64)
    imap[index] = np.arange(head_verts.shape[0])
    valid_faces = imap[valid_faces.reshape(-1)].reshape(-1, 3)

    return head_verts, valid_faces

class Body():
    def __init__(self, param_data, device,  timers=None):

        if timers: timers.tic('forward_skinning')
        param_data = torch.load(param_data, map_location='cuda:0')
        self.faces = param_data['faces'].to(device)
        self.posed_verts = param_data['posed_verts'].to(device)
        self.T = param_data['T'].to(device)
        if timers:timers.toc('forward_skinning')


        # smpl surface
        faces = param_data['faces'].cpu().numpy()
        posed_verts = param_data['posed_verts'].detach().cpu().numpy()[0]
        body = o3d.geometry.TriangleMesh()
        body.vertices = o3d.utility.Vector3dVector(posed_verts)
        body.triangles = o3d.utility.Vector3iVector(faces)
        body.compute_vertex_normals()
        self.body_manifold = body

        # head surface


        head_index = expand_ring( posed_verts, faces, hair_index, iter=2 )
        head_verts, head_faces = extract_sub_mesh( V=posed_verts, F=faces, index=head_index)
        head = o3d.geometry.TriangleMesh()
        head.vertices = o3d.utility.Vector3dVector(head_verts)
        head.triangles = o3d.utility.Vector3iVector(head_faces)
        head.compute_vertex_normals()
        self.head = head
