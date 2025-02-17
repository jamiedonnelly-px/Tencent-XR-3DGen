import copy

import torch
import trimesh
from .body_config.smpl_config import part_wrap_ref_index_map

from .utils.utils import load_json, Scene2Trimesh, compute_scale_diff, scale_aware_warp


def transfer_hair(T_s2t, m, T_body):
    pass


def transfer_shoes(T_s2t, m, T_body):
    pass

def save_cloth( m,  save_path ):
    mtl_name = save_path.split("/")[-1]
    # print("save mesh material", m.visual.material.name)
    m.visual.material.name = mtl_name
    m.export(save_path)
    # print("save mesh material", m.visual.material.name)
    print("save mesh", save_path)

    # if self.label == "hair":
    #     print("label:", self.label)
    #     fix_hair_mtl_names(save_path)


def scale_diff(S_body, T_body, scale_ref_index ):
    '''
    :param S_body:
    :param T_body:
    :param part_key:
    :return:
    '''
    src_torsol = S_body.posed_verts[:, scale_ref_index]  # .detach().cpu().numpy()[0]
    tgt_torsol = T_body.posed_verts[:, scale_ref_index]  # .detach().cpu().numpy()[0]
    scale = compute_scale_diff(src_torsol, tgt_torsol)
    return scale


def transfer_verts(S_body, T_body, verts, transform, label, scale, T_s2t=None):
    '''
    :param S_body:
    :param T_body:
    :param verts:
    :param transform:
    :param scale:
    :param T_s2t:
    :return:
    '''

    if T_s2t is None:
        T_s2t = T_body.T @ torch.inverse(S_body.T)

    vert = torch.from_numpy(verts).to(T_s2t)
    if transform is not None:
        vert = (transform[:3, :3] @ vert.T + transform[:3, 3:]).T

    vert_stand = copy.deepcopy(vert)

    src_full_cloth = S_body.posed_verts[:, part_wrap_ref_index_map[label] ]  #detach().cpu().numpy()[0]
    vert_deform, ref_idx = scale_aware_warp(T_s2t, S_body.posed_verts, part_wrap_ref_index_map[label], scale, src_full_cloth, vert, K=4)
    vert_deform = vert_deform.squeeze().detach().cpu().numpy()

    return vert_deform, vert_stand








