import torch
import numpy as np
from pytorch3d.ops.knn import knn_points
import json
import trimesh

def load_json(j):
    with open( j) as f:
        data = json.load(f)
    return data

def Scene2Trimesh(m):
    meshes = []
    for k in m.geometry.keys():
        ms = m.geometry[k]
        meshes.append(ms)
    m = trimesh.util.concatenate(meshes)
    return m

def batch_transform(P, v, pad_ones=True):
    if pad_ones:
        homo = torch.ones((*v.shape[:-1], 1), dtype=v.dtype, device=v.device)
    else:
        homo = torch.zeros((*v.shape[:-1], 1), dtype=v.dtype, device=v.device)
    v_homo = torch.cat((v, homo), dim=-1)
    v_homo = torch.matmul(P, v_homo.unsqueeze(-1))
    v_ = v_homo[..., :3, 0]
    return v_

def compute_scale_diff(src_head, tgt_head):
    src_mean = src_head.mean(dim=1, keepdims=True)
    tgt_mean = tgt_head.mean(dim=1, keepdims=True)
    src_len = torch.norm(src_head - src_mean, dim=-1)
    tgt_len = torch.norm(tgt_head - tgt_mean, dim=-1)
    scale = (tgt_len / src_len).mean()
    return scale


def scale_aware_warp( T_s2t, posed_verts,  part_index, scale, smpl_vert_src, asset_vert, K=10):
    '''
    @T_s2t: transforms of all smpl verts
    @part_index: index of refered part
    @scale: scale
    @smpl_vert_src: smpl vert of refered part
    @asset_vert: vert on asset to be transformed
    '''
    T_left_2_tgt = T_s2t[:, part_index]
    T_scale_2_left = torch.eye(4)[None, None].repeat(1, T_left_2_tgt.shape[1], 1, 1).to(smpl_vert_src)
    T_scale_2_left[..., :3, 3:] = (smpl_vert_src - scale * smpl_vert_src).reshape(1, -1, 3, 1)
    T_scale_2_tgt = T_left_2_tgt @ T_scale_2_left
    asset_vert_scaled = asset_vert * scale
    asset_vert_deform, ref_idx = interp_transform_scaled(asset_vert, posed_verts[:, part_index], asset_vert_scaled, T_scale_2_tgt, K=K)
    return asset_vert_deform, ref_idx


def concatenent_meshes ( V_lst, F_lst ):
    assert len(V_lst) == len(F_lst)
    assert len(V_lst) > 1
    size = [len(e) for e in V_lst[:-1]]
    id_offset = [0]+ list(np.cumsum(size))
    for i in range(len(F_lst)):
        F_lst[i] = F_lst[i] + id_offset[i]
    V = np.concatenate( V_lst, axis=0 )
    F = np.concatenate( F_lst, axis=0 )
    return V, F




def interp_transform(points, template_points, template_transform, K=6):
    results = knn_points(points[None], template_points, K=K)
    dists, idxs = results.dists, results.idx
    neighbs_weight = torch.exp(-dists)
    neighbs_weight = neighbs_weight / neighbs_weight.sum(-1, keepdim=True)
    neighbs_transform = template_transform[:, idxs[0], :, :].view(1, -1, 4, 4)
    points_K = points[:, None].repeat(1, K, 1).view(-1, 3)
    points_K_warpped = batch_transform(neighbs_transform, points_K).reshape(1, -1, K, 3)
    points_merge = (neighbs_weight[..., None] * points_K_warpped).sum(dim=-2)
    return points_merge.squeeze(), idxs.squeeze()


def interp_transform_scaled(points, template_points, point_scaled, template_transform, K=6):
    '''
    @point: asset point
    @template_points: smpl verts
    @point_scaled: asset point scaled
    @template_transform: smpl transformations
    '''

    results = knn_points(points[None], template_points, K=K)
    dists, idxs = results.dists, results.idx
    neighbs_weight = torch.exp(-dists)
    neighbs_weight = neighbs_weight / neighbs_weight.sum(-1, keepdim=True)

    neighbs_transform = template_transform[:, idxs[0], :, :].view(1, -1, 4, 4)
    points_K = point_scaled[:, None].repeat(1, K, 1).view(-1, 3)
    points_K_warpped = batch_transform(neighbs_transform, points_K).reshape(1, -1, K, 3)
    points_merge = (neighbs_weight[..., None] * points_K_warpped).sum(dim=-2)
    return points_merge.squeeze(), idxs.squeeze()


def interp_transform_geodesic(points, template_points, template_transform, geo_dists, K=6):
    v_2_arm, arms_index = geo_dists["arm"]
    v_2_torsol, torsol_index = geo_dists["torsol"]

    arm_pts = template_points[:, arms_index]
    arm_trans = template_transform[:, arms_index]

    torsol_pts = template_points[:, torsol_index]
    torsol_trans = template_transform[:, torsol_index]

    arm_deform, _ = interp_transform(points, arm_pts, arm_trans)  # points, template_points, template_transform, K = 6
    torsol_deform, _ = interp_transform(points, torsol_pts,
                                        torsol_trans)  # points, template_points, template_transform, K = 6

    temprature = 1
    v_2_arm = torch.exp(-v_2_arm / temprature)
    v_2_torsol = torch.exp(-v_2_torsol / temprature)
    gsum = v_2_arm + v_2_torsol
    w_arm = v_2_arm / gsum
    w_torsol = v_2_torsol / gsum

    p_deform = w_arm[:, None] * arm_deform + w_torsol[:, None] * torsol_deform

    return p_deform