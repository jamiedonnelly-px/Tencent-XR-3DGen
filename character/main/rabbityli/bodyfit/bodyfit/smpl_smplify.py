import importlib.util
import os, sys
import argparse
import shutil
import glob
import trimesh
import torch
from lib.utils.util import batch_transform
from pathlib import Path
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from lib.registration import SMPLRegistration

from lib.smpl_w_scale import SMPL_with_scale




import pathlib

class LayeredAvatar():
    def __init__(self, param_data,  smplxs,  trns=None ):

        self.transform = trns

        if type(param_data) == str or type(param_data) == pathlib.PosixPath:
            param_data = torch.load(param_data)
        self.faces, self.template, self.T = smplxs.forward_skinning(param_data)

        if self.transform is not None:
            self.transform = torch.from_numpy(self.transform).float().to(self.T.device)

        part_idx_dict = smplxs.smplx.get_part_index()
        face_idx = part_idx_dict['face']
        ndp_offset = param_data["offset"].view(1, -1, 3, 1).detach()
        ndp_offset[:, face_idx] = 0
        self.T[..., :3, 3:] = self.T[..., :3, 3:] + ndp_offset

        self.posed_verts = batch_transform(self.T, self.template)


if __name__ == '__main__':
    # args = parse_args()
    import numpy as np

    base_dir = "/aigc_cfs/rabbityli/base_bodies/"

    lst = glob.glob ( os.path.join(base_dir, "*/smplx_and_offset.npz"))

    print( lst )

    from lib.configs.config_vroid import get_cfg_defaults

    cfg = get_cfg_defaults()

    smplxs = SMPL_with_scale(cfg).to(torch.device(0))
    #


    for ele in lst :
        print( ele )



        T = LayeredAvatar(ele ,  smplxs)


        data = {
            "posed_verts": T.posed_verts,
            "faces": T.faces,
            "T": T.T
        }

        dump = os.path.join( str(pathlib.Path(ele).parent) , "smplx_and_offset_smplified.npz" )

        torch.save(data, dump)