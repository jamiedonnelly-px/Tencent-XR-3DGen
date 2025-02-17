
# import pdb; pdb.set_trace()

import os, sys
import trimesh
import argparse
import numpy as np
import shutil
from glob import glob
import pathlib

import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scipy.spatial.transform import Rotation as R
from lib.registration import SMPLRegistration
import open3d as o3d
from lib.utils.util import batch_transform
from lib.screenshotrenderer import ScreenShotRenderer, screenshot
import  cv2


def parse_args():

    parser = argparse.ArgumentParser()
    # parser.add_argument("--mesh_path", type=str, default = "./8_4290756309810844328/split/body/body.obj")
    # parser.add_argument("--dump_path", type=str, default = "./8_4290756309810844328/split/body/smplx_and_offset_smplified.npz")
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--dump_path", type=str, required=True)
    parser.add_argument("--matches", type=str, default=None)
    parser.add_argument("--rank", type=int, default=1)
    args = parser.parse_args()
    return args

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


def main():
    args = parse_args()


    from lib.configs.config_vroid import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.matches = args.matches
    cfg.mesh_path = args.mesh_path

    # load meshes
    m = trimesh.load(args.mesh_path)
    if isinstance(m, trimesh.scene.scene.Scene):  # need to handle uv pieces separately
        # merge scene to a single mesh
        meshes = []
        for k in m.geometry.keys():
            ms = m.geometry[k]
            meshes.append(ms)
        m = trimesh.util.concatenate(meshes)
    r = R.from_euler('x', 90, degrees=True).as_matrix()
    verts = (r @ np.asarray(m.vertices).T).T
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(m.faces))



    reg = SMPLRegistration(config=cfg, mesh = mesh, rank=args.rank)

    data, warpped_mesh = reg.align_vroid( viz=False)


    T = LayeredAvatar(data, reg.model)
    data = { "posed_verts": T.posed_verts, "faces": T.faces, "T": T.T }
    # dump = os.path.join(str(pathlib.Path(args.mesh_path).parent.parent), "smplx_and_offset_smplified.npz")
    torch.save(data, args.dump_path)

    smpl_deformed = os.path.join( pathlib.Path ( args.dump_path ).parent, "smpl_deformed.ply" )
    o3d.io.write_triangle_mesh( smpl_deformed, warpped_mesh )
    print("fitting done")

    #ScreenShot
    screenshot_path = os.path.join( pathlib.Path ( args.dump_path ).parent, "screenshot.jpg" )
    img = screenshot(  args.mesh_path,  smpl_deformed )
    cv2.imwrite(screenshot_path, img)
    print("screenshot done")




if __name__ == '__main__':
    import  time
    start = time.time( )
    main( )
    end = time.time( )
    print( "time used", end-start )