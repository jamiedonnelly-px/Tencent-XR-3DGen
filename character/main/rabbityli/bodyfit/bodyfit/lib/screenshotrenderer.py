import trimesh
import os

import cv2
import glob
import numpy as np
import pathlib
from pathlib import Path
import  json

from scipy.spatial.transform import Rotation as R
import os
import torch
import open3d as o3d
import glob
import torch
import numpy as np
from tqdm.notebook import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from pytorch3d.io import load_obj,load_objs_as_meshes, load_ply

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex, SoftPhongShader, AmbientLights)



class ScreenShotRenderer:

    def __init__(self, ):

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        cameras = FoVPerspectiveCameras(device=device, fov=60)

        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        # We can add a point light in front of the object.
        lights = PointLights(device=device, location=((2.0, 2.0, 2.0),))
        phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
        )

        self.phong_renderer = phong_renderer
        self.device = device


    def render_obj(self, verts, faces):

        device =self.device



        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(device))

        # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
        teapot_mesh = Meshes(
            verts=[verts.to(device)],
            faces=[faces.to(device)],
            textures=textures
        )

        # Initialize a perspective camera.

        # Select the viewpoint using spherical angles
        distance = 8  # distance from camera to the object
        elevation = 0.0  # angle of elevation in degrees
        azimuth = 0  # No rotation so the camera is positioned on the +Z axis.

        # Select the viewpoint using spherical angles
        distance = 2  # distance from camera to the object
        elevation = 0.0  # angle of elevation in degrees
        azimuth = 90  # No rotation so the camera is positioned on the +Z axis.

        # Get the position of the camera based on the spherical angles
        R, T = look_at_view_transform(distance, elevation, azimuth, device=device)  # , at =( (0,1,0)))
        # T[..., 1] = -8

        # exit(0)
        image_ref = self.phong_renderer (meshes_world=teapot_mesh, R=R, T=T)

        return image_ref.cpu().numpy()



# if __name__ == '__main__':

def screenshot( part_mesh1, part_mesh2=None,  ss=None):

    if ss is None:
        ss = ScreenShotRenderer()

    # part_mesh1 = "/home/rabbityl/workspace/auto_rig/bodyfit/bodyfit/daz_example/Female_Scout_1_0_0/split/body/Female_Scout__body.obj"
    verts, faces_idx, _ = load_obj(part_mesh1)
    faces = faces_idx.verts_idx
    img1 = ss.render_obj( verts, faces ).squeeze()
    # plt.imshow(img.squeeze())
    # plt.show()

    # if part_mesh2 :
    # part_mesh2 = "/home/rabbityl/workspace/auto_rig/bodyfit/bodyfit/daz_example/Female_Scout_1_0_0/split/smpl_deformed.ply"
    verts, faces  = load_ply(part_mesh2)
    r = torch.from_numpy ( R.from_euler('x', -90, degrees=True).as_matrix() ).float()
    verts = (r @ verts.T).T
    img2 = ss.render_obj( verts, faces ).squeeze()
    # plt.imshow(img.squeeze())
    # plt.show()

    img = np.concatenate ( [img1, img2], axis=1 )[..., :-1] * 255
    return img
    # plt.imshow(img)
    # plt.show()


def main():

    def load_json(j):
        with open(j) as f:
            data = json.load(f)
        return data


    ss = ScreenShotRenderer()

    data_json = "/aigc_cfs_11/Asset/active_list/layered_data/20240618/layer_embedding_20240618_all_with_all_glb.json"
    # data_json = "./layer_embedding_20240618_all_with_all_glb.json"
    data_json = load_json(data_json)

    screenshot_root = "/aigc_cfs_2/rabbityli/screenshot"

    clses = ["DAZ_DAZ_Top", "DAZ_DAZ_Shoe", "DAZ_DAZ_Outfit", "DAZ_DAZ_Bottom", ]

    all = 1432
    cnt = 0
    for cls in clses:

        screenshot_path = os.path.join( screenshot_root, cls)
        Path(screenshot_path).mkdir(exist_ok=True)

        for asset in data_json["data"][cls].keys():
            cnt += 1
            print( asset, cnt, "/", all)

            obj = data_json["data"][cls][asset]["Mesh"]

            # obj = obj.replace("(", "\(")
            # obj = obj.replace("&", "\&")
            # obj = obj.replace(")", "\)")

            # check if body is processed
            objp = pathlib.Path(obj).parent.parent.parent

            # print( "objp", objp)
            # print("glob", glob.glob(os.path.join(objp, "split/body", "*body.obj")))

            body = glob.glob(os.path.join(objp, "split/body", "*body.obj")) [0]
            smpl_param_path = os.path.join(objp, "split/", "smplx_and_offset_smplified.npz")
            smpl_ply = os.path.join(objp, "split/", "smpl_deformed.ply")

            # body = body.replace("(", "\(")
            # body = body.replace("&", "\&")
            # body = body.replace(")", "\)")
            #
            # smpl_param_path = smpl_param_path.replace("(", "\(")
            # smpl_param_path = smpl_param_path.replace("&", "\&")
            # smpl_param_path = smpl_param_path.replace(")", "\)")
            #
            # smpl_ply = smpl_ply.replace("(", "\(")
            # smpl_ply = smpl_ply.replace("&", "\&")
            # smpl_ply = smpl_ply.replace(")", "\)")

            if Path(smpl_param_path).exists()  and Path(smpl_ply).exists() :

                # print("#################Body Exist################")
                pass

                # import  pdb
                # pdb.set_trace()
                img = screenshot(  body, smpl_ply, ss=ss)
                cv2.imwrite( os.path.join( screenshot_path, asset + ".jpg" ), img )

                # if cnt > 10 :
                #     exit(0)

            else:

                print( "#################@@@@@@@@@@@@@@@##############")
                print( asset, cnt, "/", all)


if __name__ == '__main__':


    ss = ScreenShotRenderer()


    verts, faces_idx, _ = load_obj(""
                                   "/home/rabbityl/workspace/auto_rig/bodyfit/bodyfit/mcwy_example/part_00.obj")
    faces = faces_idx.verts_idx
    img1 = ss.render_obj( verts, faces ).squeeze()[..., :3]* 255

    plt.imshow(img1)
    plt.show()