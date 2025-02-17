import copy
import pdb
import cv2
import trimesh
import os
import glob
import numpy as np
import pathlib
import  json
import sys
import  torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt

from pytorch3d.io import load_obj,load_objs_as_meshes, load_ply

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex, SoftPhongShader, AmbientLights)


def load_json(j):
    with open( j) as f:
        data = json.load(f)
    return data

def write_json(fname , data  ):
    json_object = json.dumps(data, indent=4)
    with open(fname, "w") as outfile:
        outfile.write(json_object)

class ScreenShotRenderer:

    def __init__(self, ):

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        self.device = device

        distance = 1  # distance from camera to the object
        elevation = -90  # angle of elevation in degrees
        azimuth = 180  # No rotation so the camera is positioned on the +Z axis.

        # Get the position of the camera based on the spherical angles
        ref_R, ref_T = look_at_view_transform(distance, elevation, azimuth, device=device)
        # ref_T[]
        # print(ref_T)
        ref_T[..., 1] = -0.6

        print(ref_T)

        cameras = FoVPerspectiveCameras(device=device, R=ref_R, T=ref_T)


        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # We can add a point light in front of the object.
        # lights = PointLights(device=device, location=((2.0, 2.0, 2.0),))
        lights = PointLights(device=device, location=((2.0, -2.0, 2.0),))
        phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
        )

        self.phong_renderer = phong_renderer



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


        # exit(0)
        image_ref = self.phong_renderer ( teapot_mesh)

        return image_ref.cpu().numpy()


if __name__ == '__main__':

    ss = ScreenShotRenderer()

    # print (screenshot_path)
    # verts, faces_idx, _ = load_obj("/home/rabbityl/workspace/auto_rig/bodyfit/bodyfit/mcwy_example/part_00.obj")
    # faces = faces_idx.verts_idx
    # img = ss.render_obj(verts, faces).squeeze() * 255
    #
    # print( img.max() )


    # render_obj

    screenshot_root = "/aigc_cfs_2/rabbityli/screenshot/mcwy_filer"
    dump_path = "/aigc_cfs_2/rabbityli/screenshot/mcwy_ss"

    dirs = glob.glob( os.path.join( screenshot_root, "*" ) )

    for idx, d in  enumerate (dirs):

        print( "-------", idx, "/", len(dirs),"-------")

        obj_name = d.split("/")[-1]
        mesh_path = os.path.join( d, "part_00/part_00.obj" )

        screenshot_path = os.path.join(pathlib.Path(dump_path),  obj_name + ".jpg")


        print (screenshot_path)
        verts, faces_idx, _ = load_obj(mesh_path)
        faces = faces_idx.verts_idx
        img = ss.render_obj(verts, faces).squeeze()*255

        print( "img.max()", img.max() )

        cv2.imwrite(screenshot_path, img)
        print("screenshot done")
        # exit(0)

