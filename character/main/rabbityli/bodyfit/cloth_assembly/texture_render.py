import os
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    AmbientLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    HardPhongShader
)

# add path for demo utils functions
import sys
import os
sys.path.append(os.path.abspath(''))

from plot_image_grid import image_grid


# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Set paths
DATA_DIR = "./tripo"
# obj_filename = os.path.join(DATA_DIR, "mesh/full.obj")
# obj_filename = os.path.join(DATA_DIR, "mesh/full.obj")

obj_filename = "/home/rabbityl/workspace/auto_rig/bodyfit/bodyfit/mcwy_example/part_00.obj"
# obj_filename = "./tripo/mesh/full.obj"

# Load obj file
mesh = load_objs_as_meshes([obj_filename], device=device)






# Load the obj and ignore the textures and materials.
verts, faces_idx, _ = load_obj(obj_filename)
faces = faces_idx.verts_idx

# Initialize each vertex to be white in color.
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = TexturesVertex(verts_features=verts_rgb.to(device))

# Create a Meshes object for the teapot. Here we have only one mesh in the batch.
teapot_mesh = Meshes(
    verts=[verts.to(device)],
    faces=[faces.to(device)],
    textures=textures
)









# Select the viewpoint using spherical angles
distance = 1  # distance from camera to the object
elevation = -90   # angle of elevation in degrees
azimuth = 180 # No rotation so the camera is positioned on the +Z axis.

# Get the position of the camera based on the spherical angles
ref_R, ref_T = look_at_view_transform(distance, elevation, azimuth, device=device)
# ref_T[]
# print(ref_T)
ref_T [...,1] = -0.6

print(ref_T)

cameras = FoVPerspectiveCameras(device=device, R=ref_R, T=ref_T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
# the difference between naive and coarse-to-fine rasterization.
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the
# -z direction.
# lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

lights = AmbientLights(device=device)

# loc = ( (-0.0000, -0.6000,  1.0000))
lights = PointLights(device=device,  location=((2.0, -2.0, 2.0),))

# Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
# interpolate the texture uv coordinates for each vertex, sample from a texture image and
# apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    # shader=SoftPhongShader(
    #     device=device,
    #     cameras=cameras,
    #     lights=lights
    # )
    shader = HardPhongShader(device=device, cameras=cameras, lights=lights)

)

images = renderer(teapot_mesh)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off");
plt.show()