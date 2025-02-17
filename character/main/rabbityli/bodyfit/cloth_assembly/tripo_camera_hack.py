import os
import open3d as o3d

import torch
import numpy as np
from tqdm.notebook import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from pytorch3d.io import load_obj,load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex, SoftPhongShader, AmbientLights)

import cv2


def obtain_reference_img ( img_path ) :
    ref_image = cv2.imread( img_path , -1 )
    raw_h, raw_w, _ = ref_image.shape
    ratio_h, ratio_w = raw_h / 256. , raw_w / 256.
    ratio = ratio_h if ratio_h > ratio_w else ratio_w
    h, w =int(raw_h/ratio), int(raw_w/ratio)
    ref_image = cv2.resize(ref_image, (w, h), interpolation=cv2.INTER_CUBIC)
    ref_silluette = (ref_image[..., 3] > 0 ).astype(float)
    ref_color = ref_image[..., :3]
    ref_color = cv2.cvtColor(ref_color, cv2.COLOR_BGR2RGB)/255.0
    ref_color[ref_silluette < 1] = np.array([1.,1.,1.])
    return ref_color, ref_silluette, h, w

# part_path = "/home/rabbityl/Dropbox/tripo/legs"
part_path = "./tripo"

import glob
part_mesh = glob.glob(  os.path.join( part_path, "*/*.obj") ) [0]

ref_color, ref_silluette, h, w = obtain_reference_img( os.path.join( part_path,  "img.png" )  )
img_size = (h, w)

# Set the cuda device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Load the obj
teapot_mesh = load_objs_as_meshes([ part_mesh ], device=device)

# Initialize a perspective camera.
cameras = FoVPerspectiveCameras(device=device, fov=10)



# To blend the 100 faces we set a few parameters which control the opacity and the sharpness of
blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
raster_settings = RasterizationSettings(
    image_size=img_size,
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
    faces_per_pixel=100,
)

# Create a silhouette mesh renderer by composing a rasterizer and a shader.
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)


# We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.

# We can add a point light in front of the object.
lights = AmbientLights(device=device)
# lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

raster_settings = RasterizationSettings(
    image_size=img_size,
    blur_radius=0.0,
    faces_per_pixel=1,
)
pixel_rasterizer= MeshRasterizer( cameras=cameras, raster_settings=raster_settings)

texture_renderer = MeshRenderer(
    rasterizer=pixel_rasterizer,
    shader=SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights
    )
)

# Select the viewpoint using spherical angles
distance = 8  # distance from camera to the object
elevation = 0.0   # angle of elevation in degrees
azimuth = 90 # No rotation so the camera is positioned on the +Z axis.

# Get the position of the camera based on the spherical angles
ref_R, ref_T = look_at_view_transform(distance, elevation, azimuth, device=device)
# R2, T2 = look_at_view_transform(distance, elevation, 90, device=device)

# exit(0)

class Model(nn.Module):
    def __init__(self, meshes, silhouette_renderer, texture_renderer,  ref_silluette, ref_color):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.silhouette_renderer = silhouette_renderer
        self.texture_renderer = texture_renderer

        # Get the silhouette of the reference RGB image by finding all non-white pixel values.
        self.ref_silluette = torch.from_numpy(ref_silluette.astype(np.float32))[None].to(self.device)
        self.ref_color = torch.from_numpy(ref_color.astype(np.float32))[None].to(self.device)
        # self.register_buffer('image_ref', image_ref)

        # Create an optimizable parameter for the x, y, z position of the camera.
        self.camera_position = nn.Parameter(
            torch.from_numpy(
                np.array([0, 0, 5],
                # np.array([-3.4248e-02,  1.6270e-03,  3.9797e+00],
                         dtype=np.float32)).to(meshes.device))

    def forward(self):

        # Render the image using the updated camera position. Based on the new position of the
        # camera we calculate the rotation and translation matrices
        # R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        # T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)

        R = ref_R
        T = self.camera_position[None]

        sillouette_image = self.silhouette_renderer(meshes_world=self.meshes.clone(), R=R, T=T)[..., 3]

        color_image = self.texture_renderer(meshes_world=self.meshes.clone(), R=R, T=T)[..., :3]

        color_diff = (color_image - self.ref_color).abs()
        loss_color = torch.sum( color_diff**2 )

        # Calculate the silhouette loss
        loss_silhouette = torch.sum((sillouette_image  - self.ref_silluette) ** 2)


        return loss_silhouette,  loss_color, color_diff, sillouette_image, R, T


    def unproject(self):

        R = ref_R
        T = self.camera_position[None]
        sillouette_image = self.silhouette_renderer(meshes_world=self.meshes.clone(), R=R, T=T)[..., 3]

        valid_region = torch.logical_and( sillouette_image, self.ref_silluette)

        fragments = pixel_rasterizer(self.meshes.clone(), R=R, T=T)
        # pcd = fragments.zbuf.squeeze()

        bary_coords = fragments.bary_coords

        valid_region = torch.logical_and( valid_region , (fragments.bary_coords.sum(-1) == 1 ).squeeze(-1))

        verts = self.meshes._verts_list[0]
        faces = self.meshes._faces_list[0]
        hit_face = faces [fragments.pix_to_face.squeeze(-1)]

        face_verts = verts [ hit_face ]

        face_verts = face_verts
        bary_coords = torch.swapaxes(bary_coords, -1, -2)
        # bary_coords = bary_coords.reshape(-1,3,1)

        point_cloud = (face_verts * bary_coords).sum(dim=-2)


        # point_cloud[~valid_region] =  torch.Tensor ( [ float("-INF"),float("-INF"),float("-INF")]).to(self.device)
        point_cloud, valid_region = point_cloud.detach().cpu().numpy().squeeze(), valid_region.detach().cpu().numpy().squeeze()


        #
        pc = o3d.geometry.PointCloud()
        # pc.points = o3d.utility.Vector3dVector( point_cloud.detach().cpu().numpy().squeeze()  )
        pc.points = o3d.utility.Vector3dVector( point_cloud[valid_region]   )
        # pc.points = o3d.utility.Vector3dVector(face_verts.squeeze().cpu().numpy().mean(axis=-2).reshape(-1,3))
        # pc.points = o3d.utility.Vector3dVector(pc)
        pc.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw([pc])

        # def vis_heatmap( img):
        #     plt.imshow(img)
        #     plt.show()

        return point_cloud, valid_region


# We will save images periodically and compose them into a GIF.
filename_output = os.path.join( part_path, "fit.gif" )
writer = imageio.get_writer(filename_output, mode='I', duration=0.3)


# Initialize a model using the renderer, mesh and reference image
model = Model(meshes=teapot_mesh, silhouette_renderer=silhouette_renderer, texture_renderer=texture_renderer,
              ref_silluette=ref_silluette, ref_color = ref_color).to(device)


# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

plt.figure(figsize=(10, 10))

_, _,_ ,image_init, R, T = model()
plt.subplot(1, 2, 1)
plt.imshow(image_init.detach().squeeze().cpu().numpy())
plt.grid(False)
plt.title("Starting position")

plt.subplot(1,2, 2)
plt.imshow(model.ref_silluette.cpu().numpy().squeeze())
plt.grid(False)
plt.title("Reference silhouette")

plt.subplot(1,3,3)
plt.imshow(ref_color)
plt.grid(False)
plt.title("Reference silhouette")

plt.show()





# exit(0)



n_iter = 500  #500

loop = tqdm(range(n_iter))
for i in loop:
    optimizer.zero_grad()
    loss_sil, loss_color, color_diff, _, R, T = model()

    loss = loss_sil + loss_color

    loss.backward()
    optimizer.step()

    loss_desc = "loss_sil:" + f'={loss_sil.item():.5f}| ' + "loss_color:" + f'={loss_color.item():.5f}| '

    # print( loss_desc )

    # loop.set_description('Optimizing (loss %.4f)' % loss.data)

    if loss.item() < 200:
        break

    # Save outputs to create a GIF.
    if i % 10 == 0:

        color_diff = color_diff[0].detach().squeeze().cpu().numpy()
        # image_ref = image_ref[0, ..., :3].detach().squeeze().cpu().numpy()
        image = np.concatenate ( [color_diff, ref_color], axis=0)
        image = img_as_ubyte(image)
        writer.append_data(image)

writer.close()

print( loss_desc)
print("R", R)
print("T", T)

point_cloud, valid_region = model.unproject()

with open( os.path.join(part_path, "part.npy") , 'wb') as f:
    np.save(f, point_cloud)
    np.save(f, valid_region)
