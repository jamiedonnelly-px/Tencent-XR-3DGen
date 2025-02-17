import torch
import pytorch3d
from torchvision.transforms import Compose, Resize, GaussianBlur, InterpolationMode

from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj, IO

from pytorch3d.transforms import RotateAxisAngle, Transform3d, matrix_to_euler_angles, euler_angles_to_matrix
from pytorch3d.structures import Meshes

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform, FoVPerspectiveCameras, FoVOrthographicCameras, AmbientLights,
                                PointLights, DirectionalLights, Materials, RasterizationSettings, MeshRenderer,
                                MeshRasterizer, TexturesUV)
import numpy as np

from geometry import HardGeometryShader
from shader import HardNChannelFlatShader

import os
from utils_render import save_rgba_geom_images

def rotate_mesh_x_axis(mesh: Meshes, angle_degrees: float):
    """
    加载 mesh 并沿 x 轴旋转指定角度。

    参数:
        mesh (Meshes): load_objs_as_meshes([mesh_path], device)
        angle_degrees (float): 旋转角度（以度为单位）

    返回:
        transformed_mesh (pytorch3d.structures.Meshes): 旋转后的 Meshes 对象
    """
    rotation = RotateAxisAngle(angle_degrees, axis="X", device=mesh.device)
    verts = mesh.verts_packed()

    transformed_verts = rotation.transform_points(verts)
    transformed_mesh = mesh.update_padded(transformed_verts.unsqueeze(0))
    return transformed_mesh


def make_pers_cameras(
        dist_list,
        elevation_list,
        azimuth_list,
        fov,
        use_blender_coord=False,
        device="cuda",
):
    R, T = look_at_view_transform(dist=dist_list, elev=elevation_list, azim=azimuth_list)
    # if use_blender_coord, need rot cams x from y-up to z-up
    if use_blender_coord:
        rotation = RotateAxisAngle(-90, axis="X").get_matrix()
        R = torch.matmul(rotation[:, :3, :3], R)

    cameras = FoVPerspectiveCameras(device=device, fov=fov, R=R, T=T)

    return cameras


def make_ortho_cameras(
    dist_list,
    elevation_list,
    azimuth_list,    
    # num_frames=4,
    # elevation=0,
    # azimuth_start=0,
    # azimuth_span=360,
    scale_xyz=1.0,
    # camera_distance=2.7,
    use_blender_coord=False,
    device="cuda",
):
    R, T = look_at_view_transform(dist=dist_list, elev=elevation_list, azim=azimuth_list)
    # if use_blender_coord, need rot cams x from y-up to z-up
    if use_blender_coord:
        rotation = RotateAxisAngle(-90, axis="X").get_matrix()
        R = torch.matmul(rotation[:, :3, :3], R)    
    # when mesh in [-0.5, 0.5], scale_xyz=1.0 means fit full? TODO
    cameras = FoVOrthographicCameras(device=device, R=R, T=T, scale_xyz=((scale_xyz, scale_xyz, scale_xyz), ))

    return cameras

@torch.no_grad()
def get_conditioning_images(uvp, output_size, render_size=512, blur_filter=5, cond_type="depth"):
    verts, normals, depths, cos_maps, texels, fragments = uvp.render_geometry(image_size=render_size)
    masks = normals[...,3][:,None,...]
    masks = Resize((output_size//8,)*2, antialias=True)(masks)
    normals_transforms = Compose([
        Resize((output_size,)*2, interpolation=InterpolationMode.BILINEAR, antialias=True),
        GaussianBlur(blur_filter, blur_filter//3+1)]
    )

    if cond_type == "normal":
        view_normals = uvp.decode_view_normal(normals).permute(0,3,1,2) *2 - 1
        conditional_images = normals_transforms(view_normals)
    # Some problem here, depth controlnet don't work when depth is normalized
    # But it do generate using the unnormalized form as below
    elif cond_type == "depth":
        view_depths = uvp.decode_normalized_depth(depths).permute(0,3,1,2)
        conditional_images = normals_transforms(view_depths)

    return conditional_images, masks


def get_geom_texture(result_views, render_):
    resize = Resize((render_.render_size, ) * 2, interpolation=InterpolationMode.NEAREST_EXACT, antialias=True)
    result_views = result_views.permute(0, 3, 1, 2)[:, :-1, :, :]
    result_views = resize(result_views / 2 + 0.5).clamp(0, 1).unbind(0)
    textured_views_rgb, result_tex_rgb, visibility_weights = render_.bake_texture(views=result_views,
                                                                                  main_views=[],
                                                                                  exp=6,
                                                                                  noisy=False)
    result_tex_rgb_output = result_tex_rgb.permute(1, 2, 0).cpu().numpy()[None, ...]
    return result_tex_rgb, result_tex_rgb_output


class GeomRender():
    """easy geom render without baking and so on..
    """
    def __init__(self, render_size=1024, sampling_mode="nearest", channels=3, device=None):
        self.channels = channels
        self.device = device or torch.device("cuda")
        self.lights = AmbientLights(ambient_color=((1.0, ) * channels, ), device=self.device)
        self.render_size = render_size
        self.sampling_mode = sampling_mode

    # Load obj mesh, rescale the mesh to fit into the bounding box [-1, 1] when scale_factor=1.0
    def load_mesh(self, mesh_path, scale_factor=1.0, transformation=None, auto_center=True):
        mesh : Meshes = load_objs_as_meshes([mesh_path], device=self.device)
        # -> in [-scale_factor, scale_factor] cube
        if transformation is not None:
            verts = mesh.verts_packed()
            transformation = torch.tensor(transformation, dtype=torch.float32, device=verts.device)

            print(f"raw verts between: {torch.min(verts, dim=0)[0]} -- {torch.max(verts, dim=0)[0]}")

            # pw, [N, 4]
            raw_points = torch.cat([verts, torch.ones_like(verts[..., 0:1])], dim=-1)

            # pc = Tcw * pw then select [N, 3]
            new_points = torch.bmm(transformation.unsqueeze(0), raw_points.permute(
                1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)[..., :3]

            # transformed_vertices = (raw_points @ transformation.t())[:, :3]

            ####
            mesh = mesh.update_padded(new_points.unsqueeze(0))
            verts = mesh.verts_packed()
            print(f"new verts between: {torch.min(verts, dim=0)[0]} -- {torch.max(verts, dim=0)[0]}")

        elif auto_center:
            verts = mesh.verts_packed()
            max_bb = (verts - 0).max(0)[0]
            min_bb = (verts - 0).min(0)[0]
            raw_len = (max_bb - min_bb).max() / 2
            center = (max_bb + min_bb) / 2
            mesh.offset_verts_(-center)
            mesh.scale_verts_((scale_factor / float(raw_len)))
        else:
            mesh.scale_verts_((scale_factor))

        self.mesh = mesh

        verts = self.mesh.verts_packed()
        max_bb = (verts - 0).max(0)[0]
        min_bb = (verts - 0).min(0)[0]

    # Save obj mesh
    def save_mesh(self, mesh_path, texture):
        save_obj(mesh_path,
                 self.mesh.verts_list()[0],
                 self.mesh.faces_list()[0],
                 verts_uvs=self.mesh.textures.verts_uvs_list()[0],
                 faces_uvs=self.mesh.textures.faces_uvs_list()[0],
                 texture_map=texture)

    # Set texture for the current mesh.
    def set_texture_map(self, texture):
        new_map = texture.permute(1, 2, 0)
        new_map = new_map.to(self.device)
        new_tex = TexturesUV([new_map],
                             self.mesh.textures.faces_uvs_padded(),
                             self.mesh.textures.verts_uvs_padded(),
                             sampling_mode=self.sampling_mode)
        self.mesh.textures = new_tex

    ########
    # Set all necessary internal data for rendering and texture baking
    # Can be used to refresh after changing camera positions
    def set_cameras_and_render_settings(self,
                                        cameras,
                                        render_size=None):
        """set self.cameras and self.renderer

        Args:
            cameras: pytorch3d cameras
            render_size: _description_. Defaults to None.
        """
        self.cameras = cameras
        if render_size is None:
            render_size = self.render_size
        if not hasattr(self, "renderer"):
            self.setup_renderer(size=render_size)


    def setup_renderer(self,
                       size=1024):
        raster_settings = RasterizationSettings(
            image_size=size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=FoVPerspectiveCameras(device=self.device),
                raster_settings=raster_settings
            ),
            shader=HardGeometryShader(device=self.device)
        )


    @torch.no_grad()
    def render_geometry(self, image_size=None):
        """render geometry: verts, normals, depths, cos_angles, texels, fragments
        [nc, image_size, image_size, c+1]  c + alpha
        Args:
            image_size: _description_. Defaults to None.
        """
        if image_size:
            size = self.renderer.rasterizer.raster_settings.image_size
            self.renderer.rasterizer.raster_settings.image_size = image_size
        shader = self.renderer.shader
        self.renderer.shader = HardGeometryShader(device=self.device,
                                                  cameras=self.cameras[0],
                                                  lights=self.lights)
        tmp_mesh = self.mesh.clone()

        verts, normals, depths, cos_angles, texels, fragments = self.renderer(
            tmp_mesh.extend(len(self.cameras)),
            cameras=self.cameras,
            lights=self.lights)
        self.renderer.shader = shader

        if image_size:
            self.renderer.rasterizer.raster_settings.image_size = size

        return verts, normals, depths, cos_angles, texels, fragments


    #####
    def make_pers_cameras(
        self,
        dist_list,
        elevation_list,
        azimuth_list,
        fov,
        up=((0, 1, 0),),
        use_blender_coord = False,
    ):
        R, T = look_at_view_transform(dist=dist_list, elev=elevation_list, azim=azimuth_list, up=up)
        # if use_blender_coord, need rot cams x from y-up to z-up
        if use_blender_coord:
            rotation = RotateAxisAngle(-90, axis="X").get_matrix()
            R = torch.matmul(rotation[:, :3, :3], R)


        cameras = FoVPerspectiveCameras(device=self.device, fov=fov, R=R, T=T)

        return cameras

    def make_ortho_cameras(self,
                           num_frames=4,
                           elevation=0,
                           azimuth_start=0,
                           azimuth_span=360,
                           scale_xyz=1.0,
                           camera_distance=2.7):

        azim = np.linspace(azimuth_start, azimuth_start + azimuth_span, num_frames + 1)[:num_frames]
        elev = np.array([elevation] * azim.shape[0])
        R, T = look_at_view_transform(dist=camera_distance, elev=elev, azim=azim)
        # when mesh in [-0.5, 0.5], scale_xyz=1.0 means fit full? TODO
        cameras = FoVOrthographicCameras(device=self.device, R=R, T=T, scale_xyz=((scale_xyz, scale_xyz, scale_xyz), ))

        return cameras

    def render_mesh_geom(self, out_dir):
        # depths: [nv, 256, 256, 2=d+a]
        verts, normals, depths, cos_angles, texels, fragments = self.renderer(self.mesh.extend(len(self.cameras)),
                                                                              cameras=self.cameras)
        print('depths ', depths.shape, depths.min(), depths.max())
        save_rgba_geom_images(verts, os.path.join(out_dir, "position.png"))
        save_rgba_geom_images(normals, os.path.join(out_dir, "normal.png"))
        return

    # Normalize absolute depth to inverse depth
    @torch.no_grad()
    def decode_normalized_depth(self, depths, batched_norm=False):
        view_z, mask = depths.unbind(-1)
        view_z = view_z * mask + 100 * (1 - mask)
        inv_z = 1 / view_z
        inv_z_min = inv_z * mask + 100 * (1 - mask)
        if not batched_norm:
            max_ = torch.max(inv_z, 1, keepdim=True)
            max_ = torch.max(max_[0], 2, keepdim=True)[0]

            min_ = torch.min(inv_z_min, 1, keepdim=True)
            min_ = torch.min(min_[0], 2, keepdim=True)[0]
        else:
            max_ = torch.max(inv_z)
            min_ = torch.min(inv_z_min)
        inv_z = (inv_z - min_) / (max_ - min_)
        inv_z = inv_z.clamp(0, 1)
        inv_z = inv_z[..., None].repeat(1, 1, 1, 3)

        return inv_z
