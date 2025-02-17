import numpy as np
import os
from PIL import Image
import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (look_at_view_transform, FoVPerspectiveCameras, PointLights, RasterizationSettings,
                                MeshRenderer, MeshRasterizer, FoVOrthographicCameras, TexturesUV)
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.transforms import Rotate, Transform3d, matrix_to_euler_angles, euler_angles_to_matrix

import sys
current_script_path = os.path.abspath(__file__)
src_root = os.path.dirname(current_script_path)
project_root = os.path.dirname(src_root)
sys.path.append(src_root)
sys.path.append(project_root)

from src.geometry import HardGeometryShader
from src.utils_render import check_mesh_uv

class RenderControl:

    def __init__(self, resolution=256, device="cuda"):
        self.resolution = resolution
        self.device = device

        raster_settings = RasterizationSettings(
            image_size=self.resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        self.renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=FoVPerspectiveCameras(device=self.device),
                                                               raster_settings=raster_settings),
                                     shader=HardGeometryShader(device=self.device))

        self.renderer_ortho = MeshRenderer(rasterizer=MeshRasterizer(cameras=FoVOrthographicCameras(device=self.device),
                                                                     raster_settings=raster_settings),
                                           shader=HardGeometryShader(device=self.device))

    def render_mesh_depth(self,
                          obj_path,
                          num_frames=4,
                          elevation=15,
                          azimuth_start=0,
                          azimuth_span=360,
                          auto_center=True):
        """render depth views

        Args:
            obj_path: _description_
            num_frames: _description_. Defaults to 4.
            elevation: _description_. Defaults to 15.
            azimuth_start: _description_. Defaults to 0.
            azimuth_span: _description_. Defaults to 360.
            auto_center: _description_. Defaults to True.

        Returns:
            depths [4, self.resolution=256, 256, 3] tensor in [0, 1]
        """
        mesh = load_objs_as_meshes([obj_path], device=self.device)

        # # -> [-0.5, 0.5] cube
        mesh_min = mesh.verts_packed().min(0)[0]
        mesh_max = mesh.verts_packed().max(0)[0]
        print('mesh_min ', mesh_min)
        print('mesh_max ', mesh_max)
        # mesh_scale = (mesh_max - mesh_min).max()
        # mesh.verts_list()[0] = (mesh.verts_list()[0] - mesh_min) / mesh_scale - 0.5

        azim = np.linspace(azimuth_start, azimuth_start + azimuth_span, num_frames + 1)[:num_frames]
        print('azim ', azim)
        R, T = look_at_view_transform(dist=1.0, elev=elevation, azim=azim)
        cameras = FoVPerspectiveCameras(device=self.device, fov=60, R=R, T=T)

        # depths: [4, 256, 256, 2=d+a]
        verts, normals, depths, cos_angles, texels, fragments = self.renderer(mesh.extend(len(cameras)),
                                                                              cameras=cameras)
        depths = self.decode_normalized_depth(depths)  # in [0, 1]
        print('depths ', depths.shape, depths.min(), depths.max())
        return depths, obj_path

    def render_mesh_ortho_depth(self,
                                obj_path,
                                mesh_process="keep_raw",
                                num_frames=4,
                                image_size=512,
                                scale_factor=1.0,
                                auto_center=True,
                                azimuth_start=0,
                                azimuth_span=360,
                                camera_distance=2.0,
                                scale_xyz=1.0):
        """render ortho depth views

        Args:
            obj_path: _description_
            num_frames: _description_. Defaults to 4.
            image_size: _description_. Defaults to 512.
            scale_factor: _description_. Defaults to 2.0.
            auto_center: _description_. Defaults to True.
            azimuth_start: _description_. Defaults to 0.
            azimuth_span: _description_. Defaults to 360.
            camera_distance: _description_. Defaults to 2.0.
            scale_xyz: _description_. Defaults to 1.0.

        Returns:
            depths [4, self.resolution=256, 256, 3] tensor in [0, 1], obj_path with process
        """
        obj_path = check_mesh_uv(obj_path)
        
        mesh = load_objs_as_meshes([obj_path], device=self.device)
       
        if mesh_process == "rot2":
            # Define rotation in terms of euler angles
            rotation = torch.Tensor([-1* torch.pi / 2., 0., -1 * torch.pi / 2.]).cuda()
            rotation_matrix = euler_angles_to_matrix(rotation, "XYZ")
            rotated_verts_list = [torch.mm(vert, rotation_matrix.T) for vert in mesh.verts_list()]
            # Update mesh
            # mesh = Meshes(verts=rotated_verts_list, faces=mesh.faces_list())
            mesh._verts_list = rotated_verts_list
        elif mesh_process == "keep_raw":
            pass
        else:
            raise ValueError(f"invalid mesh_process={mesh_process}")
        
        if auto_center:
            # -> in [-scale_factor, scale_factor]
            verts = mesh.verts_packed()
            max_bb = (verts - 0).max(0)[0]
            min_bb = (verts - 0).min(0)[0]
            scale = (max_bb - min_bb).max() / 2
            center = (max_bb + min_bb) / 2
            mesh.offset_verts_(-center)
            mesh.scale_verts_((scale_factor / float(scale)))
            print('debug scale ', scale)
            print('debug center ', center)
        else:
            mesh.scale_verts_((scale_factor))
        mesh_min = mesh.verts_packed().min(0)[0]
        mesh_max = mesh.verts_packed().max(0)[0]
        print('debug mesh_min ', mesh_min)
        print('debug mesh_max ', mesh_max)

        azim = np.linspace(azimuth_start, azimuth_start + azimuth_span, num_frames + 1)[:num_frames]
        print('azim ortho: ', azim)
        elev = np.array([0] * azim.shape[0])
        R, T = look_at_view_transform(dist=camera_distance, elev=elev, azim=azim)
        # when mesh in [-1, 1], scale_xyz=1 means fit full.
        znear = 0.1  # TODO
        cameras = FoVOrthographicCameras(device=self.device, R=R, T=T, scale_xyz=((scale_xyz, scale_xyz, scale_xyz), ))

        if image_size:
            size_raw = self.renderer.rasterizer.raster_settings.image_size
            self.renderer.rasterizer.raster_settings.image_size = image_size
        # depths: [num_frames, 256, 256, 2=d+a] in meter
        verts, normals, depths, cos_angles, texels, fragments = self.renderer(mesh.extend(len(cameras)),
                                                                              cameras=cameras)
        # depths: [num_frames, 256, 256, 3] in [0, 1]
        depths = self.decode_normalized_depth(depths)  #
        print('depths ', depths.shape, depths.min(), depths.max())
        if image_size:
            self.renderer.rasterizer.raster_settings.image_size = size_raw

        return depths, obj_path

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


def test_render_control(obj_path, output_image_path, use_ortho, image_size=256, scale_factor=0.8, scale_xyz=1.0):
    render_control = RenderControl()

    if use_ortho:
        depth_maps, new_obj_path = render_control.render_mesh_ortho_depth(obj_path,
                                                            image_size=image_size,
                                                            scale_factor=scale_factor,
                                                            scale_xyz=scale_xyz)
    else:
        depth_maps, new_obj_path = render_control.render_mesh_depth(obj_path)

    depth_maps = depth_maps.cpu().numpy()
    depth_maps *= 255.0
    # depth_maps = (depth_maps - depth_maps.min()) / (depth_maps.max() - depth_maps.min()) * 255
    depth_maps = depth_maps.astype(np.uint8)  # 4, 256, 256, 3
    print('depth_maps ', depth_maps.shape)
    depth_maps_row = np.hstack(depth_maps)

    Image.fromarray(depth_maps_row).save(output_image_path)
    out_npy = output_image_path.replace(".png", ".npy")
    np.save(out_npy, depth_maps)

if __name__ == "__main__":

    # obj_path = "/aigc_cfs_2/sz/data/mvc/top_3/uv_condition/mesh.obj"
    # obj_path = "/aigc_cfs_2/sz/data/mvc/mario_tripo/untitled.obj"
    # obj_path = "/aigc_cfs_2/sz/data/mvc/chest_tripo/untitled.obj"
    # obj_path = "/aigc_cfs_2/sz/data/mvc/horse_tripo/horse.obj"
    # obj_path = "/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/pre/readyplayerme_top/rpm_top_1/uv_condition/mesh.obj"
    obj_path = "/aigc_cfs_gdp/jiawei/data/general_generate/5a7b172a-4aa7-47d8-a57b-20a7fd388535/texture_mesh/5a7b172a-4aa7-47d8-a57b-20a7fd388535.obj"
    # obj_path = "/aigc_cfs_gdp/jiawei/data/aitexture/3edf681c-2a66-4cf2-87cf-1a4009e5f8f0/in_mesh.obj"
    # obj_path = "/aigc_cfs_gdp/sz/batch_0816/compare_z123_lrm_human_ratio/z123_ratio2/1d721fda-ca16-4e86-b103-fa7af5457ee7_z123_ratio2/verts2tex/baking/tex_mesh.obj"

    use_ortho = True
    image_size = 512  # 256 for mvdream
    scale_factor = 1.0  # 0.9 # 1.0 # for ortho tripo
    scale_xyz = 0.9
    output_image_path = obj_path.replace(".obj", f"ortho_{use_ortho}_{image_size}.png")
    test_render_control(obj_path,
                        output_image_path,
                        use_ortho,
                        image_size=image_size,
                        scale_factor=scale_factor,
                        scale_xyz=scale_xyz)
