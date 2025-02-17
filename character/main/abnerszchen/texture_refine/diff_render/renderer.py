import torch
import os
import pytorch3d
from torchvision.transforms import Compose, Resize, GaussianBlur, InterpolationMode

from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj, IO

from pytorch3d.transforms import Rotate, Transform3d, matrix_to_euler_angles, euler_angles_to_matrix
from pytorch3d.structures import Meshes

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform, FoVPerspectiveCameras, FoVOrthographicCameras, AmbientLights,
                                PointLights, DirectionalLights, Materials, RasterizationSettings, MeshRenderer,
                                MeshRasterizer, TexturesUV)
import numpy as np

import glob
from geometry import HardGeometryShader
from shader import HardNChannelFlatShader
from voronoi import voronoi_solve
from geom_renderer import rotate_mesh_x_axis, make_ortho_cameras, get_conditioning_images


class DiffRender():

    def __init__(self, texture_size=1024, render_size=1024, sampling_mode="nearest", channels=3, device=None):
        """_summary_

        Args:
            texture_size: uv resolution. Defaults to 1024.
            render_size: render resolution. Defaults to 1024.
            sampling_mode: _description_. Defaults to "nearest". nearest for geom
            channels: _description_. Defaults to 3.
            device: _description_. Defaults to None.
        """
        self.channels = channels
        self.device = device or torch.device("cuda")
        self.lights = AmbientLights(ambient_color=((1.0, ) * channels, ), device=self.device)
        self.target_size = (texture_size, texture_size)
        self.render_size = render_size
        self.sampling_mode = sampling_mode

    # Load obj mesh, rescale the mesh to fit into the bounding box [-1, 1] when scale_factor=1.0
    def load_mesh(self, mesh_path, scale_factor=1.0, transformation=None, use_blender_coord=False, auto_center=True):
        mesh = load_objs_as_meshes([mesh_path], device=self.device)
        # -> in [-scale_factor, scale_factor] cube
        if transformation is not None:
            verts = mesh.verts_packed()
            transformation = torch.tensor(transformation, dtype=torch.float32, device=verts.device)
            
            # print(f"raw verts between: {torch.min(verts, dim=0)[0]} -- {torch.max(verts, dim=0)[0]}")
            
            # pw, [N, 4]
            raw_points = torch.cat([verts, torch.ones_like(verts[..., 0:1])], dim=-1)

            # pc = Tcw * pw then select [N, 3]
            new_points = torch.bmm(transformation.unsqueeze(0), raw_points.permute(
                1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)[..., :3]
            
            # transformed_vertices = (raw_points @ transformation.t())[:, :3]
            
            ####
            mesh = mesh.update_padded(new_points.unsqueeze(0))
            verts = mesh.verts_packed()
            # print(f"new verts between: {torch.min(verts, dim=0)[0]} -- {torch.max(verts, dim=0)[0]}")
                                
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

        if use_blender_coord:
            mesh = rotate_mesh_x_axis(mesh, 90)
            print('use_blender_coord=T, rot mesh x 90')
            
        self.mesh = mesh

        verts = self.mesh.verts_packed()
        max_bb = (verts - 0).max(0)[0]
        min_bb = (verts - 0).min(0)[0]
        
    # Save obj mesh
    def save_mesh(self, mesh_path, texture):
        os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
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
    
    def calcu_geom_and_cos(self):
        """set self.mesh_uv. calcute gradient and visible mask of each camera in uv space, 
                            render normal and cos weight and baking to uv space
        self.gradient_maps, self.visible_triangles and self.cos_maps, are list of [tex_size, tex_size, c=3/1/3]
        """
        if not hasattr(self, "mesh_d"):
            self.disconnect_faces()
        if not hasattr(self, "mesh_uv"):
            self.construct_uv_mesh_with_uvz()
        self.calculate_tex_gradient()
        self.calculate_visible_triangle_mask()
        _, _, _, cos_maps, _, _ = self.render_geometry()
        self.calculate_cos_angle_weights(cos_maps)

    def setup_renderer(self,
                       size=1024,
                       blur=0.0,
                       face_per_pix=1,
                       perspective_correct=False,
                       channels=None):
        if not channels:
            channels = self.channels

        self.raster_settings = RasterizationSettings(
            image_size=size,
            blur_radius=blur,
            faces_per_pixel=face_per_pix,
            perspective_correct=perspective_correct,
            cull_backfaces=True,
            max_faces_per_bin=30000,
        )
        
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras if self.cameras is not None else FoVOrthographicCameras(device=self.device),
                raster_settings=self.raster_settings,
            ),
            # set shader = HardGeometryShader when render geom
            shader=HardNChannelFlatShader(device=self.device,
                                          cameras=self.cameras,
                                          lights=self.lights,
                                          channels=channels
                                          # materials=materials
                                          )            
        )

    '''

	'''

    def disconnect_faces(self):
        """ A functions that disconnect faces in the mesh according to
            its UV seams. The number of vertices are made equal to the
            number of unique vertices its UV layout, while the faces list
            is intact. TODO(csz)
    
        """
        mesh = self.mesh
        # ---- Nuv and Fuv may not = N and F
        verts_list = mesh.verts_list()  # list of [N, 3], len list =1
        faces_list = mesh.faces_list()  # list of [F, 3], len list =1
        verts_uvs_list = mesh.textures.verts_uvs_list()  # uv coord in [0,1]. list of [Nuv, 2], len list =1
        faces_uvs_list = mesh.textures.faces_uvs_list()  # list of [Fuv, 3], len list =1  TODO(csz)
        packed_list = [v[f] for v, f in zip(verts_list, faces_list)]    # list of [F, 3, 3], len=1
        verts_disconnect_list = [
            torch.zeros((verts_uvs_list[i].shape[0], 3),
                        dtype=verts_list[0].dtype,
                        device=verts_list[0].device)
            for i in range(len(verts_list))
        ]   # [[Nuv, 3] ]
        for i in range(len(verts_list)):
            verts_disconnect_list[i][faces_uvs_list] = packed_list[i]
        assert not mesh.has_verts_normals(
        ), "Not implemented for vertex normals"
        self.mesh_d = Meshes(verts_disconnect_list, faces_uvs_list,
                             mesh.textures)
        return self.mesh_d
    
    def construct_uv_mesh_with_uvz(self):
        """make fake mesh with verts= [u, v, z] , u,v zoom form [0, 1] to [-1, 1] like nvdiffrast
        set self.mesh_uv
        """
        verts_list = self.mesh_d.verts_list()
        verts_uvs_list = self.mesh_d.textures.verts_uvs_list()
        
        new_verts_list = []
        for i, (verts, verts_uv) in enumerate(zip(verts_list, verts_uvs_list)):
            verts = verts.clone()
            verts_uv = verts_uv.clone()
            verts[..., 0:2] = verts_uv[..., :]
            verts = (verts - 0.5) * 2
            verts[..., 2] *= 1
            new_verts_list.append(verts)
        textures_uv = self.mesh_d.textures.clone()
        self.mesh_uv = Meshes(new_verts_list, self.mesh_d.faces_list(), textures_uv)
        return self.mesh_uv


    @torch.enable_grad()
    def calculate_tex_gradient(self, channels=None):
        """precalculate this gradient strength and use it to normalize gradients when we bake textures.
        Multiple screen pixels could pass gradient to a same texel
        set self.gradient_maps, list of [tex_size, tex_size, c], len=len(cameras)
            grad in tex space of each view
        """
        if not channels:
            channels = self.channels
        tmp_mesh = self.mesh.clone()
        gradient_maps = []
        for i in range(len(self.cameras)):
            zero_map = torch.zeros(self.target_size + (channels, ),
                                   device=self.device,
                                   requires_grad=True)
            optimizer = torch.optim.SGD([zero_map], lr=1, momentum=0)
            optimizer.zero_grad()
            zero_tex = TexturesUV([zero_map],
                                  self.mesh.textures.faces_uvs_padded(),
                                  self.mesh.textures.verts_uvs_padded(),
                                  sampling_mode=self.sampling_mode)
            tmp_mesh.textures = zero_tex
            images_predicted = self.renderer(tmp_mesh,
                                             cameras=self.cameras[i],
                                             lights=self.lights)
            loss = torch.sum((1 - images_predicted)**2)
            loss.backward()
            optimizer.step()

            gradient_maps.append(zero_map.detach())

        self.gradient_maps = gradient_maps


    # Get the UV space masks of triangles visible in each view
    # First get face ids from each view, then filter pixels on UV space to generate masks
    @torch.no_grad()
    def calculate_visible_triangle_mask(self,
                                        channels=None,
                                        image_size=(512, 512)):
        """set self.visible_triangles, list of [texture_size, texture_size, 1] in uv space of each view, bool
            对应每个视角在uv上的可见部分
        Args:
            channels: _description_. Defaults to None.
            image_size: _description_. Defaults to (512, 512).
        """
        if not channels:
            channels = self.channels

        ### rasterizer in cameras space, list of [1, image_size, image_size, 1] value is -1 or face id
        pix2face_list = []
        # TODO batch
        for i in range(len(self.cameras)):
            self.renderer.rasterizer.raster_settings.image_size = image_size
            pix2face = self.renderer.rasterizer(
                self.mesh_d, cameras=self.cameras[i]).pix_to_face
            self.renderer.rasterizer.raster_settings.image_size = self.render_size
            pix2face_list.append(pix2face)

        if not hasattr(self, "mesh_uv"):
            self.construct_uv_mesh_with_uvz()

        ### rasterizer in uv space
        raster_settings = RasterizationSettings(
            image_size=self.target_size,
            blur_radius=0,
            faces_per_pixel=1,
            perspective_correct=False,
            cull_backfaces=False,
            max_faces_per_bin=30000,
        )

        R, T = look_at_view_transform(dist=2, elev=0, azim=0) 
        cameras = FoVOrthographicCameras(device=self.device, R=R, T=T)

        rasterizer = MeshRasterizer(cameras=cameras,
                                    raster_settings=raster_settings)
        # [1, texture_size, texture_size, 1] value is -1 or face id
        uv_pix2face = rasterizer(self.mesh_uv).pix_to_face

        visible_triangles = []
        for i in range(len(pix2face_list)):
            valid_faceid = torch.unique(pix2face_list[i])
            valid_faceid = valid_faceid[1:] if valid_faceid[
                0] == -1 else valid_faceid
            # [texture_size, texture_size, 1] in uv space of each view, bool
            mask = torch.isin(uv_pix2face[0],
                              valid_faceid,
                              assume_unique=False)
            # uv_pix2face[0][~mask] = -1
            triangle_mask = torch.ones(self.target_size + (1, ),
                                       device=self.device)
            triangle_mask[~mask] = 0

            # dilate one pixel
            triangle_mask[:, 1:][triangle_mask[:, :-1] > 0] = 1
            triangle_mask[:, :-1][triangle_mask[:, 1:] > 0] = 1
            triangle_mask[1:, :][triangle_mask[:-1, :] > 0] = 1
            triangle_mask[:-1, :][triangle_mask[1:, :] > 0] = 1
            visible_triangles.append(triangle_mask)

        self.visible_triangles = visible_triangles


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


    # Bake screen-space cosine weights to UV space
    # May be able to reimplement using the generic "bake_texture" function, but it works so leave it here for now
    @torch.enable_grad()
    def calculate_cos_angle_weights(self,
                                    cos_angles,
                                    fill=True,
                                    channels=None):
        """set self.cos_maps, list of [texture_size, texture_size, channels]
        把camera space的cos_angles inverse render到UV space上，每个channel都是cos值
        Args:
            cos_angles: [nv, render_size, render_size, 2]
            fill: _description_. Defaults to True.
            channels: _description_. Defaults to None.
        """
        if not channels:
            channels = self.channels
        cos_maps = []
        tmp_mesh = self.mesh.clone()
        for i in range(len(self.cameras)):

            zero_map = torch.zeros(self.target_size + (channels, ),
                                   device=self.device,
                                   requires_grad=True)
            optimizer = torch.optim.SGD([zero_map], lr=1, momentum=0)
            optimizer.zero_grad()
            zero_tex = TexturesUV([zero_map],
                                  self.mesh.textures.faces_uvs_padded(),
                                  self.mesh.textures.verts_uvs_padded(),
                                  sampling_mode=self.sampling_mode)
            tmp_mesh.textures = zero_tex

            images_predicted = self.renderer(tmp_mesh,
                                             cameras=self.cameras[i],
                                             lights=self.lights)

            loss = torch.sum(
                (cos_angles[i, :, :, 0:1]**1 - images_predicted)**2)
            loss.backward()
            optimizer.step()

            if fill:
                zero_map = zero_map.detach() / (self.gradient_maps[i] + 1E-8)
                zero_map = voronoi_solve(zero_map, self.gradient_maps[i][...,
                                                                         0])
            else:
                zero_map = zero_map.detach() / (self.gradient_maps[i] + 1E-8)
            cos_maps.append(zero_map)
        self.cos_maps = cos_maps
    
    #####
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

    #### bake

    # Bake views into a texture
    # First bake into individual textures then combine based on cosine weight
    @torch.enable_grad()
    def bake_texture(self,
                     views=None,
                     main_views=[],
                     cos_weighted=True,
                     channels=None,
                     exp=None,
                     noisy=False,
                     generator=None):
        """_summary_

        Args:
            views: list of [c, h, w]
            main_views: _description_. Defaults to [].
            cos_weighted: _description_. Defaults to True.
            channels: _description_. Defaults to None.
            exp: _description_. Defaults to None.
            noisy: _description_. Defaults to False.
            generator: _description_. Defaults to None.

        Returns:
            _description_
        """
        if not exp:
            exp = 1
        if not channels:
            channels = self.channels
        views = [view.permute(1, 2, 0) for view in views]

        tmp_mesh = self.mesh
        bake_maps = [
            torch.zeros(self.target_size + (views[0].shape[2], ),
                        device=self.device,
                        requires_grad=True) for view in views
        ]
        optimizer = torch.optim.SGD(bake_maps, lr=1, momentum=0)
        optimizer.zero_grad()
        loss = 0
        for i in range(len(self.cameras)):
            bake_tex = TexturesUV([bake_maps[i]],
                                  tmp_mesh.textures.faces_uvs_padded(),
                                  tmp_mesh.textures.verts_uvs_padded(),
                                  sampling_mode=self.sampling_mode)
            tmp_mesh.textures = bake_tex
            images_predicted = self.renderer(tmp_mesh,
                                             cameras=self.cameras[i],
                                             lights=self.lights,
                                             device=self.device)
            predicted_rgb = images_predicted[..., :-1]
            loss += (((predicted_rgb[...] - views[i]))**2).sum()
        loss.backward(retain_graph=False)
        optimizer.step()

        total_weights = 0
        baked = 0
        for i in range(len(bake_maps)):
            normalized_baked_map = bake_maps[i].detach() / (
                self.gradient_maps[i] + 1E-8)
            bake_map = voronoi_solve(normalized_baked_map,
                                     self.gradient_maps[i][..., 0])
            weight = self.visible_triangles[i] * (self.cos_maps[i])**exp
            if noisy:
                noise = torch.rand(weight.shape[:-1] + (1, ),
                                   generator=generator).type(weight.dtype).to(
                                       weight.device)
                weight *= noise
            total_weights += weight
            baked += bake_map * weight
        baked /= total_weights + 1E-8
        baked = voronoi_solve(baked, total_weights[..., 0])

        bake_tex = TexturesUV([baked],
                              tmp_mesh.textures.faces_uvs_padded(),
                              tmp_mesh.textures.verts_uvs_padded(),
                              sampling_mode=self.sampling_mode)
        tmp_mesh.textures = bake_tex
        extended_mesh = tmp_mesh.extend(len(self.cameras))
        images_predicted = self.renderer(extended_mesh,
                                         cameras=self.cameras,
                                         lights=self.lights)
        learned_views = [image.permute(2, 0, 1) for image in images_predicted]

        return learned_views, baked.permute(2, 0,
                                            1), total_weights.permute(2, 0, 1)

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
    

    
def test_from_sdxl(in_mesh_path, image_all, out_dir, render_size=256, scale_factor=1.0, scale_xyz=1.0):
    from diffusers.utils import load_image, make_image_grid, numpy_to_pil
    
    def split_image(image, rows, cols):
        width, height = image.size
        block_width = width // cols
        block_height = height // rows

        images = []
        for i in range(rows):
            for j in range(cols):
                left = j * block_width
                upper = i * block_height
                right = (j + 1) * block_width
                lower = (i + 1) * block_height
                sub_image = image.crop((left, upper, right, lower))
                images.append(sub_image)

        return images
        
    os.makedirs(os.path.join(out_dir, "bake"), exist_ok=True)
    
    diff_render = DiffRender(render_size=render_size)
    diff_render.load_mesh(in_mesh_path, scale_factor=scale_factor)
    
    cameras = diff_render.make_ortho_cameras(scale_xyz=scale_xyz)

    diff_render.set_cameras_and_render_settings(cameras)
    
    diff_render.calcu_geom_and_cos()
    
    # log depth
    conditioning_images, masks = get_conditioning_images(diff_render, render_size)
    cond = (conditioning_images/2+0.5).permute(0,2,3,1).cpu().numpy()
    cond = np.concatenate([img for img in cond], axis=1)
    numpy_to_pil(cond)[0].save(f"{out_dir}/cond.jpg")
    print('save depth done')
    

    try_cnt = 3
    res = render_size
    image_list = split_image(image_all, try_cnt * 2, 2)
    
    for i in range(try_cnt):
        result_views = []
        vis_pils = []
        for j in range(4):
            index = i * 4 + j
            view = (torch.tensor(np.array(image_list[index]), device="cuda") / 255.0).permute(2, 0, 1)
            vis_pils.append(image_list[index])
            result_views.append(view)
        make_image_grid(vis_pils, 1, len(vis_pils)).save(f"{out_dir}/infer_image_{index}.png")
        
        textured_views_rgb, result_tex_rgb, visibility_weights = diff_render.bake_texture(
            views=result_views, main_views=[], exp=6, noisy=False)
        diff_render.save_mesh(f"{out_dir}/bake/textured_{i}.obj", result_tex_rgb.permute(1,2,0))
        
        # save visibility_weights
        # result_tex_rgb_output = result_tex_rgb.permute(1,2,0).cpu().numpy()
        # Image.fromarray(result_tex_rgb_output * 255. )
        # result_tex_rgb_output.save(f"debug/tex_{i}.png")
    return




# if __name__ == "__main__":

#     from PIL import Image
#     # data_dir = "/aigc_cfs_2/sz/data/mvc/chest_tripo"
#     data_dir = "/aigc_cfs_2/sz/data/mvc/top_3/uv_condition"
#     in_mesh_path=f"{data_dir}/mesh.obj"
#     # in_infer_views = "/aigc_cfs_2/sz/proj/ControlNet-v1-1-nightly/sample_control_0.8.png"
#     # in_infer_views = f"{data_dir}/sr/sample_control_0.8_out.png"
#     in_infer_views = f"{data_dir}/sr/sr.png"
#     render_size = 1024
#     scale_factor = 1.0
#     # out_dir = f"{data_dir}/bake_mvc"
#     out_dir = f"{data_dir}/bake_sdxl_{render_size}"
#     # in_infer_views = "/aigc_cfs_2/sz/proj/tex_cq/data/mario/backofmario4view.png"

#     image_all = Image.open(in_infer_views)
#     test_from_sdxl(in_mesh_path, image_all, out_dir, render_size=render_size, scale_factor=scale_factor)


def test_from_imv(in_mesh_path, image_pils, out_dir, render_size=256, scale_factor=1.0, scale_xyz=1.0):
    from diffusers.utils import load_image, make_image_grid, numpy_to_pil
    
    os.makedirs(out_dir, exist_ok=True)
    
    diff_render = DiffRender(render_size=render_size)
    diff_render.load_mesh(in_mesh_path, scale_factor=scale_factor)
    
    cameras = diff_render.make_ortho_cameras(scale_xyz=scale_xyz)

    diff_render.set_cameras_and_render_settings(cameras)
    
    diff_render.calcu_geom_and_cos()
    
    # log depth
    conditioning_images, masks = get_conditioning_images(diff_render, render_size)
    cond = (conditioning_images/2+0.5).permute(0,2,3,1).cpu().numpy()
    cond = np.concatenate([img for img in cond], axis=1)
    numpy_to_pil(cond)[0].save(f"{out_dir}/cond.jpg")
    print('save depth done')
    

    result_views = []
    for image_pil in image_pils:
        view = (torch.tensor(np.array(image_pil), device="cuda") / 255.0).permute(2, 0, 1)
        result_views.append(view)
    make_image_grid(image_pils, 1, len(image_pils)).save(f"{out_dir}/infer_image.png")
    
    textured_views_rgb, result_tex_rgb, visibility_weights = diff_render.bake_texture(
        views=result_views, main_views=[], exp=6, noisy=False)
    diff_render.save_mesh(f"{out_dir}/textured.obj", result_tex_rgb.permute(1,2,0))
    
    return

if __name__ == "__main__":

    from PIL import Image
    job_id = "c711a5ec-e9cb-45c2-bc44-05fa4cda47cb_z123_crman"
    in_img_dir = f"/aigc_cfs_gdp/sz/batch_0828/z123_crman/z123_crman/imc/{job_id}/try_0/save_c0.8"
    in_mesh_path=f"/aigc_cfs_gdp/sz/batch_0828/z123_crman/z123_crman/{job_id}/verts2tex/baking/tex_mesh.obj"
   
    # in_img_dir = "/aigc_cfs_gdp/sz/batch_0816/compare_z123_lrm_human_ratio/z123_ratio2/1d721fda-ca16-4e86-b103-fa7af5457ee7_z123_ratio2/verts2tex/baking/try_0/save_as90_c0.8"
    # in_mesh_path=f"/aigc_cfs_gdp/sz/batch_0816/compare_z123_lrm_human_ratio/z123_ratio2/1d721fda-ca16-4e86-b103-fa7af5457ee7_z123_ratio2/verts2tex/baking/tex_mesh.obj"
   
    render_size = 1024
    scale_factor = 1.0
    scale_xyz = 0.9
    
    out_dir = f"{os.path.dirname(in_img_dir)}/bake_sync_{render_size}"
    # in_infer_views = "/aigc_cfs_2/sz/proj/tex_cq/data/mario/backofmario4view.png"

    img_paths = [os.path.join(in_img_dir, f"for_sync_{i}.png") for i in range(4)]
    image_pils = [Image.open(img) for img in img_paths]
    test_from_imv(in_mesh_path, image_pils, out_dir, render_size=render_size, 
                   scale_factor=scale_factor, scale_xyz=0.9)
