import os
import argparse
import torch
import numpy as np
import nvdiffrast.torch as dr
from PIL import Image
import time
import torch.nn.functional as F
import sys

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "render"))

from render.mesh import load_mesh, Mesh, auto_center
from render.render_mesh import resize_render, transform_pos, auto_normals
from render.util import make_4views_mvp_tensor, mse_to_psnr, safe_normalize
from render.obj import write_obj
import render.texture as texture
from src.utils_render import save_bhwc_tensor, util_print_times, check_mesh_uv, make_image_grid


class NvdiffRender():

    def __init__(self, render_res=2048, tex_res=1024) -> None:
        self.device = "cuda"
        self.glctx = dr.RasterizeCudaContext()
        self.render_res = render_res
        self.tex_res = tex_res

    def prepare_mesh_geom_and_pose(self, in_obj, cam_type="ortho", scale_factor=1.0, zoom=0.9,
                                   mvp=None, camera_centers=None,
                                   out_dir=None):
        t_list = []
        t_list.append((time.time(), "start"))
        raw_mesh: Mesh = load_mesh(in_obj)
        self.raw_mesh = auto_center(raw_mesh, scale_factor=scale_factor)
        self.raw_mesh = auto_normals(self.raw_mesh)
        vtx_pos, pos_idx = self.raw_mesh.v_pos, self.raw_mesh.t_pos_idx  # [Nv, 3],  [Nf, 3]
        vtx_uv, uv_idx = self.raw_mesh.v_tex, self.raw_mesh.t_tex_idx  # [Nv, 2] [Nf, 3]
        vtx_normal, normal_idx = self.raw_mesh.v_nrm, self.raw_mesh.t_nrm_idx  # [Nv, 3], [Nf, 3]
        print('load mesh done')
        t_list.append((time.time(), "load_mesh"))

        # [b, 4, 4], [b, 1, 3]
        if mvp is not None and camera_centers is not None:
            self.mvp = mvp
            self.camera_centers = camera_centers
        else:
            self.mvp, self.camera_centers = make_4views_mvp_tensor(cam_type=cam_type, zoom=zoom, device=self.device)
        print('generate mvp done', self.camera_centers.shape, self.camera_centers)
        t_list.append((time.time(), "generate mvp"))

        ## 1. rast
        # pos_clip [b, Nv, 4]
        # rast_out [b, res, res, 4]
        self.pos_clip = transform_pos(vtx_pos, self.mvp)
        self.rast_out, self.rast_out_db = dr.rasterize(self.glctx,
                                                       self.pos_clip,
                                                       pos_idx.int(),
                                                       resolution=[self.render_res, self.render_res])

        ## 2. interpolate, fill uv of res^2.
        # self.texc: uv coords [b, res, res, 2] range:[0, 1] or texture will auto mod to [0, 1]
        self.texc, self.texd = dr.interpolate(vtx_uv[None, ...],
                                              self.rast_out,
                                              uv_idx.int(),
                                              rast_db=self.rast_out_db,
                                              diff_attrs='all')
        t_list.append((time.time(), "rast"))

        # ## 3.1 render geom, depth..

        ## 3.2 render geom, cos..
        # [nv, 1, 3] - [1, Nv, 3]
        view_dir = self.camera_centers - vtx_pos[None, ...]
        view_dir = safe_normalize(view_dir)  # [nv, Nv, 3]

        cosine = torch.sum(view_dir * vtx_normal[None, ...], dim=-1, keepdim=True)
        cosine = cosine.abs()  # [nv, Nv, 1]

        # [nv, render_res, render_res, 1] viewcos and geom_mask
        self.viewcos, _ = dr.interpolate(cosine.contiguous(), self.rast_out, pos_idx.int())
        self.geom_mask = torch.clamp(self.rast_out[..., -1:], 0, 1)
        self.viewcos = self.viewcos * self.geom_mask

        t_list.append((time.time(), "cos"))
        if out_dir is not None:
            # debug TODO maybe have problem
            cos_exps = [1, 3, 6, 12]
            for cos_exp in cos_exps:
                save_bhwc_tensor((self.viewcos**cos_exp), os.path.join(out_dir, f"cos_{cos_exp}.jpg"))
            print('self.viewcos ', self.viewcos.shape, self.viewcos.min(), self.viewcos.max())

        util_print_times(t_list, prename="prepare")
        return

    def bake_views(self,
                   in_images,
                   out_dir=None,
                   main_views=[],
                   main_weight=5,
                   cos_exp=-1,
                   hard_masks=None,
                   max_mip_level=4,
                   save_debug=False):
        """bake multi-views to texture

        Args:
            in_images: [b, h, w, 3] tensor in [0, 1]
            out_dir: _description_. Defaults to None.
            main_views: high weight views
            cos_exp: use cos weight blend when cos_exp>0
            hard_masks: seg mask, [b, h, w, 1] tensor in [0, 1]
            max_mip_level: _description_. Defaults to 4.

        Returns:
            uv_tex [h,w,3] tensor in [0, 1]
        """
        optim_cfg = {
            "lr_base": 0.05,
            "lr_ramp": 0.1,
            "max_iter": 100,
            "vis_iter": -1,
            "max_mip_level": max_mip_level,
            "mask_sp": 4,
            "min_cos": 0.25,  # normal-view cos
        }
        t_list = []
        t_list.append((time.time(), "start"))
        if in_images.shape[1] != self.render_res:
            in_images = resize_render(in_images, self.render_res, self.render_res)

        tex_data_opt = self.init_opt_tex(self.tex_res, background='black')
        optimizer = torch.optim.Adam([tex_data_opt], lr=optim_cfg['lr_base'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda x: optim_cfg['lr_ramp']**(float(x) / float(optim_cfg['max_iter'])))
        t_list.append((time.time(), "init_opt_tex"))

        in_mesh = self.raw_mesh

        ts = time.time()
        for iter in range(optim_cfg['max_iter']):
            images_iter = in_images

            # [b, render_res, render_res, 3/1]
            color_opt, alpha_opt = self.sample_texture(tex_data_opt, max_mip_level=optim_cfg['max_mip_level'])

            weight = torch.ones_like(color_opt)  # before main views
            if main_views:
                weight[main_views, ...] *= main_weight
            if cos_exp > 0:
                weight = weight * (self.viewcos**cos_exp)
            # if ignore_views:
            #     weight[ignore_views, ...] *= ignore_weight
            if hard_masks is not None:
                alpha_opt = alpha_opt * hard_masks
                
            loss = torch.mean(weight * ((images_iter * alpha_opt - color_opt * alpha_opt)**2))  # L2 pixel loss.
            if out_dir is not None and optim_cfg['vis_iter'] > 0 and (iter % optim_cfg['vis_iter'] == 0
                                                                      or iter == (optim_cfg['max_iter'] - 1)):
                print(f'iter={iter} loss ', loss)
                save_bhwc_tensor(color_opt * alpha_opt, os.path.join(out_dir, f"vis_{iter}.jpg"))

            optimizer.zero_grad()
            # loss.backward()
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
        uv_tex = (tex_data_opt.data).clone().squeeze(0)  # [h,w,3]
        t_list.append((time.time(), "diff_render"))
        tuse = time.time() - ts
        psnr = mse_to_psnr(loss.item())
        print(f"psnr={psnr}. tuse={tuse}, avg = {tuse/optim_cfg['max_iter']}")

        if out_dir is not None and save_debug:
            new_mesh = in_mesh  #.clone()
            new_mesh.material = in_mesh.material
            new_mesh.material['kd'] = texture.Texture2D(uv_tex)
            write_obj(os.path.join(out_dir, "direct_bake"), new_mesh)
            t_list.append((time.time(), "write_obj"))
            save_bhwc_tensor(in_images, os.path.join(out_dir, "resized_infer.jpg"))

        del tex_data_opt
        del optimizer
        del scheduler
        util_print_times(t_list, "bake")
        return uv_tex

    def init_opt_tex(self, tex_res, channel=3, background='gray'):
        device = self.device
        if background == 'gray':
            tex = torch.ones((1, tex_res, tex_res, channel)).to(device) * 0.5
        elif background == 'black':
            tex = torch.zeros((1, tex_res, tex_res, channel)).to(device)
        elif background == 'white':
            tex = torch.ones((1, tex_res, tex_res, channel)).to(device)
        else:
            print('invalid init_opt_tex background ', background)
            return None
        tex_merge = torch.nn.Parameter(tex.clone(), requires_grad=True)
        return tex_merge

    def render_views(self, out_dir, max_mip_level=4):
        in_mesh = self.raw_mesh
        tex = in_mesh.material  # class Material
        tex_data = tex['kd'].data

        ts = time.time()
        color, alpha = self.sample_texture(tex_data)
        tuse = time.time() - ts
        print(f'sample_texture tuse = {tuse}')
        save_bhwc_tensor(color * alpha, os.path.join(out_dir, f"render_split.png"))

        return

    def inv_proj_masks_sum_uv(self, view_masks, binary_mode=False, out_path=None):
        """inverse project each view mask to uv mask and sum.

        Args:
            view_masks: [nv, res, res, 1]
            binary_mode: optim alpha if binary_mode, else optim mask value
        Return 
            sum_uv_mask: [1, res, res, 1]
        """
        ts = time.time()
        rast_res = self.texc.shape[1]
        if rast_res != view_masks.shape[1]:
            print(f'warn rast_res={rast_res} != view_masks.shape[1]={view_masks.shape[1]}. resize now')
            view_masks = resize_render(view_masks, rast_res, rast_res)
        target = (view_masks > 0).float() if binary_mode else view_masks

        max_iter = 100
        max_mip_level = 1  # TODO

        each_uv_masks = []
        for idx, view_mask in enumerate(view_masks):
            tex_data_opt = self.init_opt_tex(self.tex_res, channel=1, background='black')
            optimizer = torch.optim.Adam([tex_data_opt], lr=1)
            for iter in range(max_iter):
                # [b, render_res, render_res, 1/1]
                if max_mip_level and max_mip_level > 0:
                    color = dr.texture(tex_data_opt,
                                       self.texc[idx].unsqueeze(0),
                                       self.texd[idx].unsqueeze(0),
                                       filter_mode='auto',
                                       max_mip_level=max_mip_level)
                    # color = dr.texture(tex_data, texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
                else:
                    color = dr.texture(tex_data_opt, self.texc[idx].unsqueeze(0), filter_mode='linear')
                color = dr.antialias(color, self.rast_out[idx].unsqueeze(0), self.pos_clip[idx].unsqueeze(0),
                                     self.raw_mesh.t_pos_idx.int())
                alpha_opt = torch.clamp(self.rast_out[..., -1:][idx].unsqueeze(0), 0, 1)
                color_opt = color * alpha_opt  # Mask out background

                loss = torch.mean(((target[idx].unsqueeze(0) - color_opt * alpha_opt)**2))  # L2 pixel loss.
                # if out_path is not None:
                #     print(f'iter={iter} loss ', loss)
                #     # save_bhwc_tensor(color_opt, os.path.splitext(out_path)[0] + f"_render_{idx}_{iter}.jpg")

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            psnr = mse_to_psnr(loss.item())
            print(f"mask psnr={psnr}.")
            each_uv_mask = (tex_data_opt.data).clone()  # [1, h,w, 1]
            each_uv_masks.append(each_uv_mask)

            del tex_data_opt
            del optimizer

        sum_uv_mask = torch.sum(torch.stack(each_uv_masks, dim=0), dim=0)

        tuse = time.time() - ts
        print(f'inv_proj_masks tuse={tuse}')
        if out_path is not None:
            save_bhwc_tensor(sum_uv_mask, out_path)
            save_bhwc_tensor(sum_uv_mask / view_masks.shape[0], os.path.splitext(out_path)[0] + f"_avg.jpg")
            for i, each_uv_mask in enumerate(each_uv_masks):
                save_bhwc_tensor(each_uv_mask, os.path.splitext(out_path)[0] + f"_split_{i}.jpg")

        return sum_uv_mask

    def sample_texture(self, tex_data, max_mip_level=4):
        """_summary_

        Args:
            tex_data: [1, 1024, 1024, c] tensor in [0, 1]
            max_mip_level: _description_. Defaults to 4.

        Returns:
            [b, res, res, c] tensor in [0, 1]
        """
        ## 3. sample texture
        if max_mip_level and max_mip_level > 0:
            color = dr.texture(tex_data, self.texc, self.texd, filter_mode='auto', max_mip_level=max_mip_level)
            # color = dr.texture(tex_data, texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
        else:
            color = dr.texture(tex_data, self.texc, filter_mode='linear')
        color = dr.antialias(color, self.rast_out, self.pos_clip, self.raw_mesh.t_pos_idx.int())
        alpha = torch.clamp(self.rast_out[..., -1:], 0, 1)
        color = color * alpha  # Mask out background

        return color, alpha

    def bake_mesh_4views(self, in_mesh_path, image_nps, hard_masks=None, mesh_process="rot2",
                         scale_factor=1.0, scale_xyz=0.9, max_theta = 77, 
                         max_mip_level=4, out_dir=None,
                         save_debug=False):
        """direct bake 4 views with cos-weight.

        Args:
            in_mesh_path: obj path
            image_nps: [4, 1024, 1024, 3] np in [0, 1]
            mesh_process: rot2, / keep_raw / ..
            scale_factor: _description_. Defaults to 1.0.
            scale_xyz: _description_. Defaults to 0.9.
            max_theta: inpaint when cos > max_theta. Defaults to 75.
            out_dir: if not None, save obj and mid results. Defaults to None.
            save_debug

        Returns:
            mesh_path: processed mesh path
            see_mask: [h,w] numpy bool
            tex_hwc_np: [h, w, 3] numpy in [0, 1]
        """
        assert os.path.exists(in_mesh_path), f"can not find in_mesh_path {in_mesh_path}"
        t_list = []
        t_list.append((time.time(), "start"))

        mesh_path = check_mesh_uv(in_mesh_path)    # ~4.5s
        t_list.append((time.time(), "check_mesh_uv"))     
                
        ## pre-calculate geom
        self.prepare_mesh_geom_and_pose(mesh_path, scale_factor=scale_factor, 
                                        zoom=scale_xyz, out_dir=out_dir if save_debug else None)
        t_list.append((time.time(), "prepare_mesh_geom_and_pose"))     
        
        ## seg TODO set hard_masks
        in_images = torch.tensor(image_nps, dtype=torch.float32, device=self.device)

        ## run bake
        cos_exp = 3
        uv_tex = self.bake_views(in_images,
                            out_dir,
                            max_mip_level=max_mip_level,
                            main_views=[0],
                            cos_exp=cos_exp,
                            hard_masks=hard_masks,
                            save_debug=save_debug)
        tex_hwc_np = uv_tex.cpu().numpy()   # [h,w,3] in [0,1]
        t_list.append((time.time(), "bake_views"))     

        ## see mask
        cos_sum_uv_mask = self.inv_proj_masks_sum_uv(
            self.viewcos * hard_masks if hard_masks is not None else self.viewcos,  # todo with hard mask
            binary_mode=False,
            out_path=os.path.join(out_dir, "uv_viewcos.jpg") if save_debug else None)
        min_weight = (np.cos(max_theta * np.pi / 180.))
        see_mask = (cos_sum_uv_mask > min_weight).squeeze(0).squeeze(-1).cpu().numpy()  #[h,w]
        t_list.append((time.time(), "see_mask"))     
        
        if out_dir is not None:
            ts = time.time()
            image_pils = [Image.fromarray((image_np * 255.0).round().astype("uint8")).resize((512, 512)) for image_np in image_nps]
            make_image_grid(image_pils, 1, len(image_pils)).save(f"{out_dir}/infer_image.png")            
            image_pils[0].save(f"{out_dir}/first_view.png")     # for sdxl refine
            
            self.render_views(out_dir, max_mip_level=max_mip_level)
            
        util_print_times(t_list, "bake_mesh")
        return mesh_path, see_mask, tex_hwc_np
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='render obj with setting pose')
    parser.add_argument(
        '--in_obj',
        type=str,
        default="/aigc_cfs_gdp/sz/result/pipe_test/82941b42-a516-4fbc-b3ed-9bcc94754bbf/texbakeinpaint/mesh.obj")
    parser.add_argument(
        '--in_img_npy',
        type=str,
        default="/aigc_cfs_gdp/sz/result/pipe_test/82941b42-a516-4fbc-b3ed-9bcc94754bbf/d2rgb/out/imgsr/color.npy")
    parser.add_argument(
        '--out_dir',
        type=str,
        default="/aigc_cfs_gdp/sz/result/pipe_test/82941b42-a516-4fbc-b3ed-9bcc94754bbf/texbakeinpaint/nv_r_cos3")
    parser.add_argument('--render_res', type=int, default=2048)
    parser.add_argument('--tex_res', type=int, default=1024)
    parser.add_argument('--max_mip_level', type=int, default=4)
    args = parser.parse_args()

    in_obj = args.in_obj
    in_img_npy = args.in_img_npy
    out_dir = args.out_dir
    render_res = args.render_res
    tex_res = args.tex_res
    max_mip_level = args.max_mip_level

    save_debug = True
    # run once
    nv_render = NvdiffRender(render_res=render_res, tex_res=tex_res)
    nv_render.prepare_mesh_geom_and_pose(in_obj, out_dir=out_dir if save_debug else None)

    raw_image_nps = np.load(in_img_npy)  # [b, 3, h, w]
    image_nps = raw_image_nps.transpose(0, 2, 3, 1) / 255.0  # [b, h, w, 3] in [0, 1]
    in_images = torch.tensor(image_nps, dtype=torch.float32, device=nv_render.device)

    # run multi
    cos_exp = 3
    nv_render.bake_views(in_images,
                         out_dir,
                         max_mip_level=max_mip_level,
                         main_views=[0],
                         cos_exp=cos_exp,
                         save_debug=save_debug)

    # vis
    nv_render.render_views(out_dir, max_mip_level=max_mip_level)
    nv_render.inv_proj_masks_sum_uv(nv_render.geom_mask,
                                    binary_mode=True,
                                    out_path=os.path.join(out_dir, "uv_see_all.jpg") if save_debug else None)
    nv_render.inv_proj_masks_sum_uv(
        nv_render.viewcos,  # todo with hard mask
        binary_mode=False,
        out_path=os.path.join(out_dir, "uv_viewcos.jpg") if save_debug else None)
