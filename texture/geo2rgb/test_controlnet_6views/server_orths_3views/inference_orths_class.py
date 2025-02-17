import torch
import requests
from PIL import Image
import random
import numpy as np
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, ControlNetModel
from pipeline_3views_online.pipeline import Zero123PlusPipeline
import os
from pdb import set_trace as st

from pathlib import Path

import sys
# sys.path.append(os.path.abspath("/aigc_cfs/xibinsong/code/zero123plus_control/zero123plus/utils_use"))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath("/aigc_cfs_gdp/xibin/z123_control/code/z123plus_controlnet_gdp/utils_use"))

from utils_parse_dataset import pose_generation

from geom_renderer import make_pers_cameras, make_ortho_cameras,  get_geom_texture
from utils_render import concatenate_images_horizontally, load_images, save_rgba_geom_images, save_rgba_depth_images, save_rgba_normals_images

import pytorch3d
from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj, IO

from pytorch3d.transforms import RotateAxisAngle, Transform3d, matrix_to_euler_angles, euler_angles_to_matrix
from pytorch3d.structures import Meshes

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform, FoVPerspectiveCameras, FoVOrthographicCameras, AmbientLights,
                                PointLights, DirectionalLights, Materials, RasterizationSettings, MeshRenderer,
                                MeshRasterizer, TexturesUV)

from utils_use.renderer import DiffRender

from torchvision.utils import make_grid, save_image
from torchvision import transforms
from PIL import Image

import torch
import numpy as np
import kiui
# import nvdiffrast.torch as dr
import torch.nn.functional as F
from kiui.mesh import Mesh
import cv2

import time

from utils_use.utils_seg_rmbg import RMBG, repadding_rgba_image

from sam_preprocess.run_sam import process_image_path

def img_seg(img_path, seg_model, img_size=512):
    # device = torch.device(f"cuda:{'0'}" if torch.cuda.is_available() else "cpu")
    # rmbg = RMBG(seg_model, device)
    # img_path = "/aigc_cfs_gdp/sz/result/pipe_test/639c3cee-085f-4e2d-acfa-fce1150d36b6/mesh2image.png"

    rmbg = seg_model

    img = Image.open(img_path)

    img = rmbg.run_and_resize("rembg", img, 0.8, 'Remove', (255, 255, 255, 255), )
    # mvimg_0 = rmbg.run_and_resize("rembg", mvimg_0, 0.9, 'Remove', (127, 127, 127, 255), )

    img = repadding_rgba_image(img, rescale=True, ratio=0.9, bg_color=255)
    img = Image.fromarray(img)

    background = Image.new("RGBA", img.size, (127, 127, 127, 255))
    img = Image.alpha_composite(background, img).convert("RGB")
    img = img.resize((img_size, img_size))
    img.save("test_gradio_seg.png")
    return img

# def render_obj_depth_normal(obj_path, out_dir, render_size=512, use_blender_coord=True, use_ortho = False, obj_gen_type="lrm"):
def render_obj_depth_normal(obj_path, out_dir, render_size=512, use_blender_coord=True, use_ortho = False):
    # print("running here !!")
    # breakpoint()
    # Mesh, Transformation = meta_dict["Mesh"], meta_dict["Transformation"]

    Mesh = obj_path

    assert os.path.exists(Mesh), f"can not find mesh = {Mesh}"
    # assert os.path.exists(Transformation), f"can not find Transformation = {Transformation}"
    
    if use_ortho:
        # 4 pose
        azimuth_list = [0, 90, 180, 270]
        elevation_list = [0] * len(azimuth_list)
        dist_list = [3.0] * len(azimuth_list)
        print("ortho!")
    else:
        # 4 pose
        azimuth_list = [0, 90, 180, 270]
        elevation_list = [0, 0, 0, 0]
        fov_list = [10] * len(azimuth_list)
        dist_list = None
    
    # breakpoint()

    # transformation = np.loadtxt(Transformation)

    # print("render !")
    # breakpoint()
    diff_render = DiffRender(render_size=render_size)
    # diff_render.load_mesh(Mesh, transformation=transformation, use_blender_coord=use_blender_coord)
    # diff_render.load_mesh(Mesh, use_blender_coord=use_blender_coord, obj_gen_type=obj_gen_type)
    # print("render !22222")
    # breakpoint()
    diff_render.load_mesh(Mesh, use_blender_coord=use_blender_coord)
    # tex_temp = torch.zeros((1024, 1024, 3), device="cuda")
    # diff_render.save_mesh("test.obj", tex_temp)

    if not use_ortho:
        # parse dist with image_percentage
        _, _, dist_list = pose_generation(azimuth_list, elevation_list, fov_list, image_size=render_size)
        print('dist_list ', dist_list)
        cameras = make_pers_cameras(dist_list,
                                    elevation_list,
                                    azimuth_list,
                                    fov_list[0],
                                    use_blender_coord=use_blender_coord,
                                    device=diff_render.device)
    else:
        cameras = make_ortho_cameras(dist_list,
                                    elevation_list,
                                    azimuth_list,
                                    scale_xyz=0.9,
                                    use_blender_coord=use_blender_coord,
                                    device=diff_render.device)
    diff_render.set_cameras_and_render_settings(cameras, render_size=render_size)
    diff_render.calcu_geom_and_cos()

    verts, normals, depths, cos_angles, texels, fragments = diff_render.render_geometry(render_size)
    # print("normals: ", normals.shape)
    save_rgba_geom_images(verts, os.path.join(out_dir, "position.png"))
    save_rgba_normals_images(normals, os.path.join(out_dir, "normal_origin.png"))
    # save_rgba_depth_images(depths, os.path.join(out_dir, "depth.png"))
    # breakpoint()

    # mask = normals[:,:,:,3]
    mask_xyz = verts[:,:,:,3]

    # normals = (normals + 1) / 2.0
    verts = (verts + 1) / 2.0

    # normals = (normals * 125) / 255.0 * 1.1
    # normals = normals * 255.0
    
    # pils = []
    # for i in range(rgba_normalized.shape[0]):
    #     img = rgba_normalized[i]

    #     img = (img * 255).byte()

    #     img_pil = Image.fromarray(img.cpu().numpy(), 'RGBA')

    #     pils.append(img_pil)

    # print("mask: ", mask.shape)
    # normal_imgs = normals[:,:,:,:3]
    xyz_imgs = verts[:,:,:,:3]
    # print("max: ", torch.max(xyz_imgs))
    # print("min: ", torch.min(xyz_imgs))
    # print("normal: ", normal_imgs.shape)

    # normal_imgs = ((normal_imgs + 1)/2.0 + 125) / 255.0
    # normal_imgs[:,:,:,0][mask==0] = 0 
    # normal_imgs[:,:,:,1][mask==0] = 0 
    # normal_imgs[:,:,:,2][mask==0] = 0
    # normal_imgs = normal_imgs.permute(0, 3, 1, 2) 

    xyz_imgs[:,:,:,0][mask_xyz==0] = 127/255.0
    xyz_imgs[:,:,:,1][mask_xyz==0] = 127/255.0
    xyz_imgs[:,:,:,2][mask_xyz==0] = 127/255.0
    xyz_imgs = xyz_imgs.permute(0, 3, 1, 2)
    # xyz_imgs = F.interpolate(xyz_imgs, size=(512, 512), mode='bilinear', align_corners=False)

    # normal_imgs = normal_imgs.permute(0, 3, 1, 2)

    # print("max: ", torch.max(normal_imgs))
    # print("min: ", torch.min(normal_imgs))
    # print(normal_imgs.shape)
    xyz_grid = make_grid(xyz_imgs, nrow=2, padding=0)

    # images_out = make_grid(images_out[0], nrow=2, padding=0) * 0.5 + 0.5
    # save_image(normal_grid, os.path.join(out_dir, "xyz_grid.png"))

    # breakpoint()

    # return normal_imgs, xyz_imgs
    # return normal_imgs, xyz_grid
    return xyz_grid

def render_obj_depth_normal_black(obj_path, out_dir, render_size=512, use_blender_coord=True, use_ortho = False):

    Mesh = obj_path

    assert os.path.exists(Mesh), f"can not find mesh = {Mesh}"
    # assert os.path.exists(Transformation), f"can not find Transformation = {Transformation}"
    
    if use_ortho:
        # 4 pose
        azimuth_list = [0, 90, 180, 270]
        elevation_list = [0] * len(azimuth_list)
        dist_list = [3.0] * len(azimuth_list)
        print("ortho!")
    else:
        # 4 pose
        azimuth_list = [0, 90, 180, 270]
        elevation_list = [0, 0, 0, 0]
        fov_list = [10] * len(azimuth_list)
        dist_list = None

    diff_render = DiffRender(render_size=render_size)

    diff_render.load_mesh(Mesh, use_blender_coord=use_blender_coord)

    if not use_ortho:
        # parse dist with image_percentage
        _, _, dist_list = pose_generation(azimuth_list, elevation_list, fov_list, image_size=render_size)
        print('dist_list ', dist_list)
        cameras = make_pers_cameras(dist_list,
                                    elevation_list,
                                    azimuth_list,
                                    fov_list[0],
                                    use_blender_coord=use_blender_coord,
                                    device=diff_render.device)
    else:
        cameras = make_ortho_cameras(dist_list,
                                    elevation_list,
                                    azimuth_list,
                                    scale_xyz=0.9,
                                    use_blender_coord=use_blender_coord,
                                    device=diff_render.device)
    diff_render.set_cameras_and_render_settings(cameras, render_size=render_size)
    diff_render.calcu_geom_and_cos()

    verts, normals, depths, cos_angles, texels, fragments = diff_render.render_geometry(render_size)
    # print("normals: ", normals.shape)
    save_rgba_geom_images(verts, os.path.join(out_dir, "position.png"))
    save_rgba_normals_images(normals, os.path.join(out_dir, "normal_origin.png"))
    # save_rgba_depth_images(depths, os.path.join(out_dir, "depth.png"))
    # breakpoint()

    mask = normals[:,:,:,3]
    mask_xyz = verts[:,:,:,3]

    normals = (normals + 1) / 2.0
    verts = (verts + 1) / 2.0

    # print("mask: ", mask.shape)
    normal_imgs = normals[:,:,:,:3]
    xyz_imgs = verts[:,:,:,:3]
    # print("normal: ", normal_imgs.shape)

    # normal_imgs = ((normal_imgs + 1)/2.0 + 125) / 255.0
    normal_imgs[:,:,:,0][mask==0] = 0 
    normal_imgs[:,:,:,1][mask==0] = 0 
    normal_imgs[:,:,:,2][mask==0] = 0
    normal_imgs = normal_imgs.permute(0, 3, 1, 2)

    xyz_imgs = xyz_imgs * 255.0
    xyz_imgs = xyz_imgs.cpu().numpy().clip(0, 255.0).astype(np.uint8)
    xyz_imgs = torch.from_numpy(xyz_imgs).cuda()

    xyz_imgs_black = xyz_imgs.clone()

    xyz_imgs[:,:,:,0][mask_xyz==0] = 127
    xyz_imgs[:,:,:,1][mask_xyz==0] = 127
    xyz_imgs[:,:,:,2][mask_xyz==0] = 127

    xyz_imgs_black[:,:,:,0][mask_xyz==0] = 0
    xyz_imgs_black[:,:,:,1][mask_xyz==0] = 0
    xyz_imgs_black[:,:,:,2][mask_xyz==0] = 0

    xyz_imgs = xyz_imgs/255.0
    xyz_imgs_black = xyz_imgs_black/255.0

    xyz_imgs = xyz_imgs.permute(0, 3, 1, 2)
    xyz_imgs_black = xyz_imgs_black.permute(0, 3, 1, 2)
    xyz_imgs = F.interpolate(xyz_imgs, size=(512, 512), mode='bilinear', align_corners=False)
    xyz_imgs_black = F.interpolate(xyz_imgs_black, size=(512, 512), mode='bilinear', align_corners=False)

    xyz_grid = make_grid(xyz_imgs, nrow=2, padding=0)
    xyz_grid_black = make_grid(xyz_imgs_black, nrow=2, padding=0)

    # images_out = make_grid(xyz_grid[0], nrow=2, padding=0) * 0.5 + 0.5
    save_image(xyz_grid, os.path.join(out_dir, "xyz_grid.png"))
    save_image(xyz_grid_black, os.path.join(out_dir, "xyz_grid_black.png"))
    # print(os.path.join(out_dir, "xyz_grid.png"))
    # breakpoint()

    # return normal_imgs, xyz_imgs
    return normal_imgs, xyz_grid, xyz_grid_black

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def post_process_image(image, depth):
    image = np.array(image)
    depth = np.array(depth)

    height, width, channels = image.shape
    res_image = np.zeros((height, width, channels), dtype=np.uint8)

    b, g, r = cv2.split(image)
    mask_b, mask_g, mask_r = cv2.split(depth)

    mask_origin = mask_b
    mask_origin[mask_origin > 0] = 255

    mask_use = mask_b
    mask_use[mask_use > 0] = 255

    # 创建一个简单的结构元素
    kernel = np.ones((11, 11), np.uint8)

    eroded_mask = cv2.erode(mask_use, kernel, iterations=1)

    b[eroded_mask == 0] = 255
    g[eroded_mask == 0] = 255
    r[eroded_mask == 0] = 255
    eroded_image = cv2.merge([b, g, r])

    cv2.imwrite("./tmp/eroded_mask.png", eroded_mask)
    cv2.imwrite("./tmp/image_eroded.png", eroded_image)

    # 创建一个简单的结构元素
    kernel = np.ones((24, 24), np.uint8)

    # 对掩码进行膨胀操作
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)

    # 确保膨胀后的区域包括掩码区域
    # expanded_mask = cv2.bitwise_or(eroded_mask, dilated_mask)

    new_expansion = cv2.subtract(dilated_mask, eroded_mask)

    # cv2.imwrite("./tmp/image_expansion.png", new_expansion)

    # 创建一个图像用于显示结果
    result_image = image.copy()

    # 使用 inpainting 方法填充扩展区域
    result_image = cv2.inpaint(result_image, new_expansion, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    result_image[:,:,0][mask_origin == 0] = 255
    result_image[:,:,1][mask_origin == 0] = 255
    result_image[:,:,2][mask_origin == 0] = 255
    # cv2.imwrite("./tmp/image_dilated.png", result_image)

    return result_image

def center_crop(crop_width, crop_height):
    # img = Image.open(image_path)
    img_width, img_height = img.size

    left = (img_width - crop_width) / 2
    top = (img_height - crop_height) / 2
    right = (img_width + crop_width) / 2
    bottom = (img_height + crop_height) / 2

    img_cropped = img.crop((left, top, right, bottom))
    return img_cropped

def run_xyz2rgb(
    seg_model,
    controlnet,
    obj_path,  
    img_path,
    out_dir,
    vis_dir,
    in_data_type, 
    # obj_gen_type,
    seed=0, 
    cfg=3.0):

    set_seed(42)

    start_time = time.perf_counter()

    # pipeline = DiffusionPipeline.from_pretrained(
    #     # "/aigc_cfs_2/neoshang/code/diffusers_triplane/release/zero23plus_v25_4vews_abs", 
    #     # "/aigc_cfs/xibinsong/models/3view_models_51000",
    #     # "/aigc_cfs/xibinsong/models/3view_models_39000/zero23plus_v25_4vews_abs",
    #     "/aigc_cfs_gdp/xibin/z123_control/models/3view_models/zero23plus_v25_4vews_abs_39000",
    #     # "/aigc_cfs/xibinsong/models/3view_models/",
    #     # custom_pipeline="/aigc_cfs/xibinsong/code/zero123plus_control/zero123plus_gray/pipeline_3views_online",
    #     custom_pipeline="/aigc_cfs_gdp/xibin/z123_control/code/z123plus_controlnet_gdp/pipeline_3_views_online",
    #     torch_dtype=torch.float16
    #     )

    pipeline = Zero123PlusPipeline.from_pretrained(
        "/aigc_cfs_gdp/xibin/z123_control/models/3view_models/zero23plus_v25_4vews_abs_39000",
        torch_dtype=torch.float16
        )

    # controlnet = ControlNetModel.from_pretrained(
    #     "/aigc_cfs/xibinsong/code/z123_gray/diffusers_triplane/configs/zero123plus/zero123plus_v24_4views/checkpoint-24000/controlnet", torch_dtype=torch.float16
    #     )

    # Feel free to tune the scheduler
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )
    pipeline.to('cuda:0')

    pipeline.enable_xformers_memory_efficient_attention()

    # Run the pipeline

    parent_path = os.path.dirname(img_path)

    input_img_0 = os.path.join(parent_path, "input_img_0.png")
    input_img_1 = os.path.join(parent_path, "input_img_1.png")
    input_img_2 = os.path.join(parent_path, "input_img_2.png")
    input_img_3 = os.path.join(parent_path, "input_img_3.png")

    img_resolution = 416

    # if in_data_type == "3viewsgen":
    #     print("running 3viewsgen d2rgb !!!")
    # else:
    #      print("try to run 3viewsgen d2rgb !!!")
    #     raise ValueError("try to run 3viewsgen d2rgb !!! the in_data_type must be 3viewsgen !!!")

    image_path_list = [] 
    # if os.path.exists(input_img_0) and os.path.exists(input_img_1) and os.path.exists(input_img_2):
    #     print("input images exist!")
    # else:
    #     raise ValueError("input images not exist !")

    num_imgs = 0
    if os.path.exists(input_img_0):
        print("input image 0 exist!")
        image_path_list.append(input_img_0)
        num_imgs += 1
    else:
        print("input image 0 not exist!")

    if os.path.exists(input_img_1):
        print("input image 1 exist!")
        image_path_list.append(input_img_1)
        num_imgs += 1
    else:
        print("input image 1 not exist!")
        
    if os.path.exists(input_img_2):
        print("input image 2 exist!")
        image_path_list.append(input_img_2)
        num_imgs += 1
    else:
        print("input image 2 not exist!")
    
    if os.path.exists(input_img_3):
        print("input image 3 exist!")
        image_path_list.append(input_img_3)
        num_imgs += 1
    else:
        print("input image 3 not exist!")

    if num_imgs > 0:
        print("find all input images !")
    else:
        raise ValueError("input images not exist !")

    # img_npy = np.load(img_path)

    # assert(img_npy.shape[0] == 4), "shape 0 dim of input npy file must be 4 !!!"

    image_list = []
    num_idx = 0
    for image_path in image_path_list:
        image_cond, _ = process_image_path(image_path, bg_color=127, wh_ratio=0.9, use_sam=False)
        # h,w = image_cond.size
        # h = int(h*0.9)
        # w = int(h*0.9)
        # image_cond = center_crop(image_cond, w, h)
        image_cond = image_cond.resize((img_resolution, img_resolution))

        # save_seg_img_name = os.path.join(save_path_each, "seg_input_img_" + str(num_idx) + ".png")
        # image_cond.save(save_seg_img_name)

        num_idx += 1
        # image_cond.save("test.png")
        # breakpoint()
        image_list.append(image_cond)

    # image_list = []
    # for num_idx in range(img_npy.shape[0]):
    #     img_each = img_npy[num_idx, :3, :, :]
    #     img_each = torch.from_numpy(img_each)
    #     img_each = img_each.permute(1, 2, 0)
    #     img_each = Image.fromarray(img_each.numpy().astype("uint8"))
    #     img_each.resize((416, 416))
    #     image_cond.save(str(num_idx) + ".png")
    #     image_list.append(img_each)

    # normal, xyz = render_obj_depth_normal(obj_path=obj_path,out_dir=out_dir, render_size=img_resolution, use_ortho=True)
    xyz = render_obj_depth_normal(obj_path=obj_path,out_dir=out_dir, render_size=img_resolution, use_ortho=True)
    # breakpoint()

    xyz = xyz.permute(1, 2, 0) 
    xyz = xyz.cpu().numpy() * 255.0
    xyz = xyz.astype(np.uint8)
    depth = Image.fromarray(xyz)

    i = 3.5
    conditioning_scale=1.0
    inference_steps = 75
    
    result = pipeline(image_list, depth_image=depth, controlnet=controlnet, guidance_scale=i, conditioning_scale=conditioning_scale, num_inference_steps=inference_steps, width=img_resolution*2, height=img_resolution*2).images[0]

    depth = np.array(depth)
    result = np.array(result)

    ref_img = image_list[0]
    ref_img = np.array(ref_img)
    ref_img = torch.from_numpy(ref_img)
    # print("ref_img: ", ref_img.shape)
    ref_img = ref_img[:, :, :3]
    depth = torch.from_numpy(depth)
    result = torch.from_numpy(result)

    depth = torch.cat((depth[:img_resolution, :img_resolution, :].unsqueeze(0), depth[:img_resolution, img_resolution:, :].unsqueeze(0), depth[img_resolution:, :img_resolution, :].unsqueeze(0), depth[img_resolution:, img_resolution:, :].unsqueeze(0)), dim=0)
    out = torch.cat((result[:img_resolution, :img_resolution, :].unsqueeze(0), result[:img_resolution, img_resolution:, :].unsqueeze(0), result[img_resolution:, :img_resolution, :].unsqueeze(0), result[img_resolution:, img_resolution:, :].unsqueeze(0)), dim=0)

    depths = depth.permute(0, 3, 1, 2)
    out = out.permute(0, 3, 1, 2)
    
    if vis_dir is not None:
        # in_img = torch.cat([ref_img.permute(1,2,0).clip(0,1) * 255, (depths.clip(0,1) * 255).permute(2,0,3,1).reshape(512,4*512,1).expand(512,4*512,3)], dim=1)
        # in_img = torch.cat([ref_img.permute(1,2,0).clip(0,1) * 255, (depths.clip(0,1) * 255).permute(2,0,3,1).reshape(512,4*512,3)], dim=1)
        in_img = torch.cat([ref_img, (depths.clip(0,255)).permute(2,0,3,1).reshape(img_resolution,4*img_resolution,3)], dim=1)
        cv2.imwrite(os.path.join(vis_dir, "in.png"), in_img.numpy()[...,::-1])
        # cv2.imwrite(os.path.join(vis_dir, "out.png"), (out.clip(0,1) * 255).permute(2,0,3,1).reshape(512,-1,3).detach().cpu().numpy()[...,::-1])
        cv2.imwrite(os.path.join(vis_dir, "out.png"), out.permute(2,0,3,1).reshape(img_resolution,-1,3).numpy()[...,::-1])
    
    # np.save(os.path.join(out_dir, "color.npy"), (rgb.clip(0,1) * 255).detach().cpu().numpy().astype(np.uint8))
    np.save(os.path.join(out_dir, "color.npy"), out.numpy().astype(np.uint8))
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"running time: {elapsed_time} seconds")        
    print("finish 3views image gen, save color images in: ", os.path.join(out_dir, "color.npy"))
