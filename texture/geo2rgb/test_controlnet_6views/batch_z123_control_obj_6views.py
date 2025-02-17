import torch
import requests
from PIL import Image
import random
import numpy as np
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, ControlNetModel
# from pipeline_online.pipeline import Zero123PlusPipeline

import cv2
import os
from pdb import set_trace as st
import json

import sys
# sys.path.append(os.path.abspath("/aigc_cfs/xibinsong/code/zero123plus_control/zero123plus_gray/utils_use"))
sys.path.append((os.path.abspath("./utils_use")))

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
import nvdiffrast.torch as dr
import torch.nn.functional as F
from kiui.mesh import Mesh

from sam_preprocess.run_sam import process_image_path

import shutil

from pathlib import Path


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
        azimuth_list = [0, 90, 180, 270, 180, 180]
        # azimuth_list = [-30, 60, 150, 240]
        # elevation_list = [0] * len(azimuth_list)
        elevation_list = [0,0,0,0,-90, 90]
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
    # diff_render.load_mesh(Mesh, use_blender_coord=use_blender_coord)
    diff_render.load_mesh_with_rotate(Mesh, use_blender_coord=use_blender_coord)

    save_obj("./test_obj.obj",
            diff_render.mesh.verts_list()[0],
            diff_render.mesh.faces_list()[0],
            verts_uvs=diff_render.mesh.textures.verts_uvs_list()[0],
            faces_uvs=diff_render.mesh.textures.faces_uvs_list()[0],)

    # breakpoint()

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

    # normals = (normals * 125) / 255.0 * 1.1
    # normals = normals * 255.0
    
    # pils = []
    # for i in range(rgba_normalized.shape[0]):
    #     img = rgba_normalized[i]

    #     img = (img * 255).byte()

    #     img_pil = Image.fromarray(img.cpu().numpy(), 'RGBA')

    #     pils.append(img_pil)

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
    xyz_imgs = F.interpolate(xyz_imgs, size=(464, 464), mode='bilinear', align_corners=False)
    xyz_imgs_black = F.interpolate(xyz_imgs_black, size=(464, 464), mode='bilinear', align_corners=False)

    # normal_imgs = normal_imgs.permute(0, 3, 1, 2)

    # print("max: ", torch.max(normal_imgs))
    # print("min: ", torch.min(normal_imgs))
    # print(normal_imgs.shape)
    xyz_grid = make_grid(xyz_imgs, nrow=2, padding=0)
    xyz_grid_black = make_grid(xyz_imgs_black, nrow=2, padding=0)

    # images_out = make_grid(xyz_grid[0], nrow=2, padding=0) * 0.5 + 0.5
    save_image(xyz_grid, os.path.join(out_dir, "xyz_grid.png"))
    save_image(xyz_grid_black, os.path.join(out_dir, "xyz_grid_black.png"))
    # print(os.path.join(out_dir, "xyz_grid.png"))
    # breakpoint()

    # return normal_imgs, xyz_imgs
    return normal_imgs, xyz_grid, xyz_grid_black

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

    # cv2.imwrite("./tmp/eroded_mask.png", eroded_mask)
    # cv2.imwrite("./tmp/image_eroded.png", eroded_image)

    # 创建一个简单的结构元素
    kernel = np.ones((11, 11), np.uint8)

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
    result_image[:,:,0][mask_origin == 0] = 127
    result_image[:,:,1][mask_origin == 0] = 127
    result_image[:,:,2][mask_origin == 0] = 127
    # cv2.imwrite("./tmp/image_dilated.png", result_image)

    return result_image

def dilate_xyz_image(xyz):
    # 定义膨胀内核
    kernel = np.ones((11, 11), np.uint8)

    mask = xyz[:,:,0]
    mask[mask>0] = 255

    # 对每个通道进行膨胀操作，但仅在掩码所指定的区域
    dilated_image = np.zeros_like(xyz)
    for i in range(3):  # 对 B, G, R 三个通道分别进行处理
        channel = xyz[:, :, i]
        masked_channel = cv2.bitwise_and(channel, channel, mask=mask)
        dilated_channel = cv2.dilate(masked_channel, kernel, iterations=1)
        dilated_image[:, :, i] = dilated_channel

    # 显示原图和膨胀后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Dilated Image', dilated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_input_filename(folder_path):
    # 获取文件夹中的所有文件名
    filenames = os.listdir(folder_path)
    
    # 过滤出所有以 .jpg 结尾的文件名
    # jpg_filenames = [filename for filename in filenames if len(filename)]
    max_file = -999999
    for name in filenames:
        if len(name) > max_file:
            max_file=len(name)
            file_name = name
    
    return file_name

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

#### z123 model path
z123_model_path = "/aigc_cfs/xibinsong/models/zero123plus_v28_1cond_6views_abs_512"
#### controlnet model path
controlnet_model_path = "/aigc_cfs_4/xibin/code/diffusers_triplane_6views/configs/zero123plus/zero123plus_v28_1cond_6views_abs_512/checkpoint-16000/controlnet"


test_path = "./test_images"
files_path = os.listdir(test_path)

guidance_scale = 5.5
conditioning_scale = 2.0
infer_steps = 75

save_path = "../results/6views_16000_non_rotate_30_" + str(infer_steps) + "_inpaint_1e_5_guidance_scale_" + str(guidance_scale) + "_conditioning_scale_" + str(conditioning_scale)  
os.makedirs(save_path, exist_ok=True)

for folder in files_path:

    # data_each = data[key]
    # obj_name = data_each["mesh"]
    # img_path = data_each["z123_output_dir"]
    # img_name = os.path.join(img_path, "image_processed.png")

    img_path = os.path.join(test_path, folder+"/0000")
    # img_name = os.path.join(img_name, "0.png")

    img_name = get_input_filename(img_path)
    # print(img_name)
    # img_name = Path(os.path.join(img_path, img_name))
    img_name = os.path.join(img_path, img_name)
    # print(img_name)
    # breakpoint()

    obj_name = os.path.join(test_path, folder+"/0000")
    obj_name_01 = os.path.join(obj_name, "mesh.obj")
    obj_name_02 = os.path.join(obj_name, "mesh_sparse.obj")

    if os.path.exists(obj_name_01):
        # obj_name = Path(obj_name_01)
        obj_name = obj_name_01
    elif os.path.exists(obj_name_02):
        # obj_name = Path(obj_name_02)
        obj_name = obj_name_02

    z123_res_name = os.path.join(test_path, folder+"/0000")
    # z123_res_name = Path(os.path.join(z123_res_name, "mvimg.png"))
    z123_res_name = os.path.join(z123_res_name, "mvimg.png")

    if os.path.exists(img_name):

        print(img_name)
        print(obj_name)

        # pipeline = DiffusionPipeline.from_pretrained(
        #     z123_model_path, custom_pipeline="/aigc_cfs/xibinsong/code/zero123plus_control/zero123plus_gray/pipeline_online",
        #     torch_dtype=torch.float16
        #     )

        pipeline = DiffusionPipeline.from_pretrained(
            z123_model_path, custom_pipeline="./pipeline_online",
            torch_dtype=torch.float16
            )

        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_path, torch_dtype=torch.float16
            )

        # Feel free to tune the scheduler
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing='trailing'
        )
        pipeline.to('cuda:0')
        pipeline.enable_xformers_memory_efficient_attention()
        # Run the pipeline

        save_path_each = os.path.join(save_path, folder)
        os.makedirs(save_path_each, exist_ok=True)

        # save_name_inital = Path(os.path.join(save_path_each, "res.png"))
        save_name_inital = os.path.join(save_path_each, "res.png")
        # save_name_res = os.path.join(save_path_each, "res_dilated.png")

        # save_input_img_name = Path(os.path.join(save_path_each, "input_img.png"))
        save_input_img_name = os.path.join(save_path_each, "input_img.png")
        shutil.copyfile(img_name, save_input_img_name)

        # save_z123_name = Path(os.path.join(save_path_each, "z123.png"))
        save_z123_name = os.path.join(save_path_each, "z123.png")
        shutil.copyfile(z123_res_name, save_z123_name)

        # save_obj_name = Path(os.path.join(save_path_each, "mesh.obj"))
        save_obj_name = os.path.join(save_path_each, "mesh.obj")
        shutil.copyfile(obj_name, save_obj_name)

        # cond = Image.open(img_name)
        cond, _ = process_image_path(img_name, bg_color=127, wh_ratio=0.9, use_sam=False)
        obj_path = obj_name

        normal, xyz, xyz_black = render_obj_depth_normal(obj_path=obj_path,out_dir=save_path_each, render_size=1024, use_ortho=True)
        # breakpoint()

        cond = cond.resize((512, 512))
        # depth = depth.resize((1024, 1024)
        xyz = xyz.permute(1, 2, 0)
        xyz_black = xyz_black.permute(1, 2, 0)
        print("xyz: ", torch.max(xyz))
        print("xyz: ", torch.min(xyz)) 
        xyz = xyz.cpu().numpy() * 255.0
        xyz_black = xyz_black.cpu().numpy() * 255.0
        xyz = xyz.astype(np.uint8)
        xyz_black = xyz_black.astype(np.uint8)

        # xyz_black = dilate_xyz_image(xyz_black)

        print(xyz.shape)
        # breakpoint()
        depth = Image.fromarray(xyz)

        # result = pipeline(cond, depth_image=depth, num_inference_steps=36, width=640, height=960).images[0]
        # guidance_scale = 15
        # for i in range(guidance_scale):
        result = pipeline(cond, depth_image=depth, controlnet=controlnet, guidance_scale=guidance_scale, conditioning_scale=conditioning_scale, num_inference_steps=infer_steps, width=928, height=1392).images[0]
        result.save(save_name_inital)

        img_res = 464
        depth = np.array(Image.fromarray(xyz))
        result = np.array(result)

        ref_img = cond.resize((img_res, img_res))
        ref_img = np.array(ref_img)
        ref_img = torch.from_numpy(ref_img)

        ref_img = ref_img[:, :, :3]
        depth = torch.from_numpy(depth)
        result = torch.from_numpy(result)

        print(depth.shape)
        print(result.shape)
        # breakpoint()

        depth = torch.cat((depth[:img_res, :img_res, :].unsqueeze(0), depth[:img_res, img_res:, :].unsqueeze(0), depth[img_res:2*img_res, :img_res, :].unsqueeze(0), depth[img_res:2*img_res, img_res:, :].unsqueeze(0), depth[2*img_res:, :img_res, :].unsqueeze(0), depth[2*img_res:, img_res:, :].unsqueeze(0)), dim=0)
        out = torch.cat((result[:img_res, :img_res, :].unsqueeze(0), result[:img_res, img_res:, :].unsqueeze(0), result[img_res:2*img_res, :img_res, :].unsqueeze(0), result[img_res:2*img_res, img_res:, :].unsqueeze(0), result[2*img_res:, :img_res, :].unsqueeze(0), result[2*img_res:, img_res:, :].unsqueeze(0)), dim=0)

        depths = depth.permute(0, 3, 1, 2)
        out = out.permute(0, 3, 1, 2)
    
        # if vis_dir is not None:
        if True:
            in_img = torch.cat([ref_img, (depths.clip(0,255)).permute(2,0,3,1).reshape(img_res,6*img_res,3)], dim=1)
            cv2.imwrite(os.path.join(save_path_each, "in.png"), in_img.numpy()[...,::-1])
            # cv2.imwrite(os.path.join(vis_dir, "out.png"), (out.clip(0,1) * 255).permute(2,0,3,1).reshape(512,-1,3).detach().cpu().numpy()[...,::-1])
            cv2.imwrite(os.path.join(save_path_each, "out.png"), out.permute(2,0,3,1).reshape(img_res,-1,3).numpy()[...,::-1])
    
        # np.save(os.path.join(out_dir, "color.npy"), (rgb.clip(0,1) * 255).detach().cpu().numpy().astype(np.uint8))
        np.save(os.path.join(save_path_each, "color.npy"), out.numpy().astype(np.uint8))
        # end_time = time.perf_counter()
        # elapsed_time = end_time - start_time

        # print(f"running time: {elapsed_time} seconds")        
        # print("finish save color images in: ", os.path.join(out_dir, "color.npy"))
