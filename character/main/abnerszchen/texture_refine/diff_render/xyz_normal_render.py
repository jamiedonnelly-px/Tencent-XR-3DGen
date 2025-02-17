import argparse
import os
import json
import random
import numpy as np
import glob
from utils_parse_dataset import pose_generation
from geom_renderer import make_pers_cameras, make_ortho_cameras,  get_geom_texture
from renderer import DiffRender
from utils_render import concatenate_images_horizontally, load_images, save_rgba_geom_images


def load_json(in_file):
    with open(in_file, encoding='utf-8') as f:
        data = json.load(f)
    return data


# # 8 pose
# azimuth_list = [0, 45, 90, 135, 180, 225, 270, 315]
# elevation_list = [0, 30, -30, 30, 0, 30, -30, 30]
# fov_list = [10, 10, 10, 10, 10, 10, 10, 10]


def render_once_obj(meta_dict, out_dir, render_size=1024, use_blender_coord=True, use_ortho = False):
    Mesh, Transformation = meta_dict["Mesh"], meta_dict["Transformation"]
    assert os.path.exists(Mesh), f"can not find mesh = {Mesh}"
    assert os.path.exists(Transformation), f"can not find Transformation = {Transformation}"
    
    if use_ortho:
        # 4 pose
        azimuth_list = [0, 90, 180, 270]
        elevation_list = [0] * len(azimuth_list)
        dist_list = [3.0] * len(azimuth_list)
    else:
        # 4 pose
        azimuth_list = [0, 90, 180, 270]
        elevation_list = [20] * len(azimuth_list)
        fov_list = [40] * len(azimuth_list)
        dist_list = None
        

    transformation = np.loadtxt(Transformation)

    diff_render = DiffRender(render_size=render_size)
    diff_render.load_mesh(Mesh, transformation=transformation, use_blender_coord=use_blender_coord)

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
                                    scale_xyz=1.0,
                                    use_blender_coord=use_blender_coord,
                                    device=diff_render.device)
    diff_render.set_cameras_and_render_settings(cameras, render_size=render_size)
    diff_render.calcu_geom_and_cos()

    verts, normals, depths, cos_angles, texels, fragments = diff_render.render_geometry(render_size)
    save_rgba_geom_images(verts, os.path.join(out_dir, "position.png"))
    save_rgba_geom_images(normals, os.path.join(out_dir, "normal.png"))

    rtype = "ortho" if use_ortho else "pers"
    result_tex_rgb, _ = get_geom_texture(verts, diff_render)
    diff_render.save_mesh(f"{out_dir}/bake/textured_xyz_{rtype}_elev{elevation_list[0]}.obj", result_tex_rgb.permute(1, 2, 0))

    result_tex_rgb, _ = get_geom_texture(normals, diff_render)
    diff_render.save_mesh(f"{out_dir}/bake/textured_normal_{rtype}_elev{elevation_list[0]}.obj", result_tex_rgb.permute(1, 2, 0))

    try:
        color_dir = os.path.join(meta_dict["ImgDir"], 'color')
        select_png_paths = [os.path.join(color_dir, f"cam-000{i}.png") for i in range(8)]
        concatenate_images_horizontally(load_images(select_png_paths), os.path.join(out_dir, 'rgb.png'))
    except Exception as e:
        print(f"select img failed {e}")

    print(f"render xyz and normal to {out_dir} done")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='render obj global xyz and normal')
    parser.add_argument('in_dataset_json', type=str, help='path of json from senbo')
    parser.add_argument('out_root_dir', type=str, help='out root path')
    parser.add_argument('--test_cnt', type=int, default=10, help='cnt, -1 use all')
    args = parser.parse_args()

    in_dataset_json = args.in_dataset_json
    out_root_dir = args.out_root_dir
    test_cnt = args.test_cnt
    use_ortho = True

    onames_dict = load_json(in_dataset_json)["data"]["objaverse"]
    # TODO
    onames = list(onames_dict.keys())
    random.shuffle(onames)

    if test_cnt == -1:
        in_mesh_keys = onames
    else:
        in_mesh_keys = onames[:(min(test_cnt, len(onames)))]    
    # in_mesh_keys = ["74b6c33e4b32436cb2d7c9dbad5d5f04"]
    # in_mesh_keys = ["695203f9c4ac4a9f8941d8813fc33d0c"]
    # in_mesh_keys = ["00d56831f9bc49f9a668f418c1af7558"] # niuniu
    # in_mesh_keys = ["2d8606488de147b7aa4018e4f26f8e82"]

    for oname in in_mesh_keys:
        meta_dict = onames_dict[oname]
        out_dir = os.path.join(out_root_dir, oname)
        try:
            render_once_obj(meta_dict, out_dir, use_ortho=use_ortho)
        except Exception as e:
            print(f"failed {e}")
