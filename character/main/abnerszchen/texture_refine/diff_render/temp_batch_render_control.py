import os
import argparse
from PIL import Image
import numpy as np
import glob
import json
from tqdm import tqdm
from render_control import test_render_control


def save_json(json_data, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as jf:
        jf.write(json.dumps(json_data, indent=4))
    return

def run_batch(in_dir, out_json, mode, image_size, scale_factor):
    ti2_imgs = glob.glob(os.path.join(in_dir, "*/t2i.png"))
    if not ti2_imgs:
        print("ERROR no ti2_imgs")
        return

    use_ortho = True if mode == "ortho" else False
    result_dict = {}
    for ti2_img in tqdm(ti2_imgs):
        job_dir = os.path.dirname(ti2_img)
        job_id = os.path.basename(job_dir)

        obj_path = os.path.join(job_dir, "verts2tex/baking/tex_mesh.obj")
        if not os.path.exists(obj_path):
            continue

        output_image_path = obj_path.replace(".obj", f"ortho_{use_ortho}_{image_size}_{scale_factor}.png")
        output_image_np = output_image_path.replace(".png", ".npy")

        test_render_control(obj_path, output_image_path, use_ortho, image_size=image_size, scale_factor=scale_factor)

        result_dict[job_id] = {
            "job_dir": job_dir,
            "obj_path": obj_path,
            "ti2_img": ti2_img,
            "output_image_path": output_image_path,
            "output_image_np": output_image_np,
        }

    save_json(result_dict, out_json)
    print(f"save {len(result_dict)} / {len(ti2_imgs)} objs render depth to {out_json} done")
    return

def main():
    parser = argparse.ArgumentParser(
        description='render obj with setting pose, feed to TexRefine then optim new texture')
    parser.add_argument('in_dir', type=str,default="/aigc_cfs_gdp/sz/batch_0816/compare_z123_lrm_human_ratio/z123_ratio2", help='path of  ')
    parser.add_argument('out_json', type=str, help='path of ')
    parser.add_argument('--mode', type=str, default="ortho")
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--scale_factor', type=float, default=0.9)

    args = parser.parse_args()
    in_dir = args.in_dir
    out_json = args.out_json
    mode = args.mode
    image_size = args.image_size
    scale_factor = args.scale_factor

    run_batch(in_dir, out_json, mode, image_size, scale_factor)


# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
