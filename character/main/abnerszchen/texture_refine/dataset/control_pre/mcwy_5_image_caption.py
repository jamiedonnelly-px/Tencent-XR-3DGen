import os
import shutil
import argparse
import sys
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, save_json
from dataset.control_pre.run_instructblip import ImageCaption

type_prompt_map = {
    "MCWY_2_Top":"Describe the top, describe its color and style.",
    "MCWY_2_Bottom":"Describe the bottom, describe its color and style.",
    "MCWY_2_Dress":"Describe the dress, describe its color and style.",
    "MCWY_2_Shoe":"Describe the shoe, describe its color and style.",
    "readyplayerme_Top":"Describe the top, describe its color and style.",
    "readyplayerme_Footwear":"Describe the footwear, describe its color and style.",
    "readyplayerme_Hat":"Describe the hat, describe its color and style.",
    "readyplayerme_Bottom":"Describe the bottom, describe its color and style.",
    "vroid":"Describe the character.",
    "Designcenter_1":"Describe the character.",
    "Designcenter_20231201":"Describe the character.",
    "Objaverse_Avatar":"Describe the character.",
    "daz":"Describe the character.",
    "gransaga":"Describe the character.",
    "hok":"Describe the character.",
    "lol":"Describe the character.",
    "mario":"Describe the character.",
    "feihong":"Describe the character.",
    "guofenggame":"Describe the character.",
    "hanfeng":"Describe the character.",
    "honkai3":"Describe the character.",
    }
common_prompts = [
    # "Briefly describe the content of the image.",
                  "A short image description:",
                #   "A photo of",
                  ]
def image_caption(key_pair, caption_cls:ImageCaption, out_dir, objs_dict : dict):
    d_, dname, oname = key_pair
    meta = objs_dict[d_][dname][oname]
    out_caption_json = os.path.join(out_dir, dname, oname, 'caption.json')

    caption = []
    for condi_img in meta["condi_imgs_in"]:
        prompt = type_prompt_map.get(dname, "")
        generated_text = caption_cls.query_img_caption(condi_img, prompt =prompt)
        generated_text_un = caption_cls.query_img_caption(condi_img, prompt ="")
        generated_texts = [caption_cls.query_img_caption(condi_img, prompt =p) for p in common_prompts]

        img_caps = [generated_text, generated_text_un] + generated_texts
        print('condi_img ', condi_img)
        print('img_caps ', img_caps)
        caption_dict = {condi_img:img_caps}
        save_json(caption_dict, out_caption_json)
        caption += img_caps

    meta["caption"] = list(set(caption))
    return 1


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="render est obj list")
    parser.add_argument(
        "in_json",
        type=str,
        default="/aigc_cfs_3/layer_tex/mcwy_2/2024/find_image_done.json",
    )
    parser.add_argument("out_dir", type=str)
    parser.add_argument(
        "--blip_model", type=str, default="/aigc_cfs/model/instructblip-flan-t5-xl"
    )
    parser.add_argument("--out_json_name", type=str, default="image_caption_done.json")
    args = parser.parse_args()

    in_json = args.in_json
    out_dir = args.out_dir
    blip_model = args.blip_model
    out_json_name = args.out_json_name
    assert os.path.exists(in_json), in_json
    os.makedirs(out_dir, exist_ok=True)

    objs_dict, key_pair_list = parse_objs_json(in_json)
    caption_cls = ImageCaption(blip_model)

    for key_pair in tqdm(key_pair_list):
        image_caption(key_pair, caption_cls, out_dir, objs_dict)

    out_dict = os.path.join(out_dir, out_json_name)
    save_json(objs_dict, out_dict)
    print(f'image_caption done {len(key_pair_list)}, save to {out_dict}')


if __name__ == "__main__":
    main()
