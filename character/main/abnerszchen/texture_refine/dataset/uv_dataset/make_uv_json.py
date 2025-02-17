import os
import argparse
import sys
import random
from tqdm import tqdm

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, save_json, load_json
from dataset.uv_dataset.batch_copy import cp_uv_dataset, replace_dir_in_dict, copy_file


def split_uv_train_test(dname_data_map, test_ratio=0.1, min_cnt=5):
    train_dicts, test_dicts = [], []
    for dname, dicts in dname_data_map.items():
        d_cnt = len(dicts)
        sample_cnt = min(max(min_cnt, int(d_cnt * test_ratio)), d_cnt)

        random.shuffle(dicts)
        train_dicts += dicts[sample_cnt:]
        test_dicts += dicts[:sample_cnt]

        print(f"sample test {sample_cnt}/{d_cnt} from {dname}")

    return train_dicts, test_dicts


def replace_dir_in_json(in_json, out_json, src_root, new_root):
    assert os.path.exists(in_json), in_json
    data = load_json(in_json)
    data = replace_dir_in_dict(data, src_root, new_root)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    save_json(data, out_json)
    return


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="select data")
    parser.add_argument(
        "in_json", type=str, default="/aigc_cfs_3/layer_tex/mcwy_2/2024/caption_llava_right.json"
    )
    parser.add_argument(
        "out_dir", type=str, default="/aigc_cfs_3/layer_tex/uv_datasets/mcwy2_pos/right_llava"
    )
    parser.add_argument(
        "--out_910b_dir", type=str, default="/data5/sz/uv_datasets/mcwy2_pos/right_llava"
    )
    parser.add_argument("--test_ratio", type=float, default=0.05)
    args = parser.parse_args()

    in_json = args.in_json
    out_dir = args.out_dir
    out_910b_dir = args.out_910b_dir
    test_ratio = args.test_ratio
    assert os.path.exists(in_json), in_json
    os.makedirs(out_dir, exist_ok=True)
    # condi_key = "uv_normal"
    condi_key = "uv_pos"

    objs_dict, key_pair_list = parse_objs_json(in_json)

    out_list = []
    cnt = 0
    dname_data_map = {}
    for d_, dname, oname in tqdm(key_pair_list):
        meta = objs_dict[d_][dname][oname]
        image = meta["uv_kd"]
        conditioning_image = meta[condi_key]
        obj_dicts = [
            {
                "image": image,
                "conditioning_image": conditioning_image,
                "text": text,
                "dname": dname,
            }
            for text in meta["caption"]
        ]
        out_list += obj_dicts
        cnt += len(obj_dicts)

        if dname not in dname_data_map:
            dname_data_map[dname] = []
        dname_data_map[dname] += obj_dicts

    all_json = os.path.join(out_dir, "all.json")
    save_json(out_list, all_json)

    train_dicts, test_dicts = split_uv_train_test(dname_data_map, test_ratio=test_ratio)
    save_json(train_dicts, os.path.join(out_dir, "train.json"))
    save_json(test_dicts, os.path.join(out_dir, "test.json"))

    # with open(out_json, "w") as outfile:
    #     json.dump(out_list, outfile, indent=4)
    print(f"select data {cnt} pairs from {len(key_pair_list)} obj done to {out_dir}")

    if out_910b_dir:
        src_root = os.path.dirname(in_json)
        cp_uv_dataset(all_json, src_root, out_910b_dir)
        print(f"cp data pairs from {src_root}to {out_910b_dir} done")
        bak_vis_dir = os.path.join(out_dir, "910b_vis")

        for name in ["train.json", "test.json"]:
            src_json = os.path.join(out_dir, name)
            out_json = os.path.join(out_910b_dir, name)
            bak_vis_json = os.path.join(bak_vis_dir, name)
            replace_dir_in_json(
                src_json,
                out_json,
                src_root,
                out_910b_dir,
            )
            
            copy_file(out_json, bak_vis_json)
            print(f"can vis 910b json {out_json} in {bak_vis_json}")
        


if __name__ == "__main__":
    main()
