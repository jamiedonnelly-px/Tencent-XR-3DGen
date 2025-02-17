import os
import argparse
import sys
import random
from tqdm import tqdm

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, save_json, load_json


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



# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="select data")
    parser.add_argument(
        "--in_json", type=str, default="/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/train_flux/add_caption.json"
    )
    parser.add_argument(
        "--out_dir", type=str, default="/aigc_cfs/sz/dataset/flux_layer1010"
    )
  
    parser.add_argument("--test_ratio", type=float, default=0.05)
    args = parser.parse_args()

    in_json = args.in_json
    out_dir = args.out_dir
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
        conditioning_image = meta[condi_key]
        image_path = os.path.join(os.path.dirname(conditioning_image), "texture_kd.png")
        assert os.path.exists(image_path)
        obj_dicts = [
            {
                "image": image_path,
                "conditioning_image": conditioning_image,
                "text": meta[text_key],
                "dname": dname,
                "oname": oname,
                "text_key": text_key,
            }
            for text_key in ["global_hunyuan", "local_hunyuan", "combined"]
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



if __name__ == "__main__":
    main()
