import os
import argparse
import sys
import random
import copy
from tqdm import tqdm
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from dataset.utils_dataset import parse_objs_json, save_json, split_pod_json, load_json, save_lines

def split_list(lst, n):
    division = len(lst) / float(n) 
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='select data')
    parser.add_argument('in_vae_json', type=str, default="/aigc_cfs/weixuan/code/DiffusionSDF/config/stage1_sz_infer/specs_test.json")
    parser.add_argument('out_dir', type=str, default="/aigc_cfs/weixuan/code/DiffusionSDF/config/stage1_sz_infer/split")
    parser.add_argument('--split_cnt', type=int, default=1)
    parser.add_argument('--code_dir', type=str, default="/aigc_cfs/weixuan/code/DiffusionSDF")
    args = parser.parse_args()

    in_vae_json = args.in_vae_json
    out_dir = args.out_dir
    split_cnt = args.split_cnt
    code_dir = args.code_dir
    
    assert os.path.exists(in_vae_json), in_vae_json
    assert os.path.exists(code_dir), code_dir
    
    infer_dict = load_json(in_vae_json)
    in_json = infer_dict["data_config"]["dataset_json"]
    
    assert os.path.exists(in_json), in_json
    os.makedirs(out_dir, exist_ok=True)

    objs_dict, key_pair_list = parse_objs_json(in_json)
    all_data_dict = objs_dict['data']
    # tasks_pair_pod
    
    def filter_key(key_pair):
        dname = key_pair[1]
        if "objaverse" in dname:
            return False
        # if "Designcenter" in dname:
        #     return True
        return True
    
    select_pairs = [key_pair for key_pair in key_pair_list if filter_key(key_pair)]
    random.shuffle(select_pairs)
    split_pods_pairs = split_list(select_pairs, split_cnt)
    
    cmds = [f"cd {code_dir}"]
    for idx in range(split_cnt):
        tasks_pair_pod = split_pods_pairs[idx]
        print(f'debug tasks_pair_pod {idx},  {len(tasks_pair_pod)}')
        out_pod_json = os.path.join(out_dir, f"split_data_{idx}.json")
        split_pod_json(in_json, tasks_pair_pod, out_pod_json)

        temp_dict = load_json(out_pod_json)
        temp_dict["geo_config"] = objs_dict["geo_config"]
        save_json(temp_dict, out_pod_json)
        
                
        infer_dict_i = copy.deepcopy(infer_dict)
        infer_dict_i["data_config"]["dataset_json"] = out_pod_json
        
        out_infer_dir = os.path.join(out_dir, f"infer_{idx}")
        os.makedirs(out_infer_dir, exist_ok=True)
        out_infer_json_i = os.path.join(out_infer_dir, f"specs_test.json")
        
        save_json(infer_dict_i, out_infer_json_i)
        
        print('debug out_pod_json', out_pod_json)
        print('debug out_infer_json_i', out_infer_json_i)
        
        cmd = f"CUDA_VISIBLE_DEVICES={idx} python test.py --exp_dir {out_infer_dir} --num_samples 100000 &"
        print('cmd ', cmd)
        cmds.append(cmd)
        cmds.append(f"echo 'run gpu {idx}'")
        cmds.append("sleep 30s")
        
    
    save_lines(cmds, os.path.join(out_dir, 'batch_run.sh'))
        


if __name__ == "__main__":
    main()
