import os
import argparse
import copy
import random

from utils_dataset import load_json, save_json


def split_train_test(in_json, out_dir, test_ratio, min_cnt=5):
    raw_dict = load_json(in_json)
    train_dict = copy.deepcopy(raw_dict)
    test_dict = copy.deepcopy(raw_dict)
    
    data_dict = raw_dict['data']
    
    train_data_dict = {}
    test_data_dict = {}
    test_cnt = 0
    all_cnt = 0
    for dname, d_metas in data_dict.items():
        keys = list(d_metas.keys())
        d_cnt = len(keys)
        sample_cnt = min(max(min_cnt, int(d_cnt * test_ratio)), d_cnt)
        print(f'sample {sample_cnt}/{d_cnt} from {dname}')
        
        random.shuffle(keys)
        train_data_dict[dname] = {key: d_metas[key] for key in keys[sample_cnt:]}
        test_data_dict[dname] = {key: d_metas[key] for key in keys[:sample_cnt]}
        test_cnt += sample_cnt
        all_cnt += d_cnt
        
    os.makedirs(out_dir, exist_ok=True)
    key = os.path.splitext(os.path.basename(in_json))[0]
    train_json = os.path.join(out_dir, f'{key}_train.json')
    test_json = os.path.join(out_dir, f'{key}_test.json')
    train_dict['data'] =  train_data_dict
    test_dict['data'] =  test_data_dict
    save_json(train_dict, train_json)
    save_json(test_dict, test_json)
    print(f'save done to {out_dir}, test_cnt:{test_cnt}/{all_cnt} train: {train_json}')
  
    return

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='diffusion est dir -> est obj list, gt mesh path')
    parser.add_argument('in_json', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    args = parser.parse_args()

    # Run.
    split_train_test(args.in_json, args.out_dir, args.test_ratio)
    return

if __name__ == "__main__":
    main()
