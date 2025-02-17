import os
import json
import logging
import time
import random
import argparse
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from easy_interface import TexgenInterface, init_job_id

logging.basicConfig(level=print, format='%(asctime)s [%(levelname)s] %(message)s')


def load_json(in_file):
    with open(in_file, encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json(json_data, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as jf:
        jf.write(json.dumps(json_data, indent=4))
    return


def run_blocking_call_query_text(args):
    texgen_interface, job_id, in_mesh_path, in_prompts, in_mesh_key, out_objs_dir, timeout = args
    return texgen_interface.blocking_call_query_text(job_id,
                                                     in_mesh_path,
                                                     in_prompts,
                                                     in_mesh_key=in_mesh_key,
                                                     out_objs_dir=out_objs_dir,
                                                     timeout=timeout)

def process_mesh_key(in_mesh_key):
    job_id = init_job_id()
    out_objs_dir = os.path.join(out_root_dir, in_mesh_key, job_id)
    try:
        if direct_read_obj:
            in_mesh_path = in_dict[in_mesh_key]["Mesh_obj_raw"]
            use_in_mesh_key = None
        else:
            in_mesh_path = None
            use_in_mesh_key = in_mesh_key

        success_flag, result_meshs = texgen_interface.blocking_call_query_text(job_id,
                                                                                in_mesh_path,
                                                                                in_prompts,
                                                                                in_mesh_key=use_in_mesh_key,
                                                                                out_objs_dir=out_objs_dir,
                                                                                timeout=50)
    except Exception as e:
        success_flag, result_meshs = False, str(e)
        print('e:', e)
        print('in_mesh_key:', in_mesh_key)

    res_dict = {"in_mesh_key": in_mesh_key, "out_objs_dir": out_objs_dir, "result_meshs": result_meshs}
    if not success_flag:
        res_dict["meta"] = in_dict[in_mesh_key]
        failed_dict[job_id] = res_dict
        logging.error(f"[ERROR] {in_mesh_key} failed! ")
    else:
        suc_dict[job_id] = res_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='batch test web_json')
    parser.add_argument('in_web_json',
                        type=str,
                        default='/aigc_cfs_gdp/layer_tex/20240711_gdp/web_flatten_gdp_manual_fixuv.json')
    parser.add_argument('out_root_dir', type=str, help='out root')
    parser.add_argument('--client_cfg_json',
                        type=str,
                        default='client_texgen.json',
                        help='relative name in codedir/configs')
    parser.add_argument('--model_name',
                        type=str,
                        default='uv_mcwy',
                        help='select model. can be uv_mcwy, control_mcwy, imguv_mcwy, imguv_lowpoly, pipe_type_dataset')

    parser.add_argument("--test_cnt", type=int, default=10, help="test can , test all if =-1 ")
    parser.add_argument("--max_workers", type=int, default=2, help="multi-thread cnt ")
    parser.add_argument("--in_prompts", type=str, default="dragon", help="in_prompts ")
    args = parser.parse_args()

    # 1. init
    texgen_interface = TexgenInterface(args.client_cfg_json, args.model_name)
    in_dict = load_json(args.in_web_json)
    out_root_dir = args.out_root_dir
    test_cnt = args.test_cnt
    max_workers = args.max_workers
    in_prompts = args.in_prompts
    
    onames = list(in_dict.keys())
    random.shuffle(onames)

    if test_cnt == -1:
        in_mesh_keys = onames
    else:
        in_mesh_keys = onames[:(min(test_cnt, len(onames)))]

    suc_dict = dict()
    failed_dict = dict()
    direct_read_obj = False
    print(f"max_workers={max_workers}, begin run {len(in_mesh_keys)} meshs")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(in_mesh_keys)) as pbar:
            for i in range(0, len(in_mesh_keys), max_workers):
                mesh_keys_to_process = in_mesh_keys[i:i+max_workers]
                futures = [executor.submit(process_mesh_key, mesh_key) for mesh_key in mesh_keys_to_process]

                for future in as_completed(futures):
                    completed_mesh_key = future.result()
                    pbar.update()
                        
            print(f"bad cases cnt: {len(failed_dict)}")
        
    failed_json = os.path.join(out_root_dir, "failed.json")
    suc_json = os.path.join(out_root_dir, "suc.json")
    save_json(failed_dict, failed_json)
    save_json(suc_dict, suc_json)


    print("test multi-process done")

    texgen_interface.close()

    print("test done")
    print(f"suc_dict/failed_dict: {len(suc_dict)}{len(failed_dict)}")
