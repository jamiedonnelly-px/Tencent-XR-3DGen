import os
import argparse
import json
import glob

from utils_dataset import load_json, read_lines, save_lines


def parse_est_dir(in_raw_json, in_est_dir, out_dir):
    # dname/oname
    est_objs = glob.glob(os.path.join(in_est_dir, '*/*/0000/mesh.obj'))
    if not est_objs or len(est_objs) < 1:
        print('can not fin any obj in ', in_est_dir)
        return
    data_dict = load_json(in_raw_json)['data']
    
    os.makedirs(out_dir, exist_ok=True)

    valid_est_objs, proc_data_paths, mesh_paths = [], [], []
    invald_cnt = 0
    for est_obj in est_objs:
        dname, o_name = est_obj.split('/')[-4:-2]
        
        meta_dict = data_dict[dname][o_name]
        proc_data_path, mesh_path = '/'.join(meta_dict['TexPcd'].split('/')[:-1]), meta_dict['Mesh']
        if not os.path.exists(proc_data_path) or not os.path.exists(mesh_path):
            invald_cnt += 1
            print('invalid', dname, o_name)
            continue
        valid_est_objs.append(est_obj)
        proc_data_paths.append(proc_data_path)
        mesh_paths.append(mesh_path)
        
    save_lines(valid_est_objs, os.path.join(out_dir, 'est_objs.txt'))
    save_lines(proc_data_paths, os.path.join(out_dir, 'proc_data_paths.txt'))
    save_lines(mesh_paths, os.path.join(out_dir, 'mesh_paths.txt'))
    print(f'save done to {out_dir} with invald_cnt {invald_cnt}')
  
    return

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='diffusion est dir -> est obj list, gt mesh path')
    parser.add_argument('in_raw_json', type=str)
    parser.add_argument('in_est_dir', type=str)
    parser.add_argument('out_dir', type=str)
    args = parser.parse_args()

    # Run.
    parse_est_dir(args.in_raw_json, args.in_est_dir, args.out_dir)
    return

if __name__ == "__main__":
    main()
