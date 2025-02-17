"""
 # @ Copyright: Copyright 2022 Tencent Inc
 # @ Author: weizhe
 # @ Create Time: 2024-11-22 11:00:00
 # @ Description: create codes and pointcloud condition feature for Artist-Created Meshes model
 """
 
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from tqdm.contrib import tzip
import hydra
import numpy as np
import pickle
 
from dataset.dataset import exclude_from
from model.data_utils import load_process_mesh, to_mesh
from model.serializaiton import BPT_serialize
from utils import sample_pc
  
def load_data(json_path, cluster='cfs'):
    dataset_json = json.load(open(json_path, "r"))
    assert (
        cluster in dataset_json["import_paths"]
    ), f"data json does not contain any import path for cluster '{cluster}'"

    data_dict = {}
    for data_path in dataset_json["import_paths"][cluster]:
        loaded_data = json.load(open(data_path, "r"))["data"]
        for dset in loaded_data:
            if dset not in data_dict:
                data_dict[dset] = {}
            data_dict[dset].update(loaded_data[dset])

    includes = dataset_json["includes"]
    excludes = dataset_json.get("excludes", {})

    if includes == "ALL":
        includes = data_dict

    includes = exclude_from(includes, excludes)

    # Load all objs cfg
    # self.meta_pairs, self.render_dirs, self.geopcd_dirs, self.texpcd_dirs, self.h5_paths, self.cam_systems

    meta_pairs = []
    mesh_paths = []

    expected_n_objs = sum([len(includes[dtype]) for dtype in includes])
    pbar = tqdm(total=expected_n_objs)

    for dtype in includes:
        for oname in includes[dtype]:

            try:
                if dtype not in data_dict or oname not in data_dict[dtype]:
                    continue

                meta = data_dict[dtype][oname]
                mesh_path = meta["Mesh"]

                meta_pairs.append((dtype, oname))
                mesh_paths.append(mesh_path)
            except Exception as e:
                print(
                    f"error loading {dtype}.{oname}: {type(e).__name__} {e}"
                )

            pbar.update(1)

    pbar.close()

    print(f"Loaded {len(meta_pairs)} meshes.")
    print(f"Expected {expected_n_objs} objs, got {len(mesh_paths)} objs.")
    
    return meta_pairs, mesh_paths

def create_data(meta_pairs, mesh_paths, config):
    os.makedirs(config.output_dir, exist_ok=True)
    interval = len(meta_pairs)//config.chunks
    assert config.cur_inter>=0 and config.cur_inter<=config.chunks-1
    if config.cur_inter < config.chunks-1:
        meta_pairs = meta_pairs[config.cur_inter*interval:(config.cur_inter+1)*interval]
        mesh_paths = mesh_paths[config.cur_inter*interval:(config.cur_inter+1)*interval]
    else:
        meta_pairs = meta_pairs[config.cur_inter*interval:]
        mesh_paths = mesh_paths[config.cur_inter*interval:]

    print(f'current chunk index: {config.cur_inter}')
    broken_list = []
    for meta, mesh_path in tzip(meta_pairs, mesh_paths):
        print('mesh_path: ',mesh_path)
        dtype, oname = meta
        try:
            if config.augment:
                for i in range(config.augment_num):
                    processed_mesh = load_process_mesh(
                        mesh_path,
                        quantization_bits=config.quantization_bits,
                        augment=True,
                    )
                    processed_mesh["faces"] = np.array(processed_mesh["faces"])
                    processed_mesh = to_mesh(
                        processed_mesh["vertices"], processed_mesh["faces"], transpose=True
                    )
                    pc_normal = sample_pc(
                        processed_mesh, pc_num=config.pc_num, with_normal=True
                    )
                    codes = BPT_serialize(
                        processed_mesh,
                        block_size=config.block_size,
                        offset_size=config.offset_size,
                        compressed=config.compressed,
                        special_token=config.special_token,
                        use_special_block=config.use_special_block,
                    )
                    
                    data = {'pc_normal':pc_normal,'codes':codes}
                    output_path = os.path.join(config.output_dir,f'{dtype}_{oname}_{i}.pkl')
                    with open(output_path,'wb') as f:
                        pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
                        f.close()
                        
            else:
                processed_mesh = load_process_mesh(
                    mesh_path,
                    quantization_bits=config.quantization_bits,
                    augment=False,
                )
                processed_mesh["faces"] = np.array(processed_mesh["faces"])
                processed_mesh = to_mesh(
                    processed_mesh["vertices"], processed_mesh["faces"], transpose=True
                )
                pc_normal = sample_pc(
                    processed_mesh, pc_num=config.pc_num, with_normal=True
                )
                codes = BPT_serialize(
                    processed_mesh,
                    block_size=config.block_size,
                    offset_size=config.offset_size,
                    compressed=config.compressed,
                    special_token=config.special_token,
                    use_special_block=config.use_special_block,
                )
                
                data = {'pc_normal':pc_normal,'codes':codes}
                output_path = os.path.join(config.output_dir,f'{dtype}_{oname}.pkl')
                with open(output_path,'wb') as f:
                    pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
                    f.close()
        except:
            broken_list.append(mesh_path)
            continue
            
            
    with open(f'./broken_list_{config.cur_inter}.pkl','wb') as f:
        pickle.dump(broken_list,f,protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
            
            
@hydra.main(config_path="../config", config_name="process-data", version_base="1.2")
def main(config):
    meta_pairs, mesh_paths = load_data(config.data_json, config.cluster)
    create_data(meta_pairs, mesh_paths, config)

    

if __name__ == '__main__':
    main()