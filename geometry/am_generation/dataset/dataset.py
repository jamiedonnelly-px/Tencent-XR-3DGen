"""
 # @ Copyright: Copyright 2022 Tencent Inc
 # @ Author: weizhe
 # @ Create Time: 2024-11-20 11:00:00
 # @ Description: dataset for Artist-Created Meshes
 """

import random
import os
random.seed(1337)

import json
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pickle

from model.data_utils import load_process_mesh, to_mesh
from model.serializaiton import BPT_serialize
from utils import sample_pc


def exclude_from(dic, excluded_dic):
    ret_dict = {}

    if not isinstance(dic, dict):
        assert (
            excluded_dic is None or len(excluded_dic) == 0
        ), "excluded_dic is expected to be empty"
        return dic

    ## excluded_dic can be a collection or dict
    if isinstance(excluded_dic, dict):
        for key in dic:
            if key in excluded_dic:
                if excluded_dic[key] == "ALL":
                    continue
                else:
                    ret_dict[key] = exclude_from(dic[key], excluded_dic[key])
            else:
                ret_dict[key] = dic[key]

    else:
        for key in dic:
            if key in excluded_dic:
                continue
            else:
                ret_dict[key] = dic[key]

    return ret_dict


class AMDataset(Dataset):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

        self.parse_data_json(self.FLAGS)

        self.n_objs_all = len(self.meta_pairs)
        if self.n_objs_all < 1:
            print("cannot load data from ", FLAGS["data_json"])
            raise TypeError

        print(f"[AM Dataset] will use {self.n_objs_all} objs.")

    def parse_data_json(self, FLAGS):
        """Read the input JSON file and parse the (name, rendering path, camera parameters, point cloud path, triplane path, and octree path) for each obj.

        Args:
            data_json: [data][dtype][dict of oname:meta], meta is dict example: data/merge/0926_raw.json
            FLAGS: config

        Returns:
            (name, rendering path, camera parameters, point cloud path, triplane path, and octree path) for each obj.
            meta_pairs: list of tupe (dtype, oname)
        """
        dataset_json = json.load(open(FLAGS["data_json"], "r"))
        cluster = FLAGS["cluster"]
        assert (
            cluster in dataset_json["import_paths"]
        ), f"data json does not contain any import path for cluster '{cluster}'"

        self.data_json_list = dataset_json["import_paths"][cluster]

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

        self.meta_pairs = []
        self.mesh_paths = []

        expected_n_objs = sum([len(includes[dtype]) for dtype in includes])
        pbar = tqdm(total=expected_n_objs)

        for dtype in includes:
            for oname in includes[dtype]:

                try:
                    if dtype not in data_dict or oname not in data_dict[dtype]:
                        continue

                    meta = data_dict[dtype][oname]
                    mesh_path = meta["Mesh"]

                    self.meta_pairs.append((dtype, oname))
                    self.mesh_paths.append(mesh_path)
                except Exception as e:
                    print(
                        f"error loading {dtype}.{oname} from {self._get_raw_json_path(dset=dtype, oid=oname)}: {type(e).__name__} {e}"
                    )

                pbar.update(1)

        pbar.close()

        print(f"Loaded {len(self.meta_pairs)} meshes.")
        print(f"Expected {expected_n_objs} objs, got {len(self.mesh_paths)} objs.")

    def _get_obj_id(self, obj_idx):

        dataset_name, obj_id = self.meta_pairs[obj_idx]

        return dataset_name, obj_id

    def _get_raw_json_path(self, obj_idx=None, dset=None, oid=None):
        if obj_idx is not None:
            dset, oid = self._get_obj_id(obj_idx)
        for json_path in reversed(self.data_json_list):
            with open(json_path) as json_f:
                json_data = json.load(json_f)["data"]
                if dset in json_data and oid in json_data[dset]:
                    return json_path

    def _parse_item(self, obj_idx):
        # dataset_name, obj_id = self._get_obj_id(obj_idx)
        meta_pair = self.meta_pairs[obj_idx]
        dtype, oname = meta_pair
        argument_index = random.randint(0, 2)
        pickle_path = os.path.join(self.FLAGS.pkl_dir,f'{dtype}_{oname}_{argument_index}.pkl')
        if not os.path.isfile(pickle_path):
            new_index = random.randint(0, self.n_objs_all//8-1)
            print('pickle file not exist: ',pickle_path)
            return self._parse_item(new_index)
            
        with open(pickle_path,'rb') as f:
            cur_data = pickle.load(f)
            f.close()

        pc_normal = cur_data['pc_normal']
        codes = cur_data['codes']
        
        if len(codes) > self.FLAGS.max_seq_len:
            new_index = random.randint(0, self.n_objs_all//8-1)
            print('exceed max length with codes length: ',len(codes))
            return self._parse_item(new_index)

        pc_normal = torch.from_numpy(pc_normal)
        codes = torch.from_numpy(codes)

        return {"pc_normal": pc_normal, "codes": codes}

    def __len__(self):
        return self.n_objs_all

    def __getitem__(self, obj_idx):
        try:
            return self._parse_item(obj_idx)
        except Exception as e:
            raise ValueError(
                f'Error when parsing {obj_idx}-th item, with id {self._get_obj_id(obj_idx)} from "{self._get_raw_json_path(obj_idx)}"'
            ) from e

    @staticmethod
    def collate_fn(batch):
        collated_batch = {}
        max_code_len = -1
        for item in batch:
            max_code_len = max(max_code_len, len(item["codes"]))
        assert max_code_len > 0
        for key in batch[0].keys():
            if key == "codes":
                collated_batch[key] = torch.utils.data.default_collate([
                    F.pad(item[key], pad=(0, max_code_len - len(item[key])), value=-1)
                    for item in batch
                ])
            else:
                collated_batch[key] = torch.utils.data.default_collate(
                    [item[key] for item in batch]
                )
        return collated_batch
