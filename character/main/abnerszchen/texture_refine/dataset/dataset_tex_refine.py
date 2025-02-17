
import os
import torch
import numpy as np
import random
import PIL

from .utils_dataset import parse_objs_json, load_img_rgb, load_condi_img

class DatasetTexRefine(torch.utils.data.Dataset):

    def __init__(self, in_json, resolution=512, data_type='train', test_ratio=0.1):
        self.resolution = resolution
        self.data_type = data_type

        try:
            self.data_dict, key_pair_list = parse_objs_json(in_json)
        except:
            print(f'can not load in_json {in_json}')
            raise ValueError
        
        self.tex_condi_pairs = []
        for first, dname, oname in key_pair_list:
            meta_dict = self.data_dict[first][dname][oname]
            tex_pairs = meta_dict['tex_pairs']
            self.tex_condi_pairs += [tex_pair + [meta_dict['Condition_img']] for tex_pair in tex_pairs]
        
        random.shuffle(self.tex_condi_pairs)
        
        self.pairs_cnt = len(self.tex_condi_pairs)
        print('self.tex_condi_pairs ', self.pairs_cnt, self.tex_condi_pairs[0])
        print('self.resolution ', self.resolution)

    def __len__(self):
        return self.pairs_cnt

    def __getitem__(self, itr):
        """dataset item

        Args:
            itr: 

        Returns:
            [3, h, w], [3, h, w], [3, h, w]
        """
        gt_tex_path, est_tex_path, condi_img_path = self.tex_condi_pairs[itr % self.pairs_cnt]

        gt_tex = load_img_rgb(gt_tex_path, resolution=self.resolution)
        est_tex = load_img_rgb(est_tex_path, resolution=self.resolution)
        condi_img = load_condi_img(condi_img_path, resolution=224)
        
        return gt_tex, est_tex, condi_img