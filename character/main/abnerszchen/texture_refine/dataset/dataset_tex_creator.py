
import os
import torch
import numpy as np
import random
import PIL
from torchvision import transforms
import torch.nn.functional as F

from .utils_dataset import parse_objs_json, load_img_rgb, load_condi_img, load_depth

class DatasetTexCreator(torch.utils.data.Dataset):

    def __init__(self, in_json, resolution=512, data_type='train', use_img_condi = True, dataset_argum=True, test_ratio=0.1):
        self.resolution = resolution
        self.data_type = data_type
        self.use_img_condi = use_img_condi
        self.dataset_argum = dataset_argum

        try:
            self.data_dict, key_pair_list = parse_objs_json(in_json)
        except:
            print(f'can not load in_json {in_json}')
            raise ValueError
        
        self.tex_condi_pairs = []
        for first, dname, oname in key_pair_list:
            meta_dict = self.data_dict[first][dname][oname]
            tex_pairs = meta_dict['tex_pairs']
            if self.use_img_condi:
                if 'Condition_img' in meta_dict:
                    self.tex_condi_pairs += [tex_pair + [meta_dict['Condition_img']] for tex_pair in tex_pairs]
                elif 'condition_imgs' in meta_dict:
                    for condi in meta_dict['condition_imgs']:
                        self.tex_condi_pairs += [tex_pair + [condi] for tex_pair in tex_pairs]
            else:
                if "caption" in meta_dict:
                    for condi_text in meta_dict['caption']:
                        self.tex_condi_pairs += [tex_pair + [condi_text] for tex_pair in tex_pairs]
                    
        random.shuffle(self.tex_condi_pairs)
        
        self.pairs_cnt = len(self.tex_condi_pairs)
        print('[DatasetTexCreator] self.tex_condi_pairs ', self.pairs_cnt, self.tex_condi_pairs[0])
        print('[DatasetTexCreator] self.resolution ', self.resolution)
        print('[DatasetTexCreator] self.use_img_condi and self.dataset_argum ', self.use_img_condi, self.dataset_argum)

        self.transform = transforms.Compose([
            
            # transforms.CenterCrop(self.resolution),
            transforms.RandomResizedCrop(self.resolution, scale=(0.6, 1.0), ratio=(1, 1)),
            transforms.RandomRotation(10),
        ])

    def __len__(self):
        return self.pairs_cnt

    def __getitem__(self, itr):
        """dataset item, get gt texture map, render depth and condition image or condition text.

        Args:
            itr: 

        Returns:
            [3, h, w], [h, w], [3, h, w] if self.use_img_condi or condi text
        """
        gt_tex_path, depth_path, condi = self.tex_condi_pairs[itr % self.pairs_cnt]

        gt_tex = load_img_rgb(gt_tex_path, resolution=self.resolution)
        if self.use_img_condi:
            condi = load_condi_img(condi, resolution=224)
        
        if self.dataset_argum:
            render_depth = load_depth(depth_path, resolution=self.resolution)  # TODO need check resize depth
            argum = self.transform(torch.cat([gt_tex, render_depth]))
            gt_tex, render_depth = argum[:3], argum[3:]
            render_depth = F.interpolate(render_depth.unsqueeze(0), size=(self.resolution // 8, self.resolution // 8), mode='bilinear', align_corners=False).squeeze(0)
        else:
            render_depth = load_depth(depth_path, resolution=self.resolution // 8)  # TODO need check resize depth
                
        return gt_tex, render_depth, condi