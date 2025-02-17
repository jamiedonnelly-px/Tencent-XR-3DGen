import os
import torch
import numpy as np
import random
import PIL
from torchvision import transforms
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTokenizer

from .utils_dataset import parse_objs_json, load_img_rgb, load_rgba_as_rgb, load_condi_img, load_depth


class DatasetTexImgUV(torch.utils.data.Dataset):
    def __init__(
        self,
        in_json: str,
        tokenizer: CLIPTokenizer,
        resolution=512,
        t_drop_rate=0.05,
        i_drop_rate=0.05,
        ti_drop_rate=0.05,
        data_type="train",
        dataset_argum="True",
        test_ratio=0.1,
    ):
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.t_drop_rate = t_drop_rate
        self.i_drop_rate = i_drop_rate
        self.ti_drop_rate= ti_drop_rate    
        self.data_type = data_type
        self.dataset_argum = dataset_argum

        
        try:
            self.data_dict, key_pair_list = parse_objs_json(in_json)
        except:
            print(f"can not load in_json {in_json}")
            raise ValueError

        self.tex_condi_pairs = []
        for first, dname, oname in key_pair_list:
            meta_dict = self.data_dict[first][dname][oname]
            uv_normal_path = meta_dict["uv_normal"]
            uv_pos_path = uv_normal_path.replace('uv_normal.png', 'uv_pos.png') # TODO(csz) add to json
            
            uv_kd_path = meta_dict["uv_kd"]
            caption_list = meta_dict["caption"]
            condi_imgs_train = meta_dict["condi_imgs_train"]
            for condi_img in condi_imgs_train:
                for caption in caption_list:
                    self.tex_condi_pairs += [
                        [uv_kd_path, uv_pos_path, uv_normal_path, caption, condi_img]
                    ]

        random.shuffle(self.tex_condi_pairs)
   
        self.pairs_cnt = len(self.tex_condi_pairs)
        print(
            f"[DatasetTexControl] {self.pairs_cnt} self.tex_condi_pairs with {len(key_pair_list)} objs, resolution={self.resolution}",
            self.tex_condi_pairs[0],
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.transform_argum = transforms.Compose([
            
            # transforms.CenterCrop(self.resolution),
            # transforms.RandomRotation(10),
            transforms.RandomResizedCrop(self.resolution, scale=(0.6, 1.0), ratio=(1, 1)),
            transforms.RandomHorizontalFlip(0.5),
        ])
        
        self.transform_argum_post_rgb = transforms.Compose([
            transforms.Normalize([0.5], [0.5]),
        ])

        self.transform_clip = transforms.Compose([
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()
        
    def __len__(self):
        return self.pairs_cnt

    def tokenize_captions(self, caption, is_train=True):
        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    def load_geom(self, geom_path):
        uv_pil = PIL.Image.open(geom_path).resize((self.resolution, self.resolution), PIL.Image.BILINEAR)
        uv_raw = torch.tensor(np.array(uv_pil).transpose(2, 0, 1)) / 255.    #[0, 1] 3,h,w
        return uv_raw
                
    def __getitem__(self, itr):
        """dataset item

        Args:
            itr:

        Returns:
            pixel_values gt_uv_kd [1, 3, h, w] Normalize([0.5], [0.5])
            conditioning_pixel_values geom: [1, 6, h, w] in [0, 1], pos cat normal
            input_ids cap [1, 77]
        """
        gt_uv_kd_path, uv_pos_path, uv_normal_path, caption, condi_img = self.tex_condi_pairs[
            itr % self.pairs_cnt
        ]

        if self.dataset_argum:
            uv_pos_raw = self.load_geom(uv_pos_path)   #[0, 1] 3,h,w
            uv_normal_raw = self.load_geom(uv_normal_path)   #[0, 1] 3,h,w
            gt_uv_kd_raw = load_img_rgb(gt_uv_kd_path, resolution=self.resolution)   # [-1, 1], [c, h, w]
            
            train_pair = torch.stack([gt_uv_kd_raw, uv_pos_raw, uv_normal_raw])
            train_pair = self.transform_argum(train_pair)
            gt_uv_kd, uv_pos, uv_normal = train_pair.chunk(3)
            gt_uv_kd = self.transform_argum_post_rgb(gt_uv_kd)
            geom = torch.cat([uv_pos, uv_normal], dim=1)
            
        else:
            raise NotImplementedError('TODO, need argum')
            gt_uv_kd = self.transform(PIL.Image.open(gt_uv_kd_path).convert("RGB"))
            uv_normal = load_depth(uv_normal_path, resolution=self.resolution)
            uv_normal = uv_normal.squeeze(0).permute(2, 0, 1)
            # 1, 3, h,w
            gt_uv_kd = gt_uv_kd.unsqueeze(0) 
            uv_normal = uv_normal.unsqueeze(0)

        clip_image = self.clip_image_processor(images=load_rgba_as_rgb(condi_img), return_tensors="pt").pixel_values
        

        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            caption = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            caption = ""
            drop_image_embed = 1

        input_ids = self.tokenize_captions(caption)
            

        return {
            "pixel_values": gt_uv_kd,
            "conditioning_pixel_values": geom,
            "input_ids": input_ids,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
        }


def collate_fn(data):
    pixel_values = torch.cat([example["pixel_values"] for example in data])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    conditioning_pixel_values = torch.cat(
        [example["conditioning_pixel_values"] for example in data]
    )
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.cat([example["input_ids"] for example in data], dim=0)

    clip_image = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embed = [example["drop_image_embed"] for example in data]


    return {
        "pixel_values": pixel_values,   # gt tex
        "conditioning_pixel_values": conditioning_pixel_values, # input uv-geom
        "input_ids": input_ids,
        "clip_image": clip_image,
        "drop_image_embed": drop_image_embed,        
    }
