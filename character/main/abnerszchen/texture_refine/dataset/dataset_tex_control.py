import os
import torch
import numpy as np
import random
import PIL
from torchvision import transforms
from torchvision.transforms import functional as Ftf
from transformers import CLIPTokenizer
from diffusers.image_processor import VaeImageProcessor

from .utils_dataset import parse_objs_json, load_img_rgb, load_condi_img, load_rgba_as_rgb

class MyTransform:
    def __init__(self, size, p, scale=(0.8, 1.0), ratio=(1, 1)):
        self.size = size
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, rgb, geom):
        # RandomResizedCrop.forward
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            geom, scale=self.scale, ratio=self.ratio)
        geom = Ftf.resized_crop(geom, i, j, h, w, self.size)
        rgb = Ftf.resized_crop(rgb, i, j, h, w, self.size)
        
        # RandomHorizontalFlip
        if torch.rand(1) < self.p:
            rgb = Ftf.hflip(rgb)
            geom = Ftf.hflip(geom)
        
        return rgb, geom
    
class DatasetTexControl(torch.utils.data.Dataset):
    def __init__(
        self,
        in_json: str,
        tokenizer: CLIPTokenizer,
        resolution=512,
        proportion_empty_prompts=0.05,
        data_type="train",
        dataset_argum=True,
        dataset_mask_gt=False,
        test_ratio=0.1,
    ):
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.proportion_empty_prompts = proportion_empty_prompts
        self.data_type = data_type
        self.dataset_argum = dataset_argum
        self.dataset_mask_gt = dataset_mask_gt

        try:
            self.data_dict, key_pair_list = parse_objs_json(in_json)
        except:
            print(f"can not load in_json {in_json}")
            raise ValueError

        self.tex_condi_pairs = []
        for first, dname, oname in key_pair_list:
            meta_dict = self.data_dict[first][dname][oname]
            uv_condi_path = meta_dict["uv_pos"]
            uv_kd_path = meta_dict["uv_kd"]
            caption_list = meta_dict["caption"]
            self.tex_condi_pairs += [
                [uv_kd_path, uv_condi_path, caption] for caption in caption_list
            ]

        random.shuffle(self.tex_condi_pairs)

        self.pairs_cnt = len(self.tex_condi_pairs)
        print(
            f"[DatasetTexControl] {self.pairs_cnt} self.tex_condi_pairs with {len(key_pair_list)} objs, resolution={self.resolution}",
            self.tex_condi_pairs[0],
        )

        # same as pipeline, .preprocess(pil)
        self.vae_scale_factor = 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )


        self.transform_argum = MyTransform(self.resolution, 0.5, scale=(0.8, 1.0), ratio=(1, 1))
    

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
        """load geom from pil to 3,h,w in [0, 1]

        Args:
            geom_path: _description_

        Returns:
            [0, 1] 3,h,w tensor
        """
        uv_pil = PIL.Image.open(geom_path).resize((self.resolution, self.resolution), PIL.Image.BILINEAR)
        uv_raw = torch.tensor(np.array(uv_pil).transpose(2, 0, 1)) / 255.    #[0, 1] 3,h,w
        return uv_raw

    def __getitem__(self, itr):
        """dataset item

        Args:
            itr:

        Returns:
            pixel_values gt_uv_kd [1, 3, h, w] Normalize([0.5], [0.5])
            conditioning_pixel_values uv_condi: [1, 3, h, w] in [0, 1]
            input_ids cap [1, 77]
        """
        gt_uv_kd_path, uv_condi_path, caption = self.tex_condi_pairs[
            itr % self.pairs_cnt
        ]

        gt_uv_kd_pil = load_rgba_as_rgb(gt_uv_kd_path, self.resolution)  # pil
        # 1, 3, h, w [-1, 1]
        gt_uv_kd_raw = self.image_processor.preprocess(
            gt_uv_kd_pil, height=self.resolution, width=self.resolution
        )
        # 1, 3, h, w [0, 1]
        uv_condi_raw = self.control_image_processor.preprocess(
            PIL.Image.open(uv_condi_path),
            height=self.resolution,
            width=self.resolution,
        )
        
        if self.dataset_argum:
            gt_uv_kd, uv_condi = self.transform_argum(gt_uv_kd_raw, uv_condi_raw)           
        else:
            gt_uv_kd = gt_uv_kd_raw
            uv_condi = uv_condi_raw

        if self.dataset_mask_gt:
            gt_uv_kd[uv_condi == 0] = 0 # 0 in [-1, 1] means gray
        
        if self.proportion_empty_prompts > 0:
            if random.random() < self.proportion_empty_prompts:
                caption = ""
        
        caption = [caption]
        input_ids = self.tokenize_captions(caption) # [1, 77]

        return {
            "pixel_values": gt_uv_kd,
            "conditioning_pixel_values": uv_condi,
            "input_ids": input_ids,
        }


def collate_fn(data):
    pixel_values = torch.cat([example["pixel_values"] for example in data])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    conditioning_pixel_values = torch.cat(
        [example["conditioning_pixel_values"] for example in data]
    )
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in data])
  

    return {
        "pixel_values": pixel_values,   # gt tex [b, 3, h, w ] [-1, 1]
        "conditioning_pixel_values": conditioning_pixel_values, # input uv-normal [b, 3, h, w ] [0, 1]
        "input_ids": input_ids,    # [b, 1, 77]
    }
