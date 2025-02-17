from einops import rearrange

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision import transforms
import pytorch_lightning as pl


class BaseDataModuleFromConfig(pl.LightningDataModule):

    def __init__(self,
                 root_dir,
                 batch_size,
                 fov,
                 path_file='valid_paths.json',
                 image_transforms=None,
                 num_workers=4,
                 fields=['rgb', 'spherical', 'intrinsic', 'extrinsic'],
                 **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.fov = fov
        self.num_workers = num_workers
        self.path_file = path_file
        self.fields = fields

        if image_transforms is not None:
            image_transforms = [
                transforms.Resize(
                    image_transforms.size,
                    interpolation=transforms.InterpolationMode.NEAREST,
                    antialias=False)
            ]
        else:
            image_transforms = []
        image_transforms.extend([
            transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=1.0),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))
        ])
        self.image_transforms = transforms.Compose(image_transforms)

    def make_dataset(mode):
        raise NotImplementedError

    def train_dataloader(self):
        dataset = self.make_dataset(mode='train')
        sampler = RandomSampler(dataset)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           sampler=sampler)

    def val_dataloader(self):
        dataset = self.make_dataset(mode='val')
        sampler = SequentialSampler(dataset)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           sampler=sampler)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.make_dataset(mode='test'),\
            batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
