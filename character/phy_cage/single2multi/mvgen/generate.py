import argparse, pdb
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.io import imsave

from ldm.models.diffusion.sync_dreamer import SyncMultiviewDiffusion, SyncDDIMSampler
from ldm.util import instantiate_from_config, prepare_inputs


def load_model(cfg,ckpt,strict=True):
    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    print(f'loading model from {ckpt} ...')
    ckpt = torch.load(ckpt,map_location='cpu')
    model.load_state_dict(ckpt['state_dict'],strict=strict)
    model = model.cuda().eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',type=str, default='configs/syncdreamer.yaml')
    parser.add_argument('--ckpt',type=str, default='../ckpt/syncdreamer-pretrain.ckpt')
    parser.add_argument('--oname', type=str, required=True)
    parser.add_argument('--elevation', type=float, required=True)

    parser.add_argument('--sample_num', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=-1)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--batch_view_num', type=int, default=8)
    parser.add_argument('--seed', type=int, default=6033)

    parser.add_argument('--sampler', type=str, default='ddim')
    parser.add_argument('--sample_steps', type=int, default=50)
    parser.add_argument('--keep_gt', action="store_true")
    parser.add_argument('--mvgtype', type=int, required=True)
    parser.add_argument('--inpaint', type=str, default="")
    flags = parser.parse_args()

    if flags.mvgtype == 0:
        flags.input = f"../../testdata/{flags.oname}.png"
        flags.output = f"../output/mvimgs/{flags.oname}"
    elif flags.mvgtype == 1:
        assert len(flags.inpaint)>0
        flags.input = f"../output/mvimgs/{flags.oname}/inpaint/sss_{flags.inpaint}.png"
        flags.output = f"../output/mvimgs/{flags.oname}/inpaint/"
    else:
        raise NotImplementedError

    torch.random.manual_seed(flags.seed)
    np.random.seed(flags.seed)

    model = load_model(flags.cfg, flags.ckpt, strict=True)
    assert isinstance(model, SyncMultiviewDiffusion)
    Path(f'{flags.output}').mkdir(exist_ok=True, parents=True)

    # prepare data
    data = prepare_inputs(flags.input, flags.elevation, flags.crop_size)
    for k, v in data.items():
        data[k] = v.unsqueeze(0).cuda()
        data[k] = torch.repeat_interleave(data[k], flags.sample_num, dim=0)

    if flags.sampler=='ddim':
        sampler = SyncDDIMSampler(model, flags.sample_steps)
    else:
        raise NotImplementedError
    x_sample = model.sample(sampler, data, flags.cfg_scale, flags.batch_view_num)

    B, N, _, H, W = x_sample.shape
    x_sample = (torch.clamp(x_sample,max=1.0,min=-1.0) + 1) * 0.5
    x_sample = x_sample.permute(0,1,3,4,2).cpu().numpy() * 255
    x_sample = x_sample.astype(np.uint8)

    if flags.keep_gt:
        x_sample[:,0] = ((torch.clamp(data['input_image'],max=1.0,min=-1.0).cpu().numpy() + 1) * 0.5 * 255).astype(np.uint8)

    for bi in range(B):
        output_fn = Path(flags.output)/ f'{bi}.png'
        imsave(output_fn, np.concatenate([x_sample[bi,ni] for ni in range(N)], 1))

if __name__=="__main__":
    main()

