import torch

model_weight = torch.load('/aigc_cfs/weixuan/code/vae/outputs/image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6/20240827michelangelo-autoencoder+n4096+noise0.0+pfeat3+normembFalse+lr5e-05+qkvbiasFalse+nfreq8+ln_postTrue/ckpts/model.ckpt')
vae_weight = torch.load('/aigc_cfs/weixuan/code/vae/outputs/michelangelo-autoencoder/l256-e64-ne8-nd16/20240909_high_256+n4096+occupancy+rotFalse+noise0.0+fourier+dsampleFalse+pfeat3+logits1.0+kl0.001+lr5e-05/ckpts/last.ckpt')
breakpoint()