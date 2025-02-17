import torch
import torch.utils.checkpoint
from transformer_vector import SD3Transformer2DModel
from thop import profile
import os
import json

exp_dir = "configs/1view_gray_2048_flow"
config_path = os.path.join(exp_dir, "train_configs.json")
with open(config_path, 'r') as fr:
    configs = json.load(fr)

diffusion_config = configs["diffusion_config"]
unet = SD3Transformer2DModel(**diffusion_config).cuda()

noisy_latents = torch.randn(1, 1024, 64).cuda()
encoder_hidden_states = torch.randn(1, 1024, 1024).cuda()
image_latents_pool = torch.randn(1, 1024).cuda()
timesteps = torch.tensor([99], dtype=torch.long).cuda()

Flops, params = profile(unet, inputs=(
    noisy_latents, encoder_hidden_states, image_latents_pool, timesteps))  # macs
print('Flops: % .4fG' % (Flops / 1000000000))  # 计算量
# 参数量：等价与上面的summary输出的Total params值
print('params: % .4fM' % (params / 1000000))


output = unet(noisy_latents, encoder_hidden_states,
              image_latents_pool, timesteps)
print(output.sample.shape)
breakpoint()
