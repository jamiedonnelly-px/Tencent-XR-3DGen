from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import pdb
import numpy as np
from PIL import Image
from scipy import ndimage

pipe = AutoPipelineForInpainting.from_pretrained("/aigc_cfs/model/stable-diffusion-xl-1.0-inpainting-0.1/", torch_dtype=torch.float16, variant="fp16").to("cuda")

import os 
from skimage.io import imread, imsave

########## BACKGROUND REMOVAL
from PIL import Image
import numpy as np
class BackgroundRemoval:
    def __init__(self, device='cuda'):
        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=1,#5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)
        image = self.interface([image])[0]
        image = np.array(image)
        return image

mask_predictor = BackgroundRemoval()

#############

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, required=True)
parser.add_argument('-t', '--text', type=str, required=True)
parser.add_argument('-g', '--neg_text', type=str, default="")
parser.add_argument('-m', '--mask', type=int, default=0)
parser.add_argument('-a', '--alpha_t', type=int, default=200)
opt = parser.parse_args()

raw_image = imread(os.path.join("output/mvimgs", opt.name, "mvout.png"))
raw_mask_image = np.load(os.path.join("output/mvimgs", opt.name, "mvout.npy"))

image_size = 256

sem = raw_mask_image[:, :image_size]
image = np.copy(raw_image[:, :image_size])



if opt.mask == 0:
    masks = sem == 1
    masks = ndimage.binary_dilation(masks, iterations=2, structure=ndimage.generate_binary_structure(2, 2))
elif opt.mask == 1:
    masks = ~(sem == 0)
    masks = ndimage.binary_dilation(masks, iterations=1, structure=ndimage.generate_binary_structure(2, 2))
elif opt.mask == 2:
    masks = ~(sem == 0)
else:
    raise NotImplementedError

image[masks == 1] = (np.random.rand(masks.sum(), 3)*255.).astype(np.uint8)


image = Image.fromarray(image).resize((1024,1024))
mask_image = Image.fromarray(masks * 255.).convert('RGB').resize((1024,1024))


x = np.array(image)
m = (np.array(mask_image) > 0)[...,0]
x[m] = 0

savedir = os.path.join("output/mvimgs", opt.name, "inpaint")
os.makedirs(savedir, exist_ok=True)

mask_image.save(os.path.join(savedir, "maske_img.png"))

Image.fromarray(x).save(os.path.join(savedir, "masked_mvout.png"))
print(f"Check {os.path.join(savedir, 'masked_mvout.png')}...")

prompt = opt.text + ", 3d model, white background, high quality" #  "a green frog"
generator = torch.Generator(device="cuda").manual_seed(0)



for guidance_scale in [7.5, 8.0, 9.0, 12.5]:

    output = pipe(
    prompt=prompt,
    negative_prompt=opt.neg_text,
    image=image,
    mask_image=mask_image,
    guidance_scale=guidance_scale,
    num_inference_steps=30,  # steps between 15 and 30 work well for us
    strength=0.99,  # make sure to use `strength` below 1.0
    generator=generator,
    ).images[0].resize((256,256))

    output = np.copy(np.asarray(output))
    # pdb.set_trace()
    alpha = mask_predictor(output)[:,:,3:]
    alpha[alpha>opt.alpha_t] = 255
    alpha[alpha<opt.alpha_t] = 0
    
    output = np.concatenate([output, alpha], -1)

    output = Image.fromarray(output)

    output.save(os.path.join(savedir, f"sss_{guidance_scale}.png"))






