from PIL import Image
import matplotlib.pyplot as plt
from PIL.Image import Image as PILImage
import os
import torch
import numpy as np
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

class Rmgb2SegPipe():

    def __init__(self, model_path="/aigc_cfs_gdp/model/RMBG-2.0"):
        model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        model.to('cuda')
        model.eval()
        self.model = model
        
        
    def seg_img(self, input_image, cvt_gray_bg=False, out_image_path=None):
        # Data settings
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if isinstance(input_image, str):
            image = Image.open(input_image)
        elif isinstance(input_image, PILImage):
            image = input_image
        elif isinstance(input_image, np.ndarray):
            image = Image.fromarray(input_image)
        else:
            print("ERROR invalid input")
            return None
        
        input_images = transform_image(image).unsqueeze(0).to('cuda')

        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)

        if cvt_gray_bg:
            bg_v = 127
            background = Image.new("RGBA", image.size, (bg_v, bg_v, bg_v, 255))
            image = Image.alpha_composite(background, image).convert("RGB")        

        if out_image_path is not None:
            image.save(out_image_path)

        return image

if __name__ == "__main__":
    input_image_path = "mesh2image.png"
    seg_pipe = Rmgb2SegPipe()
    import time
    ts = time.time()
    seg_pipe.seg_img(input_image_path, cvt_gray_bg=True, out_image_path="no_bg_image.png")
    print('t=', time.time() - ts)


