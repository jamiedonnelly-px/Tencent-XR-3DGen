import os
import numpy as np
import torch
import time
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageSegmentation, pipeline
from torchvision.transforms.functional import normalize
from PIL.Image import Image as PILImage

# pipe = pipeline("image-segmentation", model="/aigc_cfs_gdp/neoshang/models/RMBG-1.4", trust_remote_code=True, local_files_only=True)
# pillow_mask = pipe(image_path, return_mask = True) # outputs a pillow mask
# pillow_image = pipe(image_path) # applies mask on input and returns a pillow image


def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    # orig_im_size=im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image


def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array


class RmbgMain():

    def __init__(self) -> None:
        model_dir1 = "/aigc_cfs/model/RMBG-1.4/"
        model_dir2 = "/aigc_cfs_gdp/neoshang/models/RMBG-1.4"
        if os.path.exists(model_dir1):
            model_dir = model_dir1
        elif os.path.exists(model_dir2):
            model_dir = model_dir2
        else:
            exit("==Wrong: rmbg1_4 dir not exists!")

        print("load rmbg1.4 from model_dir")
        model = AutoModelForImageSegmentation.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self.model = model
        self.device = device

    def rmbg(self, image):
        """_summary_

        Args:
            image: _description_

        Returns:
            [h,w,4] in numpy [0,255]
        """
        if isinstance(image, str) and os.path.exists(image):
            image = Image.open(image)
        elif isinstance(image, str) and not os.path.exists(image):
            print(f"{image} not exists")
            return

        orig_im_array = np.array(image)
        orig_im_size = orig_im_array.shape[0:2]
        model_input_size = [1024, 1024]
        image = preprocess_image(orig_im_array, model_input_size).to(self.device)

        # inference
        result = self.model(image)
        # post process
        mask = postprocess_image(result[0][0], orig_im_size)
        rgba_array = np.concatenate([orig_im_array, np.array(mask)[..., None]], axis=-1)
        return rgba_array


    def infer_seg(self, input, output_path=None):
        """seg and return numpy uint8 [0, 255]

        Args:
            input: str/PIL/numpy(in [0, 1] or [0, 255])
            output_path: _description_. Defaults to None.

        Returns:
            output_np: numpy uint8 [h, w, 4] [0, 255]
        """
        if isinstance(input, str):
            input_pil = Image.open(input)
            input_np = np.array(input_pil)
        elif isinstance(input, PILImage):
            input_np = np.array(input)
        elif isinstance(input, np.ndarray):
            input_np = input
        else:
            print("ERROR invalid input")
            return None

        # need as uint8 [0,. 255]
        if input_np.max() < 1.1:
            input_np = (input_np * 255.0).round().astype("uint8")

        ts = time.time()
        output_np = self.rmbg(input_np)
        tuse = time.time() - ts
        print('tuse ', tuse)

        if output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            output_pil = Image.fromarray(output_np)
            output_pil.save(output_path)

        return output_np

    def infer_image_nps(self, image_nps):
        """infer nps to rgba np

        Args:
            image_nps: [nv, 1024, 1024, 3] np in [0, 1]

        Returns:
            seg_nps [4, h, w, 4] in [0, 1] np
        """
        # TODO multi 
        ts_all = time.time()
        seg_nps = [self.infer_seg(image_np) / 255.0 for image_np in image_nps]
        tuse_all = time.time() - ts_all
        print('tuse_all ', tuse_all)
        
        print('debug ', seg_nps[0].min(), seg_nps[0].max(), seg_nps[0][..., -1].max(), seg_nps[0].shape)
        return seg_nps

if __name__ == "__main__":
    rmbg_interface = RmbgMain()
    
    ts = time.time()
    # in_path = "/aigc_cfs_gdp/sz/result/pipe_test/82941b42-a516-4fbc-b3ed-9bcc94754bbf/texbakeinpaint/infer_image.png"
    # rgba_array = rmbg_interface.rmbg(in_path)
    
    in_npy_path = "/aigc_cfs_gdp/sz/result/pipe_test/2c2deb45-5cfc-44a6-a66a-2ac08e9b0d53/d2rgb/out/imgsr/color.npy"
    in_np = np.load(in_npy_path).transpose(0, 2, 3, 1) / 255.0
    seg_nps = rmbg_interface.infer_image_nps(in_np)
    
    tuse = time.time() - ts
    print('seg_nps0 ', seg_nps[0].shape, seg_nps[0].max(), tuse)
    # Image.fromarray(rgba_array).save("debug.png")
