import concurrent.futures
from rembg import remove
from PIL import Image
from PIL.Image import Image as PILImage
import numpy as np
import os
import time


class SimpleSegPipe():

    def __init__(self):
        pass

    def infer_seg(self, input, output_path=None):
        """seg and return numpy uint8 [0, 255]

        Args:
            input: str/PIL/numpy
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
        output_np = remove(input_np)
        tuse = time.time() - ts
        print('tuse ', tuse)

        if output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            output_pil = Image.fromarray(output_np)
            output_pil.save(output_path)

        return output_np

    def seg_to_gray_bg_pil(self, input, bg_v=128):
        # numpy uint8 [h, w, 4] [0, 255]
        output_np = self.infer_seg(input)
        img = Image.fromarray(output_np, model="RGBA")
        background = Image.new("RGBA", img.size, (bg_v, bg_v, bg_v, 255))
        img = Image.alpha_composite(background, img).convert("RGB")
        return img
    
    def infer_image_nps(self, image_nps):
        # TODO multi 
        ts_all = time.time()
        seg_nps = [self.infer_seg(image_np) for image_np in image_nps]
        tuse_all = time.time() - ts_all
        print('tuse_all ', tuse_all)
        
        return seg_nps

    
def split_image(image, rows, cols):
    width, height = image.size
    block_width = width // cols
    block_height = height // rows

    images = []
    for i in range(rows):
        for j in range(cols):
            left = j * block_width
            upper = i * block_height
            right = (j + 1) * block_width
            lower = (i + 1) * block_height
            sub_image = image.crop((left, upper, right, lower))
            images.append(sub_image)

    return images

if __name__ == "__main__":

    use_sdxl = True
    use_mview = True
    seed = 1234

    seg_pipe = SimpleSegPipe()

    input_path = "/aigc_cfs/xibinsong/code/zero123plus_control/results/gray_18000_75_inpaint_1e_5_guidance_scale_3.5_conditioning_scale_0.75/c4e3a8fef092469294f0a1e94f1f811f/res.png"  #2s

    seg_pipe.seg_to_gray_bg_pil(input_path).save(f"output_gray.png")
    breakpoint()
    
    in_pil = Image.open(input_path)
    pils = split_image(in_pil, 2, 2)
    seg_nps = seg_pipe.infer_image_nps(pils)
    for i, seg_np in enumerate(seg_nps):
        Image.fromarray(seg_np).save(f"output_split_{i}.png")
    
    input = np.array(in_pil)
    output_np0255 = seg_pipe.infer_seg(input, 'output_np0255.png')
    print('output_np0255 ', output_np0255.shape, output_np0255.dtype, output_np0255.max())

    input = input / 255.0
    output_np01 = seg_pipe.infer_seg(input, 'output_np01.png')
    print('output_np01 ', output_np01.dtype, output_np01.max())

    output_nppath = seg_pipe.infer_seg(input_path, 'output_path.png')

    equal = np.array_equal(output_np0255, output_np01) and np.array_equal(output_np0255, output_nppath)

    print("Are all three arrays equal?", equal)
