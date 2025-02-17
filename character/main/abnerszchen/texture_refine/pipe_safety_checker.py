from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor
import time
import numpy as np
import os
from PIL import Image
from typing import Optional, Tuple, Union, List
from diffusers.utils import load_image

class SafetyCheckerPipe():
    def __init__(self, safety_path, feature_extractor_path, device="cuda"):
        assert os.path.exists(safety_path), print(f'can not find safety_path:{safety_path}')
        assert os.path.exists(feature_extractor_path), print(f'can not find feature_extractor_path:{feature_extractor_path}')
        self.device = device
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_path).to(self.device)
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(feature_extractor_path)


    def check_image_is_safe(self, in_img):
        """Determine if the picture is safe

        Args:
            in_img: image path

        Returns:
            safe flag, T/F
        """
        try:
            images = [load_image(in_img)]
            safety_checker_input = self.feature_extractor(images, return_tensors="pt").to(self.device)
            images_np = [np.array(img) for img in images]

            _, has_nsfw_concepts = self.safety_checker(
                images=images_np,
                clip_input=safety_checker_input.pixel_values.to(self.device),
            )

            is_safe = not has_nsfw_concepts[0]
        except Exception as e:
            print(f"ERROR: check_image_is_safe failed: ", e)
            return False      
        return is_safe

def test():
    safety_path = "/aigc_cfs_gdp/model/stable-diffusion-safety-checker/"
    feature_extractor_path = "/aigc_cfs_gdp/model/clip-vit-base-patch32/"    
    
    safety_checker = SafetyCheckerPipe(safety_path, feature_extractor_path)
    st = time.time()
    is_safe = safety_checker.check_image_is_safe("test/img2img_raw.jpg")
    print('is_safe ', is_safe, time.time() - st)

    st = time.time()
    is_safe = safety_checker.check_image_is_safe("test/sexy1.jpeg")
    print('is_safe ', is_safe, time.time() - st)
    breakpoint()
