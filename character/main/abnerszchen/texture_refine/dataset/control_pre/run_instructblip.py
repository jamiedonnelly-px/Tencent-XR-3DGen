from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
import argparse
import PIL
import requests
import os
import time

def load_rgba_as_rgb(img_path, res=None):
    """load with RGBA and convert to RGB with white backgroud, if is RGB just return

    Args:
        img_path: _description_

    Returns:
        PIL.Image [h, w, 3]
    """
    img = PIL.Image.open(img_path)
    if img.mode == "RGBA":
        background = PIL.Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = PIL.Image.alpha_composite(background, img).convert("RGB")
    if res is not None and isinstance(res, int):
        img = img.resize((res, res))
    return img  

# vicuna-7b, model 31G, run oom, run cpu
# flan-t5-xl, model 16G, run 17G

class ImageCaption:
    def __init__(self, model_path="/aigc_cfs/model/instructblip-flan-t5-xl", device="cuda"):
        assert os.path.exists(model_path), model_path
        self.model, self.processor = self.load_instructblip(model_path, device)

    # def load_instructblip(model_path="/aigc_cfs/model/instructblip-vicuna-7b"):
    def load_instructblip(self, model_path, device):
        assert os.path.exists(model_path), model_path
        model = InstructBlipForConditionalGeneration.from_pretrained(model_path)
        processor = InstructBlipProcessor.from_pretrained(model_path)

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
        
        model.to(device)
        print('load_model_done')
        return model, processor

    def query_img_caption(self, in_img_path, prompt = ""):
        if not os.path.exists(in_img_path):
            return ""
        
        image = load_rgba_as_rgb(in_img_path)
        
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=60,  # clip < 77
            min_length=6,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.5,
            temperature=1,
        )
        get_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        del outputs
        return get_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="render est obj list")
    parser.add_argument(
        "--in_img_path",
        type=str,
        default="/aigc_cfs/Asset/designcenter/clothes/render_part2/render_data/top/render_data/CN_OTR_9_F_A/Tops_F_A_CN_OTR_9_F_A_CN_OTR_9_fbx2020_output_512_MightyWSB/color/cam-0011.png",
    )
    parser.add_argument(
        "--prompt", type=str, default="Describe the dress, describe the color and style"
    )
    args = parser.parse_args()

    in_img_path = args.in_img_path
    prompt = args.prompt
    caption_cls = ImageCaption()
    
    q_st = time.time()
    generated_text = caption_cls.query_img_caption(in_img_path, prompt=prompt)
    q_time = time.time() - q_st
    print('q_time', q_time)
    # breakpoint()
    print(generated_text)
    