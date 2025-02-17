import argparse
import os, pdb
import sys

import numpy as np
import json
import torch
from PIL import Image
from skimage.io import imread, imsave

import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil



SCANNET_COLOR_MAP_20 = {-1: (0., 0., 0.), 0: (174., 199., 232.), 1: (152., 223., 138.), 2: (31., 119., 180.), 3: (255., 187., 120.), 4: (188., 189., 34.), 5: (140., 86., 75.),
                        6: (255., 152., 150.), 7: (214., 39., 40.), 8: (197., 176., 213.), 9: (148., 103., 189.), 10: (196., 156., 148.), 11: (23., 190., 207.), 12: (247., 182., 210.), 
                        13: (219., 219., 141.), 14: (255., 127., 14.), 15: (158., 218., 229.), 16: (44., 160., 44.), 17: (112., 128., 144.), 18: (227., 119., 194.), 19: (82., 84., 163.)}



def load_image(image_path, i_imgs):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    w, h = image_pil.size
    image_pil = image_pil.crop((w//num_imgs * i_imgs, 0, w//num_imgs * (i_imgs+1), h))

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, with_logits=True, device="cpu", chosen=None):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    # pdb.set_trace()
    filt_mask = logits_filt.max(dim=1)[0]
    val, ind = torch.sort(filt_mask)
    max_num = 5
    logits_filt = logits_filt[ind][-max_num:]
    boxes_filt = boxes_filt[ind][-max_num:]


    # logits_filt = logits_filt[filt_mask]  # num_filt, 256
    # boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    # logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)

    # pdb.set_trace()
    mid_split = np.where(np.array(tokenized['input_ids']) == 1012)[0][0] # not "."

    # if boxA contains boxB, then delete boxA
    info = {}
    idxs = np.arange(boxes_filt.shape[0])

    for idx in idxs:
        logit = logits_filt[idx]
        box = boxes_filt[idx]
        if args.gsatype == 1:
            cand_list = [min(logit[1:mid_split])]
        else:
            cand_list = [min(logit[1:mid_split]), min(logit[mid_split+1:len(tokenized['input_ids'])-2])]
        for text_threshold in cand_list:
            pred_phrase = get_phrases_from_posmap(logit >= text_threshold, tokenized, tokenlizer)
            if text_prompt_dict.get(pred_phrase, None) is not None:
                
                # if info.get(pred_phrase, None) is None:
                #     info[pred_phrase] = []
                # info[pred_phrase].append((box, f"({str(logit.max().item())[:6]})"))

                if info.get(pred_phrase, None) is None:
                    info[pred_phrase] = [(box, f"({str(logit.max().item())[:6]})")]
                elif float(info[pred_phrase][0][1][1:-1])<logit.max().item():
                    info[pred_phrase] = [(box, f"({str(logit.max().item())[:6]})")]

        # pdb.set_trace()

        # if pred_phrase not in info:
        #     info[pred_phrase] = box 
        # else:
        #     pre_box = info[pred_phrase]
        #     if box[0] - box[2]/2 > pre_box[0] - pre_box[2]/2 and box[1] - box[3]/2 > pre_box[1] - pre_box[3]/2 and box[0] + box[2]/2 < pre_box[0] + pre_box[2]/2 and box[1] + box[3]/2 < pre_box[1] + pre_box[3]/2:
        #         info[pred_phrase] = box 
    
    if args.debug:
    # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if len(pred_phrase) == 0:
                continue
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases
    
    else:
        pred_phrases = []
        pred_boxes = []
        # pdb.set_trace()
        for k,v in info.items():
            for i in range(len(v)):
                if chosen is None or k+v[i][1] in chosen:
                # if k+v[i][1] == "a band(0.1679)" or k+v[i][1]=="a scroll(0.2624)":
                    pred_phrases.append(k+v[i][1])
                    pred_boxes.append(v[i][0])

            # pred_phrases.append(k+v[1])
            # pred_boxes.append(v[0])
        if len(pred_boxes)>0:
            pred_boxes = torch.stack(pred_boxes)
        return pred_boxes, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):

    idxs = np.arange(len(mask_list))
    sorted_idxs = sorted(idxs, key=(lambda x: mask_list[x].sum()), reverse=True)

    mask_img = np.zeros(mask_list.shape[-2:]) - 1
    mask_img_color = np.ones([mask_list.shape[-2], mask_list.shape[-1], 3])
    # pdb.set_trace()
    # pdb.set_trace()
    for idx in sorted_idxs:
        if label_list[idx].split("(")[0] in text_prompt_dict.keys():
            val = text_prompt_dict[label_list[idx].split("(")[0]]
            m = mask_list[idx][0] # (h,w)
            color_mask = np.array(SCANNET_COLOR_MAP_20[val]) / 255.0
            mask_img_color[m] = color_mask
            mask_img[m] = val

    return mask_img, mask_img_color



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
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)
        image = self.interface([image])[0]
        image = np.array(image)
        return image


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument(
        "--config", type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, default="../ckpt/groundingdino_swint_ogc.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default="../ckpt/sam_hq_vit_h.pth", help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--debug", action="store_true", help="debug"
    )
    parser.add_argument("--oname", type=str, required=True)
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")

    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")

    parser.add_argument("--keepori", action="store_true")
    parser.add_argument("--gsatype", type=int, required=True)
    args = parser.parse_args()

    args.use_sam_hq = True
    assert args.use_sam_hq
    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    text_prompt = args.text_prompt
    
    
    if args.gsatype == 1:
        image_path = f"../output/mvimgs/{args.oname}/inpaint/0.png"
        output_dir = f"../output/mvimgs/{args.oname}/inpaint/gdsam_output"
    elif args.gsatype == 0:
        image_path = f"../output/mvimgs/{args.oname}/0.png"
        output_dir = f"../output/mvimgs/{args.oname}/gdsam_output"
    else:
        raise NotImplementedError

    device = args.device

    num_imgs = 16
    text_prompt_dict = {}
    text_prompt_list = text_prompt.split(".")[:-1]
    text_prompt_list = [x.strip() for x in text_prompt_list]
    for i, text in enumerate(text_prompt_list):
        text_prompt_dict[text] = i

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))


    # plt.figure(figsize=(10, 10))
    gdsam_mask = []
    gdsam_mask_color = []
    plt.figure(figsize=(10, 10))

    save_base = os.path.dirname(image_path)

    keepori = []
    flipori = []
    chosen = {}
    if args.keepori:
        orisems = np.load(os.path.join(save_base, f'mvout.npy'))
        orirgbs = cv2.imread(os.path.join(save_base, 'fusemask.png'))
        keepori = [4,5,6,7,8,9,10,11,12,13]
        flipori = []
        chosen = {
            # 0: ["a black camera(0.3950)", "a brown leather case(0.6281)"],
            # 1: ["a black camera(0.3872)", "a brown leather case(0.3536)"],
            # 2: ["a black camera(0.2816)", "a brown leather case(0.3139)"],
            # 3: ["a black camera(0.2657)", "a brown leather case(0.3275)"],
            # 4: ["a couch(0.7606)", "a cloth(0.2643)"],
            # 5: ["a couch(0.4099)", "a cloth(0.3214)"],
            # 6: ["a couch(0.4708)", "a cloth(0.4901)"],
            # 7: ["a couch(0.4708)", "a cloth(0.4901)"],
            # # # 8: ["a chair(0.4708)", "a pillow(0.4901)"],
            # 9: ["a couch(0.4177)", "a cloth(0.2758)"],
            # # 10: ["a couch(0.4708)", "a cloth(0.4901)"],
            # 11: ["a black camera(0.8601)", "a brown leather case(0.3972)"],
            # 12: ["a black camera(0.6605)", "a brown leather case(0.1373)"],
            # 13: ["a black camera(0.4538)", "a brown leather case(0.3238)"],
            # 14: ["a black camera(0.3569)", "a brown leather case(0.2556)"],
            15: ["a black camera(0.3534)", "a brown leather case(0.2702)"],

        }

    for i_imgs in tqdm(range(num_imgs)):

        if i_imgs in keepori:
            mask_data = orisems[:,256*i_imgs:256*(i_imgs+1)]
            mask_data_color = orirgbs[:,256*i_imgs:256*(i_imgs+1),[2,1,0]] / 255.
        elif i_imgs in flipori:
            mask_data = orisems[:,256*i_imgs:256*(i_imgs+1)]
            mask_data_color = orirgbs[:,256*i_imgs:256*(i_imgs+1),[2,1,0]] / 255.

            mask0 = mask_data == 0
            mask1 = mask_data == 1
            mask_data[mask0] = 1
            mask_data[mask1] = 0
            mask_data_color[mask0] = np.array(SCANNET_COLOR_MAP_20[1]) / 255.0
            mask_data_color[mask1] = np.array(SCANNET_COLOR_MAP_20[0]) / 255.0
        
        else:
            # load image
            image = load_image(image_path, i_imgs)
            
            # visualize raw image
            # image_pil.save(os.path.join(output_dir, f"raw_image_{i_imgs:02d}.jpg"))
            # run grounding dino model
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, text_prompt, device=device, with_logits=True, chosen=chosen.get(i_imgs,None)
            )
            print(i_imgs, ":", pred_phrases)
            # pdb.set_trace()
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            H, W = image.shape[:2]
            assert W % num_imgs == 0
            W = W // num_imgs
            image = image[:, W*i_imgs:W*(i_imgs+1)]
            predictor.set_image(image)

            for i in range(len(boxes_filt)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]
            
            boxes_filt = boxes_filt.cpu()
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

            masks, _, _ = predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(device),
                multimask_output = False,
            )

            # draw output image
            
            plt.imshow(image)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box, label in zip(boxes_filt, pred_phrases):
                show_box(box.numpy(), plt.gca(), label)

            plt.axis('off')
            plt.savefig(
                os.path.join(output_dir, f"grounded_sam_output_{i_imgs:02d}.jpg"),
                bbox_inches="tight", dpi=300, pad_inches=0.0
            )
            plt.clf()

            masks = masks.detach().cpu().numpy()

            mask_data, mask_data_color = save_mask_data(output_dir, masks, boxes_filt, pred_phrases)

        gdsam_mask.append(mask_data)
        gdsam_mask_color.append(mask_data_color)
    
    gdsam_mask = np.concatenate(gdsam_mask, axis=1)
    gdsam_mask_color = (np.concatenate(gdsam_mask_color, axis=1) * 255).astype(np.uint8)

    
    np.save(os.path.join(save_base, f'mvout.npy'), gdsam_mask.astype(np.int16))
    cv2.imwrite(os.path.join(save_base, 'fusemask.png'), gdsam_mask_color[:,:,[2,1,0]])
    shutil.copyfile(image_path, os.path.join(save_base, f'mvout.png'))

