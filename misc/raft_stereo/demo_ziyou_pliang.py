import sys
import gc
sys.path.append('core')
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2


DEVICE = 'cuda'

def load_image(imfile):
    img = Image.open(imfile)
    # img = img.resize((662,492))
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    path = os.listdir(args.img_main_folder)
    for p in path:
        if os.path.isdir(args.img_main_folder+"/"+p):
            print(p)
            left_name = args.img_main_folder+"/"+p+"/left/*.png"
            right_name = args.img_main_folder+"/"+p+"/right/*.png"
            output_directory = args.img_main_folder+"/"+p+"/output"
            if not os.path.exists(output_directory):
                output_directory = Path(output_directory)
                output_directory.mkdir(exist_ok=True)
            output_directory = args.img_main_folder+"/"+p+"/output"
            with torch.no_grad():
                left_images = sorted(glob.glob(left_name, recursive=True))
                right_images = sorted(glob.glob(right_name, recursive=True))
                print(f"Found {len(left_images)} images. Saving files to {output_directory}/")
                
                for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
                    # print(imfile1)
                    # print(imfile2)
                    image1 = load_image(imfile1)
                    image2 = load_image(imfile2)
                    # print(image1.shape)
                    # print(image2.shape)

                    padder = InputPadder(image1.shape, divis_by=32)
                    image1, image2 = padder.pad(image1, image2)

                    _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
                    file_stem = imfile1.split('/')[-1]
                    # print(output_directory +"/"+f"{file_stem}")
                    if args.save_numpy:
                        np.save(output_directory+"/"+ f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
                    plt.imsave(output_directory +"/"+f"{file_stem}", -flow_up.cpu().numpy().squeeze(), cmap='jet')
                
                disp_images = sorted(glob.glob(output_directory+"/*.png", recursive=True))
                print("video make %d %d %d"%(len(left_images),len(right_images),len(disp_images)))
                fourcc = cv2.VideoWriter_fourcc(*"XVID") 
                size = (1280,480)
                videoWrite = cv2.VideoWriter(args.img_main_folder+"/"+p+"/"+p+".avi",fourcc,30,size)

                for (imfile1, imfile2,disp_name) in tqdm(list(zip(left_images, right_images,disp_images))):
                    img1 = cv2.imread(imfile1)
                    disp = cv2.imread(disp_name)
                    img_merge=cv2.hconcat([img1,disp])
                    # print(img_merge.shape)
                    videoWrite.write(img_merge)
                videoWrite.release() 
                    


    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('--img_main_folder', help="path to all first (left) frames", default="/apdcephfs_cq2/share_1615605/xiaqiangdai/ft_local/data_scene_xinye")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)
