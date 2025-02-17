import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def visualize_and_save_depth_histogram(depth_np, save_path, num_bins=50):
    # front = depth_np[depth_np>0]
    # plt.hist(front.ravel(), bins=num_bins)
    plt.hist(depth_np.ravel(), bins=num_bins)
    plt.xlabel('Depth')
    plt.ylabel('Frequency')
    plt.title('Depth Histogram')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def check_mask(gt_dir, render_dir):
    gt_depth_dir = os.path.join(gt_dir, 'depth')
    gt_color_dir = os.path.join(gt_dir, 'color')
    id_list = range(32)
    for id in id_list:
        gt_depth = os.path.join(gt_depth_dir, f"cam-{id:04d}.png")
        gt_img = os.path.join(gt_color_dir, f"cam-{id:04d}.png")
        render_depth = os.path.join(render_dir, f"depth_cam-{id:04d}.png")
        
        img_np = np.array(Image.open(gt_img).convert('RGB'))
        gt_depth_np = np.array(Image.open(gt_depth))
        gt_alpha = gt_depth_np[..., None] > 0
        
        render_depth_np =np.array(Image.open(render_depth))
        render_alpha = render_depth_np[..., None] > 0
        
        depth_diff = np.abs(gt_depth_np - render_depth_np)
        print('depth_diff ', depth_diff.shape, np.min(depth_diff), np.max(depth_diff), np.median(depth_diff))
        visualize_and_save_depth_histogram(depth_diff, os.path.join(render_dir, f'diff_hist-{id:04d}.png'))
        
        (Image.fromarray(img_np * gt_alpha)).save(os.path.join(render_dir, f'masked_gt-{id:04d}.png'))
        (Image.fromarray(img_np * render_alpha)).save(os.path.join(render_dir, f'masked_render-{id:04d}.png'))
        
        depth_diff = (depth_diff - np.min(depth_diff)) / (np.max(depth_diff) - np.min(depth_diff))
        depth_diff = (depth_diff * 255).astype(np.uint8)
        print('depth_diff new ', depth_diff.shape, np.min(depth_diff), np.max(depth_diff), np.median(depth_diff))
        (Image.fromarray(depth_diff )).save(os.path.join(render_dir, f'depth_diff-{id:04d}.png'))
            

    
    return

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('render_dir', type=str)
    parser.add_argument('gt_dir', type=str)
    args = parser.parse_args()

    
    check_mask(args.gt_dir, args.render_dir)
    return

if __name__ == "__main__":
    main()
