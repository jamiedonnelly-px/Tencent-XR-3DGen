import os
import argparse
import torch
import numpy as np
import nvdiffrast.torch as dr
from PIL import Image
import imageio

from render.render_obj import render_obj_with_in_kd, load_obj_and_pose, save_render
from render.render_mesh import render_depth_views, save_depths

import matplotlib.pyplot as plt
def visualize_and_save_depth(depth_np, save_path):
    dmin, dmax = np.min(depth_np), np.max(depth_np)
    depth = (depth_np - dmin) / (dmax - dmin)
    
    imageio.imwrite(save_path, np.clip(np.rint(depth * 255.0), 0, 255).astype(np.uint8))
    # plt.imshow(depth, cmap='gray')
    # plt.colorbar(label='Depth')
    # plt.title('Depth Map')
    # plt.savefig(save_path, bbox_inches='tight')
    # plt.close()
def visualize_and_save_depth_histogram(depth_np, save_path, num_bins=50):
    # front = depth_np[depth_np>0]
    # plt.hist(front.ravel(), bins=num_bins)
    plt.hist(depth_np.ravel(), bins=num_bins)
    plt.xlabel('Depth')
    plt.ylabel('Frequency')
    plt.title('Depth Histogram')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    

def main():
    parser = argparse.ArgumentParser(description='render obj with setting pose')
    parser.add_argument('in_obj', type=str)
    parser.add_argument('in_kd', type=str)
    parser.add_argument('in_pose_json', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument("--lrm_mode", help="use lrm mode. temp. need remove in future!TODO", action="store_true")
    parser.add_argument('--render_res', type=int, default=512)
    parser.add_argument('--max_mip_level', type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Render kd
    kd_pil = Image.open(args.in_kd).convert('RGB')
    color, alpha = render_obj_with_in_kd(args.in_obj, kd_pil, args.in_pose_json, lrm_mode=args.lrm_mode, render_res=args.render_res, max_mip_level=args.max_mip_level)
    save_render(color, alpha, out_path=os.path.join(args.out_dir,'render.jpg'), bg_type="white", row=2)
    
    # Render depth
    raw_mesh, frames = load_obj_and_pose(args.in_obj, args.in_pose_json, lrm_mode=args.lrm_mode)
    vtx_pos, pos_idx = raw_mesh.v_pos, raw_mesh.t_pos_idx
    glctx = dr.RasterizeCudaContext()
    depth = render_depth_views(glctx, vtx_pos, pos_idx, frames['mvp'], frames['w2c'], args.render_res)
    save_depths(depth, args.out_dir, cam_name_list=frames['cam_name_list']) 
    
    
    # # Depth Debug
    # # dmin, dmax = torch.min(depth), torch.max(depth)
    # # depth = (depth - dmin) / (dmax - dmin)
    
    # # print('depth ', depth.shape)
    # # print('depth median ', torch.median(depth), torch.min(depth), torch.max(depth))
    
    
    # depth_mm = depth * 1000.
    # depth_np = depth_mm.detach().cpu().numpy().astype(np.uint16)
    # depth_0 = depth_np[0, ..., 0]
    # print('depth depth_0 raw ', np.median(depth_0), np.min(depth_0), np.max(depth_0))
    # front_np = depth_0[depth_0 > 0]
    # print('debug depth_0 front_np ', front_np.shape, np.median(front_np), np.min(front_np), np.max(front_np))
    # dimg = Image.fromarray(depth_0)
    # print('dimg ', dimg)
    # dimg.save(os.path.join(args.out_dir, 'raw_depth.png'))
    
    # # depth_np_scaled = (depth_np * 65535).astype(np.uint16)
    # # invalid_mask = depth_np_scaled > 20000
    # # depth_np_scaled[invalid_mask] = 0
    
    # # depth_image = Image.fromarray(depth_np_scaled[0, ..., 0])
    
    # # depth_image.save(os.path.join(args.out_dir, 'raw_depth.png'))
    # visualize_and_save_depth(depth_0, os.path.join(args.out_dir, 'vis_depth.png'))
    # visualize_and_save_depth_histogram(depth_0, os.path.join(args.out_dir, 'vis_depth_hist.png'))
    # # # imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8))
    # # visualize_and_save_depth(load_depth(os.path.join(args.out_dir, 'raw_depth.png')), os.path.join(args.out_dir, 're_vis_depth.png'))
    
    # debug_depth_path = '/apdcephfs_cq8/share_2909871/shenzhou/data/tex_refine/debug/blender/render_data/2538418299181565750/vroid_obj_0_2538418299181565750_manifold_full_output_512_MightyWSB/depth/cam-0089.png'
    # depth = Image.open(debug_depth_path) # PIL.PngImagePlugin.PngImageFile image mode=I
    # debug_d_np = np.array(depth)
    # print('debug Image.open(debug_depth_path) ', depth)  
    # print('debug debug_d_np ', debug_d_np.shape, np.median(debug_d_np), np.min(debug_d_np), np.max(debug_d_np))
    # front_np = debug_d_np[debug_d_np > 0]
    # print('debug front_np ', front_np.shape, np.median(front_np), np.min(front_np), np.max(front_np))
    
    # visualize_and_save_depth_histogram(debug_d_np, os.path.join(args.out_dir, 'debug_vis_depth_hist.png'))
    # visualize_and_save_depth(debug_d_np, os.path.join(args.out_dir, 'debug_vis_depth.png'))
    
    # depth.save(os.path.join(args.out_dir, 'debug_resave.png'))
    # new_res = 64
    # new_size = (new_res, new_res)
    # nn = depth.resize(new_size, Image.NEAREST)
    # bicubic = depth.resize(new_size, Image.BICUBIC)
    # nn.save(os.path.join(args.out_dir, 'debug_nn.png'))
    # bicubic.save(os.path.join(args.out_dir, 'debug_bicubic.png'))
    # nn_np = np.array(nn)
    # bicubic_np = np.array(bicubic)
    # visualize_and_save_depth(nn_np, os.path.join(args.out_dir, f'debug_vis_depth_nn_{new_res}.png'))
    # visualize_and_save_depth(bicubic_np, os.path.join(args.out_dir, f'debug_vis_depth_bicubic_{new_res}.png'))
    
    # print('debug nn_np ', nn_np.shape, np.median(nn_np), np.min(nn_np), np.max(nn_np))
    # front_np = nn_np[nn_np > 0]
    # print('debug nn_np front_np ', front_np.shape, np.median(front_np), np.min(front_np), np.max(front_np))
    

    # print('debug bicubic_np ', bicubic_np.shape, np.median(bicubic_np), np.min(bicubic_np), np.max(bicubic_np))
    # front_np = bicubic_np[bicubic_np > 0]
    # print('debug bicubic_np front_np ', front_np.shape, np.median(front_np), np.min(front_np), np.max(front_np))
    

    # ref_path = '/aigc_cfs_2/sz/proj/tex_cq/test/stormtrooper_depth.png'
    # depth_ref = Image.open(ref_path) # PIL.PngImagePlugin.PngImageFile image mode=I
    # print('depth_ref ', depth_ref)
    # ref_d_np = np.array(depth_ref)
    # visualize_and_save_depth_histogram(ref_d_np, os.path.join(args.out_dir, 'ref_vis_depth_hist.png'))
    # visualize_and_save_depth(ref_d_np, os.path.join(args.out_dir, 'ref_vis_depth.png'))
        
        
    # Done.
    print("Done.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
