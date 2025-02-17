from render_bake_utils import dilate_masks, Renderer, cvt_torch, poisson_blend
import cv2, torch, numpy as np, os, uuid, json
from call_inpaint import inference_inpaint_flux, inference_inpaint_sdxl
from shutil import copy2
from pdb import set_trace as st

class DotDict:
    def __init__(self, dictionary):
        self._dict = dictionary
    
    def __getattr__(self, key):
        try:
            return self._dict[key]
        except KeyError:
            raise AttributeError(f"'DictWrapper' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        # Allow setting the internal _dict attribute without adding to the dictionary
        if key == "_dict":
            super().__setattr__(key, value)
        else:
            self._dict[key] = value

def get_view_string(azimuth, elevation):
    ret = []
    azimuth = azimuth % 360
    elevation = elevation % 360
    if azimuth >= 350 or azimuth <= 10:
        ret.append("front")
    elif azimuth >= 160 and azimuth <= 200:
        ret.append("back")
    else:
        ret.append("side")
    if elevation >= 20:
        ret.append("top")
    elif elevation <= -20:
        ret.append("bottom")
    ret.append("view")
    return " ".join(ret)
       
def make_bake_alpha(depth, weight, exp, per_view_weight=1, erode=0):
    '''
    Inputs
        - depth: (n_views, img_res, img_res, 1), pytorch
        - weight: (n_views, img_res, img_res, 1), pytorch
        - exp: float scalar
        - per_view_weight: (n_views), list ot numpy or pytorch
        - erode: unsigned int scalar
        
    Returns:
        - weights: (n_views, img_res, img_res, 1), pytorch
    '''

    # detect depth discontinuities, i.e. occlusion boundaries
    depth_map_uint8 = depth.cpu().numpy().astype(np.uint8) # (n_views, img_res, img_res, 1)
    depth_edge = [(cv2.Canny(d, 10, 40) > 0) for d in depth_map_uint8]
    depth_edge = dilate_masks(*depth_edge, iterations=erode)
    depth_edge = (torch.from_numpy(depth_edge).cuda() > 0).float().unsqueeze(-1) # binary (n_views, img_res, img_res, 1)

    weights = weight * (1-depth_edge) # remove pixels on occlusion boundaries
    # apply weights
    weights = weights ** exp * cvt_torch(per_view_weight, device=depth.device).reshape(-1,1,1,1)
    return weights

def make_gdp_accessible(path):
    if not os.path.abspath(path).startswith("/aigc_cfs_gdp"):
        copy_path = os.path.join("/aigc_cfs_gdp", "tmp", f"{uuid.uuid1()}{os.path.splitext(path)[1]}")
        os.makedirs(os.path.dirname(copy_path), exist_ok=True)
        copy2(path, copy_path)
        return copy_path
    else:
        return path

def copy_mesh_and_mtl(path, dst_dir):
    base_name = os.path.basename(path)
    os.makedirs(dst_dir, exist_ok=True)
    copy2(path, os.path.join(dst_dir, base_name))
    copy2(path[:-3]+"mtl", os.path.join(dst_dir, base_name[:-3]+"mtl"))

def blend_textures(textures_weights):
    '''
    Inputs:
        - textures_weights: (n_views, tex_res, tex_res, 4), pytorch, last channel is weights
        
    Returns:
        - texture_weights: (1, tex_res, tex_res, 4), pytorch, last channel is summed weights clipped to [0,1]
    '''
    channels = textures_weights.shape[-1]
    texture, weights = torch.split(textures_weights, (channels-1, 1), dim=-1)
    total_weight = weights.sum(0, keepdim=True)
    texture = (texture * weights).sum(0, keepdim=True) / (total_weight + 1e-10)
    return torch.cat((texture, total_weight.clip(0,1)), dim=-1)

def make_transform(offset):
    transform = np.eye(4)
    transform[:3] *= 0.99 - np.abs(offset)
    transform[2,3] = offset
    return transform

def save_args(args, output_dir):
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(
            {k: v for k, v in vars(args).items()},
            f,
            indent=4
        )


def bake_refine(args):
    
    images = np.load(args.images_npy) 
    
    assert args.bake_weighting_method in ("view_cosine", "sqrtinv_tex_area"), f"unrecognised baking weight method {args.bake_weighting_method}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    save_args(args, args.output_dir)

    images = torch.from_numpy(images).cuda().float().permute(0,2,3,1) # [n_views, img_res, img_res, channels]
    image_resolution = images.shape[1]

    print(f"[Init] setting up scene")
    
    # set up renderer
    renderer = Renderer(image_resolution, args.texture_resolution, world_orientation="y-up")
    renderer.set_object(args.input_obj_file, bound=args.obj_bound, orientation="y-up")
    
    # bake views from npy
    renderer.set_cameras(azimuths=args.cam_azimuths, elevations=args.cam_elevations, dists=args.cam_distances, camera_type=args.camera_type, zooms=1.0, near=1e-1, far=1e1)

    # render normal and depth
    depth, mask = renderer.render_depth("absolute", normalize=(255,50), bg=0) # (n_views, img_res, img_res, 1)
    if args.bake_weighting_method == "view_cosine":
        view_weight, _ = renderer.render_view_cos("vertex")
    elif args.bake_weighting_method == "sqrtinv_tex_area":
        view_weight, _ = renderer.render_texture_area(antialias=False, inverse=True, return_singulars=False)
        view_weight = view_weight ** 0.5

    weights = make_bake_alpha(depth, view_weight, args.bake_init_exp, args.bake_init_view_weight, args.bake_init_erode) # (n_views, img_res, img_res, 1)

    print(f"[Init] baking initial views")
    
    # bake
    image_weights = torch.cat((images, weights), dim=-1)
    if args.bake_method == "raycast":
        texture_weights = renderer.bake_textures_raycast(image_weights, interpolation="bicubic", inpaint=False) # (n_views, tex_res, tex_res, 1)
    elif args.bake_method.startswith("mipmap"):
        mipmap_level = int(args.bake_method[6:]) if len(args.bake_method) > 6 else None
        texture_weights = renderer.bake_textures(image_weights, max_mip_level=mipmap_level) # (n_views, tex_res, tex_res, 1)
    else:
        raise ValueError(f"unknown baking method {args.bake_method}")
    texture_weight = blend_textures(texture_weights) # (1, tex_res, tex_res, 4)
    texture, total_weight = torch.split(texture_weight, (3,1), dim=-1) # (1, tex_res, tex_res, 3), (1, tex_res, tex_res, 1)

    # inpaint missing regions
    texture = renderer.inpaint_textures(texture, (total_weight<=1e-3), inpaint_method="laplace") # (1, tex_res, tex_res, 3)
    texture_rgba = torch.cat((texture, total_weight), dim=-1) # (1, tex_res, tex_res, 4)

    # visulization
    if args.write_intermediates:
        rendered, _ = renderer.sample_texture(texture, max_mip_level=0) # (1, img_res, img_res, 3)
        cv2.imwrite(os.path.join(args.output_dir, "inputs.png"), torch.cat(list(images), dim=1).clip(0,255).cpu().numpy()[...,::-1])
        cv2.imwrite(os.path.join(args.output_dir, "init_bake.png"), torch.cat(list(rendered), dim=1).clip(0,255).cpu().numpy()[...,::-1])
    
    # if inpaint
    for i_stage, inpaint_args in enumerate(args.inpaint_stages):
        
        inpaint_args = DotDict(inpaint_args)
        texture_rgba[...,-1] *= inpaint_args.texture_attenuation
        
        if inpaint_args.n_inpaint_views > 0:
            
            if not("azimuths" in inpaint_args.inpaint_camera_config and "elevations" in inpaint_args.inpaint_camera_config and "dists" in inpaint_args.inpaint_camera_config):
                
                
                # automatically choose cameras for inpaint
                all_azimuths, all_elevations, all_dists = np.meshgrid(inpaint_args.inpaint_camera_config['candidate_azimuths'], \
                                                                inpaint_args.inpaint_camera_config['candidate_elevations'], 
                                                                inpaint_args.inpaint_camera_config['candidate_dists'], indexing="ij")
                all_azimuths = all_azimuths.flatten()
                all_elevations = all_elevations.flatten()
                all_dists = all_dists.flatten()
                renderer.set_cameras(all_azimuths, all_elevations, all_dists, **inpaint_args.inpaint_camera_config)
                renderer.modify_res(tex_res=512) # reduce uv resolution to accelerate camera selection

                total_weight_resized = torch.nn.functional.interpolate(total_weight.permute(0,3,1,2), (512,512), mode='area').permute(0,2,3,1) # (1, 512, 512, 3)
                tex_inpaint_mask = torch.logical_and((total_weight_resized.squeeze(3) < 1e-4), renderer.get_uv_mask())
                k_cams = renderer.select_k_views_for_inpaint(tex_inpaint_mask, k=inpaint_args.n_inpaint_views)
                
                lb = "\n"
                print(f"[Inpaint] selected views with \n{lb.join(f'    - azim={all_azimuths[k_cams[i]]}, elev={all_elevations[k_cams[i]]}, dist={all_dists[k_cams[i]]}' for i in range(inpaint_args.n_inpaint_views))}")
                
                inpaint_args.inpaint_camera_config["azimuths"] = all_azimuths[k_cams].tolist()
                inpaint_args.inpaint_camera_config["elevations"] = all_elevations[k_cams].tolist()
                inpaint_args.inpaint_camera_config["dists"] = all_dists[k_cams].tolist()
            
            # config inpaint 
            renderer.modify_res(render_res=inpaint_args.inpaint_resolution, tex_res=args.texture_resolution)
            renderer.set_cameras(**inpaint_args.inpaint_camera_config)

            for i in range(inpaint_args.n_inpaint_views):
                
                camera = i
                rendered_rgba, mask = renderer.sample_texture(texture_rgba, max_mip_level=0, cameras=camera)
                depth, _ = renderer.render_depth("absolute", normalize=(255,0), bg=0, cameras=camera)

                if args.bake_weighting_method == "view_cosine":
                    view_weight, _ = renderer.render_view_cos("vertex", cameras=camera)
                elif args.bake_weighting_method == "sqrtinv_tex_area":
                    view_weight, _ = renderer.render_texture_area(antialias=False, inverse=True, return_singulars=False, cameras=camera)
                    view_weight = view_weight
                    
                inpaint_mask = mask * (1 - rendered_rgba[...,3:4] * inpaint_args.inpaint_retain_factors[i]).float()
                
                cv2.imwrite(os.path.join(args.output_dir, f"stage{i_stage}_view{i}.png"), rendered_rgba[0].clip(0,255).cpu().numpy()[...,[2,1,0]])
                cv2.imwrite(os.path.join(args.output_dir, f"stage{i_stage}_mask{i}.png"), (inpaint_mask[0]*255).clip(0,255).cpu().numpy()[...,::-1])
                cv2.imwrite(os.path.join(args.output_dir, f"stage{i_stage}_depth{i}.png"), depth[0].clip(0,255).cpu().numpy()[...,::-1])

                # inpaint
                print(f"[Inpaint] stage {i_stage} generating view {i}")
                
                view_string = get_view_string(inpaint_args.inpaint_camera_config["azimuths"][i], inpaint_args.inpaint_camera_config["elevations"][i])
                
                projection_prompt = {
                    'ortho': f'{view_string}. an orthographical image of ',
                    'pinhole': f'{view_string}. an image of ',
                    'cylindrical': 'cylindrical image. cylindrical panorama. an inverse panorama of ',
                    'spherical': 'spherical projection. spherical panorama. an inverse panorama of ',
                    'spherical_equirectangular': 'spherical equirectangular projection. spherical equirectangular panorama. an inverse panorama of ',
                }[inpaint_args.inpaint_camera_config['camera_type']]
                
                if inpaint_args.diffusion_pipeline == "FLUX":
                    inpaint_func = inference_inpaint_flux # Use FLUX inpaint
                elif inpaint_args.diffusion_pipeline == "SDXL":
                    inpaint_func = inference_inpaint_sdxl # Use SDXL inpaint
                else:
                    raise ValueError(f"unknown diffusion pipeline {inpaint_args.diffusion_pipeline}")
                
                inpaint_rgb_pil = inpaint_func(projection_prompt + args.prompt, 
                    make_gdp_accessible(os.path.join(args.output_dir, f"stage{i_stage}_view{i}.png")), \
                    make_gdp_accessible(os.path.join(args.output_dir, f"stage{i_stage}_depth{i}.png")), 
                    make_gdp_accessible(os.path.join(args.output_dir, f"stage{i_stage}_mask{i}.png")), 
                    strength=inpaint_args.inpaint_denoising_strengths[i], seed=inpaint_args.inpaint_denoising_seeds[i],
                    control_strength=inpaint_args.inpaint_control_scale,
                    width=inpaint_args.inpaint_resolution,
                    height=inpaint_args.inpaint_resolution,
                    circular_decode='x_only' if inpaint_args.inpaint_camera_config['camera_type'] in ("cylindrical", "spherical", "spherical_equirectangular") else "disable",
                    denoising_steps = inpaint_args.inpaint_denoising_steps,
                    verbose = args.write_intermediates
                )
                if args.write_intermediates:
                    cv2.imwrite(os.path.join(args.output_dir, f"stage{i_stage}_view{i}_vis.png"), np.concatenate((rendered_rgba[0].clip(0,255).cpu().numpy()[...,[2,1,0]], (1-inpaint_mask)[0].cpu().numpy()*255), axis=-1))
                    inpaint_rgb_pil.save(os.path.join(args.output_dir, f"stage{i_stage}_inpaint{i}.png"))
                    
                inpaint_rgb = np.array(inpaint_rgb_pil).astype(np.float32)
                
                print(f"[Inpaint] stage {i_stage} baking view {i}")
                
                inpaint_alpha = make_bake_alpha(depth, view_weight, inpaint_args.bake_inpaint_exp, inpaint_args.bake_inpaint_view_weight, inpaint_args.bake_inpaint_erode) # (1, img_res, img_res, 1)
                
                inpaint_rgb = inpaint_rgb[None,...] # prepend batch dim
                inpaint_rgb = poisson_blend(inpaint_rgb/255, rendered_rgba[...,:3]/255, (inpaint_alpha > 1e-5))*255
                if args.write_intermediates:
                    cv2.imwrite(os.path.join(args.output_dir, f"stage{i_stage}_poisson{i}.png"), inpaint_rgb[0].clip(0,255).cpu().numpy()[...,[2,1,0]])
                

                inpaint_rgba = torch.cat((inpaint_rgb, inpaint_alpha), dim=-1) # (1, img_res, img_res, 4)
                if inpaint_args.bake_method == "raycast":
                    inpaint_texture_rgba = renderer.bake_textures_raycast(inpaint_rgba, interpolation="bicubic", inpaint=False, cameras=camera) # (n_views, tex_res, tex_res, 1)
                elif inpaint_args.bake_method.startswith("mipmap"):
                    mipmap_level = int(inpaint_args.bake_method[6:]) if len(inpaint_args.bake_method) > 6 else None
                    inpaint_texture_rgba = renderer.bake_textures(inpaint_rgba, max_mip_level=mipmap_level, cameras=camera) # (n_views, tex_res, tex_res, 1)
                else:
                    raise ValueError(f"unknown baking method {inpaint_args.bake_method}")
    
                texture_rgba = blend_textures(torch.cat((texture_rgba, inpaint_texture_rgba))) # (1, tex_res, tex_res, 4)
                texture, total_weight = torch.split(texture_rgba, (3,1), dim=-1) # (1, tex_res, tex_res, 3), (1, tex_res, tex_res, 1)
                
                # inpaint missing regions
                texture = renderer.inpaint_textures(texture, (total_weight<=1e-3), inpaint_method="laplace") # (1, tex_res, tex_res, 3)
                texture_rgba = torch.cat((texture, total_weight), dim=-1) # (1, tex_res, tex_res, 4)
                
                if args.write_intermediates:
                    rerendered_rgba, mask = renderer.sample_texture(texture_rgba, max_mip_level=0, cameras=camera)
                    cv2.imwrite(os.path.join(args.output_dir, f"bake_stage{i_stage}_view{i}.png"), texture_rgba[0].clip(0,255).cpu().numpy()[...,[2,1,0]])
                    cv2.imwrite(os.path.join(args.output_dir, f"render_stage{i_stage}_view{i}.png"), rerendered_rgba[0].clip(0,255).cpu().numpy()[...,[2,1,0]])

            # visulization
            if args.write_intermediates:
                renderer.set_cameras(azimuths=args.cam_azimuths, elevations=args.cam_elevations, dists=args.cam_distances, camera_type=args.camera_type, zooms=1.0, near=1e-1, far=1e1)
                rerendered_rgba, mask = renderer.sample_texture(texture_rgba, max_mip_level=0)
                cv2.imwrite(os.path.join(args.output_dir, f"inpaint_bake_stage{i_stage}.png"), torch.cat(list(rerendered_rgba), dim=1).clip(0,255).cpu().numpy()[...,[2,1,0]])
                res = max(inpaint_args.inpaint_resolution, image_resolution)
                cv2.imwrite(f"before_after_stage{i_stage}.png", np.concatenate((cv2.resize(cv2.imread(os.path.join(args.output_dir, f"init_bake.png")), (res*len(args.cam_azimuths), res)), cv2.resize(cv2.imread(os.path.join(args.output_dir, f"inpaint_bake_stage{i_stage}.png")), (res*len(args.cam_azimuths), res)))))
    
    # export mesh
    export_path = os.path.join(args.output_dir, args.out_obj_name)
    renderer.export_mesh(export_path, texture_rgba[...,:3], val_range=(0,255), orientation="y-up")

    return export_path

def pipeline(args):
    from gpt_caption import gpt_caption
    args.prompt = gpt_caption(args.ref_ref_image)
    args.write_intermediates = False
    return bake_refine(args)
    

if __name__ == "__main__":
    
    from types import SimpleNamespace
    
    args = SimpleNamespace()

    # def get_ids(n=None, path="/aigc_cfs_gdp/sz/result/pipe_test/"):
    #     entries = os.listdir(path)
    #     subfolders = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
    #     subfolders_with_ctime = [
    #         (subfolder, os.path.getctime(os.path.join(path, subfolder)))
    #         for subfolder in subfolders
    #     ]
    #     sorted_subfolders = sorted(subfolders_with_ctime, key=lambda x: x[1], reverse=True)
    #     sorted_subfolder_names = [subfolder for subfolder, _ in sorted_subfolders]
    #     ids = []
    #     for subfolder in sorted_subfolder_names:
    #         npy_path = os.path.join(path, subfolder, 'd2rgb/out/imgsr/color.npy')
    #         if not os.path.exists(npy_path):
    #             continue
    #         if np.load(npy_path).shape[0] == 6:
    #             ids.append(subfolder)
    #         if n is not None and len(ids) >= n:
    #             break
    #     return ids
    
    # # ids = get_ids(50)
    # ids = ['ab71a86e-c051-4409-ae93-e7d2dc2c6cd6', 'e3f2a814-1cc6-4fcd-bd16-f3289d9fa4fb', 'f328282c-f221-4a30-8a8c-6247a2f73ddc', 'b9e7e7ad-b2c8-457c-87f6-ca087cfd2054', 'eb02b6af-8fd9-4177-bbb7-4cc8bd59b756', 'f690cdd5-abba-4c27-99fc-9e119e142915', '0f601ba6-20a7-4a5f-9a9b-a6b0a22fbdf4', '942a046e-84fa-4a37-90f5-907354f03bcd', 'bd042481-e332-4e96-af3c-a5c8d8154863', '05e61328-07b4-473d-b567-00289ed924c7', 'c82760a1-61c9-4799-8c28-cbaf705f1433', '4bd11d62-df7e-4fd0-80eb-ecfbd3e8d583', '1a2ebf22-6e2d-4ef1-afd0-350b81572f1f', '4dccc2cb-c48b-48b6-afd8-c9c37a44dee6', 'b5bf77ca-7196-4f13-9ff6-ca442e8ab69c', 'cd66ded6-4558-4ddb-b146-27d0b922b1f7', 'aeda1a45-a3b0-4f20-9f79-203ae858606c', '3105fc02-4130-4f3b-8d19-53fee2afceef', '1ccca6ce-8c98-49ce-9edc-fc02e218bcae', '3edeae21-5f2e-4cfe-aaef-461f2e8bf88d', 'b8ffb681-7383-4617-8120-5a5649d82964', 'ce87e309-d28b-4f03-9ac6-6228fa03b423', 'ecd55c57-ac4f-43e5-8afa-515490a55047', 'dc7c5de9-9e97-4ea0-98f5-9a9c172d64f6', '2663e2fb-e0b9-4eec-abc6-270a07892686', '346c31a7-4df2-4ffe-8ca5-fd6cb31bd5f1', '14ec4c0a-1f06-423f-9748-2daaf41d1d01', '2f564679-16ae-4170-b2f9-1760f8238747', '5683e496-ecb5-47f4-b93b-f05640bdc4d7', '99fa283b-48e8-451a-b372-68bf6146fbc6', '35080109-3a87-46e3-98cc-b0b8abfb0017', '753b5e45-0dd0-44a7-8e8b-143db6705fa0', '7f49eabd-4401-4954-9f26-1e99150de195', '9d2235d3-96dc-4655-b797-faf4f4a88603', 'd508bf90-8625-44f1-bed0-51bbb15d1346', '5475a8d7-29a7-4bf8-98bd-afb389babbec', 'b9c789b4-7e47-4d9a-8d1d-58431ca38863', 'f9c04fe8-bf6e-402c-8c6c-7788e3ca9eb2', '8caa875e-bfc1-4d8b-a821-f0538e9aa3f9', '8427ed64-4a83-48a9-b5aa-fbbb129263f1', '94cc121b-ab2c-41ea-9e6d-631101885926', '3281aaf2-5fdd-4b25-ac9b-57709ddf5aa3', '858431b1-e255-406c-b7e5-88737f8f936d', '18a10ae2-148b-4bbb-88a2-0870b6032e30', '065f8fae-fdc5-4790-bf14-675c0d989954', 'ccc5d487-c981-41e5-b029-d571d545502b', 'b16886a5-256c-48aa-8e4f-e3da6e2ddb1f', '111d03a2-0a07-4629-9208-bd86f5a032dc', '9c6ce454-6ba7-47dd-a78d-6965e13d3def', '5c352ae2-fbfe-4758-9420-5854b3ae37df']

    
    # args_overwrite = {
    #     "bake_weighting_method": "view_cosine",
    # }
    
    # for job_id in ids:
    #     with open(f"/aigc_cfs_gdp/sz/result/pipe_test/{job_id}/texall/args.json") as f:
    #         for attr, val in json.load(f).items():
    #             setattr(args, attr, val)
                
    #         for attr, val in args_overwrite.items():
    #             setattr(args, attr, val)
                
    #         args.write_intermediates = True
    #         args.output_dir = f"/aigc_cfs_2/zacheng/demo_render/render_bake/online_output/{job_id}"
        
    #     print(f"processing {job_id}, \"{args.prompt}\"")
    #     bake_refine(args)
        
        
        
    '''
    # object
    for id in c106fcaf-6b00-4d41-bdb6-dd11900cb6a9 160b4760-f43a-4977-9e33-9b859130789d c4174664-a928-4636-b8c9-8fe63a124a25 2810a4d2-f2b0-4709-9df4-63d85f238d66 c5ae6236-e596-4eef-a344-067c85c77979 5ed3e24a-5ae1-49c1-92bd-f4c46d8e5408 8a4c846e-cba2-4908-bc68-c3d5efb04225 ae910455-3176-4383-b64a-acc4771febef f7622aa1-941d-4662-94a1-f17338c36cb4 af40713c-742e-425b-a1c1-55ec6abacb06; do scp dc:/aigc_cfs_2/zacheng/demo_render/render_bake/output_batch/texall/"$id"_texall/texall/out/"$id"_texall.glb ~/texall2/; scp dc:/aigc_cfs_2/zacheng/demo_render/render_bake/output_batch/texall2/"$id"_texall2/texall/out/"$id"_texall2.glb ~/texall2/; done

    # human
    for id in 7e9964ca-c454-4353-bb3f-9e8ad2ee80d2 8cd973f6-e192-42ec-86ba-0ad89a91261c df4017a8-79a4-4a32-8ec4-3a6cb6d8baa1 294d69e8-f961-440d-83dd-ed4ca08d826b 27eb64ef-da95-4f2f-8019-6c1996a13e92 56d9647a-fc74-421c-83f2-afb4f3582aeb 5e2585a6-cfc7-4111-814d-ab8cc7c45bdb 7d370d18-4238-4f78-97b8-532c85ec5d39 6ee33249-6c48-4e31-b4be-ec9e9904aa87 76e4dcfa-f72e-4aac-9c9d-b2e97ec52c29; do scp dc:/aigc_cfs_2/zacheng/demo_render/render_bake/output_batch_human/texall/"$id"_texall/texall/out/"$id"_texall.glb ~/texall2_human/; scp dc:/aigc_cfs_2/zacheng/demo_render/render_bake/output_batch_human/texall2/"$id"_texall2/texall/out/"$id"_texall2.glb ~/texall2_human/; done
    
    for id in 7e9964ca-c454-4353-bb3f-9e8ad2ee80d2 8cd973f6-e192-42ec-86ba-0ad89a91261c df4017a8-79a4-4a32-8ec4-3a6cb6d8baa1 294d69e8-f961-440d-83dd-ed4ca08d826b 27eb64ef-da95-4f2f-8019-6c1996a13e92 56d9647a-fc74-421c-83f2-afb4f3582aeb 5e2585a6-cfc7-4111-814d-ab8cc7c45bdb 7d370d18-4238-4f78-97b8-532c85ec5d39 6ee33249-6c48-4e31-b4be-ec9e9904aa87 76e4dcfa-f72e-4aac-9c9d-b2e97ec52c29; do scp dc:"/aigc_cfs_2/zacheng/demo_render/render_bake/output_batch_human/texall2/"$id"_texall2/texall_v1/out/*_v1*" ~/texall2_human/; done
    
    '''

    # humanoid
    jobs = "7e9964ca-c454-4353-bb3f-9e8ad2ee80d2 8cd973f6-e192-42ec-86ba-0ad89a91261c df4017a8-79a4-4a32-8ec4-3a6cb6d8baa1 294d69e8-f961-440d-83dd-ed4ca08d826b 27eb64ef-da95-4f2f-8019-6c1996a13e92 56d9647a-fc74-421c-83f2-afb4f3582aeb 5e2585a6-cfc7-4111-814d-ab8cc7c45bdb 7d370d18-4238-4f78-97b8-532c85ec5d39 6ee33249-6c48-4e31-b4be-ec9e9904aa87 76e4dcfa-f72e-4aac-9c9d-b2e97ec52c29".split()
    
    # general
    jobs = "c106fcaf-6b00-4d41-bdb6-dd11900cb6a9 160b4760-f43a-4977-9e33-9b859130789d c4174664-a928-4636-b8c9-8fe63a124a25 2810a4d2-f2b0-4709-9df4-63d85f238d66 c5ae6236-e596-4eef-a344-067c85c77979 5ed3e24a-5ae1-49c1-92bd-f4c46d8e5408 8a4c846e-cba2-4908-bc68-c3d5efb04225 ae910455-3176-4383-b64a-acc4771febef f7622aa1-941d-4662-94a1-f17338c36cb4 af40713c-742e-425b-a1c1-55ec6abacb06".split()
    
    jobs = ["76e4dcfa-f72e-4aac-9c9d-b2e97ec52c29"]
    
    jobs = ["f4aecc70-b72d-4ed7-90f6-918f36a97f04_texall2"]
    
    
    
    for job in jobs:
        # with open(f"/aigc_cfs_2/zacheng/demo_render/render_bake/output_batch_human/texall2/{job}_texall2/texall/args.json") as f:
        # with open(f"/aigc_cfs_2/zacheng/demo_render/render_bake/output_batch/texall2/{job}_texall2/texall/args.json") as f:
        with open(f"/aigc_cfs_gdp/sz/batch_1107/newtex_img_in/texall2/{job}/texall/args.json") as f:
            conf_dict = json.load(f)
            conf_dict['texture_resolution'] = 4096
            conf_dict['bake_method'] = 'mipmap'
            conf_dict['bake_weighting_method'] = 'view_cosine'
            conf_dict["bake_init_view_weight"] = [
                    0.1,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.05
                ]
            conf_dict["inpaint_stages"] = [ \
                
                # stage 0
                {   
                    "diffusion_pipeline": "SDXL",
                    "texture_attenuation": 0.5,
                    "n_inpaint_views": 2,
                    'bake_method': 'mipmap',
                    "bake_inpaint_exp": 3.0,
                    "bake_inpaint_view_weight": 1.0,
                    "bake_inpaint_erode": 5,
                    "inpaint_resolution": 1024,
                    "inpaint_denoising_strengths": [
                        0.6,
                        0.6,
                    ],
                    "inpaint_retain_factors": [
                        0,
                        0,
                    ],
                    "inpaint_denoising_seeds": [
                        0,
                        0,
                    ],
                    "inpaint_control_scale": 0.6,
                    "inpaint_denoising_steps": 12,
                    "inpaint_camera_config": {
                        "camera_type": "ortho",
                        "zooms": 1.0,
                        "radii": 10.0,
                        "azimuths": [
                            0,
                            180,
                        ],
                        "elevations": [
                            0,
                            0,
                        ],
                        "dists": [
                            10,
                            10,
                        ]
                    },
                },
                
                # stage 1
                {   
                    "diffusion_pipeline": "FLUX",
                    "texture_attenuation": 0.5,
                    "n_inpaint_views": 1,
                    'bake_method': 'mipmap',
                    "bake_inpaint_exp": 3.0,
                    "bake_inpaint_view_weight": 1.0,
                    "bake_inpaint_erode": 5,
                    "inpaint_resolution": 1024,
                    "inpaint_denoising_strengths": [
                        0.5,
                    ],
                    "inpaint_retain_factors": [
                        0.5
                    ],
                    "inpaint_denoising_seeds": [
                        0,
                    ],
                    "inpaint_control_scale": 0.6,
                    "inpaint_denoising_steps": 12,
                    "inpaint_camera_config": {
                        "camera_type": "cylindrical",
                        "zooms": 1.0,
                        "radii": 10.0,
                        "azimuths": [
                            0,
                        ],
                        "elevations": [
                            0,
                        ],
                        "dists": [
                            0,
                        ]
                    },
                },
                
                # stage 2
                {   
                    "diffusion_pipeline": "SDXL",
                    "texture_attenuation": 0.5,
                    "n_inpaint_views": 3,
                    'bake_method': 'mipmap',
                    "bake_inpaint_exp": 3.0,
                    "bake_inpaint_view_weight": 1.0,
                    "bake_inpaint_erode": 5,
                    "inpaint_resolution": 1024,
                    "inpaint_denoising_strengths": [
                        0.7,
                        0.7,
                        0.7,
                    ],
                    "inpaint_retain_factors": [
                        1,
                        1,
                        1,
                    ],
                    "inpaint_denoising_seeds": [
                        0,
                        0,
                        0,
                    ],
                    "inpaint_control_scale": 0.6,
                    "inpaint_denoising_steps": 6,
                    "inpaint_camera_config": {
                        "camera_type": "ortho",
                        "zooms": 1.0,
                        "radii": 10.0,
                        "candidate_azimuths": [0,30,60,90,120,150,180,210,240,270,300,330],
                        "candidate_elevations": [-30,0,30],
                        "candidate_dists": [10],
                    },
                    # "inpaint_camera_config": {
                    #     "camera_type": "cylindrical",
                    #     "zooms": 1.0,
                    #     "radii": 10.0,
                    #     "candidate_azimuths": [0],
                    #     "candidate_elevations": [-30,-20,-10,0,10,20,30],
                    #     "candidate_dists": [-0.2, -0.1, 0, 0.1, 0.2],
                    # },
                },
            ]
            
            for attr, val in conf_dict.items():
                setattr(args, attr, val)
            args.write_intermediates = True
            args.output_dir = f"/aigc_cfs_2/zacheng/demo_render/render_bake/output_batch_human/{job}"
            from gpt_caption import gpt_caption
            # args.prompt = "bishop in religious robes holding a cross"
            # args.prompt = gpt_caption(f"/aigc_cfs_2/zacheng/demo_render/render_bake/output_batch_human/texall2/{job}_texall2/result_z123.png")
            args.out_obj_name = f'out/{job}_texall2_v2.obj'
        print(f"processing {job}, \"{args.prompt}\"")
        
        import time
        start_time = time.time()
        bake_refine(args)
        end_time = time.time()
        print(f"total time {end_time - start_time} sec")
        
    
    # jid = "c7ff4af3-0446-425d-877b-c9888c18b527"
    # # jid = "ad9cfebc-24b1-4e8e-af11-9d663d8d762c"

    
    # args.input_obj_file = f"/aigc_cfs_gdp/sz/result/pipe_test/{jid}/texbakeinpaint/textured.obj"
    # args.images_npy = f"/aigc_cfs_gdp/sz/result/pipe_test/{jid}/d2rgb/out/color.npy"
    # args.prompt = "Super Mario"
    # args.output_dir = f"/aigc_cfs_2/zacheng/demo_render/render_bake/output/{jid}"
    # args.out_obj_name = f"{jid}.obj"
        
    # args.texture_resolution = 2048
    # args.obj_bound = 0.9

    # args.cam_azimuths = [0, 90, 180, 270] # len is n_views
    # args.cam_elevations = [0,0,0,0] # len is n_views
    # args.cam_distances = [5,5,5,5] # len is n_views
    # args.camera_type = "ortho"

    # args.bake_init_exp = 2.0
    # args.bake_init_view_weight = [0.5, 0.2, 0.2 ,0.2]
    # args.bake_init_erode = 5
    # args.bake_weighting_method = "sqrtinv_tex_area" # "view_cosine" or "sqrtinv_tex_area"
    
    # args.n_inpaint_views = 5
    # args.bake_inpaint_exp = 3.0
    # args.bake_inpaint_view_weight = 1.0
    # args.bake_inpaint_erode = 5
    
    # args.inpaint_resolution = 1024
    # args.inpaint_denoising_strengths = [0.7, 0.7, 0.7, 0.7, 0.7]
    # args.inpaint_denoising_seeds = [0] * args.n_inpaint_views
    # args.inpaint_control_scale = 0.6
    # args.inpaint_denoising_steps = 12
    # args.write_intermediates = True
    
    # args.inpaint_camera_config = {
    #     "camera_type": "cylindrical",
    #     "zooms": 1.0,
    #     "radii": 10.,
    # }
    
    # bake_refine(args)
    