import torch
from pdb import set_trace as st
import numpy as np
    
    
def test_6view():
    
    from render_bake_utils import Renderer, dilate_masks
    import numpy as np, cv2
    from pdb import set_trace as st
    from time import time
    import trimesh
    
    image_res = 768
    uv_res = 4096
    bake_weighting_method = "view_cosine"
    bake_weight_exp = 2.0
    bake_erode_boundary = 0
    
    renderer = Renderer(image_res, uv_res)
    renderer.set_cameras([0,90,180,270,0,0], [0,0,0,0,-89.9,89.9], [5,5,5,5,5,5], zooms=1.0)
    renderer.set_object("/aigc_cfs_2/zacheng/demo_render/render_bake/b883a2a6-634a-42f5-858a-b44d9281e6ac.obj", bound=0.9, merge_verts=True)     
    renderer.unwrap_uv(padding=10, parallel_regions=5) 
    
    trimesh.Trimesh(renderer.mesh.v[0].cpu().numpy(), renderer.mesh.f.cpu().numpy()).is_watertight
    
    depth, mask = renderer.render_depth("absolute", normalize=(255,50), bg=0) # (n_views, img_res, img_res, 1)

    if bake_weighting_method == "view_cosine":
        view_weight, _ = renderer.render_view_cos("vertex")
    elif bake_weighting_method == "sqrtinv_tex_area":
        view_weight, _ = renderer.render_texture_area(antialias=False, inverse=True, return_singulars=False)
        view_weight = view_weight ** 0.5

    # detect depth discontinuities, i.e. occlusion boundaries
    depth_map_uint8 = depth.cpu().numpy().astype(np.uint8) # (n_views, img_res, img_res, 1)
    depth_edge = [(cv2.Canny(d, 10, 40) > 0) for d in depth_map_uint8]
    depth_edge = dilate_masks(*depth_edge, iterations=bake_erode_boundary)
    depth_edge = (torch.from_numpy(depth_edge).cuda() > 0).float().unsqueeze(-1) # binary (n_views, img_res, img_res, 1)

    weights = view_weight * (1-depth_edge) # remove pixels on occlusion boundaries

    # apply weights
    weights = weights ** bake_weight_exp 
    weights = renderer.bake_textures(weights, 2, inpaint=False)
    
    # apply softmax for normalization
    softmax_sigma = 10
    weights_mask = (weights != 0).float()
    weights_softmax = torch.softmax(weights * softmax_sigma + (weights_mask-1) * 1e3, dim=0)
    weights_softmax = weights_softmax * (weights_mask.sum(0) != 0).float() # set occluded parts to 0
    weights = weights_softmax
    
    images = cv2.imread("/aigc_cfs_2/zacheng/demo_render/render_bake/15_step1_pred_x0_rgb.png") / 255.
    images = images.reshape(3, image_res, 2, image_res, 3)
    images = images.transpose(0,2,1,3,4).reshape(6, image_res, image_res, 3)
    
    images = torch.from_numpy(images).cuda().float()
    images_rgba = torch.cat((images, images[...,:1]*0+1), dim=-1)
    texs = renderer.bake_textures(images_rgba, 2, inpaint=False) # [B,H,W,C]
    texs, masks = texs[...,:3], texs[...,-1] > 0.5
    texs = renderer._voronoi_inpaint(texs, masks.unsqueeze(-1) > 0.5) # inpaint tex sol gradient field is smooth on island boundaries
    cv2.imwrite("tmp_6v.png", torch.cat(list(images), dim=1).cpu().clip(0,1).numpy() * 255)
    cv2.imwrite("tmp_tex.png", torch.cat(list(texs), dim=1).cpu().clip(0,1).numpy() * 255)
    
    
    # build poisson equation to minimize following objective
    # w0(t-T)^2 + w1(∇t-∇T)^2 + w2(Δt-ΔT)^2 s.t. wc(t-T)=0 
    # where t and T are input and output textures, ∇ and Δ are graident and laplacian operators
    w0 = masks.float()
    w0 = weights[...,0] * 0.5 # we will say absolute scale value is half as important as gradient
    w1 = masks.float() * (weights[...,0]).float()
    w2 = masks.float()*0 # set laplacian term to 0 to reduce computation overhead
    wc = masks.float()*0 # no constraints
    
    poinsson_weights = torch.stack((w0, w1, w2, wc), dim=-1) # [B,H,W,4]
    
    multigrid_stages = [
        dict(uv_res=256, solver="LBFGS", solver_params=dict(maxiter=1000, damp=1e-3)),
        dict(uv_res=512, solver="AdamW", solver_params=dict(maxiter=100, lr=1e-1, damp=1e-6)),
        dict(uv_res=1024, solver="AdamW", solver_params=dict(maxiter=50, lr=1e-1, damp=1e-6)),
        dict(uv_res=2048, solver="AdamW", solver_params=dict(maxiter=20, lr=5e-2, damp=1e-6)),
        dict(uv_res=4096, solver="AdamW", solver_params=dict(maxiter=10, lr=2e-2, damp=1e-6)), 
    ] # if final tex_res is 3072, you may want to start with e.g. 384, then 768, then 1536, then 3072
    
    start = time()
    tex_poisson = renderer.solve_poisson_multigrid(texs.clone(), poinsson_weights, 0.1, 0.5, multigrid_stages)
    end = time()
    print(f"build & solve system time {end - start} sec")
    
    cv2.imwrite("tmp_texp.png", tex_poisson.cpu().clip(0,1).numpy() * 255)
    
    images = renderer.sample_texture(tex_poisson, 2)[0]
    cv2.imwrite("tmp_6vp.png", torch.cat(list(images), dim=1).cpu().clip(0,1).numpy() * 255)
    
    naive_blend = (texs * masks.unsqueeze(-1).float()).sum(0) / (masks.float().sum(0).unsqueeze(-1) + 1e-5)
    cv2.imwrite("tmp_texn.png", naive_blend.cpu().clip(0,1).numpy() * 255)
    images = renderer.sample_texture(naive_blend, 2)[0]
    cv2.imwrite("tmp_6vn.png", torch.cat(list(images), dim=1).cpu().clip(0,1).numpy() * 255)
    
if __name__ == "__main__":
    from render_bake_utils import Renderer, dilate_masks
    import numpy as np, cv2
    from pdb import set_trace as st
    from time import time
    import trimesh
    
    test_6view()

    
            
        
        
        
    
        
        
    
    
    
    
    
    