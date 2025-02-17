from render_bake_utils import dilate_masks, Renderer
import cv2, torch, numpy as np
    
# inputs and parameters

texture_resolution = 2048
obj_path = "/aigc_cfs_gdp/sz/result/pipe_test/c7ff4af3-0446-425d-877b-c9888c18b527/texbakeinpaint/mesh.obj"
obj_bound = 0.9

cam_azimuths = [0, 90, 180, 270] # len is n_views
cam_elevations = [0,0,0,0] # len is n_views
cam_distances = [5,5,5,5] # len is n_views
camera_type = "ortho" # OR "pinhole"
camera_zoom = 1.0 # zooming ratio, i.e. sensor size for ortho, or NDC focal for pinhole

bake_weight_exp = 3.0
bake_front_view_weight = 5.0
bake_erode_boundary = 5
bake_weighting_method = "view_cosine" # "view_cosine" or "sqrtinv_tex_area"

images = np.load("/aigc_cfs_gdp/sz/result/pipe_test/c7ff4af3-0446-425d-877b-c9888c18b527/d2rgb/out/color.npy") 
images = torch.from_numpy(images).cuda().float().permute(0,2,3,1) # [n_views, img_res, img_res, channels]
image_resolution = images.shape[1]


# set up renderer
renderer = Renderer(image_resolution, texture_resolution, world_orientation="y-up")
renderer.set_object(obj_path, bound=obj_bound, orientation="y-up")
renderer.set_cameras(azimuths=cam_azimuths, elevations=cam_elevations, dists=cam_distances, camera_type=camera_type, zooms=camera_zoom, near=1e-1, far=1e1)

# render normal and depth
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
weights[0] *= bake_front_view_weight

# bake
image_weights = torch.cat((images, weights), dim=-1)
texture_weights = renderer.bake_textures_raycast(image_weights, interpolation="bicubic", inpaint=False)
textures, weights = torch.split(texture_weights, (3,1), dim=-1)

# blend textures by weights
total_weights = torch.sum(weights, dim=0, keepdim=True) # (1, img_res, img_res, 1)
texture = torch.sum(textures*weights, dim=0, keepdim=True) / (total_weights + 1e-10) # (1, img_res, img_res, 3)

# inpaint missing regions, optional
texture = renderer.inpaint_textures(texture, (total_weights<=1e-5), inpaint_method="laplace") # (1, img_res, img_res, 3)

# re-render image from textures
rendered, mask = renderer.sample_texture(texture, max_mip_level=None)

# output
cv2.imwrite("texture.png", texture[0].clip(0,255).cpu().numpy()[...,::-1])
cv2.imwrite("inputs.png", torch.cat(list(images), dim=1).clip(0,255).cpu().numpy()[...,::-1])
cv2.imwrite("rendered.png", torch.cat(list(rendered), dim=1).clip(0,255).cpu().numpy()[...,::-1])

renderer.export_mesh("output.obj", texture, val_range=(0,255))
    
    