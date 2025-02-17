import os
import sys
import torch
import xatlas
import numpy as np
import nvdiffrast.torch as dr
import time
from PIL import Image
import faiss

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_script_path)
sys.path.append(project_root)

from render.mesh import load_mesh, Mesh, auto_normals
from render.obj import write_obj, new_texed_mesh


def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(),
                          rast,
                          attr_idx,
                          rast_db=rast_db,
                          diff_attrs=None if rast_db is None else 'all')


def resize_render_tensor(in_tensor, new_size):
    """resize bhwc tensor

    Args:
        in_tensor: b, h, w, c
        new_size: [h, w]

    Returns:
        b, h, w, c
    """
    in_tensor = in_tensor.permute(0, 3, 1, 2)
    in_tensor = torch.nn.functional.interpolate(in_tensor, new_size, mode="bilinear", align_corners=False)
    in_tensor = in_tensor.permute(0, 2, 3, 1).contiguous()
    return in_tensor


def save_np_hwc(np_hwc, out_img, scale=255.0):
    """_summary_

    Args:
        np_hwc: [h,w,c](rgb) or [h, w](gray)
        out_img: _description_
    """
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    Image.fromarray(np.clip(np_hwc * scale, 0, 255).astype(np.uint8)).save(out_img)


def compute_inpaint_colors(see_points, inpaint_points, see_colors, n_neighbors=1000):
    """
    使用 FAISS GPU 进行 KNN 搜索，并计算 inpaint_points 的填充颜色。

    参数:
    - see_points: numpy array, 形状为 [N1, 3] 的点云数据。
    - inpaint_points: numpy array, 形状为 [N2, 3] 的查询点数据。
    - see_colors: numpy array, 形状为 [N1, 3] 的颜色数据。
    - n_neighbors: int, 最近邻的数量。

    返回:
    - inpaint_colors: numpy array, 形状为 [N2, 3] 的填充颜色。
    """

    see_points = see_points.astype('float32')
    inpaint_points = inpaint_points.astype('float32')
    see_colors = see_colors.astype('float32')
    print('debug inpaint_points/see_points ', inpaint_points.shape, see_points.shape)

    ts = time.time()

    res = faiss.StandardGpuResources()

    dim = see_points.shape[1]
    index_flat = faiss.IndexFlatL2(dim)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

    gpu_index_flat.add(see_points)

    distances, indices = gpu_index_flat.search(inpaint_points, n_neighbors)

    tsearch = time.time() - ts
    
    # inpaint_colors = np.mean(see_colors[indices], axis=1)

    device="cuda"
    see_colors_torch = torch.tensor(see_colors, device=device)
    indices_torch = torch.tensor(indices, device=device)

    inpaint_colors_torch = torch.mean(see_colors_torch[indices_torch], dim=1)
    inpaint_colors = inpaint_colors_torch.cpu().numpy()

    del gpu_index_flat
    del res
    torch.cuda.empty_cache()
    
    tuse = time.time() - ts

    print(f"KNN {tsearch:.4f}s , add mean: {tuse:.4f} 秒")
    print("indices", indices.shape)
    print("inpaint_colors", inpaint_colors.shape)  
    
    return inpaint_colors

def raw_scipy_knn(see_points, inpaint_points, see_colors, n_neighbors=1000):
    from sklearn.neighbors import NearestNeighbors
    # see_points [N1, 3], inpaint_points [N2, 3]
    ts = time.time()
    n_neighbors = 1000
    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(see_points)
    _, indices = knn.kneighbors(inpaint_points)  #  indices [N1, n_neighbors], in [0, N2)
    tsearch = time.time() - ts

    inpaint_colors = np.mean(see_colors[indices], axis=1)
    tuse = time.time() - ts
    print(f'tsearch={tsearch}, tuse={tuse}')
    
    print("indices", indices.shape)
    print("inpaint_colors", inpaint_colors.shape)    
        
    return inpaint_colors

def render_geometry_uv(ctx, mesh: Mesh, resolution, ssaa=-1):
    """render geometry uv

    Args:
        ctx: RasterizeCudaContext
        mesh(mesh.Mesh): class Mesh
        resolution(int): render uv res
        ssaa: super-sample anti-a

    Returns:
        gb_geom(xyz), gb_normal([-1, 1]), gb_mask(0 or 1). all are [h, w, x=3/3/1] tensor
    """
    render_res = ssaa * resolution if ssaa > 1 else resolution
    render_res = min(render_res, 2048)

    # clip space transform
    uv_clip = mesh.v_tex[None, ...] * 2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[..., 0:1]), torch.ones_like(uv_clip[..., 0:1])), dim=-1)

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh.t_tex_idx.int(), (render_res, render_res), grad_db=False)

    # Interpolate world space position [1, h, w, 3] and mask [1, h, w, 1]
    gb_geom, _ = interpolate(mesh.v_pos[None, ...], rast, mesh.t_pos_idx.int())
    if isinstance(mesh.v_nrm, torch.Tensor):
        pass
    elif mesh.v_nrm is None:
        mesh = auto_normals(mesh)

    gb_normal, _ = interpolate(mesh.v_nrm[None, ...], rast, mesh.t_pos_idx.int())
    # gb_normal = (gb_normal + 1.0)*0.5   # [-1, 1] -> [0, 1] as img
    gb_mask = (rast[..., 3:4] > 0)

    if ssaa > 1:
        gb_geom = resize_render_tensor(gb_geom, [resolution, resolution])
        gb_normal = resize_render_tensor(gb_normal, [resolution, resolution])
        gb_mask = resize_render_tensor(gb_mask.float(), [resolution, resolution])

    gb_mask = gb_mask.to(torch.uint8)
    return gb_geom.squeeze(0), gb_normal.squeeze(0), gb_mask.squeeze(0)


def set_uv_value(uv_tex_np, set_mask, set_values):
    max_idx = set_mask.shape[0] - 1
    set_indices = np.stack(np.nonzero(set_mask), axis=-1)

    uv_tex_np[set_indices[:, 0], set_indices[:, 1]] = set_values
    uv_tex_np[np.clip(set_indices[:, 0] + 1, 0, max_idx), set_indices[:, 1]] = set_values
    uv_tex_np[np.clip(set_indices[:, 0] - 1, 0, max_idx), set_indices[:, 1]] = set_values
    uv_tex_np[set_indices[:, 0], np.clip(set_indices[:, 1] + 1, 0, max_idx)] = set_values
    uv_tex_np[set_indices[:, 0], np.clip(set_indices[:, 1] - 1, 0, max_idx)] = set_values

    return uv_tex_np


def inpaint_pos_knn(uv_tex_np, see_mask, gb_xyz, gb_mask, debug_dir=None, save_debug=False):
    """Find the nearest visible point and inpaint the color of the invisible point

    Args:
        uv_tex_np: [h, w, 3/c] numpy
        see_mask: [h, w]  render-see mask in uv numpy
        gb_xyz: [h, w, 3] tensor
        gb_mask: [h, w, 1] tensor
        debug_dir
        save_debug
    Returns:
        uv_tex_new: [h, w, 3/c] numpy
    """
    gb_xyz = gb_xyz.cpu().numpy()

    # [h, w] gb_mask: all need uv
    gb_mask = gb_mask.to(torch.bool).squeeze(-1).cpu().numpy()
    see_mask = (see_mask > 0).astype(bool)

    # [h, w] mask
    inpaint_mask_raw = gb_mask & (~see_mask)
    # from scipy.ndimage import binary_dilation, binary_erosion
    # inpaint_mask = binary_dilation(inpaint_mask_raw, iterations=5)
    inpaint_mask = inpaint_mask_raw

    # [N1, 3]
    inpaint_points = gb_xyz[inpaint_mask]
    # [N2, 3]
    see_points = gb_xyz[see_mask]
    see_colors = uv_tex_np[see_mask]

    debug = False
    
    inpaint_colors = compute_inpaint_colors(see_points, inpaint_points, see_colors)
    # if debug:
    #     inpaint_colors= np.ones_like(inpaint_colors)

    # [N2, 2]
    uv_tex_np = set_uv_value(uv_tex_np, inpaint_mask, inpaint_colors)
    # inpaint_indices = np.stack(np.nonzero(inpaint_mask), axis=-1)
    # uv_tex_np[inpaint_indices[:, 0], inpaint_indices[:, 1]] = inpaint_colors

    # if debug:
    #     uv_tex_np = set_uv_value(uv_tex_np, gb_mask, (1,0,0))

    #     see_indices = np.stack(np.nonzero(see_mask), axis=-1)
    #     uv_tex_np[see_indices[:, 0], see_indices[:, 1]] = 0.5

    uv_tex_new = uv_tex_np

    if debug_dir is not None:
        save_np_hwc(inpaint_mask, os.path.join(debug_dir, "inpaint_mask.jpg"))
        if save_debug:
            save_np_hwc(see_mask, os.path.join(debug_dir, "see_mask.jpg"))
            save_np_hwc(gb_mask, os.path.join(debug_dir, "gb_mask.jpg"))
            save_np_hwc(inpaint_mask_raw, os.path.join(debug_dir, "inpaint_mask_raw.jpg"))
            inpaint_tex = np.zeros_like(uv_tex_np)
            inpaint_tex = set_uv_value(inpaint_tex, inpaint_mask, inpaint_colors)
            save_np_hwc(inpaint_tex, os.path.join(debug_dir, "inpaint_tex.jpg"))
            save_np_hwc(uv_tex_np * see_mask[..., None], os.path.join(debug_dir, "see_tex.jpg"))

    return uv_tex_new


def inpaint_laplacian_loop(uv_tex_np, see_mask, gb_xyz, gb_mask, debug_dir=None):
    """Efficient Invisible Region Color Completion Algorithm

    Args:
        uv_tex_np: [h, w, 3/c] numpy
        see_mask: [h, w]  render-see mask in uv numpy
        gb_xyz: [h, w, 3] tensor
        gb_mask: [h, w, 1] tensor
        debug_dir
    Returns:
        uv_tex_new: [h, w, 3/c] numpy
    """

    pass
    return


def main_inpaint_mesh(mesh, see_mask, uv_tex_np, out_dir, resolution=1024, 
                      mesh_process="keep_raw",
                      save_debug=False,
                      ):
    """_summary_

    Args:
        mesh: class Mesh
        see_mask: [h,w] numpy
        uv_tex_np: [h,w,3] numpy in [0, 1]
        out_dir: _description_
        resolution: _description_. Defaults to 1024.
        mesh_process: keep_raw / rot2
        save_debug: _description_. Defaults to False.

    Returns:
        out_obj_path out obj path
        uv_tex_new
    """
    glctx = dr.RasterizeCudaContext()
    
    if mesh_process == "rot2":
        rotation_matrix = np.array([[0, 1, 0],
                                    [0, 0, 1],
                                    [1, 0, 0]])
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32, device=mesh.v_pos.device)
        mesh.v_pos = torch.mm(mesh.v_pos, rotation_matrix.T)  # N, 3
    elif mesh_process == "keep_raw":
        pass
    else:
        raise ValueError(f"invalid mesh_process = {mesh_process}")
    
    # [h, w, 3/3/1] tensor
    gb_xyz, gb_normal, gb_mask = render_geometry_uv(glctx, mesh, resolution, ssaa=-1)

    method = "knn"  #"bake"
    if method == "knn":
        uv_tex_new = inpaint_pos_knn(uv_tex_np,
                                     see_mask,
                                     gb_xyz,
                                     gb_mask,
                                     debug_dir=os.path.join(out_dir, "outvis"),
                                     save_debug=save_debug)
    else:
        raise NotImplementedError
    

    new_mesh = new_texed_mesh(mesh, uv_tex_new)
    write_obj(out_dir, new_mesh)
    out_obj_path = os.path.join(out_dir, 'mesh.obj')

    if save_debug:
        save_np_hwc(gb_xyz.cpu().numpy(), os.path.join(out_dir, "gb_xyz.png"))
        save_np_hwc((gb_mask[..., 0]).float().cpu().numpy(), os.path.join(out_dir, "gb_mask.png"))

        save_np_hwc(uv_tex_new, os.path.join(out_dir, "tex_new.png"))

    return out_obj_path, uv_tex_new


if __name__ == "__main__":
    job_id = "c711a5ec-e9cb-45c2-bc44-05fa4cda47cb_z123_crman"
    # in_mesh_path=f"/aigc_cfs_gdp/sz/batch_0828/z123_crman/z123_crman/{job_id}/verts2tex/baking/tex_mesh.obj"
    # in_mesh_path = f"/aigc_cfs_gdp/sz/batch_0828/z123_crman/z123_crman/imc/{job_id}/try_0/bake_sync_1024/textured.obj"
    in_mesh_path = f"/aigc_cfs_gdp/sz/batch_0828/z123_crman/z123_crman/imc/{job_id}/try_0/bake_remesh_sync_1024/textured.obj"
    in_dir = os.path.dirname(in_mesh_path)
    in_tex_png = os.path.join(in_dir, "textured.png")
    in_see_mask_png = os.path.join(in_dir, "see_mask_0.001.png")
    # in_bake_texture_png = os.path.join(in_dir, "bake_texture.png")

    resolution = 1024
    out_dir = os.path.join(in_dir, f"res{resolution}")
    save_debug = True

    see_mask = np.array(Image.open(in_see_mask_png).resize((resolution, resolution)).convert("L"))  # [h,w]
    uv_tex_np = np.array(Image.open(in_tex_png).resize((resolution, resolution))) / 255.0
    
    mesh = load_mesh(in_mesh_path, skip_mtl=True)
    main_inpaint_mesh(mesh, see_mask, uv_tex_np, out_dir, resolution=resolution, save_debug=save_debug)
