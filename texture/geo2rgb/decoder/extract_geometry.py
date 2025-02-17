import torch
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
    Sequence,
)
import trimesh
import time
import numpy as np
from skimage import measure
from einops import repeat
from decoder.customized_marching_cubes import sparse_marching_cubes


def generate_dense_grid_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    octree_depth: int,
    indexing: str = "ij"
):
    length = bbox_max - bbox_min
    num_cells = np.exp2(octree_depth)
    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    xyz = xyz.reshape(-1, 3)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length

@torch.no_grad()
def extract_geometry(vae,
                    latents: torch.FloatTensor,
                    bounds: Union[Tuple[float], List[float], float] = (-1.05, -1.05, -1.05, 1.05, 1.05, 1.05),
                    octree_depth: int = 8,
                    num_chunks: int = 10000,
                    method = 'sparse',
                    ):
    
    if isinstance(bounds, float):
        bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

    
    bbox_min = np.array(bounds[0:3])
    bbox_max = np.array(bounds[3:6])
    bbox_size = bbox_max - bbox_min

    xyz_samples, grid_size, length = generate_dense_grid_points(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        octree_depth=octree_depth,
        indexing="ij"
    )
    xyz_samples = torch.FloatTensor(xyz_samples)
    batch_size = latents.shape[0]

    # breakpoint()
    start_time = time.time()
    if method == 'sparse':
        
        init_depth = 5
        final_depth = 8
        surface_in = -50
        surface_out = 50
        surface_range_decay = 0.7
        
        vertices, faces = sparse_marching_cubes(vae.sparse_query, latents, init_depth, final_depth, 0.0, surface_in, surface_out, surface_range_decay, bounds=(-1,1), verbose=True)

        mesh_v_f = []
        vertices = vertices.detach().cpu().numpy()
        # vertices = vertices / grid_size * bbox_size + bbox_min
        
        faces = faces[:, [2, 1, 0]].detach().cpu().numpy()
        faces = np.ascontiguousarray(faces)
        vertices = vertices.astype(np.float32)

        mesh_v_f.append((vertices, faces))
        has_surface = np.zeros((batch_size,), dtype=np.bool_)
        has_surface[0] = True
        
        print("--- %s seconds ---" % (time.time() - start_time))
        return mesh_v_f, has_surface, vertices, faces
        # trimesh.Trimesh(mesh_verts.detach().cpu().numpy(), mesh_faces.cpu().numpy()).export('sparse_mc_sphere.obj')
        
    else:
        
        batch_logits = []
        for start in range(0, xyz_samples.shape[0], num_chunks):
            queries = xyz_samples[start: start + num_chunks, :].to(latents)
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)

            logits = vae.query(batch_queries, latents)
            batch_logits.append(logits.cpu())

        grid_logits = torch.cat(batch_logits, dim=1).view((batch_size, grid_size[0], grid_size[1], grid_size[2])).float().numpy()

        # breakpoint()
        
        mesh_v_f = []
        has_surface = np.zeros((batch_size,), dtype=np.bool_)
        for i in range(batch_size):
            try:
                occupancy = np.argwhere(grid_logits[i] > 0)
                occupancy_cloud=trimesh.PointCloud(occupancy)

                vertices, faces, normals, _ = measure.marching_cubes(grid_logits[i], 0, method="lewiner")
                # vertices, faces = mcubes.marching_cubes(grid_logits[i], 0)
                vertices = vertices / grid_size * bbox_size + bbox_min
                faces = faces[:, [2, 1, 0]]
                faces = np.ascontiguousarray(faces)
                vertices = vertices.astype(np.float32)

                mesh_v_f.append((vertices, faces))
                has_surface[i] = True
            except:
                mesh_v_f.append((None, None))
                has_surface[i] = False

        print("--- %s seconds ---" % (time.time() - start_time))
        return mesh_v_f, has_surface, occupancy_cloud, vertices, faces