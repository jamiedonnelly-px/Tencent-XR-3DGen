## sparse marching cubes

### dependencies
if CUDA is available, install the custom mc33 package for guaranteed watertightness. otherwise the classic marching cubes is used.
```
cd mc33 && pip install -e .
```

### usage

#### vae
```python

vae = ...
latents = ...

def query_func(x):
    sdf = vae.sparse_query(x, latents).to(x).detach()
    sdf[sdf.abs() < 1e-10] = 1e-10
    return sdf

init_depth = 5
final_depth = 8
surface_in = -30
surface_out = 30
surface_range_decay = 0.6
box_v = 1.05

vertices_sparse, faces_sparse = sparse_marching_cubes(query_func, init_depth, final_depth, 0.0, surface_in, surface_out, surface_range_decay, bounds=(-box_v,box_v), verbose=False, device=latents.device, flip_faces=True)
        
```
