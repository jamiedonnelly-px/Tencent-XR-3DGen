#### Generate proxy mesh for 3D assets, hair & cloth
 
<img src="proxy.jpg" alt="drawing" width="900"/>



### step 1, generate manifold mesh from visual mesh
```shell
blender --background -P  _1_manifold/blender_manifold.py  -- example_data/visual/asset.obj  example_data/proxy_path
```



### step 2, convert manifold to single_layer
```shell
python _2_single_layer_manifold/single_layer.py --v example_data/visual/asset.obj --p example_data/proxy_path  --n 600
```
```-v``` visual mesh; ```-p``` proxy folder path, ```-n``` number of verts for proxy mesh
  
  




### step 3, Run ACVD to remesh
- First install ACVD: https://github.com/valette/ACVD 
```shell
python _3_proxy_gen/proxy.py --p /home/rabbityl/workspace/clothProxyGeneration/example_data/proxy_path --acvd_path /home/rabbityl/workspace/ACVD/bin
```
```-p``` proxy folder path (need absolut path here for acvd)

### step 3.1, Remove occluded face 
```shell
blender -P  _3_proxy_gen/proxy_filer_occluded.py  -- example_data/proxy_path/proxy/part-0.ply
```


### step 4, Compute Skinning
```shell
python _4_skinning/skinning.py --v example_data/visual/asset.obj --p example_data/proxy_path
```
