# Mesh curve annotation tool

## install
build addon, then install addon ```addon_2024XXXX.zip``` on blender
```shell
python build/make_addon.py
```



## try on a pair of face mesh

### step 1: convert mesh obj/ply mesh to np-array
```python mesh_to_ndarray.py```



### step 2: annotate source and target mesh using blender addon

- need to prepare a ```examples/face_src/config.json``` to define the ```curve_names``` and ```mesh_npy_path```
```shell
{
  "mesh_npy_path" : "/home/rabbityl/Desktop/curve_match_addon/examples/face_src/face_src.npy",
  "curve_names" : [
    "Eye-Brow_Left",
    "Eye-Brow_Right",
    "Nose-top2bot"
 ]
}
```

- in blender top right  select ```LoadMeshNPY``` panel --> ```load mesh config json``` button, load the above json file  
<img src="figs/load_config.png" alt="drawing" width="400"/> 

- select the mesh and turn to edit mode, work with Annotation3D panel to select the curves.  
<img src="figs/edit_mode.png" alt="drawing" width="400"/> 

- example of labeled curves  
<img src="figs/src.png" alt="drawing" width="400"/> <img src="figs/tgt.png" alt="drawing" width="400"/>


### step 3: convert curve matches to dense point matches
```shell
python curve_to_point_match.py \
 --src_mesh_npy examples/face_src/face_src.npy \
 --src_curves_json examples/face_src/curve.json \
 --tgt_mesh_npy examples/face_tgt/face_tgt.npy \
 --tgt_curves_json examples/face_tgt/curve.json
```
<img src="figs/point_match_viz.png" alt="drawing" width="800"/>