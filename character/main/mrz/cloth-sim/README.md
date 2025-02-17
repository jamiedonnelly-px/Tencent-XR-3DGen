# PBD based cloth simulator

A Position Based Dynamics (PBD/XPBD) based GPU cloth simulator.

- [x] GPU
- [x] Collision with static rigid
- [x] Cloth self-collision
- [ ] Cloth with moving objects
- [ ] Differentiability
- [ ] Sophiscated velocity damping


## Installation

```
pip install taichi
pip install trimesh
pip install open3d
```
pytorch3d (optional)

MacOS metal doesn't support int64/f64, `ti.gpu` may raise error when running on Mac, use `ti.cpu` instead.

### Non-Interactive garment fit mode
```
python run.py --garment_mesh ./test_meshes/garment_deformed_correspondence_test_cloth.obj --smpl_mesh ./test_meshes/smpl_deformed_test_cloth_opt_0000_0100.obj --textured_mesh ./test_meshes/test_textured_mesh/textured.obj --io_type trimesh --garment_name test_a_pose
```

```
python run.py --garment_mesh ./test_meshes/garment_test_correspondence_shirt_fix.obj --smpl_mesh ./test_meshes/smpl_deformed_shirt_fix_opt_0000.obj --io_type trimesh --garment_name test_t_pose
```
### Interactive mode
<p align="center">
  <img src="./gifs/interactive_mode_compressed.gif" width="50%" height="50%"/><br>
</p>
```
python run.py --garment_mesh ./test_meshes/garment_deformed_correspondence_test_cloth.obj --smpl_mesh ./test_meshes/smpl_deformed_test_cloth_opt_0000_0100.obj --io_type trimesh --garment_name test_a_pose -v
```

[Vulkan](https://vulkan.lunarg.com/sdk/home) is needed for [Taichi GGUI]((https://docs.taichi-lang.org/docs/ggui))

- Garment Fit 
1. The garment and SMPL body should be in a roughly fitted position after the previous landmark based fit process
2. Keep pressing "space" on keyboard for continously garment fit
3. When there the garment is a good fit on the SMPL body, click "export_obj" to export the garment and SMPL mesh

- Garment Fit and Simulation 
1. Perform the same procedure as Garment Fit
2. click "ready_for_sim", this is important for setting up a correct initial state for the simulator
3. Then click "run_sim" or press "r" on keyboard for starting the simulation