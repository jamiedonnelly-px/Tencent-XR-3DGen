## Install add-on and activate license

1. Depending on OS, zip either `quad_remesher_1_2/` for Ubuntu or `quad_remesher_1_3/` for Windows/Mac (you should zip the folder not their contents).
2. Open blender and select `Edit`->`Preferences`->`Add-ons`->`Install an add-on`, then select the zip file and install. If reinstalling, first remove the existing add-on.
3. In the same window, tick the add-on `Mesh: Quad Remesher 1.X` to enable it.
4. Click the top-right `<` button on the workspace window to open the add-on panel, then go to QuadRemesh and click `License Manager` to activate license.

## Example Usage
```
conda activate base
pip install -r requirements.txt
```

### run server and client
see `/tdmq/README.md`

### run plug in
tencent local ubuntu need use sudo blender + sudo quad_remesh activate..

To run the add-on you need to run blender in foreground with GUI enable; that is, __without__ the `-b` flag.
```bash
    # Mac
    /Applications/Blender.app/Contents/MacOS/Blender -P quad_remesh_and_bake.py -- \
        --source_mesh_path example/input.obj \
        --destination_mesh_path example/output.glb \
        --target_faces 3000 \
        --adaptive_size 0.9 \
        --tex_resolution 1024

    # linux
    /home/tencent/blender-3.6.14-linux-x64/blender -P quad_remesh_and_bake.py -- \
        --source_mesh_path example/input.obj \
        --destination_mesh_path example/output.glb \
        --target_faces 3000 \
        --adaptive_size 0.9 \
        --tex_resolution 1024

    # Windows
    "C:\\Program Files\\Blender Foundation\\Blender 3.6\\blender-launcher.exe" -P quad_remesh_and_bake_test.py -- ^
        --source_mesh_path example\\input.obj ^
        --destination_mesh_path example\\output.glb ^
        --target_faces 3000 ^
        --adaptive_size 0.9 ^
        --tex_resolution 1024
```