# Manul Correspondence

Create sparse manual correspondence between a character and SMPL model


### step 1. render character

Follow the instructions in data_disposition/render_for_registration

### step 2. pick points

- Assure the character folder follows the structure
```
|--garen
    |--render_for_registration
        |--color
        |--depth
        |--cam_parameters.json
    |--manifold_full.obj
```

- Run pick points
```shell
python pick_points.py  \
--sdir /home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/mcwy_male \
--tdir /home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/smpl_mesh_1280/smpl 
--sid 0 --tid 0
```
`--sid` and `--tid` in [0,1,2,3,4] stands for the rendering viewpoint of the RGB-D image, 
click ```s``` to save current match, click ```d``` to dump all matches.
The mathces will be saved to ```./data/mcwy_male/correspondence```





python pick_points.py --sdir /home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/yuanmeng  --id 0  --reuse