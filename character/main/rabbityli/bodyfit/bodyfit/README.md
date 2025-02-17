## SmplX + offset registration

Download smplx weights from  ```/apdcephfs_cq8/share_2909871/rabbityli/smplx_data/data.zip``` and extract under `./delta`.

Key requirements: ```pytorch3d, open3d```

## Align without keypoint match

- Align vroid
```shell
cd delta
python align_vroid.py --mesh_path ./vroid_example/vroid.obj
```


## Align with keypoint match


- Align mcwy1
```shell
cd delta
python main.py \
--mesh_path /home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/mcwy_male/body.obj  \
--matches /home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/mcwy_male/correspondence/
```

    

- Align mcwy2
```shell
cd delta
python main.py \
--mesh_path /home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/MCWY2_F_T/body.obj  \
--matches /home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/MCWY2_F_T/correspondence/
```
    


- Align timer
```shell
cd delta
python main.py \
--mesh_path /home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/timer/untitled.obj --matches /home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/timer/correspondence/
```



- Align yuanmeng
```shell
cd delta
python main.py --mesh_path /home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/yuanmeng/naked/body.obj  --matches /home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/yuanmeng/correspondence/
```



- Align pubg
```shell
cd delta
python main.py --mesh_path /home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/pubg_male/naked/body.obj --matches /home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/pubg_male/correspondence/
```




    