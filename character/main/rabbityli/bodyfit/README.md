### Fit SMPL body to scan with keypoint correspondence


### Install
- Download SMPL weights ```smplx```  from ```/apdcephfs_cq3/share_2909871/rabbityli/smpl_weights``` to ```./smpl_weights```.  
- ```pip install smplx```, other requirements include ```open3d```, and ```pytorch```


### Pick key points
Follow instructions in ```./Manual_Correspondence```.



### Fit scan with keypoint matches
```shell
python fit_scan.py \
--visual True \
--folder_path ../Manual_Correspondence/data/garen/manifold_full.obj \
--matches ../Manual_Correspondence/data/garen/correspondence/obj2smpl_0000_0000.json
```


