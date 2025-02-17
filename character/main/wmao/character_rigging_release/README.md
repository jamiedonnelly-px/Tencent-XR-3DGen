## Character rigging from 2D annotation

#### Environment and Models
```shell
git clone https://github.com/Pointcept/Pointcept
cd Pointcept
cd libs/pointops
python setup.py install
```

#### Train hand segmentation model
need to first download all characters from mixamo website and obtain the hand point cloud and segmenttion.
```shell
cd hand_rigging
python train_point_transformer_contrast_new.py --rot_aug
```

#### run character rigging
replace the mesh path and web joint 2d path in the main function with the sample data in `data` folder and then run
```shell
python character_rigging_main.py
```


