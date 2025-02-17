## An extendable rigging and animation method

#### Environment and Models
install pointops
```shell
git clone https://github.com/Pointcept/Pointcept
cd Pointcept
cd libs/pointops
python setup.py install
```

#### Train point feature extractor
```shell
bash run.sh
```

#### run igging
```shell
python main_fit_v2.py --config ./config/fitting_elephant.yaml --target_file TARGET_MESH_DIR
```


