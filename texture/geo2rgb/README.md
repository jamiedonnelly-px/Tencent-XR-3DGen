# Env

1. Install requirements first
```bash
pip install -r requirements.txt
```
blender 安装： wget https://mirrors.tuna.tsinghua.edu.cn/blender/blender-release/Blender3.6/blender-3.6.15-linux-x64.tar.xz

    需要解压到/root 路径下

下载 pytorch3d.  cd 地址 & pip install -e .

# Download ptetrain models
- Download 'facebook/dinov2-large' from huggingface and replace 'preprocessor_config.json' with 'pretrain_ckpts/dinov2-large/preprocessor_config.json'

- Download 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k' from huggingface

- Download vae pretrain ckpt
- Download diffusion pretrain ckpt
- Download RMBG-1.4 ckpt from huggingface

# Logs
We use wandb offline as logger here. you need to change the wandb private key to yours. in training scripts.
```python
os.environ["WANDB_API_KEY"] = "your_wandb_key" #### change to your own wandb key
```

# Data
- Images, pcd and json data sample, please refer './data' folder

- The dataloader will copy the origin data in json to the local machine during training, for speedup. If you don't need it just feel free to close it.

# Train
```bash
bash sh/train_geo2rgb.sh
```

# Test
```python
cd test_controlnet_6views
python batch_z123_control_obj_6views_non_rotate.py
```
