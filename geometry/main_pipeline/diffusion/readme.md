# Env

1. Install requirements first
```bash
pip install -r requirements.txt
```

2. Install marching cube for test
```bash
cd ./utils/mc/sparse_mc/mc33
pip instal -e .
```

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
Training with one machine
```bash
bash sh/train_1gpu.sh
```

Training with distribution
```bash
bash sh/train_dist.sh $master_address $nodes_num $node_rank # runing on each machine, specify your own master_address, nodes_num, and node_rank
```

# Test
After traning, the checkpoint will be saved in 'configs/1view_gray_2048_flow/checkpoint-*'.
```bash
bash sh/test.sh
```