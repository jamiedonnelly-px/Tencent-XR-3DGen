# Artisits-Created Meshes Generation


## Summary
This repo contains code for Artist-Created Meshes Generations, current version support pointclouds as condition, and the encoding and decoding strategy is the same as BPT. Code is tested with python 3.10 and CUDA 11.8.

## Installation Guide for Linux

```bash
pip install -r requirements.txt
```


## Training

```
accelerate launch train.py
```
you could modify the training settings with config/train-8k-8-16.yaml

## Inference conditioned on point clouds
```python
python infer.py \
    --config 'config/BPT-open-8k-8-16.yaml' \
    --model_path /path/to/model/ckpt \
    --output_path output/ \
    --batch_size 1 \
    --temperature 0.5 \
    --input_type mesh \
    --input_dir /path/to/your/dense/meshes
```
It requires ~12GB VRAM to run with fp16 precision. It takes averagely 2mins to generate a single mesh.


## Evaluation

```bash
python metrics.py \
    --input_dir /path/to/dense/meshes \
    --output_dir /path/to/output/meshes
```
