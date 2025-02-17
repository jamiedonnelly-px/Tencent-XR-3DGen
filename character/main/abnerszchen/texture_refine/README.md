# Docker:
mirrors.tencent.com/diffrender/get3d:tri
运行代码需要挂载ceph和cfs, 在docker中运行:
```
bash /usr/mount_ceph_cos.sh
```
# Bake UV
```
bash sd_tex_control/bake_uv.sh
```

# TexUV
SDXL + UV
```
bash texuv_obj.sh
```

# TexControl
SD + UV
```
bash uv_obj.sh
```

# TexGen WebUI
Unified UV-based services such as TexUV, TexControl and TexImgUV.

## grpc and gradio
1. set model of server in `configs/tex_gen.json`. 
 
2. Set the server address and run parameters in the file `client_texgen.json`

3. run server (and gradio)

```
cd grpc_backend
## run XL + mcwy
bash run_server_texgen.sh uv_mcwy 8986
## or run sd + mcwy
bash run_server_texgen.sh control_mcwy 8080

## then run gradio
bash run_gradio.sh
```
4. run client

```
# test client, select model in cfg json:
source /aigc_cfs_2/sz/grpc/bin/activate
python client_texgen.py --client_cfg_json ../configs/client_texgen.json --model_key uv_mcwy

```


------


# TexCreator
render and bake based.

## Generate dataset
### 生成训练/测试数据(depth-image pairs)
会在`out_dir`目录下生成`tex_creator.json`文件包含所有数据, 并自动且分成`tex_creator_train.json`, `tex_creator_val.json`, `tex_creator_test.json` 供训练和推理使用   
train/val是vae训练集做划分, test是vae的测试集

```
# 如果输入指定测试序列txt,严格按制定的划分测试集 
python dataset/creator_pre/make_human_json_wo_est.py {in_standard_json} {out_dir} --test_list_txt {test_list_txt}
# 如果不指定测试序列, 按名称排序后取前1%作为测试集,和vae一致
python dataset/creator_pre/make_human_json_wo_est.py {in_standard_json} {out_dir}
# Example:
python dataset/creator_pre/make_human_json_wo_est.py /aigc_cfs_2/weizhe/code_clean/rendering_free_onetri/scripts/lowpoly_0119.json /aigc_cfs_3/sz/data/tex/human/low_poly --test_list_txt /aigc_cfs/weixuan/code/DiffusionSDF/low_poly_test.txt
```

### 生成推理格式(来自vae/diffusion的obj和condition图)

```
python dataset/creator_pre/make_diffusion_json.py {obj_dir} {out_json}
# Example
python dataset/creator_pre/make_diffusion_json.py /aigc_cfs/weixuan/code/DiffusionSDF/config/stage1_vae_geotri_people_finetune_20240119/recon2024-01-22-15:39:18 /aigc_cfs_3/sz/data/tex/human/low_poly/test_infer_obj.json
```

merge input meshs, batch render with nvdiffrast. Then merge to json and split train/val/test. 

<details>
  <summary> 旧方法(bak) </summary>

  ## LRM dataset
  ```
  bash dataset/creator_pre/lrm_objaverse.sh ${diffusion_train_obj_dir} ${diffusion_test_obj_dir} ${out_data_dir}
  # Example
  bash dataset/creator_pre/lrm_objaverse.sh /aigc_cfs/sz/data/tex/diffusion_weapons \ 
  /aigc_cfs/neoshang/code/diffusers_triplane/configs/triplane_conditional_sdfcolor_objaverse_kl_v0.0.0/triplane_2023-12-21-20:39:16 \ 
  /aigc_cfs/sz/data/tex/objaverse_weapon
  ```

  ## Human dataset
  ```
  bash dataset/pre_render.sh ${diffusion_obj_dir} ${out_data_dir}
  # Example
  bash dataset/pre_render.sh /apdcephfs_cq3/share_2909871/shenzhou/proj/DiffusionSDF/config/sz_diffusion_4096_v0_test/recon2023-12-06-15:46:04 /apdcephfs_cq8/share_2909871/shenzhou/data/tex_refine/debug
  ```
</details>



## Train TexCreator SD Model
```
cd sd_tex_creator
bash exp_cq_g8.sh
```
## Infer TexCreator and optim texture map 
```
bash optim_obj.sh
```


# TexRefine
## Train TexRefine SD Model
```
cd sd_tex_refine
bash exp_cq_g8.sh
```

## Infer TexRefine and optim texture map
```
bash optim_obj_refine.sh
```

# grpc
## Begin server
```
cd grpc_backend
bash scripts/gen_code.sh
# 支持人物和物体两种模型:
python server_texcreator.py --model_key tex_creator_weapon
python server_texcreator.py --model_key tex_creator_human_design
```
## client test
```
cd grpc_backend
# 支持人物和物体两种模型,model_key需要和server对应(对应只是为了debug,实际上grpc直接调用就行):
python client_texcreator.py --model_key tex_creator_weapon
python client_texcreator.py --model_key tex_creator_human_design
```



----For Debug----

# Render obj
## Render LRM
```
python run_render_obj.py in_obj, in_pose_json, out_dir --render_res 512 --lrm
```

## Render Human
```
python run_render_obj.py in_obj, in_pose_json, out_dir --render_res 512
## Example:
python run_render_obj.py data/C0041_clean_15700_BuZhiHuoWu_Show1/manifold_full.obj data/cams/cam_parameters_select.json out/render_only
```

# Render obj and Optim texture
```
python run_optim_obj_texture.py in_obj, in_pose_json, out_dir
## Example:
python run_optim_obj_texture.py data/C0041_clean_15700_BuZhiHuoWu_Show1/manifold_full.obj data/cams/cam_parameters_select.json out/render_and_optim
```