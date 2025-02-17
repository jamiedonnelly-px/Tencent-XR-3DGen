docker: 
`mirrors.tencent.com/diffrender/get3d:tri_1126`  

use model:


```
helloworldXL70: /aigc_cfs_2/model/helloworldXL70, converted from https://civitai.com/models/43977/leosams-helloworld-xl
sdxl-vae-fp16-fix: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/tree/main
xinsir/controlnet-depth-sdxl-1.0: https://huggingface.co/xinsir/controlnet-depth-sdxl-1.0/tree/main
IP-Adapter-plus: https://huggingface.co/h94/IP-Adapter/tree/main

RMBG-2.0: https://huggingface.co/briaai/RMBG-2.0/tree/main


## local path:
sd_path="/aigc_cfs_2/model/helloworldXL70",
vae_path="/aigc_cfs_gdp/model/sdxl-vae-fp16-fix",
control_path="/aigc_cfs_gdp/model/xinsir/controlnet-depth-sdxl-1.0",
ip_adapter_model_path="/aigc_cfs_gdp/model/IP-Adapter-plus",
use_ip_mode="plus_vit-h",

/aigc_cfs_gdp/model/RMBG-2.0
```


## 整体pipeline
整体pipeline参考 `pipeline_interface.py` 里的：
- `main_call_step1_generate_geom_mesh` 生几何  
- `main_call_step2_mesh_gen_texture` 生纹理
- `call_generate_texture_pipe` 白模着色标签页

## 白模着色frontal
把输入的各种奇怪mesh统一成pandorax用的obj, 然后生成d2rgb的参考图：
```
## in text mode
bash run_in_text.sh

## in image mode
bash run_in_image.sh
```

渲染gif：
```
blender_utils/blender_render_gif/render_debug.sh
```

线上的顺序：

- `blender_utils/anything_obj_converter.py` 把输入的各种奇怪mesh统一成pandorax用的obj   

- `tex_frontal/src/step0_depth_text2image.py` 输入图像分割 + mesh2image生成纹理的参考图   

- (d2rgb)  

- `blender_utils/glb_obj_converter.py`  obj->glb

- `blender_utils/render_gif/linux_main_renderRotatorMesh.py` 渲染动图gif




