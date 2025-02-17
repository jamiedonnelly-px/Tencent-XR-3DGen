raw_cfs1="/aigc_cfs"
raw_cfs2="/aigc_cfs_2"
raw_cfs3="/aigc_cfs_3"


gdp_cfs="/aigc_cfs_gdp"


cp_sh="/aigc_cfs_2/sz/lanjing_cfs/large_rclone.sh"



# bash ${cp_sh} ${raw_cfs1}/model/stable-diffusion-xl-base-1.0 ${gdp_cfs}/model/stable-diffusion-xl-base-1.0 
# bash ${cp_sh} ${raw_cfs1}/model/sdxl-vae-fp16-fix ${gdp_cfs}/model/sdxl-vae-fp16-fix
# bash ${cp_sh} ${raw_cfs1}/model/IP-Adapter ${gdp_cfs}/model/IP-Adapter

bash ${cp_sh} ${raw_cfs3}/sz/result/tex_control_2024/xl_mcwy2_manual/g4_pre_pos_4class_blip_1e-5/checkpoint-2000/controlnet \
 ${gdp_cfs}/sz/runtime_model/xl_mcwy2_manual/g4_pre_pos_4class_blip_1e-5/checkpoint-2000


python cp_meshs_to_gdp_cfs.py /aigc_cfs_gdp/sz/data/layer_human/web_0507/web_flatten.json /aigc_cfs_gdp/sz/data/layer_human/web_0507/web_flatten_gdp.json
