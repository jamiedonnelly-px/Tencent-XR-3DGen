import subprocess

mesh_path = "/aigc_cfs/weixuan/output/craftsman/2024-08-27-14:45:57/01508624-6c1c-4bb4-8ed2-ed363a3658bf.obj"
save_obj_name = "./tmp/01508624-6c1c-4bb4-8ed2-ed363a3658bf.obj"

mesh_path = "/aigc_cfs/weixuan/output/craftsman/2024-08-27-14:45:57/03633dd2-26cf-49ac-9caa-440ef14455c9.obj"
save_obj_name = "./tmp/03633dd2-26cf-49ac-9caa-440ef14455c9.obj"
cmd = f"/root/blender-3.6.15-linux-x64/blender -b -P /aigc_cfs/xibinsong/code/MMD_NPU_code/MMD_NPU_depth_2_rgb/MMD_NPU/tdmq/utils/obj_convert.py -- \
    --mesh_path '{mesh_path}' \
    --output_mesh_path '{save_obj_name}' \
    --process_stages 'smart_uv+add_image'"

subprocess.run(cmd, shell=True)