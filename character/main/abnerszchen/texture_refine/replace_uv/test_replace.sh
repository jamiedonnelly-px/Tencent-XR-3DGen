/usr/blender-3.6.2-linux-x64/blender -b -P replace_glb_part_uvtex.py -- \
--source_mesh_path "/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/23e04bc2-feee-57f4-b525-3684e08a95a9/mesh/mesh.glb" \
--input_image_paths '/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/23e04bc2-feee-57f4-b525-3684e08a95a9/part_01/part_01.obj.png' '/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/23e04bc2-feee-57f4-b525-3684e08a95a9/part_01/part_01.obj.png' \
--object_part_names 'SM_Shoe_Left' 'SM_Shoe_Right' \
--output_mesh_path '/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/23e04bc2-feee-57f4-b525-3684e08a95a9/mesh/replace_mesh_a.glb' \
| tee ./log.txt

