cd /aigc_cfs_2/sz/proj/tex_cq/test_uv
# in_json="/aigc_cfs_3/layer_tex/readyplayerme/right_llava_3class_test.json"
# out_root="/aigc_cfs_3/sz/result/compare_c_cxs/vis_ready_test"
in_json="/aigc_cfs_3/layer_tex/mcwy_2/manual_4class_0416/right_test.json"
out_root="/aigc_cfs_3/sz/result/compare_c_cxs/vis_mcwy2_test"
# CUDA_VISIBLE_DEVICES=0 python batch_run_my.py ${in_json} "${out_root}/my" 

# python imgs_vis_merge.py "${out_root}/my/out.json" "${out_root}/my/out_vis" --select_key infer_uv_sdxl
# python imgs_to_pdf.py "${out_root}/my/out.json"  "${out_root}/my/out.pdf" --select_key infer_uv_sdxl

CUDA_VISIBLE_DEVICES=1 python batch_run_canny.py ${in_json} "${out_root}/pos_canny" 
python imgs_vis_merge.py "${out_root}/pos_canny/out.json" "${out_root}/pos_canny/out_vis" --select_key infer_uv_canny_sdxl
python imgs_to_pdf.py "${out_root}/pos_canny/out.json"  "${out_root}/pos_canny/out.pdf" --select_key infer_uv_canny_sdxl