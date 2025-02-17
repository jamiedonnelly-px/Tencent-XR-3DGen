cd /aigc_cfs_2/sz/proj/tex_cq/test_uv
in_json="/aigc_cfs_3/layer_tex/mcwy_2/manual_4class_0416/right_test.json"
out_root="/aigc_cfs_3/sz/result/compare_c_cxs/vis_mcwy2_test"
CUDA_VISIBLE_DEVICES=0 python batch_run_depth.py ${in_json} "${out_root}/raw_c_0.8_white" &
# CUDA_VISIBLE_DEVICES=3 python batch_run_t2i.py ${in_json} "${out_root}/t2i" &

echo "done"