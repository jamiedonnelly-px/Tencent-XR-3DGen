#!/bin/bash

# python mp_face_landmarker.py
# python face_texture_baking.py

# # Define a list of items
# items=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11")

# # Loop through the list
# for item in "${items[@]}"; do
#     face_dir="/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/cute_you${item}_uv.obj"
#     out_dir="/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/cute_you${item}_uv_aligned.obj"
#     python align_face.py --face_dir $face_dir --out_dir $out_dir

#     face_dir="/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/cute_you${item}_uv_aligned.obj"
#     face_texture_dir="/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/cute_you${item}_texture_map.png"
#     out_dir="/aigc_cfs_2/weimao/avatar_face_generation/output_face_baking_uv_edited/cute_you${item}_uv_aligned_combined.obj"
#     echo "Current item: $face_dir"
#     python face_head_sewing_v2.py --face_dir $face_dir --face_texture_dir $face_texture_dir --out_dir $out_dir
# done

# # Define a list of items
# items=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11")

# # Loop through the list
# for item in "${items[@]}"; do
#     img_dir="/aigc_cfs_2/weimao/avatar_face_generation/test_data/cute_you${item}.png"
#     python overall_pipeline_v2.py --file_path $img_dir
# done


#!/bin/bash

# Define the folder path
folder_path="/aigc_cfs_2/weimao/avatar_face_generation/test_data/celebrities"

# Loop through all .jpg and .jpeg files in the folder
for file in "$folder_path"/*.{jpg,jpeg}; do
  # Check if the file exists to avoid errors
  if [ -e "$file" ]; then
    echo "Processing $file"
    # Add any processing commands here
    python overall_pipeline_v2.py --file_path $file
  fi
done