#!/bin/bash

# Define the folder path
folder_path="/aigc_cfs_2/weimao/avatar_face_generation/test_data/celebrities"

# Loop through all .jpg and .jpeg files in the folder
for file in "$folder_path"/*/; do
  # Check if the file exists to avoid errors
  if [ -e "$file" ]; then
    echo "Processing $file"
    # Add any processing commands here
    python quest_head_deform_pipeline.py --file_path "$folder_path/$file/"
  fi
done