input_file=debug/raw2.obj
output_file=debug/out/quad2-nonom.glb
    /home/tencent/blender-3.6.14-linux-x64/blender -P quad_remesh_and_bake.py -- \
        --source_mesh_path ${input_file} \
        --destination_mesh_path ${output_file} \
        --target_faces -25000 \
        --adaptive_size 0.9 \
        --tex_resolution 1024 \
        --geom_only


#   job_id=ffec2d03-1911-4902-9051-ca1e8bf3d16c
#   input_file=/mnt/aigc_bucket_4/pandorax/quad_remesh/ffec2d03-1911-4902-9051-ca1e8bf3d16c/obj_mesh_mesh.obj
#   output_file=/mnt/aigc_bucket_4/pandorax/quad_remesh/ffec2d03-1911-4902-9051-ca1e8bf3d16c/quad_remesh/quad_mesh.glb
#   target_faces=-10000
#   adaptive_size=90
#   tex_resolution=1024
#   job_dir=/mnt/aigc_bucket_4/pandorax/quad_remesh/ffec2d03-1911-4902-9051-ca1e8bf3d16c
#   geom_only=True

# N=10  #

# for ((i=0; i<N; i++)); do
#     echo "Running - iteration $((i+1))"
#     /home/tencent/blender-3.6.14-linux-x64/blender -P quad_remesh_and_bake.py -- \
#         --source_mesh_path ${input_file} \
#         --destination_mesh_path ${output_file} \
#         --target_faces -10000 \
#         --adaptive_size 0.9 \
#         --tex_resolution 1024
# done


    # /home/tencent/blender-3.6.14-linux-x64/blender -P quad_remesh_and_bake.py -- \
    #     --source_mesh_path /mnt/aigc_bucket_4/pandorax/quad_remesh/debug.glb \
    #     --destination_mesh_path /mnt/aigc_bucket_4/pandorax/quad_remesh/out_local.glb \
    #     --target_faces 3000 \
    #     --adaptive_size 0.9 \
    #     --tex_resolution 1024

    # ll -hl /mnt/aigc_bucket_4/pandorax/quad_remesh/out_local.glb



    #    sudo /home/tencent/blender-3.6.14-linux-x64/blender -P quad_remesh_and_bake.py -- \
    #     --source_mesh_path /home/tencent/Downloads/timi_mesh/1/bake.glb \
    #     --destination_mesh_path /home/tencent/Downloads/timi_mesh/1/1_quad_bake.glb \
    #     --target_faces 3000 \
    #     --adaptive_size 0.9 \
    #     --tex_resolution 1024


    #          sudo   /home/tencent/blender-3.6.14-linux-x64/blender -P quad_remesh_and_bake.py -- \
    #     --source_mesh_path /home/tencent/Downloads/timi_mesh/2/bake.glb \
    #     --destination_mesh_path /home/tencent/Downloads/timi_mesh/2/2_quad_bake.glb \
    #     --target_faces 3000 \
    #     --adaptive_size 0.9 \
    #     --tex_resolution 1024


    #     sudo   /home/tencent/blender-3.6.14-linux-x64/blender -P quad_remesh_and_bake.py -- \
    #     --source_mesh_path /home/tencent/Downloads/timi_mesh/mesh_0/tdmq_out.glb \
    #     --destination_mesh_path /home/tencent/Downloads/timi_mesh/mesh_0/0_quad_bake.glb \
    #     --target_faces 3000 \
    #     --adaptive_size 0.9 \
    #     --tex_resolution 1024