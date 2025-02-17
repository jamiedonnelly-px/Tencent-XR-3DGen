# 3D AIGC Dataset Processing Pipeline

This repo contains codes for generating dataset used in 3D DiT and LRM training.

## Introduction

The repo contains the following functions:
- Render mesh in format obj/fbx/glb/vrm/pmx
- Robust 3D format converters that preserve material information:
   - glb / fbx / dae / pmx -> obj
   - max -> fbx
   - everything -> glb
- Remove parts in mesh with certain material
- Transform mesh into manifold
- Baking textures onto mesh
- Merge multiple materials into one material
- Dataset verification and organization


## Environments

### Docker file

We use some docker codes from [get3d](https://github.com/nv-tlabs/GET3D). 
It also contains a modified version of [occupancy network](https://github.com/autonomousvision/occupancy_networks).

### Install step-by-step

1. Install dependency in blender's python environment. An exampled command for Blender 4.2.5 ([Download address](https://mirrors.tuna.tsinghua.edu.cn/blender/blender-release/Blender4.2/blender-4.2.5-linux-x64.tar.xz)) is as follows: 
   ```/root/blender-4.2.5-linux-x64/4.2/python/bin/python3.11 -m pip install miniball scipy easydict trimesh pillow tqdm```。
2. Install dependencies in requirements.txt. Note that this part of requirements is installed in system python environments, 
therefore is not related with dependencies in blender's python environments.
3. Install OpenEXR and its python bindings in system python environments. 
First install openexr's binary lib: ```apt-get install openexr libopenexr-dev```.
For python versions higher than 3.9 you can install python binding directly using:```pip install openexr-python```; 
for version older we recommand using conda: ```conda install conda-forge::openexr-python```.
4. Install mesh-to-sdf from github source codes: ```git clone https://github.com/marian42/mesh_to_sdf && cd mesh_to_sdf && python setup.py install```.
Note that we do not recommend install this library from pip as its dependency may be broken.
5. Install intersection module in modules/intersection using pip.
6. Copy code in [this repo](https://github.com/autonomousvision/occupancy_networks) to /root folder.

## Usage

### Distributed processing

We propose a very simple distributed processing framework based on kubectl provided by k8s cluster in simple_batch_job.py. 
For readers that are not aware of k8s or kubectl, please refer to [this link](https://kubernetes.io/docs/home/). 
Taking the simplest job in tools/pipeline/h5_conversion.py as an example, we detail how to run this job distributedly on k8s cluster here:
- Setup a bash script as the following format:
```shell
pod_num=$1
pod_id=$2
pipeline_folder=$3
cd $pipeline_folder

python $pipeline_folder/tools/pipeline/h5_conversion.py \
--pod_num $pod_num \
--pod_id $pod_id \
--json_path 'json_file_containing_render_folder' \
--save_path 'output_folder' --pool_cnt 12
```
- Get the kubeconfig and yaml file of k8s cluster. 
Kubeconfig file is used to control access to the cluster, see [this link](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig/) for more information.
Yaml file is exactly the same file you used to create your deployment, 
like the nginx deployment file in [this link](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/).
- Run simple_batch_job.py file like this:
```shell
python tools/simple_batch_job.py \
--pod_cmd_folder 'TEMP_FOLDER_FOR_DEBUGGING' \
--pod_script_path 'ABSPATH_TO_H5_JOB_FILE' \
--kubeconfig_path 'ABSPATH_TO_kubeconfig_FILE' \
--pod_yaml_path 'ABSPATH_TO_k8s_yaml_FILE'
```


### Format conversion

#### Common conversion

Most conversion tasks can be done using process_conversion.py. 
Exampled conversion command is:
```shell
python process_conversion.py \
--data_json_path 'data_json_abspath' \
--output_folder 'output_converted_mesh_folder' \
--blender_root 'blender_exec_file_path' \
--log_folder 'conversion_log_folder' \
--pool_cnt 12 --copy_texture
```
Using the "copy_texture" in conversion commands will help rebuild texture paths for newly converted files.

#### Dealing with 3dsmax files

We currently only support converting 3dsmax files on Windows platform.
Note that 3dsmax software should be installed to its default path, 
and 3dsmaxbatch.exe should be in Windows's PATH.
Use the conversion/batch_max_fbx.py for automatically converting a list of 3dsmax files.

### Manifold data generation

Please use the process_standard_conflatio_data.py. Typical command is:
```shell
python process_standard_conflatio_data.py \
--output_folder 'output_converted_standard_mesh_folder' \
--fusion_temp_folder 'temp_folder_for_data_storage' \
--data_json_path 'data_json_abspath' \
--config_json_path 'conflatio_standard_json_path' \
--blender_root 'blender_exec_file_path' \
--log_folder 'conversion_log_folder_path' \
--pool_cnt 2 --triangulate --decimate --clean_mesh
```

### Dataset generation command

Typical commands for generating dataset is:

```shell
python ABSOLUTE_PATH/render_mesh_batch.py \
--data_json_path 'data_json_abspath' \
--output_folder 'output_render_results_folder' \
--config_json_path 'render_config_json_path' \
--generate_pose_config_json_path 'pose_generation_script_abspath' \
--pool_cnt 8 \
--silent \
--parse_exr \
--apply_render \
--apply_preprocess_mesh \
--preprocess_scale_mesh \
--log_folder 'conversion_log_folder_path' \
--proc_data_output_folder 'output_sample_results_folder' \
--pose_generation_mode 'RC/RSVC/RTriVC' \
--render_stage_string ‘render_stage_String' \
--blender_root 'blender_exec_file_path' 
```

### Data json structure

Typical json structure is：

```json
{
  "data":{
    "data_name":{
      "mesh_name":{
        "Mesh": "mesh_abspath",
        "Manifold": "watertight+manifold_mesh_abspath_after_remesh",
        "Fine": "watertight+manifold_mesh_abspath_before_remesh",
        "Original": "original_mesh_abspath"
      }
    }
  }
}
```


## Biography

1. Fischer, Kaspar, et al. "Fast smallest-enclosing-ball computation in high dimensions." European Symposium on
   Algorithms. Berlin, Heidelberg: Springer Berlin Heidelberg, 2003.
2. Alexa, Marc. "Super-fibonacci spirals: Fast, low-discrepancy sampling of SO(3)." Proceedings of the IEEE/CVF
   Conference on Computer Vision and Pattern Recognition. 2022.
3. Keinert, Benjamin, et al. "Spherical fibonacci mapping." ACM Transactions on Graphics (TOG) 34.6 (2015): 1-7.
4. Zhang, Longwen, et al. "CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets." ACM Transactions on Graphics (TOG) 43.4 (2024): 1-20.
5. Hong, Yicong, et al. "Lrm: Large reconstruction model for single image to 3d." arXiv preprint arXiv:2311.04400 (2023).
6. Gao, Jun, et al. "Get3d: A generative model of high quality 3d textured shapes learned from images." Advances In Neural Information Processing Systems 35 (2022): 31841-31854.
7. Mescheder, Lars, et al. "Occupancy networks: Learning 3d reconstruction in function space." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.
