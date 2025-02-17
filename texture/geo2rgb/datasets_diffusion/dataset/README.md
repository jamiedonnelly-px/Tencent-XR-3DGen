# IO-Optimized Dataloader for Large Reconstruction Model (LRM)

## Direction for use
1. install dependencies [optional, only required for online PCD sampling]
    ```bash
    # install pysdf
    pip install pysdf
    # download embree to any folder that embree will be later installed to
    wget https://github.com/RenderKit/embree/releases/download/v2.17.7/embree-2.17.7.x86_64.linux.tar.gz
    tar xzf embree-2.17.7.x86_64.linux.tar.gz 
    source /workspace/embree-2.17.7.x86_64.linux/embree-vars.sh
    echo 'source /workspace/embree-2.17.7.x86_64.linux/embree-vars.sh' >> ~/.bashrc
    # build pyembree
    git clone https://github.com/scopatz/pyembree.git
    cython_ver=$(python3 -m cython --version 2>&1 | awk '{print $NF}') 
    cd pyembree/
    python3 -m pip install cython==0.29.36 
    python3 setup.py install
    cd ..
    pip install cython==$cython_ver
    # verify python import
    mkdir -p tmp; cd tmp # cd into an empty folder to prevent relative import
    python3 - <<END
    try:
        import pyembree
        from pyembree import rtcore_scene
    except ImportError as e:
        print(f"Error: pyembree installation has failed: {e}")
        exit(1)
    END
    ```
    if import failed in the last step, locate the build folder `pyembree/build/lib.linux-x86_64-*/pyembree` and manually 
    copy it to your python site packages directory. e.g.
    ```bash
    cp -r pyembree/build/lib.linux-x86_64-*/pyembree /usr/local/lib/python3.8/dist-packages/
    ```
    if error persists, ask @zacheng to install to your docker image 
2. write example yaml file for dataloader. modify include following lines to sample point cloud data (e.g. surface point normals, space points visibility and sdf) from new manifold meshes
    ```yaml
      n_surface_pts: 2048 # number of surface points to sample, returned as "surface_points" and "surface_normals"
      n_near_surface_pts: 2048 # number of near surface points to sample, returned as "near_surface_points", "near_surface_visibility", "near_surface_sdf"
      n_space_pts: 2048 # online sample space points to sample, returned as "space_points", "space_visibility", "space_sdf"
      
      online_sample_pcd: False # whether or not to sample points on the fly, or load pre-sampled point clouds; it's recommended to turn off online sampling for new manifold meshes 

      offline_sample_pcd_strategy: consecutive # "consecutive" or "uniform" to specify how training pcds are selected among all 500k pre-sampled points. because all offline samples are shuffled and stored in chunked h5 files, "consecutive" local reads are much faster but less random while "uniform" is fully random

      pcd_coordinate_system: "refcam, xyz" # a string that defines the coordinate convention of point cloud data including points and point normals
    ```
3. instantiate dataloader from yaml
    ```python
    from dataset import MVLRMDataset
    from omegaconf import OmegaConf

    dset = MVLRMDataset(OmegaConf.load('dataset/example_config.yaml'))
    dataloader = DataLoader(dset, batch_size=XXX, shuffle=True, num_workers=X, collate_fn=dset.collate_fn, **kwargs)

    for batch in dataloader:
        
        # commonly used attributes

        # images
        rgb_imgs = batch["color"] # [batch, 3, h, w] RGB images, in range [0,1]
        masks = batch["mask"] # [batch, 1, h, w] masks, in range [0,1]
        normal_images = batch["normal"] # [batch, 3, h, w] normal images, in range [0,1]

        # transform matrices
        cam2worlds = batch["cam2world"] # [batch, multiview, 4, 4] world is always relative to reference camera
        obj2worlds = batch["obj2world"] # [batch, 4, 4]
        intrinsics = batch["intrinsics"] # [batch, multiview, 3, 4] either second last or last column is zeros, depending on pinhole or orthographic
        mvp = batch["mvp"] # [batch, multiview, 4, 4] world to ndc transform

        # point clouds
        surface_pts = batch["surface_points"] # [batch, surf_points, 3]
        surface_nrm = batch["surface_normals"] # [batch, surf_points, 3]
        near_surface_pts = batch["near_surface_points"] # [batch, near_points, 3]
        near_surface_occ = batch["near_surface_occupancy"] # [batch, near_points, 1] 
        near_surface_sdf = batch["near_surface_sdf"] # [batch, near_points, 1], outside is negative, inside is positive
        space_pts = batch["space_points"] # [batch, space_points, 3]
        space_occ = batch["space_occupancy"] # [batch, space_points, 1] 
        space_sdf = batch["space_sdf"] # [batch, space_points, 1], outside is negative, inside is positive

    ```


