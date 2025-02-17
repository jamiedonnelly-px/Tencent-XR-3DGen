## All in one renderer class for forward rendering and baking

### requirements

  - nvdiffrast (at minimum 0.3.3 to support image or uv resolution higher than 2048)
      ```
        pip install git+https://github.com/NVlabs/nvdiffrast.git@729261d
        python -c "import nvdiffrast.torch as dr; dr.RasterizeCudaContext()"
      ```
  - kiui (for loading obj/glb/fbx)
      ```
      pip install kiui
      ```
  - pyembree and by extension, embree (for ray tracing)
      ```
        wget "https://github.com/RenderKit/embree/releases/download/v2.17.7/embree-2.17.7.x86_64.linux.tar.gz"
        tar xzf embree-2.17.7.x86_64.linux.tar.gz 
        source embree-2.17.7.x86_64.linux/embree-vars.sh
        echo "source $(pwd)/embree-2.17.7.x86_64.linux/embree-vars.sh" >> ~/.bashrc
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
  - cupy (for solving voronoi diagram in texture inpainting)
    ```
      pip install cupy
    ```
    
### usage
    see example*.py