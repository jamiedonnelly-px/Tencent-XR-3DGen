## Overview 
This repo contains a refactored verison of [OpenVDB](https://www.openvdb.org) and its python interface. The goal is to provide python API for conversion between vdb and mesh, and allows sparse voxel sampling from VDB.

Following functionalities are supported:
- load/export vdb files
- convert levelset vdb to/from watertight mesh with triangle and/or quad faces
- perform sparse sampling of active vdb voxels

## Build

Tested with following dependencies
- cmake 3.25.2
- oneTBB 2021.13.0 built with g++7 (g++11 errored out)
- boost 1.86
- c-blosc 1.21.7 built with gcc 7

Note building OpenVDB may require a higher g++ version (e.g. g++ 11). 

```bash
mkdir -p openvdb/build && cd openvdb/build
cmake -DBOOST_STATIC_ASSERT=static_assert ..
make -j4 && make install
```

If you cannot import pyopenvdb from python, you may have to manually copy the binary to your python package path, e.g. 
```bash
cp openvdb/build/openvdb/openvdb/python/pyopenvdb.cpython-38-x86_64-linux-gnu.so /usr/local/lib/python3.8/dist-packages/ # replace 3.8 with your python version
```

## API 

The file `pyvdb.py` provides a class `VDBGrid` that implements above functions (see docstring). Or you can directly interface the compiled openvdb module like below.

### 
```python

import pyopenvdb as vdb

# read vdb
grid = vdb.read("xxx.vdb")

# OR construct TSDF vdb from mesh, with voxel size of 1/256 world unit and active bandwidth of 3 voxels
# SDF is truncated at active bandwidth
transform = vdb.createLinearTransform(voxelSize=1/256)
grid = vdb.FloatGrid.createLevelSetFromPolygons(verts, faces, transform=transform, halfWidth=3)

# write vdb
vdb.write("xxxx.vdb", grid=grid)

# export TSDF vdb to mesh, non-zero adaptivity creates a simplified mesh with some triangle faces
# when adaptivity=0 mesh contains only quad faces
points, triangles, quads = grid.convertToPolygons(isovalue=0, adaptivity=0.5)

# get active voxles that are within the active bandwidth from isosurface
# returns a numpy integer array of shape (N,3) where N is number of active voxels
voxel_coords = grid.activeIndices()

# convert voxel coordinates to world coordinates, or world to voxel with grid.transform.worldToIndex
# in this case world_coords = voxel_coords * voxelSize
world_coords = [grid.transform.indexToWorld(vc) for vc in voxel_coords]

# query TSDF value and gradient given integer voxel coordinates as an (M,3) array
sdf, sdf_grad = grid.queryValGradInt(voxel_coords)
```

