
## Dependency

### Operating system

- tlinux (tencentos) 3.1 or higher
- ubuntu 20.04 or higher

### Mesa

On Centos/tlinux:

```
yum install libGL libGL-devel mesa-libGL
```

On Ubuntu:

```
apt-get install mesa-common-dev mesa-utils freeglut3-dev ninja-build 
```

## Usage

Remesh the file to approximately 50000 vertices (roughly 35000 faces):

```
python /path/to/decimate_ACVD.py \
--input_mesh_path 'some obj mesh path here' \
--output_mesh_path 'some obj mesh path here' \
--point_number 50000
```
