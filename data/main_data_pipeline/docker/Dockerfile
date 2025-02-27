# Copyright (c) 2023-2024 Tencent.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.08-py3
FROM $BASE_IMAGE

RUN apt-get update && apt-get install -y ca-certificates curl wget
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y pkg-config libglvnd0 libgl1 libglx0 \
                                                                        libegl1 libgles2 libglvnd-dev \
                                                                        libgl1-mesa-dev libegl1-mesa-dev \
                                                                        libsm6 libxext6 libxkbcommon-x11-dev cmake \
                                                                        wget bzip2 git libeigen3-dev zsh screenfetch

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

ENV PYOPENGL_PLATFORM egl

RUN /bin/sh -c echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf &&     echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN DEBIAN_FRONTEND=noninteractive apt-get update --fix-missing && apt-get install -y libxi6 libgconf-2-4 libfontconfig1 libxrender1 nfs-common net-tools nfs-kernel-server \
                                                                                        libgoogle-perftools4 libnss3 libnspr4 libfuse2 xvfb freeglut3-dev libglew-dev \
                                                                                        libxext-dev libx11-dev x11proto-gl-dev htop openexr libopenexr-dev libpng-dev \
                                                                                        libtbb-dev libblosc-dev libfuse-dev curl build-essential

RUN pip install gputil h5py gdown meshzoo openexr-python Cython pybind11 pyrender tensorboard ninja plyfile trimesh ipdb imageio tensorboardX xatlas miniball open3d -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN cd /root && wget https://mirrors.tuna.tsinghua.edu.cn/blender/blender-release/Blender3.6/blender-3.6.12-linux-x64.tar.xz && tar -xvf blender-3.6.12-linux-x64.tar.xz && /root/blender-3.6.12-linux-x64/3.6/python/bin/python3.10 -m pip install miniball scipy easydict trimesh pillow tqdm
RUN cd /root && wget https://mirrors.tuna.tsinghua.edu.cn/blender/blender-release/Blender4.2/blender-4.2.1-linux-x64.tar.xz && tar -xvf blender-4.2.1-linux-x64.tar.xz && /root/blender-4.2.1-linux-x64/4.2/python/bin/python3.11 -m pip install miniball scipy easydict trimesh pillow tqdm
RUN cd /root && rm -rf blender-3.6.12-linux-x64.tar.xz blender-4.2.1-linux-x64.tar.xz

RUN cd /root && git clone https://github.com/marian42/mesh_to_sdf && cd mesh_to_sdf && python setup.py install

COPY occupancy_network /root/occupancy_network
RUN cd /root &&  cd occupancy_network/external/mesh-fusion && \
    cd libfusiongpu && mkdir build && cd build && cmake .. && make -j8 && cd .. && python setup.py build_ext --inplace && \
    cd .. && cd librender && python setup.py build_ext --inplace && \
    cd .. && cd libmcubes && python setup.py build_ext --inplace