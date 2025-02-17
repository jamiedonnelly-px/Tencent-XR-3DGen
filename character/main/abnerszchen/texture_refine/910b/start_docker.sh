#!/bin/bash
name="sz"
docker_name="mirrors.tencent.com/xr3d/910b_ubuntu18.04:diffuser0.21"

docker run -itd \
    --name ${name} \
    --net=host \
    --ipc=host \
    --privileged \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci8 \
    --device=/dev/davinci9 \
    --device=/dev/davinci10 \
    --device=/dev/davinci11 \
    --device=/dev/davinci12 \
    --device=/dev/davinci13 \
    --device=/dev/davinci14 \
    --device=/dev/davinci15 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
    -v /var/log/npu/slog/:/var/log/npu/slog \
    -v /var/log/npu/profiling/:/var/log/npu/profiling \
    -v /var/log/npu/dump/:/var/log/npu/dump \
    -v /var/log/npu/:/usr/slog \
    -v /data0:/data0 \
    -v /data1:/data1 \
    -v /data2:/data2 \
    -v /data3:/data3 \
    -v /data4:/data4 \
    -v /data5:/data5 \
    -v /data6:/data6 \
    -v /data7:/data7 \
    -v /aigc_cfs:/aigc_cfs \
    -v /aigc_cfs_2:/aigc_cfs_2 \
    -v /aigc_cfs_3:/aigc_cfs_3 \
    -v /aigc_cfs_4:/aigc_cfs_4 \
    -v /aigc_cfs_5:/aigc_cfs_5 \
    -v /aigc_cfs_6:/aigc_cfs_6 \
    -v /aigc_cfs_7:/aigc_cfs_7 \
    -v /aigc_cfs_8:/aigc_cfs_8 \
    -v /aigc_cfs_9:/aigc_cfs_9 \
    ${docker_name} /bin/bash

