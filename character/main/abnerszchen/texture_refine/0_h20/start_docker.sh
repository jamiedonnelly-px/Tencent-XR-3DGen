#!/bin/bash
name="sz1"
docker_name="mirrors.tencent.com/diffrender/3dd:flux"

docker run -itd \
    --name ${name} \
    --privileged \
	--cap-add=IPC_LOCK \
    --ipc=host --net=host \
    --gpus all \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /data0:/data0 \
    -v /data1:/data1 \
    -v /data2:/data2 \
    -v /data3:/data3 \
    -v /aigc_cfs:/aigc_cfs \
    -v /aigc_cfs_2:/aigc_cfs_2 \
    -v /aigc_cfs_3:/aigc_cfs_3 \
    -v /aigc_cfs_4:/aigc_cfs_4 \
    -v /aigc_cfs_5:/aigc_cfs_5 \
    -v /aigc_cfs_6:/aigc_cfs_6 \
    -v /aigc_cfs_7:/aigc_cfs_7 \
    -v /aigc_cfs_8:/aigc_cfs_8 \
    -v /aigc_cfs_9:/aigc_cfs_9 \
    -v /aigc_cfs_10:/aigc_cfs_10 \
    -v /aigc_cfs_11:/aigc_cfs_11 \
    -v /aigc_cfs_12:/aigc_cfs_12 \
    -v /aigc_cfs_13:/aigc_cfs_13 \
    -v /apdcephfs_cq8/share_2909871:/apdcephfs_cq8/share_2909871 \
    -v /apdcephfs_cq8/share_1615605:/apdcephfs_cq8/share_1615605 \
    -v /apdcephfs_cq10/share_1615605:/apdcephfs_cq10/share_1615605 \
    ${docker_name} /bin/bash
