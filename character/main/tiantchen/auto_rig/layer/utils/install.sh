#!/bin/bash
root="/root/tinatchen/blender-3.6.13-linux-x64"
version="3.6"
python_version="python3.10"

${root}/${version}/python/bin/${python_version} -m pip install -U venus-api-base -i https://mirrors.cloud.tencent.com/pypi/simple/ --trusted-host mirrors.cloud.tencent.com

${root}/${version}/python/bin/${python_version} -m pip install rpyc -i https://pypi.tuna.tsinghua.edu.cn/simple

${root}/${version}/python/bin/${python_version} -m pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
