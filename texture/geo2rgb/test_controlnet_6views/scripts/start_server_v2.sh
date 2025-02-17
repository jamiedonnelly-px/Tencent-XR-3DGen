# for azimuth[0, 45, 90, 135, 180, 225, 270, 315]
source /opt/anaconda3/bin/activate py39

unset http_proxy
unset https_proxy

cd /data/code

config_name=$1   ## config json name
model_name=$2   ## chekcpoint dir name in aigc_cfs_gdp
result_rmbg=$3  ## if the result use result_rmbg: string "true" or "false"

python server_v2.py --cfg_json ${config_name} --model_name ${model_name} --result_rmbg ${result_rmbg}