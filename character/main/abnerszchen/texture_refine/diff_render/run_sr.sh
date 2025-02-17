input_img=$1
outdir=$2
cd /aigc_cfs_2/sz/proj/Real-ESRGAN
python inference_realesrgan.py -n RealESRGAN_x4plus -i ${input_img} -o ${outdir} -s 4