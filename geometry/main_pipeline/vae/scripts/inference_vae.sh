
python inference_vae.py \
    --model /aigc_cfs_14/weixuan/weights/vae/michelangelo-autoencoder/l2048-e64-ne8-nd16/20241227_h20_linear+n8192+occupancy+rotFalse+noise0.0+fourier+dsampleFalse+pfeat3+logits1.0+kl0.001+lr0.0001/ckpts \
    --output "outputs/" \
    --input "sample/sample.h5" \
    --device 6 \
    --octree_depth 8 \
    --mc_method old \
