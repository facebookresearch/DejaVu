
export MODEL_FLAGS_256="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --use_fp16 True"
export MODEL_FLAGS_128="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

#Can do batch size 8 at 128
#Can do batch size 2 at 256

python3 run_with_submitit.py \
	--feat_cond \
	--ssl_from_ckpt True \
	--ssl_loss 'simclr' \
	--ssl_arch 'resnet101' \
	--mlp '2048-256' \
	--data_dir '/datasets01/imagenet_full_size/061417/train/' \
	--batch_size 8 \
	--out_dir  '/checkpoint/caseymeehan/experiments/rcdm_2/simclr/rcdm_simclr_1000ep_300pc_A' \
	--ssl_model_pth '/checkpoint/caseymeehan/experiments/ssl_sweep/simclr/simclr_dssweep_300pc_A/model_ep1000.pth' \
	--use_supervised_activations 0 \
        --dataset_indices  '/private/home/caseymeehan/SSL_reconstruction/imgnet_partition/500_per_class/public.npy'  \
	--nodes 4 \
	--ngpus 8 \
	--timeout 4320 \
	--use_volta32 \
	--partition 'learnlab' \
	$MODEL_FLAGS_128
	
