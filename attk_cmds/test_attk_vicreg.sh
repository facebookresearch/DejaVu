d=100
e=1000

python3  label_inference_attack.py \
	--use_corner_crop 1 \
	--local 1 \
	--partition "learnlab" \
	--loss "vicreg" \
	--use_volta32 \
	--output_dir "/private/home/caseymeehan/SSL_reconstruction/test_attack/" \
	--model_A_pth "/checkpoint/caseymeehan/experiments/ssl_sweep/vicreg/vicreg_dssweep_${d}pc_A/model_ep${e}.pth" \
	--model_B_pth "/checkpoint/caseymeehan/experiments/ssl_sweep/vicreg/vicreg_dssweep_${d}pc_B/model_ep${e}.pth" \
	--bbox_A_idx_pth "/private/home/caseymeehan/imgnet_splits/${d}_per_class/bbox_A.npy" \
	--bbox_B_idx_pth "/private/home/caseymeehan/imgnet_splits/${d}_per_class/bbox_B.npy" \
	--public_idx_pth "/private/home/caseymeehan/imgnet_splits/500_per_class/public.npy" \
	--k 100 \
	--k_attk 100 

