#attack vicreg at 50, 100, 250, 500, 750, and 1000 epochs for 
#100k, 200k, 300k, and 400k dataset sizes

#temperature=( .05 .075 .10 .125 .15 .175 .25 .30 .50 .99 )
frac=( .20 .25 .30 .35 .40 .45 .50)
epochs=( 1000 )


for f in "${frac[@]}"
do 
  for e in "${epochs[@]}"
  do 
	python3  label_inference_attack.py \
		--use_corner_crop \
		--corner_crop_frac ${f} \
		--local 0 \
		--partition "learnlab" \
		--loss "vicreg" \
		--use_volta32 \
		--output_dir /checkpoint/caseymeehan/experiments/ssl_sweep/vicreg/attack_sweeps_cropsize/NN_attk_vicreg_crop_p${f//./}_${e}ep \
		--model_A_pth /checkpoint/caseymeehan/experiments/ssl_sweep/vicreg/vicreg_dssweep_100pc_A/model_ep${e}.pth \
		--model_B_pth /checkpoint/caseymeehan/experiments/ssl_sweep/vicreg/vicreg_dssweep_100pc_B/model_ep${e}.pth \
		--bbox_A_idx_pth "/private/home/caseymeehan/imgnet_splits/100_per_class/bbox_A.npy" \
		--bbox_B_idx_pth "/private/home/caseymeehan/imgnet_splits/100_per_class/bbox_B.npy" \
		--public_idx_pth "/private/home/caseymeehan/imgnet_splits/500_per_class/public.npy" \
		--k 100 \
		--k_attk 100  
  done
done  
