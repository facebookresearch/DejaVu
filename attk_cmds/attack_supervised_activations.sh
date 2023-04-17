#attack supervised at 50, 100, 250, 500, 750, and 1000 epochs for 
#100k, 200k, 300k, and 400k dataset sizes

dataset_size=( 100 200 300 400 500 )
epochs=( 50 100 250 500 750 1000 )


for d in "${dataset_size[@]}"
do 
  for e in "${epochs[@]}"
  do 
	python3  label_inference_attack.py \
		--local 0 \
		--partition "learnlab" \
		--loss "supervised" \
		--use_supervised_linear 1 \
		--use_volta32 \
		--output_dir "/checkpoint/caseymeehan/experiments/ssl_sweep/supervised/attack_sweeps_w_linear_layer_activations/NN_attk_supervised_${d}pc_${e}ep" \
		--model_A_pth "/checkpoint/caseymeehan/experiments/ssl_sweep/supervised/supervised_dssweep_${d}pc_A/model_ep${e}.pth" \
		--model_B_pth "/checkpoint/caseymeehan/experiments/ssl_sweep/supervised/supervised_dssweep_${d}pc_B/model_ep${e}.pth" \
		--bbox_A_idx_pth "/private/home/caseymeehan/imgnet_splits/${d}_per_class/bbox_A.npy" \
		--bbox_B_idx_pth "/private/home/caseymeehan/imgnet_splits/${d}_per_class/bbox_B.npy" \
		--public_idx_pth "/private/home/caseymeehan/imgnet_splits/500_per_class/public.npy" \
		--k 100 \
		--k_attk 100 
  done
done  
