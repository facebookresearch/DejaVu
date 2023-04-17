#attack simclr using backbone at 50, 100, 250, 500, 750, and 1000 epochs for 
#100k, 200k, 300k, 400k, 500k dataset sizes

#UNCOMMENT ONE OF THE TWO SWEEPS 

#FOR RUNNING DATASET SWEEP 
dataset_size=( 100 200 400 500 )
epochs=( 1000 )

#FOR RUNNING EPOCHS SWEEP 
# dataset_size=( 300 )
# epochs=( 50 100 250 500 750 1000 )



for d in "${dataset_size[@]}"
do 
  for e in "${epochs[@]}"
  do 
	python3  label_inference_attack.py \
		--use_corner_crop \
		--corner_crop_frac 0.40 \
		--local 0 \
		--partition "learnlab" \
		--loss "simclr" \
		--use_volta32 \
		--output_dir "/checkpoint/caseymeehan/experiments/ssl_sweep/simclr/attack_sweeps_corner_crop/NN_attk_simclr_${d}pc_${e}ep" \
		--model_A_pth "/checkpoint/caseymeehan/experiments/ssl_sweep/simclr/simclr_dssweep_${d}pc_A/model_ep${e}.pth" \
		--model_B_pth "/checkpoint/caseymeehan/experiments/ssl_sweep/simclr/simclr_dssweep_${d}pc_B/model_ep${e}.pth" \
		--bbox_A_idx_pth "/private/home/caseymeehan/SSL_reconstruction/imgnet_partition/${d}_per_class/bbox_A.npy" \
		--bbox_B_idx_pth "/private/home/caseymeehan/SSL_reconstruction/imgnet_partition/${d}_per_class/bbox_B.npy" \
		--public_idx_pth "/private/home/caseymeehan/SSL_reconstruction/imgnet_partition/500_per_class/public.npy" \
		--k 100 \
		--k_attk 100 \
		--mlp '2048-256' 
  done
done  
