#attack vicreg at various parameter values 

similarity=( 10 25 50 65 80 100 )
epochs=( 1000 )


for s in "${similarity[@]}"
do 
  for e in "${epochs[@]}"
  do 
	python3  label_inference_attack.py \
		--local 0 \
		--partition "learnlab" \
		--loss "vicreg" \
		--use_volta32 \
		--output_dir "/checkpoint/caseymeehan/experiments/ssl_sweep/vicreg/attack_sweeps/NN_attk_vicreg_sim${s}_${e}ep" \
		--model_A_pth "/checkpoint/caseymeehan/experiments/ssl_sweep/vicreg/vicreg_simsweep_sim${s}_A/model_ep${e}.pth" \
		--model_B_pth "/checkpoint/caseymeehan/experiments/ssl_sweep/vicreg/vicreg_simsweep_sim${s}_B/model_ep${e}.pth" \
		--bbox_A_idx_pth "/private/home/caseymeehan/SSL_reconstruction/imgnet_partition/100_per_class/bbox_A.npy" \
		--bbox_B_idx_pth "/private/home/caseymeehan/SSL_reconstruction/imgnet_partition/100_per_class/bbox_B.npy" \
		--public_idx_pth "/private/home/caseymeehan/SSL_reconstruction/imgnet_partition/500_per_class/public.npy" \
		--k 100 \
		--k_attk 100 \
		--mlp '8192-8192-8192' 
  done
done  
