#attack simclr at various temperature parameter values 

temperature=( .05 .075 .10 .125 .15 .175 .25 .30 .50 .99 )
epochs=( 1000 )


for t in "${temperature[@]}"
do 
  for e in "${epochs[@]}"
  do 
	python3  label_inference_attack.py \
		--local 0 \
		--partition "learnlab" \
		--loss "simclr" \
		--use_volta32 \
		--output_dir /checkpoint/caseymeehan/experiments/ssl_sweep/simclr/attack_sweeps/NN_attk_simclr_p${t//./}t_${e}ep \
		--model_A_pth /checkpoint/caseymeehan/experiments/ssl_sweep/simclr/simclr_tempsweep_p${t//./}_A/model_ep${e}.pth \
		--model_B_pth /checkpoint/caseymeehan/experiments/ssl_sweep/simclr/simclr_tempsweep_p${t//./}_B/model_ep${e}.pth \
		--bbox_A_idx_pth "/private/home/caseymeehan/SSL_reconstruction/imgnet_partition/100_per_class/bbox_A.npy" \
		--bbox_B_idx_pth "/private/home/caseymeehan/SSL_reconstruction/imgnet_partition/100_per_class/bbox_B.npy" \
		--public_idx_pth "/private/home/caseymeehan/SSL_reconstruction/imgnet_partition/500_per_class/public.npy" \
		--k 100 \
		--k_attk 100  \
		--mlp '2048-256' 
  done
done  
