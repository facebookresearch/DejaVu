#train supervised vs. dataset size 
#100k, 200k, 300k, and 400k dataset sizes

dataset_size=( 100 200 300 400 500 )

for d in "${dataset_size[@]}"
do 
	for s in A B
	do
	  python train_ssl_2_bnrelu.py \
	  	--config-file configs/supervised_cfg.yaml \
	       	--dist.timeout 4000  \
		--dist.use_submitit 1 \
	  	--data.train_dataset /checkpoint/caseymeehan/ffcv_datasets/${d}_per_class_${s}.beton \
		--logging.test_attack 1 \
	  	--logging.folder /checkpoint/caseymeehan/experiments/ssl_sweep/supervised/supervised_dssweep_${d}pc_${s} 0 
	done
done  
