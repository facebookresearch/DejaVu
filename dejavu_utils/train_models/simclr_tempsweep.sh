#train simclr sweeping the temperature parameter 
#train on 100k examples 

temperature=( .05 .075 .10 .125 .15 .175 .25 .30 .50 .99 )

for t in "${temperature[@]}"
do 
for s in A B
    do
	  python train_ssl_2.py \
	  	--config-file configs/simclr_cfg.yaml \
	  	--simclr.temperature ${t} \
	       	--dist.timeout 3000  \
		--dist.use_submitit 1 \
	  	--data.train_dataset /checkpoint/caseymeehan/ffcv_datasets/100_per_class_${s}.beton \
	  	--logging.folder /checkpoint/caseymeehan/experiments/ssl_sweep/simclr/simclr_tempsweep_p${t//./}_${s} 0 
    done  
done
