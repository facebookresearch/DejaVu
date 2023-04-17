
#train vicreg sweeping the similarity parameter
#100k, 200k, 300k, and 400k dataset sizes

#similarity=( 10 25 50 100 )
similarity=( 65 80 )



for s in "${similarity[@]}"
do 
for e in A B
    do
    python train_ssl_2.py \
    	--config-file configs/vicreg_cfg.yaml \
    	--vicreg.sim_coeff ${s} \
       	--dist.timeout 3000  \
  	--dist.use_submitit 1 \
    	--data.train_dataset /checkpoint/caseymeehan/ffcv_datasets/100_per_class_${e}.beton \
    	--logging.folder /checkpoint/caseymeehan/experiments/ssl_sweep/vicreg/vicreg_simsweep_sim${s//./}_${e} 0 
    done
done  
