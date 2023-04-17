
#train vicreg sweeping the similarity parameter
#100k, 200k, 300k, and 400k dataset sizes

datasets=( 100 200 300 400 500 )

for d in "${datasets[@]}"
do 
for s in A B 
  do
    python train_ssl_2.py \
    	--config-file configs/vicreg_cfg.yaml \
       	--dist.timeout 3000  \
  	--dist.use_submitit 1 \
    	--data.train_dataset /checkpoint/caseymeehan/ffcv_datasets/${d}_per_class_${s}.beton \
  	--logging.test_attack 0 \
    	--logging.folder /checkpoint/caseymeehan/experiments/ssl_sweep/vicreg/vicreg_dssweep_${d}pc_${s} 0 
  done  
done
