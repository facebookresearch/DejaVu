#train simclr sweeping dataset size

datasets=( 100 200 300 400 500 )

for d in "${datasets[@]}"
do 
for s in A B 
  do
    python train_ssl_2.py \
    	--config-file configs/simclr_cfg.yaml \
	--model.arch resnet101 \
       	--dist.timeout 3000  \
  	--dist.use_submitit 1 \
    	--data.train_dataset /checkpoint/caseymeehan/ffcv_datasets/${d}_per_class_${s}.beton \
	--data.bboxA_dataset /checkpoint/caseymeehan/bbox_ffcv_datasets/${d}_per_class_bbox_A.beton \
	--data.bboxB_dataset /checkpoint/caseymeehan/bbox_ffcv_datasets/${d}_per_class_bbox_B.beton \
  	--logging.test_attack 1 \
    	--logging.folder /checkpoint/caseymeehan/experiments/ssl_sweep/simclr/simclr_dssweep_${d}pc_${s} 0 
  done  
done
