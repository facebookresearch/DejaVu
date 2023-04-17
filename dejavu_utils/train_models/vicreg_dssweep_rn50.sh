#train vicreg vs. dataset size with rn50 model 

#datasets=( 100 200 300 400 500 )
datasets=( 300 )

for d in "${datasets[@]}"
do 
for s in A B 
  do
    python train_model.py \
    	--config-file configs/vicreg_cfg.yaml \
	--model.arch resnet50 \
       	--dist.timeout 1440  \
  	--dist.use_submitit 1 \
    	--data.train_dataset /private/home/caseymeehan/SSL_reconstruction/betons/${d}_per_class_${s}.beton \
	--data.bboxA_dataset /private/home/caseymeehan/SSL_reconstruction/betons/${d}_per_class_bbox_A.beton \
	--data.bboxB_dataset /private/home/caseymeehan/SSL_reconstruction/betons/${d}_per_class_bbox_B.beton \
	--data.val_dataset /private/home/caseymeehan/SSL_reconstruction/betons/val.beton \
  	--logging.test_attack 1 \
  	--training.eval_freq 100 \
    	--logging.folder /checkpoint/caseymeehan/experiments/ssl_sweep/vicreg_rn50/vicreg_dssweep_${d}pc_${s} 0 
  done  
done
