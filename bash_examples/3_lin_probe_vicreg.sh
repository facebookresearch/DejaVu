# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#sweep linear probe results for vicreg across epochs and dataset sizes 

#UNCOMMENT ONE OF THE TWO SWEEPS 

#FOR RUNNING DATASET SWEEP 
# dataset_size=( 100 200 400 500 )
# epochs=( 1000 )

#FOR RUNNING EPOCHS SWEEP 
dataset_size=( 300 )
epochs=( 50 100 250 500 750 1000 )

for d in "${dataset_size[@]}"
do 
  for e in "${epochs[@]}"
  do 
    python lin_probe.py \
	--local 0 \
    	--partition '' \
    	--model_pth $LOGGING_FOLDER/vicreg/vicreg_dssweep_${d}pc_A/model_ep${e}.pth \
    	--end_lr 0.1 \
    	--epochs 20 \
    	--loss vicreg \
	--imgnet_train_pth $IMAGENET_DATASET_DIR \
    	--train_idx_pth $INDEX_FOLDER/${d}_per_class/train_A.npy \
    	--val_idx_pth $INDEX_FOLDER/val_set.npy \
    	--output_dir $LOGGING_FOLDER/vicreg/lin_probe_sweeps_with_aug/lp_vicreg_${d}pc_${e}ep 
  done
done  

