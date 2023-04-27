# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#train vicreg sweeping the similarity parameter
#100k, 200k, 300k, and 400k dataset sizes

datasets=( 100 200 300 400 500 )
for d in "${datasets[@]}"
do 
for s in A B 
  do
    python train_model.py \
		--config-file configs/vicreg_cfg.yaml \
		--dist.use_submitit 1 \
		--data.train_dataset $BETONS_FOLDER/${d}_per_class_${s}.beton \
		--data.val_dataset $BETONS_FOLDER/val.beton \
		--logging.folder $LOGGING_FOLDER/vicreg/vicreg_dssweep_${d}pc_${s} 0 
  done  
done
