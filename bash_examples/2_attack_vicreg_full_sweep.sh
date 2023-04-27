# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#attack vicreg at 50, 100, 250, 500, 750, and 1000 epochs for 
#100k, 200k, 300k, and 400k dataset sizes

#UNCOMMENT ONE OF THE TWO SWEEPS 

#FOR RUNNING DATASET SWEEP 
# dataset_size=( 100 200 300 400 500 )
# epochs=( 1000 )

#FOR RUNNING EPOCHS SWEEP 
dataset_size=( 300 )
epochs=( 750 1000 ) #50 100 250 ) # 500 750 1000 )


for d in "${dataset_size[@]}"
do 
  for e in "${epochs[@]}"
  do 
	python label_inference_attack.py \
		--local 0 \
		--loss vicreg \
		--output_dir $LOGGING_FOLDER/vicreg/attack_sweeps/NN_attk_vicreg_${d}pc_${e}ep \
		--model_A_pth $LOGGING_FOLDER/vicreg/vicreg_dssweep_${d}pc_A/model_ep${e}.pth \
		--model_B_pth $LOGGING_FOLDER/vicreg/vicreg_dssweep_${d}pc_B/model_ep${e}.pth \
		--bbox_A_idx_pth $INDEX_FOLDER/${d}_per_class/bbox_A.npy \
		--bbox_B_idx_pth $INDEX_FOLDER/${d}_per_class/bbox_B.npy \
		--public_idx_pth $INDEX_FOLDER/500_per_class/public.npy \
		--imgnet_train_pth $IMAGENET_DATASET_DIR \
		--imgnet_bbox_pth $IMAGENET_BBOX_ANNOTATIONS \
		--k 100 \
		--k_attk 100 \
  done
done  
