# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#PICK WHICH SET OF MODELS TO PLOT 
models=( vicreg  )
plot_lin_probe=1
attack_fname=attack_sweeps #standard attack sweep 

for m in "${models[@]}"
do
echo "plotting ${m}" 
python plot_quant_results.py \
	--model ${m}\
	--epoch_sweep_dataset 300 \
	--dataset_sweep_epoch 1000  \
	--pct_confidences '20,100'  \
	--attack_fname ${attack_fname}  \
	--plot_lin_probe ${plot_lin_probe} \
	--attk_folder_path $LOGGING_FOLDER/vicreg/attack_sweeps \
	--lin_probe_folder_path $LOGGING_FOLDER/vicreg/lin_probe_sweeps_with_aug \
	--save_path $LOGGING_FOLDER/vicreg/plot/
done
