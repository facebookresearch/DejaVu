#PICK WHICH SET OF MODELS TO PLOT 
models=( vicreg simclr supervised )
plot_lin_probe=1
attack_fname=attack_sweeps #standard attack sweep 

#RN50 PLOTS (ONLY ATTACK NO LINPROBE) 
#models=( vicreg_rn50 simclr_rn50 )
#plot_lin_probe=0
#attack_fname=attack_sweeps #standard attack sweep 

#CORNER CROP 
#models=( vicreg simclr supervised )
#attack_fname=attack_sweeps_corner_crop
#plot_lin_probe=0

#BACKBONE
#models=( vicreg simclr supervised )
# models=supervised
# attack_fname=attack_sweeps_backbone
# plot_lin_probe=0


for m in "${models[@]}"
do
echo "plotting ${m}" 
python3 plot_quant_results.py \
	--model ${m}\
	--epoch_sweep_dataset 300 \
	--dataset_sweep_epoch 1000  \
	--pct_confidences '20,100'  \
	--attack_fname ${attack_fname}  \
	--plot_lin_probe ${plot_lin_probe}
done
