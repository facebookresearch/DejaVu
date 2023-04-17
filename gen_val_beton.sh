echo "writing validation beton..."
python3 write_ffcv_dataset.py \
	 --idx-path /private/home/caseymeehan/SSL_reconstruction/imgnet_partition/val_set.npy \
	 --write-path /private/home/caseymeehan/SSL_reconstruction/betons/val.beton
