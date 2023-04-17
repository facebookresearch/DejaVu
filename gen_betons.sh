
#datasets=( 200 400 500 )
datasets=( 300 )


for d in "${datasets[@]}"
do 
  for s in A B
  do
    echo "writing ${d} per-class set ${s} train beton..."
    python3 write_ffcv_dataset.py \
	 --idx-path /private/home/caseymeehan/SSL_reconstruction/imgnet_partition/${d}_per_class/train_${s}.npy \
	 --write-path /private/home/caseymeehan/SSL_reconstruction/betons/${d}_per_class_${s}.beton
  done 
done  
