
datasets=( 300 )


for d in "${datasets[@]}"
do 
  for s in A B
  do
    echo "writing ${d} per-class set ${s} bbox beton..."
    python3 write_ffcv_dataset_bboxes.py \
	 --idx-path /private/home/caseymeehan/SSL_reconstruction/imgnet_partition/${d}_per_class/bbox_${s}.npy \
	 --write-path /private/home/caseymeehan/SSL_reconstruction/betons/${d}_per_class_bbox_${s}.beton
  done 
done  
