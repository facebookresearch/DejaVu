
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

datasets=( 100 200 300 400 500 )
# datasets=( 300 )

for d in "${datasets[@]}"
do 
  for s in A B
  do
    echo "writing ${d} per-class set ${s} train beton..."
    python write_ffcv_dataset.py \
	 --idx-path $INDEX_FOLDER/${d}_per_class/train_${s}.npy \
	 --write-path $BETONS_FOLDER/${d}_per_class_${s}.beton
  done 
done  
