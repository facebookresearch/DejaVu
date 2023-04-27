# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

echo "writing validation beton..."
python3 write_ffcv_dataset.py \
	 --idx-path $INDEX_FOLDER/val_set.npy \
	 --write-path $BETONS_FOLDER/val.beton
