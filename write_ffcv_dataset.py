#script to turn bbox set A into ffcv dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
import torch 
import torchvision 
import numpy as np
import argparse



def main(args): 
    print('\nusing indices: ', args.idx_path)

    # Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
    dataset = torchvision.datasets.ImageFolder(args.imgnet_path)
    ds_idxs = np.load(args.idx_path)
    dataset = torch.utils.data.Subset(dataset, ds_idxs) 
    
    # Pass a type for each data field
    writer = DatasetWriter(args.write_path, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': RGBImageField(
            max_resolution=512,
            jpeg_quality=90,
        ),
        'label': IntField()
    })

    # Write dataset
    writer.from_indexed_dataset(dataset)
    print('written to: ', args.write_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description = 'write .beton', 
                        add_help = False
                        )
    parser.add_argument('--imgnet-path', 
                        default='/datasets01/imagenet_full_size/061417/train',
                        type=str,
                        help='path to imgnet train set'
                        )
    parser.add_argument('--idx-path', 
                        type=str,
                        help='path to dataset idxs'
                        )
    parser.add_argument('--write-path', 
                        type=str,
                        help='path to write .beton'
                        )
    args = parser.parse_args()
    main(args)


