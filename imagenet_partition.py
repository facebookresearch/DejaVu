# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch, torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import datasets
from collections import defaultdict
from matplotlib import pyplot as plt
import os, sys
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser("create imgnet partitions")
    parser.add_argument("--imgnet_dir", default='', 
                        type=str, help="imgnet train folder")
    parser.add_argument("--bbox_dir", default='', 
                        type=str, help="bounding box annotation folder")
    parser.add_argument("--save_dir", default='', 
                        type=str, help="bounding box annotation folder")             
    return parser.parse_args()



def main(args): 
    print('FIRST MAP IMGNET ID (e.g. n01440764_10040) TO ImageFolder DATASET INDEX...') 
    dataset = ImageFolder(args.imgnet_dir, transform = transforms.ToTensor())
    
    #get list of class IDs (SSL training classes)
    keys = dataset.class_to_idx.keys()
    idx_to_class = {dataset.class_to_idx[k]:k for k in keys}
    SSL_train_classes = list(idx_to_class.values())
    print('idx to class key:', list(idx_to_class.keys())[0], ', value:', list(idx_to_class.values())[0])
    print('SSL train classes:', *SSL_train_classes[:5], '...')
    
    #Map image ID e.g. 'n01440764_10040' to its master index 
    ID_to_idx = {}

    for idx, samp in enumerate(dataset.samples): 
        pth = samp[0]
        ID = pth.split('/')[-1][:-5]
        ID_to_idx[ID] = idx
        
        
    print('ID to idx keys:', *list(ID_to_idx.keys())[:5], '...')
    print('ID to idx vals:', *list(ID_to_idx.values())[:5], '...')
        
    #generate dict mapping class to set of all indices in class 
    class_to_all_idxs = defaultdict(set)
    for i,s in enumerate(dataset.samples): 
        class_to_all_idxs[ idx_to_class[s[1]] ] |= {i}

    print('class to all idxs key:', list(class_to_all_idxs.keys())[0], ', value:', 
          *list(list(class_to_all_idxs.values())[0])[:5], '...')
    
    print('\nGET ALL IDXS WITH BOUNDING BOX ANNOTATIONS...') 
    #For each class, get all idxs that have bounding boxes 
    class_to_bbox_idxs = defaultdict(list) #dict mapping class ID to list of all indices with bbox files 
    bbox_not_in_ds = 0
    bbox_in_ds = 0

    for CLS in SSL_train_classes: 
        #get bbox directory/files for this class of images
        bbox_cls_dir = os.path.join(args.bbox_dir, 'Annotation', CLS)
        files = os.listdir(bbox_cls_dir)

        #get master index of each file
        for f in files: 
            f = f[:-4] #trim off '.xls'
            in_class = f[:len(CLS)] == CLS
            in_keys = f in ID_to_idx.keys()
            #every file should exist in keys, but a small number do not for some reason 
            if in_class and in_keys: 
                idx = ID_to_idx[f]
                class_to_bbox_idxs[CLS].append(idx)
                bbox_in_ds += 1
            else: 
                bbox_not_in_ds += 1

    print('Class to bbox idxs key:', list(class_to_bbox_idxs.keys())[0] )
    print('Class to bbox idxs val:', *list(class_to_bbox_idxs.values())[0][:5], '...' )
    print('\nbbox files in dataset:', bbox_in_ds)
    print('bbox files not in dataset:', bbox_not_in_ds)
    
    
    #check median number in each class
    num_in_class = [len(v) for v in class_to_bbox_idxs.values()]
    print('median number bbox annotated examples per class:', np.median(num_in_class))
    
    print('\nBREAK EACH CLASS INTO BBOX A, BBOX B, SHARED SET, AND PUBLIC SET...')
    #break each class into bbox_A, bbox_B, shared training set, and public set 
    def gen_datasets(num_per_class = 100): 
        #train A/B sets are given by bbox_A/B \cup shared
        shared = defaultdict(list) #shared set of training indices used by A & B
        bbox_A = defaultdict(list) #subset of train A
        bbox_B = defaultdict(list) #subset of train B
        public = defaultdict(list) #disjoint from train A \cup train B

        def cat(list_a, list_b): 
            return np.concatenate((list_a, list_b))

        for cls in SSL_train_classes: 
            all_idx = list(class_to_all_idxs[cls])
            all_bbox = class_to_bbox_idxs[cls]
            #get non-bbox idxs: 
            no_bbox = list(set(all_idx) - set(all_bbox)) 

            #split bounding boxes in 2
            bbox_split = int(len(all_bbox) / 2) 
            num_bbox = min(bbox_split, num_per_class) #include as many bbox examples as allowed 
            bbox_A[cls] = cat(bbox_A[cls], all_bbox[:num_bbox])
            bbox_B[cls] = cat(bbox_B[cls], all_bbox[num_bbox:2*num_bbox])

            #put the rest of the bounding boxes examples into public data 
            if 2*num_bbox < len(all_bbox): 
                public[cls] = cat(public[cls], all_bbox[2*num_bbox:])

            #if bbox examples do not exceed num_per_class, then add to shared set 
            num_to_add = num_per_class - num_bbox #number of non-bbox examples to add to SSL training sets 
            if num_to_add > 0: 
                shared[cls] = cat(shared[cls], no_bbox[:num_to_add])

            #build public set with remaining non-bbox examples 
            public[cls] = cat(public[cls], no_bbox[num_to_add:])

        return bbox_A, bbox_B, shared, public
    
    
    print('\nfirst generate all sets and test to make sure no intersection between bbox_A, bbox_B, and public')
    # Run tests on generated sets to ensure 
    #     - equality of bboxes / class 
    #     - equality of examples / class 
    #     - disjoint-ness of set A, set B, and public set
    
    npcs = [100, 200, 300, 400, 500]
    old_pub = None

    #get public set of 500k train set 
    #(smallest public set -- used for all tests)
    _, _, _, public_500 = gen_datasets(500)
    public_500 = np.concatenate([a for a in public_500.values()])
    pub_500 = set(public_500)


    for npc in npcs: 
        print(f"\n{npc} per class:")
        bbox_A, bbox_B, shared, public = gen_datasets(npc)

        #per-class tests: 
        min_cls_pub = 1e6 #minimum number of examples in any class 
        for cls in bbox_A.keys(): 
            s, p = shared[cls], public[cls]
            bA, bB = bbox_A[cls], bbox_B[cls]

            #A and B have same number 
            if len(bA) != len(bB): 
                print(f'set A and B different size in class {cls}')
                print(f"bA {bA}, bB {bB}")

            if len(p) < min_cls_pub: 
                min_cls_pub = len(p)

        print(f"STAT: public set minimum class size: {min_cls_pub}")

        shared = np.concatenate([a for a in shared.values()])
        bbox_A = np.concatenate([a for a in bbox_A.values()])
        bbox_B = np.concatenate([a for a in bbox_B.values()])
        public = np.concatenate([a for a in public.values()])
        sha = set(shared)
        b_A = set(bbox_A)
        b_B = set(bbox_B)
        pub = set(public)

        print(f"STAT: shared:{len(sha)} bbox A:{len(b_A)} bbox B:{len(b_B)} train:{len(b_A) + len(sha)} pub:{len(pub)}")

        #A and B are disjoint 
        if len(b_A & b_B) > 0: 
            print('BAD: bbox A and bbox B intersect')
        else: 
            print('GOOD: bbox A and bbox B do not intersect')

        #A and B are disjoint from shared
        if len( (b_A | b_B) & sha ) > 0: 
            print('BAD: bboxes intersect shared')
        else: 
            print('GOOD: bbox (attack) sets do not intersect with the shared set')

        #public is disjoint from the training sets 
        if len( (b_A | b_B | sha) & pub ) > 0: 
            print('BAD: the training sets intersect with public set')
        else: 
            print('GOOD: the training sets do not intersect with public set')

        #check that the public set of this iteration is included in public set of previous 
        #public sets shrink as training sets grow. This test confirms that we can use public set
        #of a larger train set split if we want 
        if old_pub: 
            if len(old_pub & pub) == len(pub): 
                print('GOOD: this pub. included in pub. set of previous iter')
        old_pub = pub

        #sanity check -- training sets are disjoint from 500k public set 
        if len( (b_A | b_B | sha) & pub_500 ) > 0: 
            print('BAD: the training sets intersect with public set')
        else: 
            print('GOOD: the training sets do not intersect with 500k public set')
            
    print('\nGENERATE SETS AND SAVE INDICES...')
    base_dir = Path(args.save_dir)
    #Regenerate and save set indices 
    for npc in npcs: 
        print(f"\n{npc} per class:")
        bbox_A, bbox_B, shared, public = gen_datasets(npc)

        pth = base_dir / f"{npc}_per_class"
        if not os.path.exists(pth): 
            os.makedirs(pth)

        # produce lists of indices: 

        bbox_A_list = np.concatenate([a for a in bbox_A.values()])
        print(f"bbox A size: {len(bbox_A_list)}")
        np.save(pth / 'bbox_A', bbox_A_list.astype(int))

        bbox_B_list = np.concatenate([b for b in bbox_B.values()])
        print(f"bbox B size: {len(bbox_B_list)}")
        np.save(pth / 'bbox_B', bbox_B_list.astype(int))

        public_list = np.concatenate([b for b in public.values()])
        print(f"public size: {len(public_list)}")
        np.save(pth / 'public', public_list.astype(int))

        if len(shared) > 0:
            shared_list = np.concatenate([b for b in shared.values()])
            np.save(pth / 'shared', shared_list.astype(int))
        else: 
            shared_list = []
            np.save(pth / 'shared', shared_list)
        print(f"shared size: {len(shared_list)}")

        train_A_list = np.concatenate([bbox_A_list, shared_list])
        print(f"train A size: {len(train_A_list)}")
        np.save(pth / 'train_A', train_A_list.astype(int))

        train_B_list = np.concatenate([bbox_B_list, shared_list])
        print(f"train B size: {len(train_B_list)}")
        np.save(pth / 'train_B', train_B_list.astype(int))
        
        
    print('\nGENERATE CLASS BALANCED VALIDATION SET...')
    #generate class-balance validation set for linear probe 
    _, _, _, public = gen_datasets(500)

    min_per_class = np.min([len(v) for v in public.values()])
    print(f'Minimum # examples in any class: {min_per_class}')

    #balance classes: 
    for cls in public.keys(): 
        public[cls] = public[cls][:min_per_class]

    public_list = np.concatenate([b for b in public.values()])

    print(f"val set size size: {len(public_list)}")
    np.save(base_dir  / 'val_set', public_list.astype(int))


if __name__ == "__main__":
    args = parse_args()
    main(args)       
