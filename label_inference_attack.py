# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#modified model for gen ssl trained files

import numpy as np
import torch, torchvision
from torchvision import transforms
from torch import nn, optim
from torchvision.datasets import ImageFolder
from torchvision import datasets
from torchvision.models import resnet50, resnet101
from torch.utils.data import DataLoader, Subset
from matplotlib import pyplot as plt
import argparse
import os, sys
from pathlib import Path
import time
import xmltodict
import faiss
from scipy.stats import entropy
import submitit
import uuid

#for loading datasets: 
from dejavu_utils.utils import (aux_dataset, crop_dataset, 
                                ImageFolderIndex, stopwatch, 
                                SSLNetwork, SSL_Transform)

def parse_args():
    parser = argparse.ArgumentParser("Submitit for NN Attack")

    parser.add_argument("--local", default = 0, type=int, help="whether to run on devfair")
    parser.add_argument("--local_gpu", default = 1, type=int, help="which device to use during local run")
    #slurm args 
    parser.add_argument("--timeout", default=180, type=int, help="Duration of the job")
    parser.add_argument("--partition", default="learnlab", type=str, help="Partition where to submit")
    parser.add_argument("--mem_gb", default=250) 
    parser.add_argument("--use_volta32", action='store_true')
    parser.add_argument("--output_dir", type=Path) 
    
    #attack args
    parser.add_argument("--model_A_pth", type=Path) 
    parser.add_argument("--model_B_pth", type=Path) 
    parser.add_argument("--mlp", type=str, default='8192-8192-8192') 
    parser.add_argument("--use_backbone", type=int, default=0) 
    parser.add_argument("--loss", type=str, default='barlow') 
    parser.add_argument("--use_supervised_linear", type=int, default=0) 
    parser.add_argument("--use_corner_crop", action='store_true')
    parser.add_argument("--corner_crop_frac", type=float, default=0.3)
    parser.add_argument("--bbox_A_idx_pth", type=Path) 
    parser.add_argument("--bbox_B_idx_pth", type=Path) 
    parser.add_argument("--public_idx_pth", type=Path) 
    parser.add_argument("--imgnet_train_pth", type=Path, default="/datasets01/imagenet_full_size/061417/train") 
    parser.add_argument("--imgnet_bbox_pth", type=Path, default="/private/home/caseymeehan/imgnet_bboxes") 
    parser.add_argument("--k", type=int, default=100, 
            help="number of neighbors to search when building index") 
    parser.add_argument("--k_attk", type=int, default=100, 
            help="number of neighbors to use in attack") 
    parser.add_argument("--resnet50", action='store_true')
             
    
    return parser.parse_args()

#NN_adversary(model_A, public_loader, args.gpu, args.k, args.k_attk, args.use_backbone)
class NN_adversary: 
    def __init__(self, model, public_DL, args):# gpu, k = 100, k_attk = None, use_backbone = 0): 
        self.model = model
        self.use_backbone = args.use_backbone
        self.public_DL = public_DL
        self.use_supervised_linear = args.use_supervised_linear and (args.loss == 'supervised')

        self.gpu = args.gpu

        #Nearest neighbor index 
        self.k = args.k #number of neighbors to collect, not necessarily use in attack 
        self.index = None
        self.public_idxs = []
        self.public_labels = []
        
        #Nearest neighbor data on attk set 
        self.neighb_idxs = [] 
        self.neighb_labels = []
        self.attk_idxs = []
        self.class_cts = []
        
        #activation attack 
        self.activations = []
        
        #attack uncertainty on attk set 
        self.attk_uncert = []
        self.topk_preds = []
        
        #num neighbs 
        if not args.k_attk: 
            self.k_attk = self.k
        else: 
            self.k_attk = args.k_attk

    def get_embed(self, x): 
        embed = self.model.module.net(x)
        if not self.use_backbone:
            embed = self.model.module.projector(embed)
            if self.use_supervised_linear: 
                embed = self.model.module.fc(embed)
        return embed

        
    def build_index(self): 
        DS = []
        n = len(self.public_DL)
        print_every = int(n / 10)
        sw = stopwatch(n)
        sw.start()
        print('gathering public embeddings...')
        for i, (x,y,idx) in enumerate(self.public_DL): 
            x = x.cuda()
            with torch.no_grad(): 
                embed = self.get_embed(x)

                DS.append(embed.cpu().numpy())
                self.public_labels.append(y.numpy())
                self.public_idxs.append(idx.numpy())
            if (i+1) % print_every == 0: 
                print(f"progress: {i/n:.2f}, min remaining: {sw.time_remaining(i)/60:.1f}")

        DS = np.concatenate(DS, axis = 0)
        self.public_labels = np.concatenate(self.public_labels, axis = 0)
        self.public_idxs = np.concatenate(self.public_idxs, axis = 0)
        
        print("building faiss index...")
        nlist = 1000 #number of voronoi cells in NN indexer
        dim = DS.shape[1]
        quantizer = faiss.IndexFlatL2(dim)
        
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, self.gpu, quantizer)
        
        self.index.train(DS)
        self.index.add(DS)
        
    def get_neighbors(self, aux_DL): 
        n = len(aux_DL)
        print_every = int(n / 10)
        sw = stopwatch(n)
        sw.start()
        print("getting neighbors...")
        for i, (x,y,idx) in enumerate(aux_DL): 
            with torch.no_grad():
                embed = self.get_embed(x.cuda())
            
            #get idxs with usable bounding boxes 
            good_idx = y>-1
            embed = embed[good_idx].cpu().numpy()
            y = y[good_idx].numpy()
            idx = idx[good_idx]
            
            if len(y) > 0: 
                #get indices of nearest neighbors
                D,I = self.index.search(embed, self.k) 
                self.neighb_idxs.append(self.public_idxs[I])
            
                #get labels of nearest neighbors 
                k_neighb_labels = self.public_labels[I.ravel()]
                k_neighb_labels = k_neighb_labels.reshape(I.shape)
                self.neighb_labels.append(k_neighb_labels)
                
                #get indices of the examples attacked 
                #(that we just got neighbors of)
                self.attk_idxs.append(idx)
                
            if (i+1) % print_every == 0: 
                print(f"progress: {i/n:.2f}, min remaining: {sw.time_remaining(i)/60:.1f}")
                
        self.neighb_idxs = np.concatenate(self.neighb_idxs)
        self.neighb_labels = np.concatenate(self.neighb_labels)
        self.attk_idxs = np.concatenate(self.attk_idxs)
        
        #get class counts 
        self.class_cts = np.apply_along_axis(np.bincount, axis=1, 
                                arr=self.neighb_labels[:,:self.k_attk], minlength=1000)
        
        #get confidence 
        self.attk_uncert = entropy(self.class_cts, axis = 1)
        
    def get_activations(self, aux_DL): 
        n = len(aux_DL)
        print_every = int(n / 10)
        sw = stopwatch(n)
        sw.start()
        print("getting neighbors...")
        for i, (x,y,idx) in enumerate(aux_DL): 
            with torch.no_grad():
                embed = self.get_embed(x.cuda())
            
            #get idxs with usable bounding boxes 
            good_idx = y>-1
            embed = embed[good_idx].cpu().numpy()
            y = y[good_idx].numpy()
            idx = idx[good_idx]
            
            if len(y) > 0: 
                self.activations.append(embed)
                self.attk_idxs.append(idx)

            if (i+1) % print_every == 0: 
                print(f"progress: {i/n:.2f}, min remaining: {sw.time_remaining(i)/60:.1f}")
                
        self.activations = np.concatenate(self.activations)
        self.attk_idxs = np.concatenate(self.attk_idxs)
        
        #get class counts 
        self.class_cts = self.activations
        
        #get confidence 
        self.attk_uncert = - np.max(self.class_cts, axis = 1)
        
    def compute_topk_preds(self, k): 
        """get topk NN predictions on all attacked examples
        Input:
            k: compute top k NN predictions 
        """
        topk_cts, topk_preds = torch.topk(torch.Tensor(self.class_cts), k, dim = 1)
        self.topk_preds = np.array(topk_preds).astype(int)
        
    def attack_p_frac(self, most_conf_frac): 
        """get topk NN predictions on the most confident fraction
        of attacked examples. Run after compute_topk_preds
        Input:
            most_conf_frac: scalar [0,1], most confident frac of examps
        Return: 
            frac_idxs: indices of the most confident examples
            preds: topk predictions of these examples 
        """
        n_most_conf = int(most_conf_frac * len(self.attk_uncert))
        
        #get most confident subset of indices
        most_conf_idxs = np.argsort(self.attk_uncert)[:n_most_conf]
        
        #get predictions 
        most_conf_preds = self.topk_preds[most_conf_idxs, :]
        
        return self.attk_idxs[most_conf_idxs], most_conf_preds

#Run attack code
def main(args): 
    #init distributed process because saved models need this
    print('Initializing process group...') 
    torch.distributed.init_process_group(
       backend='nccl', init_method=args.dist_url,
       world_size=args.world_size, rank=args.rank)
   
    torch.cuda.set_device(args.gpu)

    print('Loading models...')

    #load up models A and B 
    if args.resnet50: 
        arch = 'resnet50'
    else:
        arch = 'resnet101'

    model_A = SSLNetwork(arch = arch, 
                          remove_head = 0, 
                          mlp = args.mlp, 
                          fc = 0,
                          patch_keep = 1.0,
                          loss = args.loss).cuda()
    if (args.loss == 'supervised') and args.use_supervised_linear: 
        model_A.fc = nn.Sequential(
            nn.BatchNorm1d(model_A.num_features), 
            nn.ReLU(inplace = True), 
            nn.Linear(model_A.num_features, 1000)
            ).cuda()
    model_A = torch.nn.parallel.DistributedDataParallel(model_A, device_ids=[args.gpu])
    ckpt = torch.load(args.model_A_pth, map_location='cpu')
    model_A.load_state_dict(ckpt['model'], strict = False)
    _ = model_A.eval()
    
    model_B = SSLNetwork(arch = arch, 
                          remove_head = 0, 
                          mlp = args.mlp, 
                          fc = 0,
                          patch_keep = 1.0,
                          loss = args.loss).cuda()
    if (args.loss == 'supervised') and args.use_supervised_linear: 
        model_B.fc = nn.Sequential(
            nn.BatchNorm1d(model_A.num_features), 
            nn.ReLU(inplace = True), 
            nn.Linear(model_A.num_features, 1000)
            ).cuda()
    model_B = torch.nn.parallel.DistributedDataParallel(model_B, device_ids=[args.gpu])
    ckpt = torch.load(args.model_B_pth, map_location='cpu')
    model_B.load_state_dict(ckpt['model'], strict = False)
    _ = model_B.eval()    

    #Get datasets
    #bbox set A 
    print('initializing datasets...')
    bbox_A_idx = np.load(args.bbox_A_idx_pth)#[:n]
    if not args.use_corner_crop: 
        aux_set_A = aux_dataset(args.imgnet_train_pth, args.imgnet_bbox_pth, bbox_A_idx)
    else: 
        aux_set_A = crop_dataset(args.imgnet_train_pth, bbox_A_idx, crop_frac = args.corner_crop_frac)
    aux_loader_A = DataLoader(aux_set_A, batch_size = 64, num_workers = 8, shuffle = True)
    
    #bbox set B 
    bbox_B_idx = np.load(args.bbox_B_idx_pth)#[:n]
    if not args.use_corner_crop: 
        aux_set_B = aux_dataset(args.imgnet_train_pth, args.imgnet_bbox_pth, bbox_B_idx)
    else: 
        aux_set_B = crop_dataset(args.imgnet_train_pth, bbox_B_idx, crop_frac = args.corner_crop_frac)
    aux_loader_B = DataLoader(aux_set_B, batch_size = 64, num_workers = 8, shuffle = True)
    
    #public set
    public_idx = np.load(args.public_idx_pth)#[:n]
    public_set = ImageFolderIndex(args.imgnet_train_pth, SSL_Transform(), public_idx)
    public_loader = DataLoader(public_set, batch_size = 64, shuffle = False, num_workers=8) 
    
    #full imgnet dataset 
    imgnet_set = ImageFolder(args.imgnet_train_pth, transform = transforms.ToTensor())

    #Now attack sets A & B with adversary A 
    adv_A_attk_A = NN_adversary(model_A, public_loader, args)
    adv_A_attk_B = NN_adversary(model_A, public_loader, args)

    if args.use_supervised_linear: 
        adv_A_attk_A.get_activations(aux_loader_A)
        adv_A_attk_B.get_activations(aux_loader_B)
    else: 
        #build adversary A index
        print("Building index for adversary A")
        adv_A_attk_A.build_index()
        adv_A_attk_B.index = adv_A_attk_A.index
        adv_A_attk_B.public_labels = adv_A_attk_A.public_labels
        adv_A_attk_B.public_idxs = adv_A_attk_A.public_idxs
    
        #attack sets A and B
        print("Adversary A attacking set A using auxiliary data from A")
        adv_A_attk_A.get_neighbors(aux_loader_A)
        print("Adversary A attacking set B using auxiliary data from B")
        adv_A_attk_B.get_neighbors(aux_loader_B)
    
        #free gpu memory 
        adv_A_attk_A.index.reset()

    #Now attack sets A & B with adversary B 
    adv_B_attk_A = NN_adversary(model_B, public_loader, args)
    adv_B_attk_B = NN_adversary(model_B, public_loader, args)

    if args.use_supervised_linear: 
        adv_B_attk_A.get_activations(aux_loader_A)
        adv_B_attk_B.get_activations(aux_loader_B)
    else: 
        #build adversary A index
        print("Building index for adversary B")
        adv_B_attk_A.build_index()
        adv_B_attk_B.index = adv_B_attk_A.index
        adv_B_attk_B.public_labels = adv_B_attk_A.public_labels
        adv_B_attk_B.public_idxs = adv_B_attk_A.public_idxs
    
        #attack sets A and B
        print("Adversary B attacking set A using auxiliary data from A")
        adv_B_attk_A.get_neighbors(aux_loader_A)
        print("Adversary B attacking set B using auxiliary data from B")
        adv_B_attk_B.get_neighbors(aux_loader_B)
    
        #free gpu memory 
        adv_B_attk_A.index.reset()

    def get_labels(idxs): 
        #get ground-truth labels of examples indices
        return np.array([imgnet_set.samples[i][1] for i in idxs])[:,None]

    #Save attack data
    #A attk A
    np.save(args.output_dir / 'A_attk_A_neighb_idxs', adv_A_attk_A.neighb_idxs)
    np.save(args.output_dir / 'A_attk_A_neighb_labels', adv_A_attk_A.neighb_labels)
    np.save(args.output_dir / 'A_attk_A_attk_idxs', adv_A_attk_A.attk_idxs)
    np.save(args.output_dir / 'A_attk_A_labels', get_labels(adv_A_attk_A.attk_idxs))
    #A attk B 
    np.save(args.output_dir / 'A_attk_B_neighb_idxs', adv_A_attk_B.neighb_idxs)
    np.save(args.output_dir / 'A_attk_B_neighb_labels', adv_A_attk_B.neighb_labels)
    np.save(args.output_dir / 'A_attk_B_attk_idxs', adv_A_attk_B.attk_idxs)
    np.save(args.output_dir / 'A_attk_B_labels', get_labels(adv_A_attk_B.attk_idxs))
    #B attk A
    np.save(args.output_dir / 'B_attk_A_neighb_idxs', adv_B_attk_A.neighb_idxs)
    np.save(args.output_dir / 'B_attk_A_neighb_labels', adv_B_attk_A.neighb_labels)
    np.save(args.output_dir / 'B_attk_A_attk_idxs', adv_B_attk_A.attk_idxs)
    np.save(args.output_dir / 'B_attk_A_labels', get_labels(adv_B_attk_A.attk_idxs))
    #B attk B
    np.save(args.output_dir / 'B_attk_B_neighb_idxs', adv_B_attk_B.neighb_idxs)
    np.save(args.output_dir / 'B_attk_B_neighb_labels', adv_B_attk_B.neighb_labels)
    np.save(args.output_dir / 'B_attk_B_attk_idxs', adv_B_attk_B.attk_idxs)
    np.save(args.output_dir / 'B_attk_B_labels', get_labels(adv_B_attk_B.attk_idxs))

    #Get test results on set A and B


    def get_acc(idxs, preds): 
        #get array indicating whether topk preds are correct
        true_labels = get_labels(idxs) 
        return (true_labels == preds).sum(axis = 1)    

    topks = [1,5]

    for topk in topks: 
        print(f"top-{topk} results:")

        adv_A_attk_A.compute_topk_preds(topk)
        adv_A_attk_B.compute_topk_preds(topk)
        adv_B_attk_A.compute_topk_preds(topk)
        adv_B_attk_B.compute_topk_preds(topk)

        print("Attack stats on set A")

        #get marginal accuracy on set A 
        idxs_A, preds_A = adv_A_attk_A.attack_p_frac(most_conf_frac = 1)
        acc = get_acc(idxs_A, preds_A).mean()
        print(f"Model A Acc: {acc:.3f}")
        
        idxs_B, preds_B = adv_B_attk_A.attack_p_frac(most_conf_frac = 1)
        acc = get_acc(idxs_B, preds_B).mean()
        print(f"Model B Acc: {acc:.3f}")
        
        #get conf conditioned accuracy on set A 
        idxs_A_p05, preds_A_p05 = adv_A_attk_A.attack_p_frac(most_conf_frac = .05)
        acc = get_acc(idxs_A_p05, preds_A_p05).mean()
        print(f"Model A 5% most conf Acc: {acc:.3f}")
        
        idxs_B_p05, preds_B_p05 = adv_B_attk_A.attack_p_frac(most_conf_frac = .05)
        acc = get_acc(idxs_B_p05, preds_B_p05).mean()
        print(f"Model B 5% most conf Acc: {acc:.3f}")

        idxs_A_p20, preds_A_p20 = adv_A_attk_A.attack_p_frac(most_conf_frac = .20)
        acc = get_acc(idxs_A_p20, preds_A_p20).mean()
        print(f"Model A 20% most conf Acc: {acc:.3f}")
        
        idxs_B_p20, preds_B_p20 = adv_B_attk_A.attack_p_frac(most_conf_frac = .20)
        acc = get_acc(idxs_B_p20, preds_B_p20).mean()
        print(f"Model B 20% most conf Acc: {acc:.3f}")

        #swap test set A
        print("Swap test on set A")
        #first align adv B predictions to adv A's idxs 
        permute_B = [np.where(idxs_B == i)[0][0] for i in idxs_A]
        preds_B_ = preds_B[permute_B]
        
        #get A/B correct 
        correct_A = get_acc(idxs_A, preds_A)
        correct_B = get_acc(idxs_A, preds_B_)
        n = len(correct_A)
        
        yes_A_yes_B = correct_A[correct_B == 1].sum() / n
        yes_A_no_B = correct_A[correct_B == 0].sum() / n
        no_A_no_B = (correct_A == 0)[correct_B == 0].sum() / n
        no_A_yes_B = (correct_A == 0)[correct_B == 1].sum() /n
        
        print(f"A success, B success: {yes_A_yes_B:.3f}")
        print(f"A success, B fail: {yes_A_no_B:.3f}")
        print(f"A fail, B fail: {no_A_no_B:.3f}")
        print(f"A fail, B success: {no_A_yes_B:.3f}")

        #sweep confidence and save accuracy 
        fracs = np.linspace(0.1, 1, 100)
        accs_A = []
        accs_B = []
        for f in fracs: 
            idxs, preds = adv_A_attk_A.attack_p_frac(most_conf_frac = f)
            accs_A.append(get_acc(idxs, preds).mean())
            
            idxs, preds = adv_B_attk_A.attack_p_frac(most_conf_frac = f)
            accs_B.append(get_acc(idxs, preds).mean())

        np.save(args.output_dir / f"top_{topk}_conf_sweep_A_attk_A", accs_A)
        np.save(args.output_dir / f"top_{topk}_conf_sweep_B_attk_A", accs_B)

        print("Attack stats on set B")

        #get marginal accuracy on set B 
        idxs_A, preds_A = adv_A_attk_B.attack_p_frac(most_conf_frac = 1)
        acc = get_acc(idxs_A, preds_A).mean()
        print(f"Model A Acc: {acc:.3f}")
        
        idxs_B, preds_B = adv_B_attk_B.attack_p_frac(most_conf_frac = 1)
        acc = get_acc(idxs_B, preds_B).mean()
        print(f"Model B Acc: {acc:.3f}")

        #get conf conditioned accuracy on set B  
        idxs_A_p05, preds_A_p05 = adv_A_attk_B.attack_p_frac(most_conf_frac = .05)
        acc = get_acc(idxs_A_p05, preds_A_p05).mean()
        print(f"Model A 5% most conf Acc: {acc:.3f}")
        
        idxs_B_p05, preds_B_p05 = adv_B_attk_B.attack_p_frac(most_conf_frac = .05)
        acc = get_acc(idxs_B_p05, preds_B_p05).mean()
        print(f"Model B 5% most conf Acc: {acc:.3f}")

        idxs_A_p20, preds_A_p20 = adv_A_attk_B.attack_p_frac(most_conf_frac = .20)
        acc = get_acc(idxs_A_p20, preds_A_p20).mean()
        print(f"Model A 20% most conf Acc: {acc:.3f}")
        
        idxs_B_p20, preds_B_p20 = adv_B_attk_B.attack_p_frac(most_conf_frac = .20)
        acc = get_acc(idxs_B_p20, preds_B_p20).mean()
        print(f"Model B 20% most conf Acc: {acc:.3f}")

        #swap test set B
        print("Swap test on set B")
        #first align adv B predictions to adv A's idxs 
        permute_B = [np.where(idxs_B == i)[0][0] for i in idxs_A]
        preds_B_ = preds_B[permute_B]
        
        #get A/B correct 
        correct_A = get_acc(idxs_A, preds_A)
        correct_B = get_acc(idxs_A, preds_B_)
        n = len(correct_A)
        
        yes_A_yes_B = correct_A[correct_B == 1].sum() / n
        yes_A_no_B = correct_A[correct_B == 0].sum() / n
        no_A_no_B = (correct_A == 0)[correct_B == 0].sum() / n
        no_A_yes_B = (correct_A == 0)[correct_B == 1].sum() /n
        
        print(f"A success, B success: {yes_A_yes_B:.3f}")
        print(f"A success, B fail: {yes_A_no_B:.3f}")
        print(f"A fail, B fail: {no_A_no_B:.3f}")
        print(f"A fail, B success: {no_A_yes_B:.3f}")

        #sweep confidence and save accuracy 
        accs_A = []
        accs_B = []
        for f in fracs: 
            idxs, preds = adv_A_attk_B.attack_p_frac(most_conf_frac = f)
            accs_A.append(get_acc(idxs, preds).mean())
            
            idxs, preds = adv_B_attk_B.attack_p_frac(most_conf_frac = f)
            accs_B.append(get_acc(idxs, preds).mean())
   
        np.save(args.output_dir / f"top_{topk}_conf_sweep_A_attk_B", accs_A)
        np.save(args.output_dir / f"top_{topk}_conf_sweep_B_attk_B", accs_B)
    

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        self._setup_gpu_args()
        main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")



def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(f"/checkpoint/{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


def file_submitit_job(args): 
    if args.output_dir == "":
        args.output_dir = get_shared_folder() / "%j"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'

    executor.update_parameters(
            mem_gb=args.mem_gb,
            gpus_per_node=1,
            tasks_per_node=1,  # one task per GPU
            cpus_per_task=10,
            nodes=1,
            timeout_min=args.timeout,  # max is 60 * 72
            # Below are cluster dependent parameters
            slurm_partition=args.partition,
            slurm_signal_delay_s=120,
            **kwargs
        )

    executor.update_parameters(name="NN attack")

    args.dist_url = get_init_file().as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    if args.local == 1: 
        if args.output_dir == "":
            args.output_dir = get_shared_folder() / "%j"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args.dist_url = get_init_file().as_uri()
        args.gpu = args.local_gpu
        args.world_size = 1
        args.rank = 0
        main(args)       
    else: 
        file_submitit_job(args)
