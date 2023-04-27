# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy.stats import entropy
from scipy.stats import mode
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm_notebook as tqdm
import os
import pickle
import xmltodict
import json
from mpl_toolkits.axes_grid1 import ImageGrid

from RCDM.image_sample import create_argparser
from guided_diffusion import dist_util, logger
from guided_diffusion.get_ssl_models import get_model
from guided_diffusion.get_rcdm_models import get_dict_rcdm_model
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from .utils import ( 
        aux_dataset, SSL_Transform, 
        ImageFolderIndex, get_confidence_and_topk, 
        SSLNetwork
)

class InverseTransform:
    """inverses normalization of SSL transform
    """
    def __init__(self): 
        self.invTrans = torchvision.transforms.Compose([
                  torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                  std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                  torchvision.transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                  ])
                            
    def __call__(self, x): 
        return self.invTrans(x)


def get_attack_data(model, dataset, epoch, k_neighb, attk_path):
    #get attack data
    base_dir = Path(attk_path)
    traces = defaultdict(list)
    ds = dataset
    ep = epoch

    #load in attack data
    folder = base_dir / f'NN_attk_{model}_{ds}pc_{ep}ep'
    #attack on set A and B
    for attk_set in ('A', 'B'):
        A_idxs = np.load(folder / f'A_attk_{attk_set}_attk_idxs.npy')
        B_idxs = np.load(folder / f'B_attk_{attk_set}_attk_idxs.npy')
        B_order = [np.where(B_idxs == i)[0][0] for i in A_idxs]
        A_labels = np.load(folder / f'A_attk_{attk_set}_labels.npy')
        B_labels = np.load(folder / f'B_attk_{attk_set}_labels.npy')[B_order]
        A_neighb_labels = np.load(folder / f'A_attk_{attk_set}_neighb_labels.npy')
        B_neighb_labels = np.load(folder / f'B_attk_{attk_set}_neighb_labels.npy')[B_order]
        A_neighb_idxs = np.load(folder / f'A_attk_{attk_set}_neighb_idxs.npy')
        B_neighb_idxs = np.load(folder / f'B_attk_{attk_set}_neighb_idxs.npy')[B_order]

        #get confidences
        A_conf, A_preds = get_confidence_and_topk(A_neighb_labels, k_neighb)
        B_conf, B_preds = get_confidence_and_topk(B_neighb_labels, k_neighb)

        traces[f'set_{attk_set}_idxs_{ep}ep_{ds}pc'] = A_idxs
        traces[f'set_{attk_set}_labels_{ep}ep_{ds}pc'] = A_labels
        traces[f'A_attk_{attk_set}_neighb_idxs_{ep}ep_{ds}pc'] = A_neighb_idxs
        traces[f'B_attk_{attk_set}_neighb_idxs_{ep}ep_{ds}pc'] = B_neighb_idxs
        traces[f'A_attk_{attk_set}_conf_{ep}ep_{ds}pc'] = A_conf
        traces[f'B_attk_{attk_set}_conf_{ep}ep_{ds}pc'] = B_conf
        traces[f'A_attk_{attk_set}_preds_{ep}ep_{ds}pc'] = A_preds
        traces[f'B_attk_{attk_set}_preds_{ep}ep_{ds}pc'] = B_preds

    return traces

def load_ssl_models(model_A_pth, model_B_pth, mlp, loss, arch='resnet101', activations = False):
    gpu = torch.cuda.current_device()
    model_A = SSLNetwork(arch = arch, 
                          remove_head = 0, 
                          mlp = mlp, 
                          fc = 0,
                          patch_keep = 1.0,
                          loss = loss).cuda()
    if (loss == 'supervised') and activations:
        model_A.fc = nn.Linear(model_A.num_features, 1000).cuda()
    model_A = torch.nn.parallel.DistributedDataParallel(model_A, device_ids=[gpu])
    ckpt = torch.load(model_A_pth, map_location='cpu')
    model_A.load_state_dict(ckpt['model'], strict = False)
    _ = model_A.eval()

    model_B = SSLNetwork(arch = arch, 
                          remove_head = 0, 
                          mlp = mlp, 
                          fc = 0,
                          patch_keep = 1.0,
                          loss = loss).cuda()
    if (loss == 'supervised') and activations:
        model_B.fc = nn.Linear(model_B.num_features, 1000).cuda()
    model_B = torch.nn.parallel.DistributedDataParallel(model_B, device_ids=[gpu])
    ckpt = torch.load(model_B_pth, map_location='cpu')
    model_B.load_state_dict(ckpt['model'], strict = False)
    _ = model_B.eval()    

    return model_A, model_B

def get_rcdm_args_128(): 
    parser = create_argparser()
    RCDM_args = parser.parse_args('')
    RCDM_args.rank = 0
    RCDM_args.gpu = torch.cuda.current_device()
    RCDM_args.world_size = 1
    RCDM_args.image_size = 128
    RCDM_args.clipped_denoised = True
    RCDM_args.num_channels = 256
    RCDM_args.learn_sigma = True
    RCDM_args.attention_resolutions = '32,16,8'
    RCDM_args.resblock_updown = True
    RCDM_args.class_cond = False
    RCDM_args.diffusion_steps = 1000
    RCDM_args.num_res_blocks = 2
    RCDM_args.use_scale_shift_norm = True
#     RCDM_args.use_fp16 = True
    RCDM_args.noise_schedule = 'linear'
    RCDM_args.use_head = False
    RCDM_args.num_images = 1
    return RCDM_args

def get_rcdm_args_256(): 
    parser = create_argparser()
    RCDM_args = parser.parse_args('')
    RCDM_args.rank = 0
    RCDM_args.gpu = torch.cuda.current_device()
    RCDM_args.world_size = 1
    RCDM_args.image_size = 256
    RCDM_args.clipped_denoised = True
    RCDM_args.num_channels = 256
    RCDM_args.num_head_channels = 64
    RCDM_args.learn_sigma = True
    RCDM_args.attention_resolutions = '32,16,8'
    RCDM_args.resblock_updown = True
    RCDM_args.class_cond = False
    RCDM_args.diffusion_steps = 1000
    RCDM_args.num_res_blocks = 2
    RCDM_args.use_scale_shift_norm = True
#     RCDM_args.use_fp16 = True
    RCDM_args.noise_schedule = 'linear'
    return RCDM_args

def load_rcdm_model(rcdm_pth, ssl_dim, res = 128): 
    if res == 128: 
        RCDM_args = get_rcdm_args_128()
    else: 
        RCDM_args = get_rcdm_args_256()
    
    #Get model A
    RCDM_args.model_path = rcdm_pth
    RCDM, diffusion = create_model_and_diffusion(
        **args_to_dict(RCDM_args, model_and_diffusion_defaults().keys()), G_shared=RCDM_args.no_shared, 
                        feat_cond=True, ssl_dim=ssl_dim)
    
    # Load model
    trained_model = torch.load(rcdm_pth, map_location="cpu")
    RCDM.load_state_dict(trained_model, strict=True)
    RCDM = RCDM.cuda()
    _ = RCDM.eval()

    return RCDM, diffusion


####
#GENERATE CODE
####

def gen_samples(
    indices, diffusion_A, diffusion_B,
    ssl_A, ssl_B,
    RCDM_A, RCDM_B,
    ssl_epoch, ssl_ds,
    attk_data, 
    num_NNs = 5,
    n_samples = 4,
    attk_set = 'A',
    imgnet_dir = '',
    bbox_dir = '', 
    just_neighbs = False, 
    only_use_model = False, 
    activations = False, 
    res = 128
    ): 
    
    if res == 128: 
        RCDM_args = get_rcdm_args_128()
    else: 
        RCDM_args = get_rcdm_args_256()
    
    with open("imgnet_classes.json") as f:
        imgnet_classes = json.load(f)

    NNs_ds = ImageFolder(imgnet_dir, transform = SSL_Transform())
    attack_idxs = attk_data[f'set_{attk_set}_idxs_{ssl_epoch}ep_{ssl_ds}pc']
    crop_ds = aux_dataset(imgnet_dir, bbox_dir, attack_idxs, return_im_and_tgt = True)
        
    im_dict = defaultdict(list)
    
    ssl_NNs_A = attk_data[f'A_attk_{attk_set}_neighb_idxs_{ssl_epoch}ep_{ssl_ds}pc']
    ssl_NNs_B = attk_data[f'B_attk_{attk_set}_neighb_idxs_{ssl_epoch}ep_{ssl_ds}pc']
    
    model_kwargs = {}
    
    iTrans = InverseTransform()
    
    def avg_backbone(model, idxs): 
        xs = []
        xs_vis = []
        ys = []
        for idx in idxs: 
            x,y = NNs_ds[idx]
            x_vis = iTrans(x)
            ys.append(y)
            xs.append(x)
            xs_vis.append(x_vis)
        xs = torch.stack(xs).cuda()
        xs_vis = torch.stack(xs_vis)
        embeds = model.module.net(xs) 
        avg = embeds.mean(dim = 0).unsqueeze(0)
        return avg, xs_vis, ys
    
    if only_use_model == 'A' or not only_use_model:
        sample_fn_A = (
            diffusion_A.p_sample_loop if not RCDM_args.use_ddim else diffusion_A.ddim_sample_loop
        )
    if only_use_model == 'B' or not only_use_model:
        sample_fn_B = (
            diffusion_B.p_sample_loop if not RCDM_args.use_ddim else diffusion_B.ddim_sample_loop
        )
    
    def show_im(im): 
        plt.imshow(im.permute(1,2,0))
        plt.axis('off')
        plt.show()
        
    
    
    for i in indices: 
        #get attack info 
        NNs_A = ssl_NNs_A[i][:num_NNs]
        NNs_B = ssl_NNs_B[i][:num_NNs]
    
        #get image & patch 
        patch, samp, target = crop_ds[i]
        print(f"Original image, attk set idx {i}:")
        show_im(samp)
        im_dict['samps'].append(samp)
        print("Patch:")
        patch_view = iTrans(patch)
        show_im(patch_view)
        im_dict['patches'].append(patch_view)

        if only_use_model == 'A' or not only_use_model:
            if not activations: 
                #get SSL projector embedding 
                bb = ssl_A.module.net(patch.unsqueeze(0).cuda())
                pj = ssl_A.module.projector(bb)
                #get avg of SSL nearest neighbors
                bb, xs_vis, ys = avg_backbone(ssl_A, NNs_A)
                NN_vote = np.argmax(np.bincount(ys))
                emb = torch.cat((bb,pj), dim = 1)

                print(f"{'Class:':<30} {'A NN vote:':<30}")
                print(f"{imgnet_classes[target]:<30} {imgnet_classes[NN_vote]:<30}")
                print(f"k={num_NNs} nearest neighbs")
                fig = plt.figure(figsize=(16,4)) #width x height
                grid = ImageGrid(fig, 111,  # similar to subplot(111)
                                 nrows_ncols=(1, num_NNs),  # creates 2x2 grid of axes
                                 axes_pad=0.0,  # pad between axes in inch.
                                 )
                for ax, s in zip(grid, xs_vis):
                    # Iterating over the grid returns the Axes.
                    ax.set_axis_off()
                    ax.imshow(s.permute(1,2,0))
                plt.show()
                im_dict['NN_sets_A'].append(xs_vis)
            else: 
                #get model activations
                bb = ssl_A.module.net(patch.unsqueeze(0).cuda())
                pj = ssl_A.module.projector(bb)
                emb = ssl_A.module.fc(pj)
                pred = torch.argmax(emb, dim = 1).item()

                print(f"{'Class:':<30} {'A activation vote:':<30}")
                print(f"{imgnet_classes[target]:<30} {imgnet_classes[pred]:<30}")                
        
            #sample
            if not just_neighbs: 
                print('Sampling model A...')
                model_kwargs["feat"] = emb.repeat(n_samples, 1)
                sample = sample_fn_A(
                    RCDM_A,
                    (n_samples, 3, RCDM_args.image_size, RCDM_args.image_size),
                    clip_denoised=RCDM_args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
            
                fig = plt.figure(figsize=(16,4)) #width x height
                grid = ImageGrid(fig, 111,  # similar to subplot(111)
                                 nrows_ncols=(1, 4),  # creates 2x2 grid of axes
                                 axes_pad=0.0,  # pad between axes in inch.
                                 )
                for ax, s in zip(grid, sample.cpu()):
                    # Iterating over the grid returns the Axes.
                    ax.set_axis_off()
                    ax.imshow(s.permute(1,2,0))
                plt.show()
                im_dict['rcdm_sets_A'].append(sample.cpu())
    
    
        if only_use_model == 'B' or not only_use_model:
            #Now model B
            if not activations: 
                #get SSL projector embedding 
                bb = ssl_B.module.net(patch.unsqueeze(0).cuda())
                pj = ssl_B.module.projector(bb)
                #get avg of SSL nearest neighbors
                bb, xs_vis, ys = avg_backbone(ssl_B, NNs_B)
                NN_vote = np.argmax(np.bincount(ys))
                emb = torch.cat((bb,pj), dim = 1)

                print(f"{'Class:':<30} {'B NN vote:':<30}")
                print(f"{imgnet_classes[target]:<30} {imgnet_classes[NN_vote]:<30}")
                print(f"k={num_NNs} nearest neighbs")
                fig = plt.figure(figsize=(16,4)) #width x height
                grid = ImageGrid(fig, 111,  # similar to subplot(111)
                                 nrows_ncols=(1, num_NNs),  # creates 2x2 grid of axes
                                 axes_pad=0.0,  # pad between axes in inch.
                                 )
                for ax, s in zip(grid, xs_vis):
                    # Iterating over the grid returns the Axes.
                    ax.set_axis_off()
                    ax.imshow(s.permute(1,2,0))
                plt.show()
                im_dict['NN_sets_B'].append(xs_vis)
            else: 
                #get model activations
                bb = ssl_B.module.net(patch.unsqueeze(0).cuda())
                pj = ssl_B.module.projector(bb)
                emb = ssl_B.module.fc(pj)
                pred = torch.argmax(emb, dim = 1).item()

                print(f"{'Class:':<30} {'B activation vote:':<30}")
                print(f"{imgnet_classes[target]:<30} {imgnet_classes[pred]:<30}")
        
            if not just_neighbs: 
                #sample
                print('Sampling model B...')
                model_kwargs["feat"] = emb.repeat(n_samples, 1)
                sample = sample_fn_B(
                    RCDM_B,
                    (n_samples, 3, RCDM_args.image_size, RCDM_args.image_size),
                    clip_denoised=RCDM_args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
            
                fig = plt.figure(figsize=(16,4)) #width x height
                grid = ImageGrid(fig, 111,  # similar to subplot(111)
                                 nrows_ncols=(1, 4),  # creates 2x2 grid of axes
                                 axes_pad=0.0,  # pad between axes in inch.
                                 )
                for ax, s in zip(grid, sample.cpu()):
                    # Iterating over the grid returns the Axes.
                    ax.set_axis_off()
                    ax.imshow(s.permute(1,2,0))
                plt.show()
                im_dict['rcdm_sets_B'].append(sample.cpu())                

    return im_dict


def tile_reconstructions(model, im_dict, save_pth, examp_subset = []): 

    if len(examp_subset) == 0: 
        examp_subset = np.arange(len(im_dict['samps']))
        
    n = len(examp_subset)
    
    subset_dict = {k:[v[i] for i in examp_subset] for k,v in im_dict.items()}
    Path(save_pth).mkdir(parents=True, exist_ok=True)   
    
    #save full im_dict
    with open(save_pth + 'im_dict.pkl', 'wb') as handle:
        pickle.dump(im_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #plot tile
    resize_xfrm = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor()])
    
    #first plot sample + patch
    fig = plt.figure(figsize=(2*2,2*n)) #width x height
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(n, 2),  # creates 2x2 grid of axes
                     axes_pad=(0.1, 0),  # pad between axes in inch.
                     )
    for ct, (s, p) in enumerate(zip(subset_dict['samps'], subset_dict['patches'])):
        # Iterating over the grid returns the Axes.
        ax_l = grid[2*ct]
        ax_l.set_axis_off()
        s = resize_xfrm(s)
        ax_l.imshow(s.permute(1,2,0))
        
        ax_r = grid[2*ct+1]
        ax_r.set_axis_off()
        ax_r.imshow(p.permute(1,2,0))
    plt.savefig(save_pth + 'samps_patches')
    print('target samples and patches')
    plt.show()
    
    #then plot neighbors for A  
    num_neighbs = len(subset_dict['NN_sets_A'][0])
    fig = plt.figure(figsize=(2*num_neighbs,2*n)) #width x height
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(n, num_neighbs),  # creates 2x2 grid of axes
                     axes_pad=0.0,  # pad between axes in inch.
                     )
    for ct, neighb_set in enumerate(subset_dict['NN_sets_A']):
        for i, im in enumerate(neighb_set): 
            idx = num_neighbs*ct + i
            grid[idx].set_axis_off()
            grid[idx].imshow(im.permute(1,2,0))
    plt.savefig(save_pth + 'NNs_A')
    print('ssl A nearest neighbs')
    plt.show()
    
    #then plot neighbors for B  
    fig = plt.figure(figsize=(2*num_neighbs,2*n)) #width x height
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(n, num_neighbs),  # creates 2x2 grid of axes
                     axes_pad=0.0,  # pad between axes in inch.
                     )
    for ct, neighb_set in enumerate(subset_dict['NN_sets_B']):
        for i, im in enumerate(neighb_set): 
            idx = num_neighbs*ct + i
            grid[idx].set_axis_off()
            grid[idx].imshow(im.permute(1,2,0))
    plt.savefig(save_pth + 'NNs_B')
    print('ssl B nearest neighbs')
    plt.show()
    
    #then plot rdcms for A  
    num_rcdm = len(im_dict['rcdm_sets_A'][0])
    fig = plt.figure(figsize=(2*num_rcdm,2*n)) #width x height
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(n, num_rcdm),  # creates 2x2 grid of axes
                     axes_pad=0.0,  # pad between axes in inch.
                     )
    for ct, neighb_set in enumerate(subset_dict['rcdm_sets_A']):
        for i, im in enumerate(neighb_set): 
            idx = num_rcdm*ct + i
            grid[idx].set_axis_off()
            grid[idx].imshow(im.permute(1,2,0))
    plt.savefig(save_pth + 'recon_A')
    print('rcdm A reconstructions')
    plt.show()
    
    #then plot rdcms for B
    num_rcdm = len(im_dict['rcdm_sets_B'][0])
    fig = plt.figure(figsize=(2*num_rcdm,2*n)) #width x height
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(n, num_rcdm),  # creates 2x2 grid of axes
                     axes_pad=0.0,  # pad between axes in inch.
                     )
    for ct, neighb_set in enumerate(subset_dict['rcdm_sets_B']):
        for i, im in enumerate(neighb_set): 
            idx = num_rcdm*ct + i
            grid[idx].set_axis_off()
            grid[idx].imshow(im.permute(1,2,0))
    plt.savefig(save_pth + 'recon_B')
    print('rcdm B reconstructions')
    plt.show()



def print_class_statistics(attk_data, attk_set, epoch, ds, imgnet_classes, k = 10): 
    #examine accuracy gap -- compare with confidence
    ref_set = 'B' if attk_set == 'A' else 'A'
    labels = attk_data[f'set_A_labels_{epoch}ep_{ds}pc']
    cls = np.unique(labels)
    accs, ref_accs = [], []
    confs_avs, ref_confs_avs = [], []
    conf_diffs, conf_mins = [], []
    
    preds = np.array(attk_data[f'{attk_set}_attk_{attk_set}_preds_{epoch}ep_{ds}pc'][f'top_1'])
    ref_preds = np.array(attk_data[f'{ref_set}_attk_{attk_set}_preds_{epoch}ep_{ds}pc'][f'top_1'])
    confs = np.array(attk_data[f'{attk_set}_attk_{attk_set}_conf_{epoch}ep_{ds}pc'])
    ref_confs = np.array(attk_data[f'{ref_set}_attk_{attk_set}_conf_{epoch}ep_{ds}pc'])
    tgt_acc = preds == labels #where target model is accurate
    ref_acc = ref_preds == labels #where ref model is accurate
    
    for cl in cls:
        in_class_idxs = np.where(labels == cl)[0]
        in_class_tgt_idxs = np.where((labels == cl) & tgt_acc)[0]
        in_class_ref_idxs = np.where((labels == cl) & ref_acc)[0]
        #memorized set -- only target is correct
        in_class_mem_idxs = list(set(in_class_tgt_idxs) - set(in_class_ref_idxs))
        #correlated set -- both models are correct
        in_class_cor_idxs = list(set(in_class_tgt_idxs) & set(in_class_ref_idxs))
    
        class_acc = (preds[in_class_idxs] == labels[in_class_idxs]).mean()
        accs.append(class_acc)
        ref_class_acc = (ref_preds[in_class_idxs] == labels[in_class_idxs]).mean()
        ref_accs.append(ref_class_acc)
    
        #if memorized set is > k, report average confidence agreement or discrepancy
        if len(in_class_mem_idxs) < k:
            confs_avs.append(-np.inf)
            conf_diffs.append(0)
        else:
            class_conf = np.sort(confs[in_class_mem_idxs])[-k:].mean()
            confs_avs.append(class_conf)
            #record avg disagreement between models in top k memorized examples
            conf_diffs_class = confs[in_class_mem_idxs] - ref_confs[in_class_mem_idxs]
            conf_diffs.append(np.sort(conf_diffs_class)[-k:].mean())
    
        if len(in_class_cor_idxs) < k:
            ref_confs_avs.append(-np.inf)
            conf_mins.append(-np.inf)
        else:
            ref_class_conf = np.sort(ref_confs[in_class_cor_idxs][-k:]).mean()
            ref_confs_avs.append(ref_class_conf)
            #record avg agreement between models in top k correlated examples
            conf_min_class = np.minimum(confs[in_class_cor_idxs], ref_confs[in_class_cor_idxs])
            conf_mins.append(np.sort(conf_min_class)[-k:].mean())
    
    accs, ref_accs = np.array(accs), np.array(ref_accs)
    confs_avs, ref_confs_avs = np.array(confs_avs), np.array(ref_confs_avs)
    conf_diffs, conf_mins = np.array(conf_diffs), np.array(conf_mins)
    
    acc_gap = accs - ref_accs
    plt.hist(acc_gap, bins = 50, density = False)
    plt.xlabel('Class Accuracy Gap')
    plt.ylabel('# Classes')
    plt.title('Tgt v. Ref model accuracy per class')
    plt.show()
    
    print(f"{'Order':<6}",
          f"{'Class ID':<9}",
          f"{'Class Name':<20}",
          f"{'Tgt. Acc.':<11}",
          f"{'Ref. Acc.':<11}",
          f"{'Mem set, topk conf diff':<22}",
          f"{'Corr set, topk conf min':<22}")
    
    accs_order = np.argsort(conf_diffs)[::-1]
    for ct, (cl, acc, ref_acc, cdiff, cmin) in enumerate(zip(
                                                cls[accs_order],
                                                accs[accs_order],
                                                ref_accs[accs_order],
                                                conf_diffs[accs_order],
                                                conf_mins[accs_order],
                                                )):
        print(f"{ct:<6}",
          f"{cl:<9}",
          f"{imgnet_classes[cl][:20]:<20}",
          f"{acc:<11.3f}",
          f"{ref_acc:<11.3f}",
          f"{cdiff:<22.3f}",
          f"{cmin:<22.3f}")

def mem_v_corr_show_class_examples(attk_data, attk_set, epoch, ds, cl, mem_set, crop_ds, imgnet_classes, k = 10): 
    ref_set = 'B' if attk_set == 'A' else 'A'
    labels = attk_data[f'set_A_labels_{epoch}ep_{ds}pc']
    confs = np.array(attk_data[f'{attk_set}_attk_{attk_set}_conf_{epoch}ep_{ds}pc'])
    ref_confs = np.array(attk_data[f'{ref_set}_attk_{attk_set}_conf_{epoch}ep_{ds}pc'])
    
    preds = np.array(attk_data[f'{attk_set}_attk_{attk_set}_preds_{epoch}ep_{ds}pc']['top_1'])
    ref_preds = np.array(attk_data[f'{ref_set}_attk_{attk_set}_preds_{epoch}ep_{ds}pc']['top_1'])
    
    #get confident examples in class 
    in_class_idxs = np.where((labels == cl))[0]# & (ref_preds != labels))[0]
    in_class_tgt_idxs = np.where((labels == cl) & (preds == labels))[0]
    in_class_ref_idxs = np.where((labels == cl) & (ref_preds == labels))[0]
    in_class_mem_idxs = np.array(list(set(in_class_tgt_idxs) - set(in_class_ref_idxs)))
    in_class_cor_idxs = np.array(list(set(in_class_tgt_idxs) & set(in_class_ref_idxs)))
    
    if mem_set: 
        idxs = in_class_mem_idxs
        sort_criterion = confs[idxs] - ref_confs[idxs]
    else: 
        idxs = in_class_cor_idxs
        sort_criterion = np.minimum(confs[idxs], ref_confs[idxs])
    
    order = np.argsort(sort_criterion)[::-1]
    print('attk set idxs:\n', idxs[order][:k], '\n')
    patches = []
    iTrans = InverseTransform()
    for examp in idxs[order][:k]: 
        print(f'model {attk_set} confidence:', confs[examp])
        print(f'model {ref_set} confidence:', ref_confs[examp])
        patch, samp, tgt = crop_ds[examp]
        patch = iTrans(patch)
        print('label:', imgnet_classes[tgt])
        print('examp idx:', examp)
        print('Tgt correct:', (preds[examp] == tgt)[0])
        print('Ref correct:', (ref_preds[examp] == tgt)[0])
        print('patch:')
        patch = patch.permute(1,2,0)
        plt.imshow(patch)
        patches.append(patch)
        plt.axis('off')
        plt.show()
        print('full im:')
        plt.imshow(samp.permute(1,2,0))
        plt.axis('off')
        plt.show()

    return patches, idxs[order][:k]

def print_class_statistics_sort_conf(attk_data, attk_set, epoch, ds, imgnet_classes, k = 10): 
    #examine accuracy gap -- sort by confidence 
    ref_set = 'B' if attk_set == 'A' else 'A'
    labels = attk_data[f'set_A_labels_{epoch}ep_{ds}pc']
    cls = np.unique(labels)
    accs, ref_accs = [], []
    confs_avs, ref_confs_avs = [], []
    
    preds = np.array(attk_data[f'{attk_set}_attk_{attk_set}_preds_{epoch}ep_{ds}pc'][f'top_1'])
    ref_preds = np.array(attk_data[f'{ref_set}_attk_{attk_set}_preds_{epoch}ep_{ds}pc'][f'top_1'])
    confs = np.array(attk_data[f'{attk_set}_attk_{attk_set}_conf_{epoch}ep_{ds}pc'])
    ref_confs = np.array(attk_data[f'{ref_set}_attk_{attk_set}_conf_{epoch}ep_{ds}pc'])
    
    for cl in cls:
        in_class_idxs = np.where(labels == cl)[0]

        class_acc = (preds[in_class_idxs] == labels[in_class_idxs]).mean()
        accs.append(class_acc)
        ref_class_acc = (ref_preds[in_class_idxs] == labels[in_class_idxs]).mean()
        ref_accs.append(ref_class_acc)

        class_conf = np.sort(confs[in_class_idxs])[-k:].mean()
        confs_avs.append(class_conf)
        ref_class_conf = np.sort(ref_confs[in_class_idxs])[-k:].mean()
        ref_confs_avs.append(ref_class_conf)
    
    accs, ref_accs = np.array(accs), np.array(ref_accs)
    confs_avs, ref_confs_avs = np.array(confs_avs), np.array(ref_confs_avs)
    
    acc_gap = accs - ref_accs
    plt.hist(acc_gap, bins = 50, density = False)
    plt.xlabel('Class Accuracy Gap')
    plt.ylabel('# Classes')
    plt.title('Tgt v. Ref model accuracy per class')
    plt.show()
    
    print(f"{'Order':<6}",
          f"{'Class ID':<9}",
          f"{'Class Name':<20}",
          f"{'Tgt. Acc.':<11}",
          f"{'Ref. Acc.':<11}",
          f"{'Tgt model ave conf':<22}",
          f"{'Ref model ave conf':<22}")
    
    accs_order = np.argsort(acc_gap)[::-1]
    for ct, (cl, acc, ref_acc, tgt_conf, ref_conf) in enumerate(zip(
                                                cls[accs_order],
                                                accs[accs_order],
                                                ref_accs[accs_order],
                                                confs_avs[accs_order],
                                                ref_confs_avs[accs_order],
                                                )):
        print(f"{ct:<6}",
          f"{cl:<9}",
          f"{imgnet_classes[cl][:20]:<20}",
          f"{acc:<11.3f}",
          f"{ref_acc:<11.3f}",
          f"{tgt_conf:<22.3f}",
          f"{ref_conf:<22.3f}")
        
def top_conf_show_class_examples(attk_data, attk_set, epoch, ds, cl, crop_ds, imgnet_classes, k = 40): 
    ref_set = 'B' if attk_set == 'A' else 'A'
    labels = attk_data[f'set_{attk_set}_labels_{epoch}ep_{ds}pc']
    confs = np.array(attk_data[f'{attk_set}_attk_{attk_set}_conf_{epoch}ep_{ds}pc'])
    ref_confs = np.array(attk_data[f'{ref_set}_attk_{attk_set}_conf_{epoch}ep_{ds}pc'])
    
    preds = np.array(attk_data[f'{attk_set}_attk_{attk_set}_preds_{epoch}ep_{ds}pc']['top_1'])
    ref_preds = np.array(attk_data[f'{ref_set}_attk_{attk_set}_preds_{epoch}ep_{ds}pc']['top_1'])
    
    #get confident examples in class 
    idxs = np.where((labels == cl))[0]# & (ref_preds != labels))[0]
    confs_in_class = confs[idxs]
    order = np.argsort(confs_in_class)[::-1]
    
    print('attk set idxs:\n', idxs[order][:k], '\n')
    patches = []
    iTrans = InverseTransform()
    for examp in idxs[order][:k]: 
        print(f'model {attk_set} confidence:', confs[examp])
        print(f'model {ref_set} confidence:', ref_confs[examp])
        patch, samp, tgt = crop_ds[examp]
        patch = iTrans(patch)
        print('label:', imgnet_classes[tgt])
        print('examp idx:', examp)
        print('Tgt correct:', (preds[examp] == tgt)[0])
        print('Ref correct:', (ref_preds[examp] == tgt)[0])
        print('patch:')
        patch = patch.permute(1,2,0)
        plt.imshow(patch)
        patches.append(patch)
        plt.axis('off')
        plt.show()
        print('full im:')
        plt.imshow(samp.permute(1,2,0))
        plt.axis('off')
        plt.show()

    return patches, idxs[order][:k]
