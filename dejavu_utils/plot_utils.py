import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pathlib import Path
from collections import defaultdict
from scipy.stats import entropy
from scipy.stats import mode
import torch
import torchvision
import os

import sys 
import ast

from .utils import get_confidence_and_topk

def get_attack_data(model, datasets, epochs, k_neighb, pct_confidences, topks, fname = 'attack_sweeps'): 
    #get attack data 
    base_dir = Path(f'/checkpoint/caseymeehan/experiments/ssl_sweep/{model}/{fname}')
        
    traces = defaultdict(list)
    
    for ds in datasets: 
        for ep in epochs: 
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

                #get confidences 
                A_conf, A_preds = get_confidence_and_topk(A_neighb_labels, k_neighb, topk = topks)
                A_conf_order = np.argsort(A_conf)[::-1]
                B_conf, B_preds = get_confidence_and_topk(B_neighb_labels, k_neighb, topk = topks)
                B_conf_order = np.argsort(B_conf)[::-1]
                n = len(A_conf_order)

                #get top-p pct most confident accuracies
                for p in pct_confidences: 
                    num = int(n * p / 100)
                    A_conf_idxs = A_conf_order[:num]
                    B_conf_idxs = B_conf_order[:num]
                    for tk in topks: 
                        A_acc = (A_preds[f'top_{tk}'][A_conf_idxs] == A_labels[A_conf_idxs]).sum(axis = 1).mean()
                        traces[f'A_attk_{attk_set}_top{tk}_{p}conf_{ds}pc'].append(A_acc)
                        B_acc = (B_preds[f'top_{tk}'][B_conf_idxs] == B_labels[B_conf_idxs]).sum(axis = 1).mean()
                        traces[f'B_attk_{attk_set}_top{tk}_{p}conf_{ds}pc'].append(B_acc)

                #get 4 quadrant dataset breakdown 
                for tk in topks: 
                    A_accs = (A_preds[f'top_{tk}'] == A_labels).sum(axis = 1) == 1
                    B_accs = (B_preds[f'top_{tk}'] == B_labels).sum(axis = 1) == 1
                    yes_A_yes_B = (A_accs & B_accs).mean()
                    traces[f'yes_A_yes_B_attk_{attk_set}_top{tk}_{ds}pc'].append(yes_A_yes_B)
                    yes_A_no_B = (A_accs & ~B_accs).mean()
                    traces[f'yes_A_no_B_attk_{attk_set}_top{tk}_{ds}pc'].append(yes_A_no_B)
                    no_A_yes_B = (~A_accs & B_accs).mean()
                    traces[f'no_A_yes_B_attk_{attk_set}_top{tk}_{ds}pc'].append(no_A_yes_B)
                    no_A_no_B = (~A_accs & ~B_accs).mean()
                    traces[f'no_A_no_B_attk_{attk_set}_top{tk}_{ds}pc'].append(no_A_no_B)
                    
    return traces

def plot_attack_sweep_epochs(traces, dataset, epochs, pct_confidences, topks, pth):
    #first plot top-p percent versus epochs for single dataset 
    ds = dataset
    for tk in topks:
        for p_ct, p in enumerate(pct_confidences): 
            #get target model accuracies 
            AA = np.array(traces[f'A_attk_A_top{tk}_{p}conf_{ds}pc'])
            BB = np.array(traces[f'B_attk_B_top{tk}_{p}conf_{ds}pc'])
            tgt_model = (AA + BB) / 2
            plt.plot(epochs, tgt_model, '-p', color = f'C{p_ct}')
            if p != 100: 
                plt.plot([],[], 'o', color = f'C{p_ct}', label = f'accuracy on top {p}% conf.')
            else: 
                plt.plot([],[], 'o', color = f'C{p_ct}', label = f'accuracy on all examples')
            #get ref model accuracies
            AB = np.array(traces[f'A_attk_B_top{tk}_{p}conf_{ds}pc'])
            BA = np.array(traces[f'B_attk_A_top{tk}_{p}conf_{ds}pc'])
            ref_model = (AB + BA) / 2
            plt.plot(epochs, ref_model, '--o', color = f'C{p_ct}')
        plt.ylim(0,1)
        plt.xlabel('epochs', fontsize = 25)
        plt.grid(alpha = 0.75)
        plt.plot([], [], '--', color = 'black', label='ref. model correlation baseline')
        plt.ylabel('Label Inference Accuracy')
        plt.savefig(pth + f'attk_epochs_acc_top{tk}', pad_inches=0)
        plt.legend(loc = 'upper left')
#         plt.plot()
        plt.savefig(pth + f'attk_epochs_acc_top{tk}_legend', pad_inches=0)
        plt.show()
        plt.clf()

    #second plot quadrant breakdown 
    for tk in topks:
        #get 'memorized' examples
        mem_A = np.array(traces[f'yes_A_no_B_attk_A_top{tk}_{ds}pc'])
        mem_B = np.array(traces[f'no_A_yes_B_attk_B_top{tk}_{ds}pc'])
        mem = (mem_A + mem_B) / 2
        #get 'bad embed' examples
        bad_A = np.array(traces[f'no_A_yes_B_attk_A_top{tk}_{ds}pc'])
        bad_B = np.array(traces[f'yes_A_no_B_attk_B_top{tk}_{ds}pc'])
        bad = (bad_A + bad_B) / 2
        #get 'correlation' examples 
        corr_A = np.array(traces[f'yes_A_yes_B_attk_A_top{tk}_{ds}pc'])
        corr_B = np.array(traces[f'yes_A_yes_B_attk_B_top{tk}_{ds}pc'])
        corr = (corr_A + corr_B) / 2

        #plot the share of each 
        plt.fill_between(epochs, np.zeros(len(epochs)), corr, color = 'C3', label = 'correlated')
        plt.plot(epochs, corr, color = 'black')
        plt.fill_between(epochs, corr, corr+bad, color = 'C4', label = 'forgotten')
        plt.plot(epochs, corr+bad, color = 'black')
        plt.fill_between(epochs, corr+bad, corr+bad+mem, color = 'C5', label = 'memorized')
        plt.plot(epochs, corr+bad+mem, color = 'black')
#         plt.fill_between(epochs, corr+bad+mem, 1, color="gray",
#                                  edgecolor="black", alpha = 0.2, label = 'unassociated')

        plt.xlabel('epochs', fontsize = 25)
        plt.ylim(0,0.5)
        plt.ylabel('Share of total')
        plt.grid(alpha = 0.75)
        plt.savefig(pth + f'attk_epochs_partition_top{tk}', pad_inches=0)
        plt.legend(loc = 'upper left')
#         plt.plot()
        plt.savefig(pth + f'attk_epochs_partition_top{tk}_legend', pad_inches=0)
        plt.show()
        plt.clf()
        
def plot_attack_sweep_datasets(traces, datasets, pct_confidences, topks, pth):
    #first plot top-p percent versus dataset for one epoch checkpoint 
    for tk in topks:
        for p_ct, p in enumerate(pct_confidences): 
            tgt_model = []
            ref_model = []
            for ct, ds in enumerate(datasets): 
                #get target model accuracies 
                AA = np.array(traces[f'A_attk_A_top{tk}_{p}conf_{ds}pc'])
                BB = np.array(traces[f'B_attk_B_top{tk}_{p}conf_{ds}pc'])
                tgt_model.append((AA + BB)[0] / 2)
                #get ref model accuracies
                AB = np.array(traces[f'A_attk_B_top{tk}_{p}conf_{ds}pc'])
                BA = np.array(traces[f'B_attk_A_top{tk}_{p}conf_{ds}pc'])
                ref_model.append((AB + BA)[0] / 2)
            plt.plot(datasets, tgt_model, '-o', color = f'C{p_ct}')
            plt.plot(datasets, ref_model, '--o', color = f'C{p_ct}')
            if p != 100: 
                plt.plot([],[], 'o', color = f'C{p_ct}', label = f'accuracy on top {p}% conf.')
            else: 
                plt.plot([],[], 'o', color = f'C{p_ct}', label = f'accuracy on all examples')
#             plt.plot([],[], 'o', color = f'C{p_ct}', label = f'{p}% conf')
        plt.ylim(0,1)
        plt.xlabel('train set size (thousands)', fontsize = 25)
        plt.ylabel('Label Inference Accuracy')
        plt.plot([], [], '--', color = 'black', label='ref. model correlation baseline')
        plt.grid(alpha = 0.75)
        plt.savefig(pth + f'attk_datasets_acc_top{tk}', pad_inches=0)
        plt.legend(loc = 'upper left')
        plt.savefig(pth + f'attk_datasets_acc_top{tk}_legend', pad_inches=0)
        plt.show()
        plt.clf()

    #second plot quadrant breakdown 
    for tk in topks:
        mem, bad, corr = [], [], []
        for ct, ds in enumerate(datasets): 
            #get 'memorized' examples
            mem_A = np.array(traces[f'yes_A_no_B_attk_A_top{tk}_{ds}pc'])
            mem_B = np.array(traces[f'no_A_yes_B_attk_B_top{tk}_{ds}pc'])
            mem.append((mem_A + mem_B)[0] / 2)
            #get 'bad embed' examples
            bad_A = np.array(traces[f'no_A_yes_B_attk_A_top{tk}_{ds}pc'])
            bad_B = np.array(traces[f'yes_A_no_B_attk_B_top{tk}_{ds}pc'])
            bad.append((bad_A + bad_B)[0] / 2)
            #get 'correlation' examples 
            corr_A = np.array(traces[f'yes_A_yes_B_attk_A_top{tk}_{ds}pc'])
            corr_B = np.array(traces[f'yes_A_yes_B_attk_B_top{tk}_{ds}pc'])
            corr.append((corr_A + corr_B)[0] / 2)
            
        mem, bad, corr = np.array(mem), np.array(bad), np.array(corr)

        #plot the share of each 
        plt.fill_between(datasets, np.zeros(len(datasets)), corr, color = 'C3', label = 'correlated')
        plt.plot(datasets, corr, color = 'black')
        plt.fill_between(datasets, corr, corr+bad, color = 'C4', label = 'forgotten')
        plt.plot(datasets, corr+bad, color = 'black')
        plt.fill_between(datasets, corr+bad, corr+bad+mem, color = 'C5', label = 'memorized')
        plt.plot(datasets, corr+bad+mem, color = 'black')
#         plt.fill_between(datasets, corr+bad+mem, 1, color="gray",
#                              edgecolor="black", alpha = 0.2, label = 'unassociated')

        plt.xlabel('train set size (thousands)', fontsize = 25)
        plt.ylim(0,0.5)
        plt.ylabel('Share of total')
        plt.grid(alpha = 0.75)
        plt.savefig(pth + f'attk_datasets_partition_top{tk}', pad_inches=0)
        plt.legend(loc = 'upper left')
#         plt.plot()
        plt.savefig(pth + f'attk_datasets_partition_top{tk}_legend', pad_inches=0)
        plt.show()
        plt.clf()

def get_linprobe_plots_epochs(model, dataset, epochs, pth): 
    #get attack data 
    base_dir = Path(f'/checkpoint/caseymeehan/experiments/ssl_sweep/{model}/lin_probe_sweeps')
    ds = dataset
    
    traces = defaultdict(list)

    for ep in epochs: 
        folder = base_dir / f'lp_{model}_{ds}pc_{ep}ep' 
        files = os.listdir(folder)
        pid = files[0].split('_')[0]
        file = folder / (pid + '_0_log.out')
        with open(file) as f:
            lines = f.readlines()
        trn = ast.literal_eval(lines[-2])
        val = ast.literal_eval(lines[-6])
        #get train acc
        traces[f'train_top_1_{ds}pc'].append(trn['top1'])
        traces[f'train_top_5_{ds}pc'].append(trn['top5'])
        #get val acc
        traces[f'val_top_1_{ds}pc'].append(val['top1'])
        traces[f'val_top_5_{ds}pc'].append(val['top5'])

    # plot linear probe for each topk 
    topks = [1,5]
    for tk in topks:
        #plot train performance 
        plt.plot(epochs, traces[f'train_top_{tk}_{ds}pc'], '-o', color = 'C6', label='train')
        #plot val performance 
        plt.plot(epochs, traces[f'val_top_{tk}_{ds}pc'], '-o', color = 'C7',  label='validation')
#         #plot gap
#         plt.plot(epochs, np.array(traces[f'train_top_{tk}_{ds}pc']) - np.array(traces[f'val_top_{tk}_{ds}pc']), 
#                      '-o', color = 'C7',  label='gap')

        plt.xlabel('epochs', fontsize = 25)
        plt.ylim(0,1.1)
        plt.ylabel('Linear Probe Accuracy')
        plt.grid(alpha = 0.75)
        plt.savefig(pth + f'lp_epochs_top{tk}', pad_inches=0)
        plt.legend(loc = 'lower left')
#         plt.plot()
        plt.savefig(pth + f'lp_epochs_top{tk}_legend', pad_inches=0)
        plt.show()
        plt.clf()


    return traces 

def get_linprobe_plots_datasets(model, datasets, epoch, pth):
    #get saved data 
    base_dir = Path(f'/checkpoint/caseymeehan/experiments/ssl_sweep/{model}/lin_probe_sweeps')# Get linear probe results
    traces = defaultdict(list)
    ep = epoch
    for ds in datasets: 
        folder = base_dir / f'lp_{model}_{ds}pc_{ep}ep' 
        files = os.listdir(folder)
        pid = files[0].split('_')[0]
        file = folder / (pid + '_0_log.out')
        with open(file) as f:
            lines = f.readlines()
        trn = ast.literal_eval(lines[-2])
        val = ast.literal_eval(lines[-6])
        #get train acc
        traces[f'train_top_1_pc'].append(trn['top1'])
        traces[f'train_top_5_pc'].append(trn['top5'])
        #get val acc
        traces[f'val_top_1_pc'].append(val['top1'])
        traces[f'val_top_5_pc'].append(val['top5'])
            
    # plot linear probe for each topk 
    topks = [1,5]
    for tk in topks:
        #plot train performance 
        plt.plot(datasets, traces[f'train_top_{tk}_pc'], '-o', color = 'C6', label='train')
        #plot val performance 
        plt.plot(datasets, traces[f'val_top_{tk}_pc'], '-o', color = 'C7',  label='validation')
#         #plot gap
#         plt.plot(datasets, np.array(traces[f'train_top_{tk}_pc']) - np.array(traces[f'val_top_{tk}_pc']), 
#                      '-o', color = 'C7',  label='gap')

        plt.xlabel('train set size (thousands)', fontsize = 25)
        plt.ylim(0,1.1)
        plt.ylabel('Linear Probe Accuracy')
        plt.grid(alpha = 0.75)
        plt.savefig(pth + f'lp_datasets_top{tk}', pad_inches=0)
        plt.legend(loc = 'lower left')
#         plt.plot()
        plt.savefig(pth + f'lp_datasets_top{tk}_legend', pad_inches=0)
        plt.show()
        plt.clf()

    return traces 


def print_class_statistics(attk_data, attk_set, epoch, ds, k = 10): 
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
    plt.xlabel('Class Accuracy Gap', fontsize = 25)
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


def get_attack_data_v_params(model, param_vals, epochs, k_neighb, pct_confidences, topks): 
    #get attack data 
    base_dir = Path(f'/checkpoint/caseymeehan/experiments/ssl_sweep/{model}/attack_sweeps')
    traces = defaultdict(list)
    
    for pv in param_vals: 
        for ep in epochs: 
            #load in attack data
            if model == 'simclr': 
                pvstr = str(pv)[2:]
                if len(pvstr) < 2: 
                    pvstr += '0'
                folder = base_dir / f"NN_attk_{model}_p{pvstr}t_{ep}ep"
            else: 
                folder = base_dir / f'NN_attk_{model}_sim{pv}_{ep}ep'
            #attack on set A and B
            for attk_set in ('A', 'B'): 
                A_idxs = np.load(folder / f'A_attk_{attk_set}_attk_idxs.npy')
                B_idxs = np.load(folder / f'B_attk_{attk_set}_attk_idxs.npy')
                B_order = [np.where(B_idxs == i)[0][0] for i in A_idxs]
                A_labels = np.load(folder / f'A_attk_{attk_set}_labels.npy')
                B_labels = np.load(folder / f'B_attk_{attk_set}_labels.npy')[B_order]
                A_neighb_labels = np.load(folder / f'A_attk_{attk_set}_neighb_labels.npy')
                B_neighb_labels = np.load(folder / f'B_attk_{attk_set}_neighb_labels.npy')[B_order]

                #get confidences 
                A_conf, A_preds = get_confidence_and_topk(A_neighb_labels, k_neighb, topk = topks)
                A_conf_order = np.argsort(A_conf)[::-1]
                B_conf, B_preds = get_confidence_and_topk(B_neighb_labels, k_neighb, topk = topks)
                B_conf_order = np.argsort(B_conf)[::-1]
                n = len(A_conf_order)

                #get top-p pct most confident accuracies
                for p in pct_confidences: 
                    num = int(n * p / 100)
                    A_conf_idxs = A_conf_order[:num]
                    B_conf_idxs = B_conf_order[:num]
                    for tk in topks: 
                        A_acc = (A_preds[f'top_{tk}'][A_conf_idxs] == A_labels[A_conf_idxs]).sum(axis = 1).mean()
                        traces[f'A_attk_{attk_set}_top{tk}_{p}conf_{pv}'].append(A_acc)
                        B_acc = (B_preds[f'top_{tk}'][B_conf_idxs] == B_labels[B_conf_idxs]).sum(axis = 1).mean()
                        traces[f'B_attk_{attk_set}_top{tk}_{p}conf_{pv}'].append(B_acc)

                #get 4 quadrant dataset breakdown 
                for tk in topks: 
                    A_accs = (A_preds[f'top_{tk}'] == A_labels).sum(axis = 1) == 1
                    B_accs = (B_preds[f'top_{tk}'] == B_labels).sum(axis = 1) == 1
                    yes_A_yes_B = (A_accs & B_accs).mean()
                    traces[f'yes_A_yes_B_attk_{attk_set}_top{tk}_{pv}'].append(yes_A_yes_B)
                    yes_A_no_B = (A_accs & ~B_accs).mean()
                    traces[f'yes_A_no_B_attk_{attk_set}_top{tk}_{pv}'].append(yes_A_no_B)
                    no_A_yes_B = (~A_accs & B_accs).mean()
                    traces[f'no_A_yes_B_attk_{attk_set}_top{tk}_{pv}'].append(no_A_yes_B)
                    no_A_no_B = (~A_accs & ~B_accs).mean()
                    traces[f'no_A_no_B_attk_{attk_set}_top{tk}_{pv}'].append(no_A_no_B)
    return traces


def plot_attack_sweep_params(model, traces, param_vals, pct_confidences, topks, pth):
    if model == 'simclr':
        xlabel = 'Temperature, $\\tau$'
    else: 
        xlabel = 'Invariance, $\lambda$'
    #first plot top-p percent versus dataset for one epoch checkpoint 
    for tk in topks:
        for p_ct, p in enumerate(pct_confidences): 
            tgt_model = []
            ref_model = []
            for ct, pv in enumerate(param_vals): 
                #get target model accuracies 
                AA = np.array(traces[f'A_attk_A_top{tk}_{p}conf_{pv}'])
                BB = np.array(traces[f'B_attk_B_top{tk}_{p}conf_{pv}'])
                tgt_model.append((AA + BB)[0] / 2)
                #get ref model accuracies
                AB = np.array(traces[f'A_attk_B_top{tk}_{p}conf_{pv}'])
                BA = np.array(traces[f'B_attk_A_top{tk}_{p}conf_{pv}'])
                ref_model.append((AB + BA)[0] / 2)
            plt.plot(param_vals, tgt_model, '-o', color = f'C{p_ct}')
            plt.plot(param_vals, ref_model, '--o', color = f'C{p_ct}')
            if p != 100: 
                plt.plot([],[], 'o', color = f'C{p_ct}', label = f'accuracy on top {p}% conf.')
            else: 
                plt.plot([],[], 'o', color = f'C{p_ct}', label = f'accuracy on all examples')
#             plt.plot([],[], 'o', color = f'C{p_ct}', label = f'{p}% conf')
        plt.ylim(0,1)
        plt.xlabel(xlabel, fontsize = 25)
        plt.ylabel('Label Inference Accuracy')
        plt.plot([], [], '--', color = 'black', label='ref. model correlation baseline')
        plt.grid(alpha = 0.75)
        plt.savefig(pth + f'attk_params_acc_top{tk}', pad_inches=0)
        plt.legend(loc = 'upper left')
        plt.savefig(pth + f'attk_params_acc_top{tk}_legend', pad_inches=0)
        plt.show()
        plt.clf()

    #second plot quadrant breakdown 
    for tk in topks:
        mem, bad, corr = [], [], []
        for ct, pv in enumerate(param_vals): 
            #get 'memorized' examples
            mem_A = np.array(traces[f'yes_A_no_B_attk_A_top{tk}_{pv}'])
            mem_B = np.array(traces[f'no_A_yes_B_attk_B_top{tk}_{pv}'])
            mem.append((mem_A + mem_B)[0] / 2)
            #get 'bad embed' examples
            bad_A = np.array(traces[f'no_A_yes_B_attk_A_top{tk}_{pv}'])
            bad_B = np.array(traces[f'yes_A_no_B_attk_B_top{tk}_{pv}'])
            bad.append((bad_A + bad_B)[0] / 2)
            #get 'correlation' examples 
            corr_A = np.array(traces[f'yes_A_yes_B_attk_A_top{tk}_{pv}'])
            corr_B = np.array(traces[f'yes_A_yes_B_attk_B_top{tk}_{pv}'])
            corr.append((corr_A + corr_B)[0] / 2)
            
        mem, bad, corr = np.array(mem), np.array(bad), np.array(corr)

        #plot the share of each 
        plt.fill_between(param_vals, np.zeros(len(param_vals)), corr, color = 'C3', label = 'correlated')
        plt.plot(param_vals, corr, color = 'black')
        plt.fill_between(param_vals, corr, corr+bad, color = 'C4', label = 'forgotten')
        plt.plot(param_vals, corr+bad, color = 'black')
        plt.fill_between(param_vals, corr+bad, corr+bad+mem, color = 'C5', label = 'memorized')
        plt.plot(param_vals, corr+bad+mem, color = 'black')
#         plt.fill_between(param_vals, corr+bad+mem, 1, color="gray",
#                              edgecolor="black", alpha = 0.2, label = 'unassociated')

        if model == 'simclr':
            plt.xlabel(xlabel, fontsize = 25)
        else: 
            plt.xlabel(xlabel, fontsize = 25)
        plt.ylim(0,0.5)
        plt.ylabel('Share of total')
        plt.grid(alpha = 0.75)
        plt.savefig(pth + f'attk_params_partition_top{tk}', pad_inches=0)
        plt.legend(loc = 'upper left')
#         plt.plot()
        plt.savefig(pth + f'attk_params_partition_top{tk}_legend', pad_inches=0)
        plt.show()
        plt.clf()
        
        
def get_linprobe_plots_params(model, param_vals, epoch, pth):
    if model == 'simclr':
        xlabel = 'Temperature, $\\tau$'
    else: 
        xlabel = 'Invariance, $\lambda$'
    #get saved data 
    base_dir = Path(f'/checkpoint/caseymeehan/experiments/ssl_sweep/{model}/lin_probe_sweeps')# Get linear probe results
    traces = defaultdict(list)
    ep = epoch
    for pv in param_vals: 
        if model == 'simclr': 
            pvstr = str(pv)[2:]
            if len(pvstr) < 2: 
                pvstr += '0'
            folder = base_dir / f"lp_{model}_p{pvstr}t_{ep}ep"
        else: 
            folder = base_dir / f'lp_{model}_sim{pv}_{ep}ep'
        files = os.listdir(folder)
        pid = files[0].split('_')[0]
        file = folder / (pid + '_0_log.out')
        with open(file) as f:
            lines = f.readlines()
        trn = ast.literal_eval(lines[-2])
        val = ast.literal_eval(lines[-6])
        #get train acc
        traces[f'train_top_1_pc'].append(trn['top1'])
        traces[f'train_top_5_pc'].append(trn['top5'])
        #get val acc
        traces[f'val_top_1_pc'].append(val['top1'])
        traces[f'val_top_5_pc'].append(val['top5'])
            
    # plot linear probe for each topk 
    topks = [1,5]
    for tk in topks:
        #plot train performance 
        plt.plot(param_vals, traces[f'train_top_{tk}_pc'], '-o', color = 'C6', label='train')
        #plot val performance 
        plt.plot(param_vals, traces[f'val_top_{tk}_pc'], '-o', color = 'C7',  label='validation')
#         #plot gap
#         plt.plot(param_vals, np.array(traces[f'train_top_{tk}_pc']) - np.array(traces[f'val_top_{tk}_pc']), 
#                      '-o', color = 'C7',  label='gap')

        plt.xlabel(xlabel, fontsize = 25)
        plt.ylim(0,1.1)
        plt.ylabel('Linear Probe Accuracy')
        plt.grid(alpha = 0.75)
        plt.savefig(pth + f'lp_param_vals_top{tk}', pad_inches=0)
        plt.legend(loc = 'lower left')
#         plt.plot()
        plt.savefig(pth + f'lp_param_vals_top{tk}_legend', pad_inches=0)
        plt.show()
        plt.clf()

    return traces 

def neighbor_and_RCDM_tiles(SSL_dict, SUP_dict, example_idx, num_neighbs = 4, num_RCDM_samples = 4): 
    print('Neighbors')
    #First do neighbor grid 
    fig = plt.figure(figsize=(16, 12)) #width x height
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, num_neighbs),  # creates grid of axes
                     axes_pad=(0.05, 0.05),  # pad between axes in inch.
                     )
    A_neighbs = SSL_dict['NN_sets_A'][example_idx][:num_neighbs]
    B_neighbs = SSL_dict['NN_sets_B'][example_idx][:num_neighbs]
    sup_neighbs = SUP_dict['NN_sets_A'][example_idx][:num_neighbs]
    all_neighbs = torch.cat((A_neighbs, B_neighbs, sup_neighbs))
    for ct, im in enumerate(all_neighbs): 
        grid[ct].set_axis_off()
        grid[ct].imshow(im.permute(1,2,0))

    plt.show()

    #Then do RCDM grid 
    print('\nRCDM reconstructions')
    fig = plt.figure(figsize=(16, 12)) #width x height
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, num_RCDM_samples),  # creates grid of axes
                     axes_pad=(0.05, 0.05),  # pad between axes in inch.
                     )
    A_samps = SSL_dict['rcdm_sets_A'][example_idx][:num_RCDM_samples]
    B_samps = SSL_dict['rcdm_sets_B'][example_idx][:num_RCDM_samples]
    sup_samps = SUP_dict['rcdm_sets_A'][example_idx][:num_RCDM_samples]
    all_samps = torch.cat((A_samps, B_samps, sup_samps))
    for ct, im in enumerate(all_samps): 
        grid[ct].set_axis_off()
        grid[ct].imshow(im.permute(1,2,0))

    plt.show()
