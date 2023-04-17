from dejavu_utils import plot_utils as pu
import matplotlib as mpl
from matplotlib import pyplot as plt
from cycler import cycler
import numpy as np
import argparse
from pathlib import Path

def parse_args(): 
    parser = argparse.ArgumentParser("plotting args")
    
    parser.add_argument("--model", type=str,
            help="name of model to plot results for")
    parser.add_argument("--k_neighb", type=int, default=100,
            help="number of neighbors to use in attack")
    parser.add_argument("--epoch_sweep_dataset", type=int, default=300,
            help="which dataset to use in epoch sweep plots")
    parser.add_argument("--dataset_sweep_epoch", type=int, default=1000,
            help="which epoch to use in dataset sweep plots")
    parser.add_argument("--pct_confidences", type=str, default='5,10,30',
            help="comma-delimited string of confidence percentages to plot")
    parser.add_argument("--attack_fname", type=str, default='attack_sweeps',
            help="name of folder under model directory where attack data is stored. Change if plotting backbone or cornercrop results")
    parser.add_argument("--plot_lin_probe", type=int, default=1,
            help="whether to plot linear probe results, too. Set to 0 for rn50 results.")
    return parser.parse_args()

def main(args): 
    print('initializing plot params...') 

    #get param vals as list
    pct_confidences = [float(x) for x in args.pct_confidences.split(',')]

    #first get save directory 
    if args.attack_fname == "attack_sweeps": 
        #save standard attack plots used in main paper 
        savepth = f"/checkpoint/caseymeehan/experiments/ssl_sweep/{args.model}/plots/"
    else: 
        #save specialized attack plots (backbone, cornercrop test) used in appendix 
        savepth = f"/checkpoint/caseymeehan/experiments/ssl_sweep/{args.model}/plots_{args.attack_fname}/"
    Path(savepth).mkdir(parents=True, exist_ok=True)

    #set parameters 
    plt.figure()
    plt.style.use(['science'])
    mpl.rcParams['axes.prop_cycle'] = cycler(color = [
              '#0000FF', #blue
              '#00BFFF', #sky blue
              '#006400', #dark turquoise
              '#00008B', #dark blue
              '#DC143C', #crimson
              '#FF8C00', #dark orange
              '#9400D3', #dark violet
              '#228B22', #forest green
             ])
    mpl.rcParams['figure.figsize'] = (4,4)
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 20
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['grid.linewidth'] = 1.5
    mpl.rcParams['lines.markersize'] = 8
    mpl.rcParams['legend.fontsize'] = 15

    ###
    #ATTACK PLOTS
    ###
    print('getting attack plots...') 

    #Attack performance vs. epochs
    traces = pu.get_attack_data(
                model = args.model,
                datasets = [args.epoch_sweep_dataset],
                epochs = [50, 100, 250, 500, 750, 1000],
                k_neighb = args.k_neighb,
                pct_confidences = pct_confidences,
                topks = [1,5],
                fname = args.attack_fname,
    )

    pu.plot_attack_sweep_epochs(
        traces,
        args.epoch_sweep_dataset,
        epochs = [50, 100, 250, 500, 750, 1000],
        pct_confidences = pct_confidences,
        topks = [1,5],
        pth = savepth
    )

    #Attack performance vs. dataset size 
    traces_ds = pu.get_attack_data(
                model = args.model,
                datasets = [100, 200, 300, 400, 500],
                epochs = [args.dataset_sweep_epoch],
                k_neighb = args.k_neighb,
                pct_confidences = pct_confidences,
                topks = [1,5],
                fname = args.attack_fname,
    )

    pu.plot_attack_sweep_datasets(
        traces_ds, 
        datasets = [100, 200, 300, 400, 500], 
        pct_confidences = pct_confidences, 
        topks = [1,5], 
        pth = savepth
    )

    ###
    #LINEAR PROBE PLOTS
    ###
    if args.plot_lin_probe == 1:
        print('getting linear probe plots...') 
    
        #linear probe vs. epoch
        _ = pu.get_linprobe_plots_epochs(
            model = args.model,
            dataset = args.epoch_sweep_dataset,
            epochs = [50, 100, 250, 500, 750, 1000],
            pth = savepth, 
        )
    
        #linear probe vs. dataset size
        _ = pu.get_linprobe_plots_datasets(
            model = args.model,
            datasets = [100, 200, 300, 400, 500],
            epoch = args.dataset_sweep_epoch,
            pth = savepth,
        )


if __name__ == "__main__": 
    args = parse_args()
    main(args)
