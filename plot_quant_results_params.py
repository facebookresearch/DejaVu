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
    parser.add_argument("--param_vals", type=str, default='',
            help="comma-delimited list of parameter values to sweep")
    parser.add_argument("--pct_confidences", type=str, default='5,10,30',
            help="comma-delimited string of confidence percentages to plot")
    return parser.parse_args()

def main(args): 
    print('initializing plot params...') 

    #first make sure save directory is there 
    savepth = f"/checkpoint/caseymeehan/experiments/ssl_sweep/{args.model}/plots/"
    Path(savepth).mkdir(parents=True, exist_ok=True)
    
    #get param vals as list
    if args.model=='simclr': 
        param_vals = [float(x) for x in args.param_vals.split(',')]
    else: 
        param_vals = [int(x) for x in args.param_vals.split(',')]


    #get param vals as list
    pct_confidences = [float(x) for x in args.pct_confidences.split(',')]

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
    mpl.rcParams['xtick.labelsize'] = 15
    mpl.rcParams['ytick.labelsize'] = 15
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['grid.linewidth'] = 1.5
    mpl.rcParams['lines.markersize'] = 8
    mpl.rcParams['legend.fontsize'] = 15

    ###
    #ATTACK PLOTS
    ###
    print('getting attack plots...') 

    #Attack performance vs. epochs
    traces = pu.get_attack_data_v_params(
                model = args.model, 
                param_vals = param_vals, 
                epochs = [1000], 
                k_neighb = args.k_neighb, 
                pct_confidences = pct_confidences, 
                topks = [1,5]
                )

    pu.plot_attack_sweep_params(
            model = args.model, 
            traces = traces, 
            param_vals = param_vals, 
            pct_confidences = pct_confidences, 
            topks = [1,5], 
            pth = savepth
            )

    ###
    #LINEAR PROBE PLOTS
    ###
    print('getting linear probe plots...') 

    #linear probe performance vs. param values
    _ = pu.get_linprobe_plots_params(
            model = args.model, 
            param_vals = param_vals, 
            epoch = 1000, 
            pth = savepth
            )

if __name__ == "__main__": 
    args = parse_args()
    main(args)
