# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import numpy as np
import torch as th
import torch.nn as nn
import torchvision
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from torch.cuda.amp import autocast
try:
    from RCDM.RCDM_utils import get_ssl_model_from_ckpt, embed_bb, embed_proj, embed_activ, Transform
except:
    from RCDM_utils import get_ssl_model_from_ckpt, embed_bb, embed_proj, embed_activ, Transform

def main(args):
    # Init distributed setup
    dist_util.init_distributed_mode(args)
    logger.configure(dir=args.out_dir)

    # Load SSL model
    if args.feat_cond and args.ssl_from_ckpt: 
        ssl_model = get_ssl_model_from_ckpt(args.ssl_model_pth, args.gpu, 
                                            args.ssl_loss, args.ssl_arch, 
                                            args.mlp, args.use_supervised_activations
                                            ).to(args.gpu).eval()
        if args.use_supervised_activations: 
            ssl_dim = 1000
        else: 
            ssl_dim = ssl_model.module.representation_size + ssl_model.module.num_features
        print("SSL DIM Backbone + Projector:", ssl_dim)
        for _,p in ssl_model.named_parameters():
            p.requires_grad_(False)

    else:
        ssl_model = None
        ssl_dim = 2048
        print("No SSL models")

    # Create RCDM
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()), G_shared=args.no_shared, feat_cond=args.feat_cond, ssl_dim=ssl_dim
    )

    model.to(args.gpu)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # Create the dataloader
    logger.log("creating data loader...")
    data = load_ssl_data_projbackbone(
        args,
        ssl_model=ssl_model,
    )

    logger.log("training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps
    ).run_loop()

def load_ssl_data_projbackbone(args, ssl_model=None):
    dataset = torchvision.datasets.ImageFolder(args.data_dir, transform = Transform(args.image_size))
    ds_idxs = np.load(args.dataset_indices)                                          
    dataset = th.utils.data.Subset(dataset, ds_idxs) 
    sampler = th.utils.data.distributed.DistributedSampler(dataset)                                                
    loader = th.utils.data.DataLoader(                                                                          
            dataset, batch_size=args.batch_size, num_workers=args.workers,                                       
            pin_memory=True, sampler=sampler)  
    
    while True: 
        for (ssl_batch, rcdm_batch), _ in loader:
            # We add the conditioning in conditional mode
            model_kwargs = {}
            if ssl_model is not None:
                with th.no_grad():
                    with autocast(args.use_fp16):
                        backbone = embed_bb(ssl_model, ssl_batch.to(args.gpu)).detach()
                        proj = embed_proj(ssl_model, backbone).detach()
                        if not args.use_supervised_activations: 
                            model_kwargs["feat"] = th.cat((backbone, proj), dim = 1)
                        else: 
                            model_kwargs["feat"] = embed_activ(ssl_model, proj)
                yield rcdm_batch.to(args.gpu), model_kwargs
            else:
                yield rcdm_batch.to(args.gpu), model_kwargs

def create_argparser():
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        submitit=False,
        local_rank=0,
        dist_url="env://",
        #new args for loading pretrained fast_ssl 
        ssl_from_ckpt='False', 
        ssl_model_pth='', 
        dataset_indices='',
        data_dir='',
        ssl_loss='vicreg',
        ssl_arch='resnet50',
        mlp='8192-8192-8192', 
        use_supervised_activations=False, 
        workers = 8, 
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--feat_cond', action='store_true', default=False,
                        help='Activate conditional RCDM.')
    parser.add_argument('--use_head', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--no_shared', action='store_false', default=True,
                        help='Disable the shared lower dimensional projection of the representation.')
    parser.add_argument('--type_model', type=str, default="dino",
                    help='Select the type of model to use.')
    #parser.add_argument('--data_dir', type=str, default="toto",
    #                help='data directory')
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
