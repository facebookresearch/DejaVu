#linear probe code training on SSL augmentations (random crop) 
#run val loop every epoch 
#print results in better format 

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
from PIL import Image, ImageOps, ImageFilter

from dejavu_utils.utils import SSL_Transform, stopwatch, SSLNetwork
from dejavu_utils.train_models.augmentations import TrainTransform

def parse_args():
    parser = argparse.ArgumentParser("linear probe args")

    parser.add_argument("--local", default = 0, type=int, help="whether to run on devfair")
    parser.add_argument("--local_gpu", default = 1, type=int, help="which device to use during local run")
    #slurm args 
    parser.add_argument("--timeout", default=1440, type=int, help="Duration of the job")
    parser.add_argument("--partition", default="learnlab", type=str, help="Partition where to submit")
    parser.add_argument("--mem_gb", default=100) 
    parser.add_argument("--use_volta32", action='store_true')
    parser.add_argument("--output_dir", type=Path) 
    
    #lin model training args
    parser.add_argument("--model_pth", type=Path) 
    parser.add_argument("--mlp", type=str, default='8192-8192-8192') 
    parser.add_argument("--loss", type=str, default='barlow') 
    parser.add_argument("--train_idx_pth", type=Path) 
    parser.add_argument("--val_idx_pth", type=Path) 
    parser.add_argument("--imgnet_train_pth", type=Path, default="/datasets01/imagenet_full_size/061417/train") 
    parser.add_argument("--start_lr", type=float, default=10)
    parser.add_argument("--end_lr", type=float, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--resnet50", action='store_true')
    parser.add_argument("--use_projector", action='store_true')
             
    return parser.parse_args()

class lin_prober: 
    def __init__(self, model, args): 
        self.bb = model.module.net
        self.pj = model.module.projector 

        train_idxs = np.load(args.train_idx_pth)
        val_idxs = np.load(args.val_idx_pth)

        imgnet_train = ImageFolder(args.imgnet_train_pth, transform = TrainTransform())
        self.train_set = Subset(imgnet_train, train_idxs)

        imgnet_eval = ImageFolder(args.imgnet_train_pth, transform = SSL_Transform())
        self.train_eval_set = Subset(imgnet_eval, train_idxs)
        self.val_set = Subset(imgnet_eval, val_idxs)

        self.gpu = args.gpu
        self.use_projector = args.use_projector
        
        #Get backbone/projector dimension 
        test = torch.zeros(3,10,10).cuda()
        xfrm = transforms.Compose([transforms.ToPILImage(), SSL_Transform()])
        test = xfrm(test).unsqueeze(0).cuda()
        o_bb = self.bb(test)
        o_pj = self.pj(o_bb)
        self.bb_dim = o_bb.shape[1]
        self.pj_dim = o_pj.shape[1]

        #Init. variables 
        self.lin_model = None
                
    def train(self, n_epochs, start_lr, end_lr, val_every_epoch = True): 
        #instantiate model: 
        if self.use_projector: 
            self.lin_model = nn.Linear(self.pj_dim, 1000).cuda()
        else: 
            self.lin_model = nn.Linear(self.bb_dim, 1000).cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.lin_model.parameters(), lr = start_lr, momentum = 0.9)
        train_loader = DataLoader(self.train_set, batch_size = 64, num_workers = 8, shuffle = True)

        print_every = 50
        n = n_epochs*len(train_loader)
        sw = stopwatch(n)
        sw.start()        
        ct = 0
        for epoch in range(n_epochs):
            print(f"\nepoch: {epoch}")
            accs = []
            t5_accs = []
            for step, ((_, x), y) in enumerate(train_loader): 
                optimizer.zero_grad()
                x, y = x.cuda(), y.cuda()
                with torch.no_grad(): 
                    embed = self.bb(x)
                    if self.use_projector: 
                        embed = self.pj(embed)
                logits = self.lin_model(embed)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                #get top-1 acc: 
                preds = logits.argmax(dim = 1)
                acc = (preds == y).float().mean().item()
                accs.append(acc)
                
                #get top-5 acc: 
                preds = logits.topk(5, dim = 1)[1]
                corr = (preds == y[:,None])
                acc = corr.sum(dim = 1).float().mean().item()
                t5_accs.append(acc)
                
                if (ct+1) % print_every == 0: 
                    lr = optimizer.param_groups[0]['lr']
                    optimizer.param_groups[0]['lr'] = start_lr*(1-ct/n) + end_lr*(ct/n)
                    print(f"ave t1 batch acc: {np.mean(accs):.2f}, ave t5 batch acc: {np.mean(t5_accs):.2f}, LR:{lr:.2e}")
                    accs = []
                    t5_accs = []
                    
                    print(f"progress: {ct/n:.2f}, min remaining: {sw.time_remaining(ct)/60:.1f}")

                ct += 1
            if val_every_epoch: 
                print('\nchecking val accuracy...')
                acc_dict = self.score(train = False)
                acc_dict['epoch'] = epoch
                print(acc_dict)

                    
    def score(self, train = True):
        accs = []
        t5_accs = []
        if train: 
            loader = DataLoader(self.train_eval_set, batch_size = 64, num_workers = 8, shuffle = True)
        else: 
            loader = DataLoader(self.val_set, batch_size = 64, num_workers = 8, shuffle = True)
        for x,y in loader: 
            with torch.no_grad():
                x, y = x.cuda(), y.cuda()
                embed = self.bb(x)
                if self.use_projector: 
                    embed = self.pj(embed)
                logits = self.lin_model(embed)
        
                #get top-1 acc: 
                preds = logits.argmax(dim = 1)
                acc = (preds == y).float().mean().item()
                accs.append(acc)
        
                #get top-5 acc: 
                preds = logits.topk(5, dim = 1)[1]
                corr = (preds == y[:,None])
                acc = corr.sum(dim = 1).float().mean().item()
                t5_accs.append(acc)
                
        if train: 
            dset = 'train'
        else: 
            dset = 'val' 
        acc_dict = {
                'dset': dset, 
                'top1': np.mean(accs), 
                'top5': np.mean(t5_accs)
            }
        print(f"top-1_acc: {acc_dict['top1']:.3f} top-5 acc: {acc_dict['top5']:.3f}")
        return acc_dict



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

    model = SSLNetwork(arch = arch, 
                          remove_head = 0, 
                          mlp = args.mlp, 
                          fc = 0,
                          patch_keep = 1.0,
                          loss = args.loss).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    ckpt = torch.load(args.model_pth, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict = False)
    _ = model.eval()
    
    #Instantiate linear prober
    lp = lin_prober(model, args) 

    #Train linear prober will print val accuracy every epoch
    print('\ntrain linear probe...')
    lp.train(args.epochs, args.start_lr, args.end_lr)

    #Get final train accuracy of prober
    print('\nchecking final train accuracy...')
    train_acc_dict = lp.score(train = True)
    print(train_acc_dict)

#CONTINUE HERE 

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

    executor.update_parameters(name="lin_probe")

    args.dist_url = get_init_file().as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    if args.local == 1: 
        print('running locally on devfair') 
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
