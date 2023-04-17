# Adapted from https://github.com/libffcv/ffcv-imagenet to support SSL
import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)
import torchvision.transforms as transforms
import torchmetrics
import numpy as np
from tqdm import tqdm
import subprocess
import os
import time
import json
import uuid
import ffcv
import scipy
import faiss
import submitit
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser
from torchvision.utils import make_grid, save_image

from fastargs import get_current_config, set_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from utils import LARS, cosine_scheduler, learning_schedule
from scipy.stats import entropy
from augmentations import TrainTransform
import torchvision

Section('model', 'model details').params(
    arch=Param(str, 'model to use', default='resnet50'),
    remove_head=Param(int, 'remove the projector? (1/0)', default=0),
    mlp=Param(str, 'number of projector layers', default="2048-512"),
    mlp_coeff=Param(float, 'number of projector layers', default=1),
    patch_keep=Param(float, 'Proportion of patches to keep with VIT training', default=1.0),
    fc=Param(int, 'remove the projector? (1/0)', default=0),
    proj_relu=Param(int, 'Proj relu? (1/0)', default=0),
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=64),
    max_res=Param(int, 'the maximum (starting) resolution', default=224),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=30),
    start_ramp=Param(int, 'when to start interpolating resolution', default=10)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', default=""),
    bboxA_dataset=Param(str, '.dat file to use for training bbox', default=""),
    bboxB_dataset=Param(str, '.dat file to use for training bbox', default=""),
    public_dataset=Param(str, '.dat file to use as public data', default=""),
    num_workers=Param(int, 'The number of workers', default=10),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True),
    random_seed=Param(int, 'Purcentage of noised labels', default=-0),
    use_torchvision=Param(int, 'Purcentage of noised labels', default=0)
)

Section('attack', 'attack').params(
    k=Param(int, 'number of neigbors faiss', default=100),
    k_attk=Param(int, 'number of neigbors attack', default=100),
)

Section('vicreg', 'Vicreg').params(
    sim_coeff=Param(float, 'VicREG MSE coefficient', default=25),
    std_coeff=Param(float, 'VicREG STD coefficient', default=25),
    cov_coeff=Param(float, 'VicREG COV coefficient', default=1),
)

Section('simclr', 'simclr').params(
    temperature=Param(float, 'SimCLR temperature', default=0.15),
)

Section('barlow', 'barlow').params(
    lambd=Param(float, 'Barlow Twins Lambd parameters', default=0.0051),
)

Section('byol', 'byol').params(
    momentum_teacher=Param(float, 'Momentum Teacher value', default=0.996),
)

Section('dino', 'dino').params(
    warmup_teacher_temp=Param(float, 'weight decay', default=0.04),
    teacher_temp=Param(float, 'weight decay', default=0.07),
    warmup_teacher_temp_epochs=Param(int, 'weight decay', default=50),
    student_temp=Param(float, 'center momentum dino', default=0.1),
    center_momentum=Param(float, 'center momentum dino', default=0.9),
    momentum_teacher=Param(float, 'Momentum Teacher value', default=0.996),
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True),
    log_level=Param(int, '0 if only at end 1 otherwise', default=2),
    checkpoint_freq=Param(int, 'When saving checkpoints', default=5), 
    snapshot_freq=Param(int, 'How often to save a model snapshot', default=50), 
    test_attack=Param(int, 'whether to test attack', default=1)
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=256),
    resolution=Param(int, 'final resized validation image size', default=224),
)

Section('training', 'training hyper param stuff').params(
    eval_only=Param(int, 'eval only?', default=0),
    eval_freq=Param(float, 'number of epochs', default=1),
    batch_size=Param(int, 'The batch size', default=512),
    num_small_crops=Param(int, 'number of crops?', default=0),
    optimizer=Param(And(str, OneOf(['sgd', 'adamw', 'lars'])), 'The optimizer', default='adamw'),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    epochs=Param(int, 'number of epochs', default=30),
    base_lr=Param(float, 'number of epochs', default=0.0005),
    end_lr_ratio=Param(float, 'number of epochs', default=0.001),
    label_smoothing=Param(float, 'label smoothing parameter', default=0),
    distributed=Param(int, 'is distributed?', default=0),
    clip_grad=Param(float, 'sign the weights of last residual block', default=0),
    use_ssl=Param(int, 'use ssl data augmentations?', default=0),
    loss=Param(str, 'use ssl data augmentations?', default="simclr"),
    train_probes_only=Param(int, 'load linear probes?', default=0),
)

Section('dist', 'distributed training options').params(
    use_submitit=Param(int, 'enable submitit', default=0),
    world_size=Param(int, 'number gpus', default=1),
    ngpus=Param(int, 'number of gpus per nodes', default=8),
    nodes=Param(int, 'number of nodes', default=1),
    comment=Param(str, 'comment for slurm', default=''),
    timeout=Param(int, 'timeout', default=2800),
    partition=Param(str, 'partition', default="learnlab"),
    address=Param(str, 'address', default='localhost'),
    use_volta32=Param(int, 'use_volta', default=0),
    port=Param(str, 'port', default='58492')
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

################################
##### Some Miscs functions #####
################################

def get_shared_folder() -> Path:
    user = os.getenv("USER")
    path = "/checkpoint/"
    if Path(path).is_dir():
        p = Path(f"{path}{user}/experiments")
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

def exclude_bias_and_norm(p):
    return p.ndim == 1

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def gather_center(x):
    x = batch_all_gather(x)
    x = x - x.mean(dim=0)
    return x

def batch_all_gather(x):
    x_list = GatherLayer.apply(x.contiguous())
    return ch.cat(x_list, dim=0)

class GatherLayer(ch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [ch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = ch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

################################
##### Loss definitions #####
################################

class SimCLRLoss(nn.Module):
    """
    SimCLR Loss:
    When using a batch size of 2048, use LARS as optimizer with a base learning rate of 0.5, 
    weight decay of 1e-6 and a temperature of 0.15.
    When using a batch size of 256, use LARS as optimizer with base learning rate of 1.0, 
    weight decay of 1e-6 and a temperature of 0.15.
    """
    @param('simclr.temperature')
    def __init__(self, batch_size, world_size, gpu, temperature):
        super(SimCLRLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size).to(gpu)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = ch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.size(0)
        N = 2 * batch_size * self.world_size

        if self.world_size > 1:
            z_i = ch.cat(GatherLayer.apply(z_i), dim=0)
            z_j = ch.cat(GatherLayer.apply(z_j), dim=0)
        
        z = ch.cat((z_i, z_j), dim=0)

        features = F.normalize(z, dim=1)
        sim = ch.matmul(features, features.T)/ self.temperature

        sim_i_j = ch.diag(sim, batch_size * self.world_size)
        sim_j_i = ch.diag(sim, -batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = ch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        logits = ch.cat((positive_samples, negative_samples), dim=1)
        logits_num = logits
        logits_denum = ch.logsumexp(logits, dim=1, keepdim=True)
        num_sim = (- logits_num[:, 0]).sum() / N
        num_entropy = logits_denum[:, 0].sum() / N
        return num_sim, num_entropy

class VicRegLoss(nn.Module):
    """
    ViCREG Loss:
    When using a batch size of 2048, use LARS as optimizer with a base learning rate of 0.5, 
    weight decay of 1e-4 and a sim and std coeff of 25 with a cov coeff of 1.
    When using a batch size of 256, use LARS as optimizer with base learning rate of 1.5, 
    weight decay of 1e-4 and a sim and std coeff of 25 with a cov coeff of 1.
    """
    @param('vicreg.sim_coeff')
    @param('vicreg.std_coeff')
    @param('vicreg.cov_coeff')
    def __init__(self, sim_coeff, std_coeff, cov_coeff):
        super(VicRegLoss, self).__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, z_i, z_j, return_only_loss=True):
        # Repr Loss
        repr_loss = self.sim_coeff * F.mse_loss(z_i, z_j)
        std_loss = 0.
        cov_loss = 0.

        # Std Loss z_i
        x = gather_center(z_i)
        std_x = ch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = std_loss + self.std_coeff * ch.mean(ch.relu(1 - std_x))
        # Cov Loss z_i
        cov_x = (x.T @ x) / (x.size(0) - 1)
        cov_loss = cov_loss + self.cov_coeff * off_diagonal(cov_x).pow_(2).sum().div(z_i.size(1))
        
        # Std Loss z_j
        x = gather_center(z_j)
        std_x = ch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = std_loss + self.std_coeff * ch.mean(ch.relu(1 - std_x))
        # Cov Loss z_j
        cov_x = (x.T @ x) / (x.size(0) - 1)
        cov_loss = cov_loss + self.cov_coeff * off_diagonal(cov_x).pow_(2).sum().div(z_j.size(1))

        std_loss = std_loss / 2.

        loss = std_loss + cov_loss + repr_loss
        if return_only_loss:
            return loss
        else:
            return loss, repr_loss, std_loss, cov_loss

class BarlowTwinsLoss(nn.Module):
    @param('barlow.lambd')
    def __init__(self, bn, batch_size, world_size, lambd):
        super(BarlowTwinsLoss, self).__init__()
        self.bn = bn
        self.lambd = lambd
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size * self.world_size)
        ch.distributed.all_reduce(c)

        on_diag = ch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

class ByolLoss(nn.Module):
    @param('byol.momentum_teacher')
    def __init__(self, momentum_teacher):
        super().__init__()
        self.momentum_teacher = momentum_teacher

    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output.chunk(2)
        teacher_out = teacher_output.detach().chunk(2)

        student_out_1, student_out_2 = student_out
        student_out_1 = F.normalize(student_out_1, dim=-1, p=2)
        student_out_2 = F.normalize(student_out_2, dim=-1, p=2)
        teacher_out_1, teacher_out_2 = teacher_out
        teacher_out_1 = F.normalize(teacher_out_1, dim=-1, p=2)
        teacher_out_2 = F.normalize(teacher_out_2, dim=-1, p=2)
        loss_1 = 2 - 2 * (student_out_1 * teacher_out_2.detach()).sum(dim=-1)
        loss_2 = 2 - 2 * (student_out_2 * teacher_out_1.detach()).sum(dim=-1)
        return (loss_1 + loss_2).mean()


class DINOLoss(nn.Module):
    @param('dino.warmup_teacher_temp')
    @param('dino.teacher_temp')
    @param('dino.warmup_teacher_temp_epochs')
    @param('dino.student_temp')
    @param('dino.center_momentum')
    @param('dino.momentum_teacher')
    def __init__(self, out_dim, ncrops, nepochs, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, student_temp,
                 center_momentum, momentum_teacher):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.momentum_teacher = momentum_teacher
        self.ncrops = ncrops
        self.register_buffer("center", ch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = ch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @ch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = ch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

################################
##### SSL Model Generic CLass ##
################################

class SSLNetwork(nn.Module):
    @param('model.arch')
    @param('model.remove_head')
    @param('model.mlp')
    @param('model.patch_keep')
    @param('model.fc')
    @param('training.loss')
    def __init__(
        self, arch, remove_head, mlp, patch_keep, fc, loss
    ):
        super().__init__()
        if "resnet" in arch:
            import torchvision.models.resnet as resnet
            self.net = resnet.__dict__[arch]()
            if fc:
                self.net.fc = nn.Linear(2048, 256)
            else:
                self.net.fc = nn.Identity()
        elif "vgg" in arch:
            import torchvision.models.vgg as vgg
            self.net = vgg.__dict__[arch]()
            self.net.classifier = nn.Identity()
        else:
            print("Arch not found")
            exit(0)

        # Compute the size of the representation
        self.representation_size = self.net(ch.zeros((1,3,224,224))).size(1)
        print("REPR SIZE:", self.representation_size)
        # Add a projector head
        self.mlp = mlp
        if remove_head:
            self.num_features = self.representation_size
            self.projector = nn.Identity()
        else:
            self.num_features = int(self.mlp.split("-")[-1])
            self.projector = self.MLP(self.representation_size)
        self.loss = loss
        if loss == "barlow":
            self.bn = nn.BatchNorm1d(self.num_features, affine=False)
        elif loss == "byol":
            self.predictor = self.MLP(self.num_features)

    @param('model.proj_relu')
    @param('model.mlp_coeff')
    def MLP(self, size, proj_relu, mlp_coeff):
        mlp_spec = f"{size}-{self.mlp}"
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        f[-2] = int(f[-2] * mlp_coeff)
        print("MLP:", f)
        for i in range(len(f) - 2):
            layers.append(nn.Sequential(nn.Linear(f[i], f[i + 1]), nn.BatchNorm1d(f[i + 1]), nn.ReLU(True)))
        if proj_relu:
            layers.append(nn.Sequential(nn.Linear(f[-2], f[-1], bias=False), nn.ReLU(True)))
        else:
            layers.append(nn.Linear(f[-2], f[-1], bias=False))
        return nn.Sequential(*layers)

    def forward(self, inputs, embedding=False, predictor=False):
        if embedding:
            embedding = self.net(inputs)
            return embedding
        else:
            representation = self.net(inputs)
            embeddings = self.projector(representation)
            list_outputs = [representation.detach()]
            outputs_train = representation.detach()
            for l in range(len(self.projector)):
                outputs_train = self.projector[l](outputs_train).detach()
                list_outputs.append(outputs_train)
            if self.loss == "byol" and predictor:         
                embeddings = self.predictor(embeddings)
            return embeddings, list_outputs

class LinearsProbes(nn.Module):
    @param('model.mlp_coeff')
    def __init__(self, model, num_classes, mlp_coeff):
        super().__init__()
        print("NUM CLASSES", num_classes)
        mlp_spec = f"{model.module.representation_size}-{model.module.mlp}"
        f = list(map(int, mlp_spec.split("-")))
        f[-2] = int(f[-2] * mlp_coeff)
        self.probes = []
        for num_features in f:
            self.probes.append(nn.Linear(num_features, num_classes))
        self.probes = nn.Sequential(*self.probes)

    def forward(self, list_outputs, binary=False):
        return [self.probes[i](list_outputs[i]) for i in range(len(list_outputs))]


################################
##### Main Trainer ############
################################

class ImageNetTrainer:
    @param('training.distributed')
    @param('training.batch_size')
    @param('training.label_smoothing')
    @param('training.loss')
    @param('training.train_probes_only')
    @param('training.epochs')
    @param('training.num_small_crops')
    @param('data.train_dataset')
    @param('data.val_dataset')
    @param('data.bboxA_dataset')
    @param('data.bboxB_dataset')
    @param('data.public_dataset')
    @param('data.use_torchvision')
    def __init__(self, gpu, ngpus_per_node, world_size, dist_url, distributed, batch_size, label_smoothing, loss, train_probes_only, epochs, num_small_crops, train_dataset, val_dataset, bboxA_dataset, bboxB_dataset, public_dataset, use_torchvision):
        self.all_params = get_current_config()
        ch.cuda.set_device(gpu)
        self.gpu = gpu
        self.rank = self.gpu + int(os.getenv("SLURM_NODEID", "0")) * ngpus_per_node
        self.world_size = world_size
        self.seed = 50 + self.rank
        self.dist_url = dist_url
        self.batch_size = batch_size
        self.uid = str(uuid4())
        if distributed:
            self.setup_distributed()
        self.start_epoch = 0
        # Create dataLoader used for training and validation
        if use_torchvision:
            train_transforms = TrainTransform()
            ds_idxs = np.load("/private/home/caseymeehan/imgnet_splits/100_per_class/train_A.npy")
            dataset = torchvision.datasets.ImageFolder("/datasets01/imagenet_full_size/061417/train/", train_transforms)
            dataset = ch.utils.data.Subset(dataset, ds_idxs)
            self.sampler = ch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            self.train_loader = ch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=10,
                pin_memory=True,
                sampler=self.sampler,
                drop_last=True
            )
            self.num_train_exemples = len(self.train_loader.dataset)
            self.num_classes = 1000
        else:
            self.index_labels = 1
            self.train_loader = self.create_train_loader_ssl(train_dataset)
            self.num_train_exemples = self.train_loader.indices.shape[0]
            self.num_classes = 1000
        self.val_loader = self.create_val_loader(val_dataset)
        print("NUM TRAINING EXEMPLES:", self.num_train_exemples)
        # Create the dataLoader used to perform the attack
        self.bboxA_loader = self.create_attack_loader(bboxA_dataset)
        self.bboxB_loader = self.create_attack_loader(bboxB_dataset)
        self.public_loader = self.create_attack_loader(public_dataset)
        # Create SSL model
        self.model, self.scaler = self.create_model_and_scaler()
        self.num_features = self.model.module.num_features
        self.n_layers_proj = len(self.model.module.projector) + 1
        print("N layers in proj:", self.n_layers_proj)
        self.initialize_logger()
        self.classif_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.create_optimizer()
        # Create lineares probes
        self.loss = nn.CrossEntropyLoss()
        self.probes = LinearsProbes(self.model, num_classes=self.num_classes)
        self.probes = self.probes.to(memory_format=ch.channels_last)
        self.probes = self.probes.to(self.gpu)
        self.probes = ch.nn.parallel.DistributedDataParallel(self.probes, device_ids=[self.gpu])
        self.optimizer_probes = ch.optim.AdamW(self.probes.parameters(), lr=1e-4)
        # Load models if checkpoints
        self.load_checkpoint()
        # Define SSL loss
        self.do_ssl_training = False if train_probes_only else True
        self.teacher_student = False
        self.supervised_loss = False
        self.loss_name = loss
        if loss == "simclr":
            self.ssl_loss = SimCLRLoss(batch_size, world_size, self.gpu).to(self.gpu)
        elif loss == "vicreg":
            self.ssl_loss = VicRegLoss()
        elif loss == "barlow":
            self.ssl_loss = BarlowTwinsLoss(self.model.module.bn, batch_size, world_size)
        elif loss == "byol":
            self.ssl_loss = ByolLoss()
            self.teacher_student = True
            self.teacher, _ = self.create_model_and_scaler()
            self.teacher.module.load_state_dict(self.model.module.state_dict())
            self.momentum_schedule = cosine_scheduler(self.ssl_loss.momentum_teacher, 1, epochs, len(self.train_loader))
            for p in self.teacher.parameters():
                p.requires_grad = False
        elif loss == "dino":
            self.ssl_loss = DINOLoss(self.num_features, 2, epochs).to(self.gpu)
            self.teacher_student = True
            self.teacher, _ = self.create_model_and_scaler()
            self.teacher.module.load_state_dict(self.model.module.state_dict())
            self.momentum_schedule = cosine_scheduler(self.ssl_loss.momentum_teacher, 2+num_small_crops, epochs, len(self.train_loader))
            for p in self.teacher.parameters():
                p.requires_grad = False
        elif loss == "supervised":
            self.supervised_loss = True
        else:
            print("Loss not available")
            exit(1)

    # resolution tools
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param('training.use_ssl')
    @param('data.train_dataset')
    def get_dataloader(self, use_ssl, train_dataset):
        if use_ssl:
            return self.create_train_loader_ssl(160), self.create_val_loader()
        else:
            return self.create_train_loader_supervised(160), self.create_val_loader()

    def setup_distributed(self):
        dist.init_process_group("nccl", init_method=self.dist_url, rank=self.rank, world_size=self.world_size)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing):
        assert optimizer == 'sgd' or optimizer == 'adamw' or optimizer == "lars"

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k)]
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]
        if optimizer == 'sgd':
            self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        elif optimizer == 'adamw':
            # We use a big eps value to avoid instabilities with fp16 training
            self.optimizer = ch.optim.AdamW(param_groups, lr=1e-4)
        elif optimizer == "lars":
            self.optimizer = LARS(param_groups)  # to use with convnet and large batches
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optim_name = optimizer

    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    @param('data.in_memory')
    @param('training.num_small_crops')
    def create_train_loader_ssl(self, train_dataset, num_workers, batch_size,
                            distributed, in_memory, num_small_crops):
        this_device = f'cuda:{self.gpu}'
        train_path = Path(train_dataset)
        assert train_path.is_file()
        # First branch of augmentations
        self.decoder = ffcv.transforms.RandomResizedCrop((224, 224))
        image_pipeline_big: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
            ffcv.transforms.RandomGrayscale(0.2),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))
        ]

        # Second branch of augmentations
        self.decoder2 = ffcv.transforms.RandomResizedCrop((224, 224))
        image_pipeline_big2: List[Operation] = [
            self.decoder2,
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
            ffcv.transforms.RandomGrayscale(0.2),
            ffcv.transforms.RandomSolarization(0.2, 128),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        # SSL Augmentation pipeline
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True)
        ]

        pipelines={
            'image': image_pipeline_big,
            'label': label_pipeline,
            'image_0': image_pipeline_big2
        }

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        custom_field_mapper={"image_0": "image"}

        # Add small crops (used for Dino)
        if num_small_crops > 0:
            self.decoder_small = ffcv.transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.4))
            image_pipeline_small: List[Operation] = [
                self.decoder2,
                RandomHorizontalFlip(),
                ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
                ffcv.transforms.RandomGrayscale(0.2),
                ffcv.transforms.RandomSolarization(0.2, 128),
                ToTensor(),
                ToDevice(ch.device(this_device), non_blocking=True),
                ToTorchImage(),
                NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
            ]
            for j in range(1,num_small_crops+1):
                pipelines["image_"+str(j)] = image_pipeline_small
                custom_field_mapper["image_"+str(j)] = "image"

        # Create data loader
        loader = ffcv.Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines=pipelines,
                        distributed=distributed,
                        custom_field_mapper=custom_field_mapper)


        return loader

    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          resolution, distributed):
        this_device = f'cuda:{self.gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
            non_blocking=True)
        ]

        order = OrderOption.SEQUENTIAL

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
        return loader

    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    def create_attack_loader(self, val_dataset, num_workers, batch_size,
                          resolution, distributed):
        this_device = f'cuda:{self.gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
            non_blocking=True)
        ]

        index_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
            non_blocking=True)
        ]

        order = OrderOption.SEQUENTIAL

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=0,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline,
                            'index': index_pipeline,
                            'is_good': index_pipeline
                        },
                        distributed=distributed)
        return loader

    @param('training.epochs')
    @param('logging.log_level')
    @param('logging.test_attack')
    @param('training.eval_freq')
    @param('data.use_torchvision')
    def train(self, epochs, log_level, test_attack, eval_freq, use_torchvision):
        # We scale the number of max steps w.t the number of examples in the training set
        self.max_steps = epochs * self.num_train_exemples // (self.batch_size * self.world_size)
        for epoch in range(self.start_epoch, epochs):
            if not use_torchvision:
                res = self.get_resolution(epoch)
                self.res = res
                self.decoder.output_size = (res, res)
                self.decoder2.output_size = (res, res)
            train_loss, stats = self.train_loop(epoch)
            if log_level > 0:
                extra_dict = {
                    'train_loss': train_loss,
                    'epoch': epoch
                }
                self.log(dict(stats,  **extra_dict))
            if epoch % eval_freq == 0:
                # Run attack
                if test_attack == 1:
                    accA, accB = self.attack_loop(epoch)
                    extra_dict["Attack_AccA"] = accA
                    extra_dict["Attack_AccB"] = accB
                # Eval and log
                self.eval_and_log(stats, extra_dict)
            # Empty cache
            ch.cuda.empty_cache()
            # Run checkpointing
            self.checkpoint(epoch+1)
        if self.gpu == 0:
            ch.save(self.model.state_dict(), self.log_folder / 'final_weights.pt')

    def eval_and_log(self, stats, extra_dict={}):
        stats = self.val_loop()
        self.log(dict(stats, **extra_dict))
        return stats

    @param('training.loss')
    def create_model_and_scaler(self, loss):
        scaler = GradScaler()
        model = SSLNetwork()
        if loss == "supervised":
#            model.fc = nn.Linear(model.num_features, self.num_classes)
            model.fc = nn.Sequential(
                        nn.BatchNorm1d(model.num_features), 
                        nn.ReLU(inplace = True), 
                        nn.Linear(model.num_features, self.num_classes)
                        )
        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
        return model, scaler

    @param('training.train_probes_only')
    def load_checkpoint(self, train_probes_only):
        if (self.log_folder / "model.pth").is_file():
            if self.rank == 0:
                print("resuming from checkpoint")
            ckpt = ch.load(self.log_folder / "model.pth", map_location="cpu")
            self.start_epoch = ckpt["epoch"]
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if not train_probes_only:
                self.probes.load_state_dict(ckpt["probes"])
                self.optimizer_probes.load_state_dict(ckpt["optimizer_probes"])
            else:
                self.start_epoch = 0

    @param('logging.checkpoint_freq')
    @param('logging.snapshot_freq')
    @param('training.train_probes_only')
    @param('data.random_seed')
    def checkpoint(self, epoch, checkpoint_freq, snapshot_freq, train_probes_only, random_seed):
        if self.rank != 0 or (epoch % checkpoint_freq != 0 and epoch % snapshot_freq != 0) :
            return
        if train_probes_only:
            state = dict(
                epoch=epoch, 
                probes=self.probes.state_dict(), 
                optimizer_probes=self.optimizer_probes.state_dict()
            )
            save_name = f"probes_"+str(random_seed)+".pth"
        else:
            state = dict(
                epoch=epoch, 
                model=self.model.state_dict(), 
                optimizer=self.optimizer.state_dict(),
                probes=self.probes.state_dict(), 
                optimizer_probes=self.optimizer_probes.state_dict()
            )
            if epoch % snapshot_freq == 0: 
                save_name = f"model_ep{epoch}.pth" 
                ch.save(state, self.log_folder / save_name)
            if epoch % checkpoint_freq == 0: 
                save_name = f"model.pth" 
                ch.save(state, self.log_folder / save_name)

    @param('logging.log_level')
    @param('training.loss')
    @param('training.base_lr')
    @param('training.end_lr_ratio')
    @param('training.num_small_crops')
    @param('data.use_torchvision')
    def train_loop(self, epoch, log_level, loss, base_lr, end_lr_ratio, num_small_crops, use_torchvision):
        """
        Main training loop for SSL training with VicReg criterion.
        """
        model = self.model
        model.train()
        losses = []
        iterator = tqdm(self.train_loader)
        if use_torchvision:
            self.sampler.set_epoch(epoch)

        for ix, loaders in enumerate(iterator, start=epoch * len(self.train_loader)):

            # Get lr
            lr = learning_schedule(
                global_step=ix,
                batch_size=self.batch_size * self.world_size,
                base_lr=base_lr,
                end_lr_ratio=end_lr_ratio,
                total_steps=self.max_steps,
                warmup_steps=10 * self.num_train_exemples // (self.batch_size * self.world_size),
            )
            for g in self.optimizer.param_groups:
                 g["lr"] = lr

            # Get data
            if use_torchvision:
                images_big_0, images_big_1 = loaders[0]
                images_big_0 = images_big_0.to(self.gpu).half()
                images_big_1 = images_big_1.to(self.gpu).half()
                labels_big = loaders[1].to(self.gpu)
                batch_size = loaders[1].size(0)
            else:
                # Get first view
                images_big_0 = loaders[0]
                labels_big = loaders[1]
                batch_size = loaders[1].size(0)
                images_big_1 = loaders[2]
            images_big = ch.cat((images_big_0, images_big_1), dim=0)

            # SSL Training
            if self.do_ssl_training:
                self.optimizer.zero_grad(set_to_none=True)
                with autocast():
                    if self.teacher_student:
                        # Compute the teacher target
                        with ch.no_grad():
                            teacher_output, _ = self.teacher(images_big)
                            teacher_output = teacher_output.view(2, batch_size, -1)
                        if self.loss_name == "byol":
                            embedding_big, _ = model(images_big, predictor=True)
                        elif self.loss_name == "dino":
#                            embedding_big, _ = model(images_big)
#                            embedding_small, _ = model(ch.cat(loaders[3:], dim=0))
                            embedding_big, _ = model(images_big)
                            images_small = ch.cat(loaders[5:5+num_small_crops], dim=0)
                            embedding_small, _ = model(images_small)
                    elif self.supervised_loss:
                        embedding_big, _ = model(images_big_0.repeat(2,1,1,1))
                    else:
                        # Compute embedding in bigger crops
                        embedding_big, _ = model(images_big)
                    
                    # Compute SSL Loss
                    if self.loss_name == "dino":
                        embedding_big = embedding_big.view(2, batch_size, -1)
                        embedding_small = embedding_small.view(num_small_crops, batch_size, -1)
                        student_output = ch.cat((embedding_big, embedding_small), dim=0)
                        loss_train = self.ssl_loss(student_output, teacher_output, epoch)
                    elif self.loss_name == "byol":
                        embedding_big = embedding_big.view(2, batch_size, -1)
                        loss_train = self.ssl_loss(embedding_big, teacher_output)
                    elif self.loss_name == "supervised":
                        output_classif_projector = model.module.fc(embedding_big)
                        loss_train = self.classif_loss(output_classif_projector, labels_big.repeat(2))
                    else:
                        embedding_big = embedding_big.view(2, batch_size, -1)
                        if "simclr" in self.loss_name:
                            loss_num, loss_denum = self.ssl_loss(embedding_big[0], embedding_big[1])
                            loss_train = loss_num + loss_denum
                        else:
                            loss_train = self.ssl_loss(embedding_big[0], embedding_big[1])
                            
                    self.scaler.scale(loss_train).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss_train = ch.tensor(0.)
            if self.teacher_student:
                m = self.momentum_schedule[ix]  # momentum parameter
                for param_q, param_k in zip(model.module.parameters(), self.teacher.module.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # Linear probes training
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_probes.zero_grad(set_to_none=True)
            # Compute embeddings vectors
            with ch.no_grad():
                with autocast():
                    _, list_representation = model(images_big_0)
            # Train probes
            with autocast():
                # Real value classification
                list_outputs = self.probes(list_representation)
                loss_classif = 0.
                for l in range(len(list_outputs)):
                    # Compute classif loss
                    current_loss = self.loss(list_outputs[l], labels_big)
                    loss_classif += current_loss
                    self.train_meters['loss_classif_layer'+str(l)](current_loss.detach())
                    for k in ['top_1_layer'+str(l), 'top_5_layer'+str(l)]:
                        self.train_meters[k](list_outputs[l].detach(), labels_big)
            self.scaler.scale(loss_classif).backward()
            self.scaler.step(self.optimizer_probes)
            self.scaler.update()

            # Logging
            if log_level > 0:
                self.train_meters['loss'](loss_train.detach())
                losses.append(loss_train.detach())
                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.5f}')

                names = ['ep', 'iter', 'shape', 'lrs']
                values = [epoch, ix, tuple(images_big.shape), group_lrs]
                if log_level > 1:
                    names += ['loss']
                    values += [f'{loss_train.item():.3f}']
                    names += ['loss_c']
                    values += [f'{loss_classif.item():.3f}']

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)

        # Return epoch's log
        if log_level > 0:
            self.train_meters['time'](ch.tensor(iterator.format_dict["elapsed"]))
            loss = ch.stack(losses).mean().cpu()
            assert not ch.isnan(loss), 'Loss is NaN!'
            stats = {k: m.compute().item() for k, m in self.train_meters.items()}
            [meter.reset() for meter in self.train_meters.values()]
            return loss.item(), stats

    def val_loop(self):
        model = self.model
        model.eval()
        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.val_loader):
                    _, list_representation = model(images)
                    list_outputs = self.probes(list_representation)
                    loss_classif = 0.
                    for l in range(len(list_outputs)):
                        # Compute classif loss
                        current_loss = self.loss(list_outputs[l], target)
                        loss_classif += current_loss
                        self.val_meters['loss_classif_val_layer'+str(l)](current_loss.detach())
                        for k in ['top_1_val_layer'+str(l), 'top_5_val_layer'+str(l)]:
                            self.val_meters[k](list_outputs[l].detach(), target)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    def get_embeddings(self, dataloader):
        self.model.eval()
        list_emb = []
        list_labels = []
        list_idxs = []
        list_usable_images = []
        # Public set
        for images, label, index, is_good in tqdm(dataloader):
            embeddings, _ = self.model(images)
            list_emb.append(ch.cat(GatherLayer.apply(embeddings.detach()), dim=0).cpu())
            list_labels.append(ch.cat(GatherLayer.apply(label), dim=0).cpu())
            list_idxs.append(ch.cat(GatherLayer.apply(index), dim=0).cpu())
            list_usable_images.append(ch.cat(GatherLayer.apply(is_good), dim=0).cpu())
        list_emb = ch.cat(list_emb, dim=0).float().numpy()
        list_labels = ch.cat(list_labels, dim=0).numpy()
        list_idxs = ch.cat(list_idxs, dim=0).numpy()
        list_usable_images = ch.cat(list_usable_images, dim=0).numpy()
        return list_emb, list_labels, list_idxs, list_usable_images

    @param('attack.k')
    @param('attack.k_attk')
    @param('logging.folder')
    def attack_loop(self, epoch, k, k_attk, folder):
        # 1. We compute embeddings on all the data splits
        with ch.no_grad():
            with autocast():
                list_public_emb, list_public_labels, list_public_idxs, _ = self.get_embeddings(self.public_loader)
                list_bboxA_emb, list_bboxA_labels, list_bboxA_attk_idxs, list_bboxA_usable_images = self.get_embeddings(self.bboxA_loader)
                list_bboxB_emb, list_bboxB_labels, list_bboxB_attk_idxs, list_bboxB_usable_images = self.get_embeddings(self.bboxB_loader)
                usable_A = list_bboxA_usable_images == 1 
                usable_B = list_bboxB_usable_images == 1 
                list_bboxA_emb = list_bboxA_emb[usable_A]
                list_bboxB_emb = list_bboxB_emb[usable_B]
                list_bboxA_attk_idxs = list_bboxA_attk_idxs[usable_A]
                list_bboxB_attk_idxs = list_bboxB_attk_idxs[usable_B]
                list_bboxA_labels = list_bboxA_labels[usable_A]
                list_bboxB_labels = list_bboxB_labels[usable_B]

        # 2. We use faiss to create the public index
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = True
        flat_config.device = self.rank
        # quantizer = faiss.GpuIndexFlatL2(list_public_emb.shape[1])
        # index = faiss.index_cpu_to_gpu(res, 0, quantizer)
        index = faiss.GpuIndexFlatL2(res, list_public_emb.shape[1], flat_config)
        index.train(list_public_emb)
        index.add(list_public_emb)

        #3. We search the NN for bbox A in public set
        D,bboxA_idxs = index.search(list_bboxA_emb, k)
        neighb_bboxA_idxs = list_public_idxs[bboxA_idxs]
        neighb_bboxA_labels = list_public_labels[bboxA_idxs]

        #4. We search the NN for bbox B in public set
        D,bboxB_idxs = index.search(list_bboxB_emb, k)
        neighb_bboxB_idxs = list_public_idxs[bboxB_idxs]
        neighb_bboxB_labels = list_public_labels[bboxB_idxs]

        # Get attack stats 
        pred_bboxA, conf_bboxA = scipy.stats.mode(neighb_bboxA_labels[:, :k_attk],axis=1,keepdims=True)
        pred_bboxA = np.array(pred_bboxA.ravel())
        conf_bboxA = np.array(conf_bboxA.ravel())
        accs_bboxA = pred_bboxA == list_bboxA_labels
        accuracyA = (accs_bboxA).mean() * 100 
        print("Accuracy A: ", accuracyA)
        pred_bboxB, conf_bboxB = scipy.stats.mode(neighb_bboxB_labels[:, :k_attk],axis=1,keepdims=True)
        pred_bboxB = np.array(pred_bboxB.ravel())
        conf_bboxB = np.array(conf_bboxB.ravel())
        accs_bboxB = pred_bboxB == list_bboxB_labels
        accuracyB = (accs_bboxB).mean() * 100 
        print("Accuracy B: ", accuracyB)

        #save data
        attk_dir = Path(folder) / 'attack'
        np.save(attk_dir / Path(f'conf_bboxA_ep{epoch}'), conf_bboxA)
        np.save(attk_dir / Path(f'accs_bboxA_ep{epoch}'), accs_bboxA)
        np.save(attk_dir / Path(f'idxs_bboxA_ep{epoch}'), list_bboxA_attk_idxs)
        np.save(attk_dir / Path(f'neighb_idxs_bboxA_ep{epoch}'), neighb_bboxA_idxs)

        np.save(attk_dir / Path(f'conf_bboxB_ep{epoch}'), conf_bboxB)
        np.save(attk_dir / Path(f'accs_bboxB_ep{epoch}'), accs_bboxB)
        np.save(attk_dir / Path(f'idxs_bboxB_ep{epoch}'), list_bboxB_attk_idxs)
        np.save(attk_dir / Path(f'neighb_idxs_bboxB_ep{epoch}'), neighb_bboxB_idxs)
        

        # Free memory
        index.reset()
        del index, res, list_public_emb
        ch.cuda.synchronize()
        return accuracyA, accuracyB


    @param('logging.folder')
    def initialize_logger(self, folder):
        self.train_meters = {
            'loss': torchmetrics.MeanMetric(compute_on_step=False).to(self.gpu),
            'time': torchmetrics.MeanMetric(compute_on_step=False).to(self.gpu),
        }

        for l in range(self.n_layers_proj):
            self.train_meters['loss_classif_layer'+str(l)] = torchmetrics.MeanMetric(compute_on_step=False).to(self.gpu)
            self.train_meters['top_1_layer'+str(l)] = torchmetrics.Accuracy(compute_on_step=False).to(self.gpu)
            self.train_meters['top_5_layer'+str(l)] = torchmetrics.Accuracy(compute_on_step=False).to(self.gpu)

        self.val_meters = {}
        for l in range(self.n_layers_proj):
            self.val_meters['loss_classif_val_layer'+str(l)] = torchmetrics.MeanMetric(compute_on_step=False).to(self.gpu)
            self.val_meters['top_1_val_layer'+str(l)] = torchmetrics.Accuracy(compute_on_step=False).to(self.gpu)
            self.val_meters['top_5_val_layer'+str(l)] = torchmetrics.Accuracy(compute_on_step=False).to(self.gpu)

        if self.gpu == 0:
            if Path(folder + 'final_weights.pt').is_file():
                self.uid = ""
                folder = Path(folder)
            else:
                folder = Path(folder)
            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)
        self.log_folder = Path(folder)

    @param('training.train_probes_only')
    @param('data.random_seed')
    def log(self, content, train_probes_only, random_seed):
        print(f'=> Log: {content}')
        if self.rank != 0: return
        cur_time = time.time()
        name_file = 'log_probes_'+str(random_seed) if train_probes_only else 'log'
        with open(self.log_folder / name_file, 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()

    @classmethod
    @param('training.distributed')
    @param('dist.world_size')
    @param('dist.port')
    def launch_from_args(cls, distributed, world_size, port):
        if distributed:
            ngpus_per_node = ch.cuda.device_count()
            world_size = int(os.getenv("SLURM_NNODES", "1")) * ngpus_per_node
            if "SLURM_JOB_NODELIST" in os.environ:
                cmd = ["scontrol", "show", "hostnames", os.getenv("SLURM_JOB_NODELIST")]
                host_name = subprocess.check_output(cmd).decode().splitlines()[0]
                dist_url = f"tcp://{host_name}:"+port
            else:
                dist_url = "tcp://localhost:"+port
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=ngpus_per_node, join=True, args=(None, ngpus_per_node, world_size, dist_url))
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        if args[1] is not None:
            set_current_config(args[1])
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param('training.distributed')
    @param('training.eval_only')
    @param('logging.folder')
    def exec(cls, gpu, config, ngpus_per_node, world_size, dist_url, distributed, eval_only, folder):
        trainer = cls(gpu=gpu, ngpus_per_node=ngpus_per_node, world_size=world_size, dist_url=dist_url)
        if eval_only:
            trainer.eval_and_log()
        elif Path(folder + 'final_weights.pt').is_file():
            trainer.attack_loop()
        else:
            trainer.train()

        if distributed:
            trainer.cleanup_distributed()

class Trainer(object):
    def __init__(self, config, num_gpus_per_node, dump_path, dist_url, port):
        self.num_gpus_per_node = num_gpus_per_node
        self.dump_path = dump_path
        self.dist_url = dist_url
        self.config = config
        self.port = port

    def __call__(self):
        self._setup_gpu_args()

    def checkpoint(self):
        self.dist_url = get_init_file().as_uri()
        print("Requeuing ")
        empty_trainer = type(self)(self.config, self.num_gpus_per_node, self.dump_path, self.dist_url, self.port)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        from pathlib import Path
        job_env = submitit.JobEnvironment()
        self.dump_path = Path(str(self.dump_path).replace("%j", str(job_env.job_id)))
        gpu = job_env.local_rank
        world_size = job_env.num_tasks
        if "SLURM_JOB_NODELIST" in os.environ:
            cmd = ["scontrol", "show", "hostnames", os.getenv("SLURM_JOB_NODELIST")]
            host_name = subprocess.check_output(cmd).decode().splitlines()[0]
            dist_url = f"tcp://{host_name}:"+self.port
        else:
            dist_url = "tcp://localhost:"+self.port
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        ImageNetTrainer._exec_wrapper(gpu, config, self.num_gpus_per_node, world_size, dist_url)

# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast SSL training')
    parser.add_argument("folder", type=str)
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()
    return config

@param('logging.folder')
@param('dist.ngpus')
@param('dist.nodes')
@param('dist.timeout')
@param('dist.partition')
@param('dist.comment')
@param('dist.use_volta32')
@param('dist.port')
def run_submitit(config, folder, ngpus, nodes,  timeout, partition, comment, use_volta32, port):
    Path(folder).mkdir(parents=True, exist_ok=True)
    #create folder for NN attack data 
    (Path(folder) / 'attack').mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=folder, slurm_max_num_timeout=30)

    num_gpus_per_node = ngpus
    nodes = nodes
    timeout_min = timeout

    kwargs = {}
    if use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if comment:
        kwargs['slurm_comment'] = comment

    executor.update_parameters(
        mem_gb=60 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node, 
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="ffcv2")

    dist_url = get_init_file().as_uri()

    trainer = Trainer(config, num_gpus_per_node, folder, dist_url, port)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {folder}")

@param('dist.use_submitit')
def main(config, use_submitit):
    if use_submitit:
        run_submitit(config)
    else:
        ImageNetTrainer.launch_from_args()

if __name__ == "__main__":
    config = make_config()
    main(config)
