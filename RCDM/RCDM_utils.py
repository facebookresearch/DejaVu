# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch 
from torchvision import transforms
from torch import nn, optim
from torchvision.datasets import ImageFolder
from torchvision import datasets
from torchvision.models import resnet50, resnet101

class SSLNetwork(nn.Module):
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
        self.representation_size = self.net(torch.zeros((1,3,224,224))).size(1)
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

    def MLP(self, size, proj_relu=0, mlp_coeff=1):
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

#functions for making embeddings given distributed model (could have been saved in a smarter way...) 
def embed_bb(model, tensor):
    return model.module.net(tensor)

def embed_proj(model, tensor):
    return model.module.projector(tensor) 

def embed_activ(model, tensor):
    return model.module.fc(tensor)

class Transform: 
    def __init__(self, rcdm_image_size):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.ssl_xfrm = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                    ])

        self.rcdm_xfrm = transforms.Compose([
                    transforms.Resize(int(256 * rcdm_image_size / 224)),
                    transforms.CenterCrop(rcdm_image_size),
                    transforms.ToTensor()
                    ])
        
    def __call__(self, x): 
        return self.ssl_xfrm(x), self.rcdm_xfrm(x)


def get_ssl_model_from_ckpt(pth, gpu, loss = 'vicreg', arch = 'resnet50', 
        mlp = '8192-8192-8192', use_supervised_activations = False):
    ssl_model = SSLNetwork(arch = arch, 
                          remove_head = 0, 
                          mlp = mlp, 
                          fc = 0,
                          patch_keep = 1.0,
                          loss = loss).cuda()
    use_lin = (loss == 'supervised') and use_supervised_activations 
    if use_lin: 
        ssl_model.fc = nn.Linear(ssl_model.num_features, 1000).cuda()
    ssl_model = torch.nn.parallel.DistributedDataParallel(ssl_model, device_ids=[gpu])
    ckpt = torch.load(pth, map_location='cpu')
    msg = ssl_model.load_state_dict(ckpt['model'], strict = True)
    print("Loading SSL model:", msg)
    return ssl_model
