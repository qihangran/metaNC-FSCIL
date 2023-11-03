#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================
import numpy as np
import sys, os
import torch as t
import torch.nn as nn
import torch.nn.functional as F
#from dotmap import DotMap
from .embeddings.ResNet12 import ResNet12
from .embeddings.model_factory import Model

from .embeddings.ResNet20 import ResNet20
from lib.torch_blocks import fixCos, softstep, step, softabs, softrelu, cosine_similarity_multi, scaledexp
t.manual_seed(0) #for reproducability
import math
import hashlib
from tqdm import tqdm
import shutil
import tempfile
#import warnings
#import re
#import errno
from urllib.request import urlopen
from urllib.parse import urlparse  # noqa: F401
#import pdb
# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------
class MLPFFNNeck(nn.Module):
    def __init__(self, in_channels=512, out_channels=512):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.ln1 = nn.Sequential(nn.Linear(in_channels, in_channels * 2),
                                 nn.BatchNorm1d(in_channels * 2),
                                  nn.ReLU())

        self.ln2 = nn.Sequential(nn.Linear(in_channels*2, in_channels * 2),
                                  nn.BatchNorm1d(in_channels * 2),
                                  nn.ReLU())

        self.ln3 = nn.Linear(in_channels*2, in_channels*2)


    def init_weights(self):
        pass

    def forward(self, inputs):#input is embedding.forward_conv
        x = self.ln1(inputs)
        x = self.ln2(x)
        x = self.ln3(x)
        return x


class KeyValueNetwork(nn.Module):

    # ----------------------------------------------------------------------------------------------
    # Special Functions & Properties
    # ----------------------------------------------------------------------------------------------

    def __init__(self, args, mode="meta", pretrained=False, progress=True):
        super().__init__()

        self.args = args
        self.mode = mode

        # Modules
        if args.block_architecture == "mini_resnet12":
            self.embedding = ResNet12(args)
        elif args.block_architecture == "mini_resnet18": 
            # self.embedding = resnet18(num_classes=args.dim_features, pretrained=pretrained)
            self.embedding = Model(args, pretrained=pretrained)
        elif args.block_architecture == "mini_resnet20": 
            self.embedding = ResNet20(num_classes=args.dim_features)


        if args.dataset == 'cifar100' or args.dataset == 'mini_imagenet':
            #self.fc_pretrain = nn.Linear(args.dim_features, args.base_class, bias=False)
            self.fc_pretrain = nn.Linear(args.dim_features, args.base_class + args.base_class * (args.base_class - 1) // 2, bias=False)
        else:
            self.fc_pretrain = nn.Linear(args.dim_features, args.base_class ,bias=False)

        # # Activations
        # activation_functions = {
        #     'softabs':  (lambda x: softabs(x, steepness=args.sharpening_strength, K=args.num_classes)),
        #     'softrelu': (lambda x: softrelu(x, steepness=args.sharpening_strength)),
        #     'relu':     nn.ReLU,
        #     'abs':      t.abs,
        #     'scaledexp': (lambda x: scaledexp(x, s = args.sharpening_strength)),
        #     'exp':      t.exp
        # }
        # approximations = {
        #     'softabs':  'abs',
        #     'softrelu': 'relu'
        # }
        #
        # self.sharpening_activation = activation_functions[args.sharpening_activation]

        # Access to intermediate activations
        self.intermediate_results = dict()
        
        self.feat_replay = t.zeros((args.num_classes, self.embedding.n_interm_feat)).cuda(args.gpu)
        #self.feat_replay = t.zeros((args.num_classes, 1280)).cuda(args.gpu)
        self.label_feat_replay = t.diag(t.ones(self.args.num_classes)).cuda(args.gpu)

    # ----------------------------------------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------------------------------------

    def forward(self, inputs, session, nways = 5):
        '''
        Forward pass of main model

        Parameters:
        -----------
        inputs:  Tensor (B,H,W)
            Input data
        Return: 
        -------
        output:  Tensor (B,ways)
        '''
        # Embed batch
        query_vectors = self.embedding(inputs)

        if self.mode =="pretrain":
            output = self.fc_pretrain(query_vectors)
        else:
            output = F.linear(F.normalize(query_vectors, p=2, dim=1), F.normalize(self.key_mem, p=2, dim=1))

        return output

    def write_mem(self,x,y):
        '''
        Rewrite key and value memory

        Parameters:
        -----------
        x:  Tensor (B,D)
            Input data
        y:  Tensor (B,w)
            One-hot encoded classes
        ''' 
        self.key_mem = self.embedding(x)
        # self.key_mem = self.neck(self.embedding.forward_conv(x))
        self.val_mem = y

        if self.args.average_support_vector_inference:
            self.key_mem = t.matmul(t.transpose(self.val_mem,0,1), self.key_mem)
        return


    def reset_prototypes(self,args):
        if hasattr(self,'key_mem'):
            self.key_mem.data.fill_(0.0)
        else:
            self.key_mem = nn.parameter.Parameter(t.zeros(self.args.num_classes, self.args.dim_features),requires_grad=False).cuda(args.gpu)
            self.val_mem = nn.parameter.Parameter(t.diag(t.ones(self.args.num_classes)),requires_grad=False).cuda(args.gpu)
            # self.key_mem = nn.parameter.Parameter(t.zeros(self.args.num_classes, 512),
            #                                       requires_grad=False).cuda(args.gpu)
            # self.val_mem = nn.parameter.Parameter(t.diag(t.ones(self.args.num_classes)), requires_grad=False).cuda(
            #     args.gpu)



    def update_prototypes(self,x,y): 
        '''
        Update key memory  

        Parameters:
        -----------
        x:  Tensor (B,D)
            Input data
        y:  Tensor (B)
            lables 
        '''
        support_vec = self.embedding(x)
        #support_vec = self.embedding.forward_conv(x)
        y_onehot = F.one_hot(y, num_classes = self.args.num_classes).float()
        prototype_vec = t.matmul(t.transpose(y_onehot,0,1), support_vec)
        self.key_mem.data += prototype_vec

    # def bipolarize_prototypes(self):
    #     '''
    #     Bipolarize key memory
    #     '''
    #     self.key_mem.data = t.sign(self.key_mem.data)

    def ETF(self, args):
        '''
        Create ETF form  bipolarize_prototypes
        '''
        self.key_mem.data = t.sign(self.key_mem.data).permute(1, 0)
        num_classes = self.key_mem.data.shape[1]

        i_nc_nc = t.eye(num_classes).cuda(args.gpu)
        one_nc_nc: t.Tensor = t.mul(t.ones(num_classes, num_classes), (1 / num_classes)).cuda(args.gpu)
        self.key_mem.data = t.mul(t.matmul(self.key_mem.data[:, :num_classes], i_nc_nc - one_nc_nc),
                            math.sqrt(num_classes / (num_classes - 1))).permute(1, 0)


    def get_sum_support(self,x,y):
        '''
        Compute prototypes
        
        Parameters:
        -----------
        x:  Tensor (B,D)
            Input data
        y:  Tensor (B)
            lables 
        '''
        support_vec = self.embedding(x)
        y_onehot = F.one_hot(y, num_classes = self.args.num_classes).float()
        sum_cnt = t.sum(y_onehot,dim=0).unsqueeze(1)
        sum_support = t.matmul(t.transpose(y_onehot,0,1), support_vec)
        return sum_support, sum_cnt


    def update_feat_replay(self,x,y): 
        '''
        Compute feature representatin of new data and update
        Parameters:
        -----------
        x   t.Tensor(B,in_shape)
            Input raw images
        y   t.Tensor (B)
            Input labels

        Return: 
        -------
        '''
        feat_vec = self.embedding.forward_conv(x)
        y_onehot = F.one_hot(y, num_classes = self.args.num_classes).float()
        sum_cnt = t.sum(y_onehot, dim=0).unsqueeze(1)
        sum_feat_vec = t.matmul(t.transpose(y_onehot, 0, 1), feat_vec)
        # avg_feat_vec = t.div(sum_feat_vec, sum_cnt+1e-8)
        # self.feat_replay += avg_feat_vec
        #feat_replay += sum_feat_vec
        return sum_cnt, sum_feat_vec

    def update_prototype_replay(self,x,y):
        '''
        Compute feature representatin of new data and update
        Parameters:
        -----------
        x   t.Tensor(B,in_shape)
            Input raw images
        y   t.Tensor (B)
            Input labels

        Return:
        -------
        '''
        feat_vec = self.embedding(x)
        y_onehot = F.one_hot(y, num_classes = self.args.num_classes).float()
        sum_cnt = t.sum(y_onehot, dim=0).unsqueeze(1)
        sum_feat_vec = t.matmul(t.transpose(y_onehot,0,1), feat_vec)
        return sum_cnt, sum_feat_vec


    def get_feat_replay(self): 
        return self.feat_replay, self.label_feat_replay

    def update_prototypes_feat(self,feat,y_onehot, session, nways=None,):
        '''
        Update key 

        Parameters:
        -----------
        feat:  Tensor (B,d_f)
            Input data
        y:  Tensor (B)
        nways: int
            If none: update all prototypes, if int, update only nwyas prototypes
        ''' 
        support_vec = self.get_support_feat(feat, session)

        prototype_vec = t.matmul(t.transpose(y_onehot,0,1), support_vec)

        if nways is not None:
            self.key_mem.data[:nways] += prototype_vec[:nways]
        else:
            self.key_mem.data += prototype_vec


    def get_support_feat(self,feat,session):
        '''
        Pass activations through final FC 

        Parameters:
        -----------
        feat:  Tensor (B,d_f)
            Input data
        Return:
        ------
        support_vec:  Tensor (B,d)
            Mapped support vectors
        '''
        support_vec = self.embedding.fc(feat)
        return support_vec


def _download_url_to_file(url, dst, hash_prefix, progress):
    file_size = None
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overriden by a broken download.
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)