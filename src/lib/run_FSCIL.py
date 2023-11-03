#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================
import csv
import datetime
import time
from copy import copy
from operator import itemgetter

import shutil
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

import numpy as np
from dotmap import DotMap

import progressbar
from tqdm import tqdm

#from .model import KeyValueNetwork

from lib.validation import NCMValidation
from lib.criterion import AngularPenaltySMLoss
from lib.model import *
# from lib.util import csv2dict, loadmat
from lib.torch_blocks import myCosineLoss
from plot.confusion_support import plot_confusion_support, avg_sim_confusion
import os.path
import pdb
from lib.dataloader.FSCIL.data_utils import *

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def pretrain_baseFSCIL(verbose,**parameters):
    '''
    Pre-training on base session
    '''
    args = DotMap(parameters)
    args.gpu = 0 
    
    # Initialize the dataset generator and the model
    args = set_up_datasets(args)
    trainset, train_loader, val_loader = get_base_dataloader(args)

    #cub200 resnet18 miniImageNet and cifar100 resnet12
    if args.dataset == 'cub200':
        model = KeyValueNetwork(args, pretrained=True)
    else:
        model = KeyValueNetwork(args, pretrained=False)

    model.mode = 'pretrain'  
    # Store all parameters in a variable
    parameters_list, parameters_table = process_dictionary(parameters)

    # Print all parameters
    if verbose:
        print("Parameters:")
        for key, value in parameters_list:
            print("\t{}".format(key).ljust(40) + "{}".format(value))

    writer = SummaryWriter(args.log_dir)

    # Take start time
    start_time = time.time()

    criterion = nn.CrossEntropyLoss()

    if args.gpu is not None: 
        t.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)

    optimizer = t.optim.SGD(model.parameters(),lr=args.learning_rate,nesterov=args.SGDnesterov, 
                                weight_decay=args.SGDweight_decay, momentum=args.SGDmomentum) 
    scheduler = t.optim.lr_scheduler.StepLR(optimizer,step_size=args.lr_step_size,gamma=0.1)

    start_train_iter=0
    best_acc1 = 0

    # for epoch in tqdm.tqdm(range(1,args.max_train_iter),desc='Epoch'):
    for epoch in tqdm(range(1, args.max_train_iter), desc='Epoch'):
        global_count = 0

        losses = AverageMeter('Loss')
        acc = AverageMeter('Acc@1')
        
        model.train(True)
        iter_len = len(train_loader)
        for i, batch in enumerate(train_loader):
            global_count = global_count + 1
            #---------------------------two_images---------------------------------------------
            data0 = batch[0][0].cuda(args.gpu)
            data1 = batch[0][1].cuda(args.gpu)
            train_label = batch[1].cuda(args.gpu)
            batch_size = data0.shape[0]
            # if args.data_fusion:
            if args.dataset == 'cifar100' or args.dataset == 'mini_imagenet':
                data0, data1, train_label = fusion_aug_two_image(data0, data1, train_label, 0, args)

            # forward pass
            optimizer.zero_grad()
            output1 = model(data0, session=0)
            output2 = model(data1, session=0)
            loss1 = criterion(output1, train_label)
            loss2 = criterion(output2, train_label)
            loss = 0.5 * loss1 + 0.5 * loss2

            # Backpropagation
            loss.backward()
            optimizer.step()

            accuracy1 = top1accuracy(output1[:batch_size, :].argmax(dim=1), train_label[:batch_size])
            accuracy2 = top1accuracy(output2[:batch_size, :].argmax(dim=1), train_label[:batch_size])
            accuracy = 0.5 * accuracy1 + 0.5 * accuracy2
            losses.update(loss.item(), data0.size(0))
            acc.update(accuracy.item(), data0.size(0))

            # -------------------------one images end------------------------------------------
            # images = batch[0].cuda(args.gpu)
            # train_label = batch[1].cuda(args.gpu)
            # batch_size = images.shape[0]
            # # forward pass
            # optimizer.zero_grad()
            # output1 = model(images, session=0)
            # loss = criterion(output1, train_label)
            # # Backpropagation
            # loss.backward()
            # optimizer.step()
            # accuracy = top1accuracy(output1.argmax(dim=1), train_label)
            # losses.update(loss.item(), images.size(0))
            # acc.update(accuracy.item(), images.size(0))

            if not i % 10:
                print("Training... | epoch: {} | iter: {}/{} | loss: {} | acc: {}".format(epoch, i, iter_len, loss.item(), accuracy.item()))
            
        scheduler.step()

        # write to tensorboard
        writer.add_scalar('training_loss/pretrain_CEL', losses.avg, epoch)
        writer.add_scalar('accuracy/pretrain_train', acc.avg, epoch)

        val_loss, val_acc_mean,_ = validation(model,criterion,val_loader, args)
        writer.add_scalar('validation_loss/pretrain_CEL', val_loss,epoch)
        writer.add_scalar('accuracy/pretrain_val', val_acc_mean,epoch)

        print("Training... | epoch: {}  | aargmax_acc: {}".format(epoch, val_acc_mean))

        is_best = val_acc_mean > best_acc1
        best_acc1 = max(val_acc_mean, best_acc1)

        save_checkpoint({
            'train_iter': epoch + 1,
            'arch': args.block_architecture,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,savedir=args.log_dir) 

    writer.close()

def metatrain_baseFSCIL(verbose,**parameters):
    '''
    Meta-training on base session
    ''' 

    # Argument Preparation
    args = DotMap(parameters)
    args.gpu = 0 
    
    # Initialize the dataset generator and the model
    args = set_up_datasets(args)
    trainset, train_loader, val_loader = get_base_dataloader_meta(args)
    model = KeyValueNetwork(args)

    model.mode = 'meta'
    # Store all parameters in a variable
    parameters_list, parameters_table = process_dictionary(parameters)

    # Print all parameters
    if verbose:
        print("Parameters:")
        for key, value in parameters_list:
            #if key == 'lam_dis' or key == 'lam_fea':
            print("\t{}".format(key).ljust(40) + "{}".format(value))

    writer = SummaryWriter(args.log_dir)

    # Take start time
    start_time = time.time()
    criterion = nn.BCELoss()



    if args.gpu is not None: 
        t.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)


    # set all parameters except FC to trainable false
    for param in model.parameters():
        param.requires_grad = False
    for param in model.embedding.fc.parameters():
        param.requires_grad = True

    optimizer = t.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.learning_rate, nesterov=args.SGDnesterov,
                            weight_decay=args.SGDweight_decay, momentum=args.SGDmomentum)

    scheduler = t.optim.lr_scheduler.StepLR(optimizer,step_size=args.lr_step_size,gamma=0.1)

    model,optimizer,scheduler,start_train_iter,best_acc1 = load_checkpoint(model,optimizer,scheduler,args)
    best_acc1 = 0
    k = args.num_ways_training*args.num_shots_training

    # losses = AverageMeter('Loss')
    # acc = AverageMeter('Acc@1')
    train_iterator = iter(train_loader) 
    # for i in tqdm.tqdm(range(start_train_iter,args.max_train_iter), initial=start_train_iter, total=args.max_train_iter,desc='Epoch'):
    for i in tqdm(range(start_train_iter, args.max_train_iter), initial=start_train_iter,
                       total=args.max_train_iter, desc='Epoch'):
        batch = next(train_iterator)
        data, train_label = [_.cuda(args.gpu,non_blocking=True) for _ in batch]
        train_label_onehot = F.one_hot(train_label, num_classes = args.num_classes).float()
        proto, query = data[:k], data[k:]
        proto_label, query_label = train_label_onehot[:k], train_label_onehot[k:]
        model.eval()
        with t.no_grad():
            model.write_mem(proto, proto_label)
        
         # forward pass
        model.train()
        optimizer.zero_grad()
        output = model(query, session=0)

        #################losss##
        K = args.num_classes
        steepness = 10.
        s = 10.
        query_label_int = train_label[k:]
        excl = t.cat([t.cat((output[j, :y], output[j, y + 1:])).unsqueeze(0) for j, y in enumerate(query_label_int)], dim=0)
        phi = t.sigmoid(steepness * ((excl + 1. / (K - 1)) - 0.5)) + t.sigmoid(steepness * (-(excl + 1. / (K - 1)) - 0.5))
        numerator = s * t.diagonal(output.transpose(0, 1)[query_label_int])
        denominator = t.exp(numerator) + t.sum(t.exp(s * phi), dim=1)
        L = numerator - t.log(denominator)
        loss = -t.mean(L)
        ########################

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Do evaluation           
        predicted_labels, predicted_certainties, actual_labels, actual_certainties, accuracy = process_result(
                output,query_label)

        scheduler.step()

        if not i % args.summary_frequency_very_often:
            # write to tensorboard
            writer.add_scalar('training_loss/log_loss',loss.item(),i)
            writer.add_scalar('accuracy/training',accuracy*100,i)


        if not i % args.validation_frequency: 
            val_loss, val_acc_mean = validation_onehot(model,criterion,val_loader,args,
                                                    num_classes = args.num_classes)
            writer.add_scalar('validation_loss/log_loss', val_loss,i)
            writer.add_scalar('accuracy/validation', val_acc_mean,i)

            is_best = val_acc_mean > best_acc1
            best_acc1 = max(val_acc_mean, best_acc1)

            save_checkpoint({
                'train_iter': i + 1,
                'arch': args.block_architecture,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best,savedir=args.log_dir) 

    writer.close()

def train_FSCIL(verbose=False, **parameters):
    '''
    Main FSCIL evaluation on all sessions
    ''' 
    args = DotMap(parameters) 
    args = set_up_datasets(args)
    args.gpu = 0
    model = KeyValueNetwork(args,mode='meta')

    # Store all parameters in a variable
    parameters_list, parameters_table = process_dictionary(parameters)

    # Print all parameters
    if verbose:
        print("Parameters:")
        for key, value in parameters_list:
            if key == 'lam_fea' or key == 'lam_dis':
                print("\t{}".format(key).ljust(40) + "{}".format(value))

    # Write parameters to file
    if not args.inference_only:
        filename = args.log_dir + '/parameters.csv'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        #retrain
        with open(filename, 'w') as csv_file:
            writer = csv.writer(csv_file)
            keys, values = zip(*parameters_list)
            writer.writerow(keys)
            writer.writerow(values)

    writer = SummaryWriter(args.log_dir)

    # Take start time
    start_time = time.time()

    criterion = nn.CrossEntropyLoss()

    if args.gpu is not None: 
        t.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)

    # set all parameters except FC to trainable false
    for param in model.parameters():
        param.requires_grad = False
    for param in model.embedding.fc.parameters():
        param.requires_grad = True


    optimizer = t.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.learning_rate,nesterov=args.SGDnesterov, 
                            weight_decay=args.SGDweight_decay, momentum=args.SGDmomentum)
    
    scheduler = t.optim.lr_scheduler.StepLR(optimizer,step_size=args.lr_step_size,gamma=0.1)

    model,optimizer,scheduler, start_train_iter,best_acc1 = load_checkpoint(model,optimizer,scheduler,args)

    total_acc = 0.
    for session in range(args.sessions):
        nways_session = args.base_class + session*args.way
        
        train_set, train_loader, test_loader = get_dataloader(args, session)
        # update model
        batch = next(iter(train_loader))

        align(model,batch, optimizer,args,writer,session,nways_session)
        
        loss, acc, conf_fig = validation(model,criterion,test_loader,args,nways_session,session)
        print("Session {:}: {:.2f}%".format(session+1, acc))

        writer.add_scalar('accuracy/cont', acc, session)
        total_acc += acc

    mean_acc = total_acc / args.sessions
    print("mean_acc {:.2f}%".format(mean_acc))
    writer.close()

def align(model,data,optimizer,args,writer,session,nways_session):
    
    '''
    Alignment of FC using MSE Loss and feature replay
    '''
    losses = AverageMeter('Loss')
    criterion = myCosineLoss(args.retrain_act)
    dataset = myRetrainDataset(data[0], data[1])
    dataloader = DataLoader(dataset=dataset, batch_size = args.batch_size_training)

    # Stage 1: Compute feature representation of new data
    model.eval()
    with t.no_grad():
        sum_cnt = t.zeros(args.num_classes, 1).cuda(args.gpu).float()
        sum_feat_vec = t.zeros(args.num_classes, model.embedding.n_interm_feat).cuda(args.gpu).float()
        for x,target in dataloader:
            x = x.cuda(args.gpu,non_blocking=True)
            target = target.cuda(args.gpu,non_blocking=True)
            cnt, feat_vec = model.update_feat_replay(x, target)
            sum_cnt += cnt
            sum_feat_vec += feat_vec
        avg_feat_vec = t.div(sum_feat_vec, sum_cnt+1e-8)
        model.feat_replay += avg_feat_vec

    # Stage 2: Compute prototype based on GAAM
    feat, label = model.get_feat_replay()
    model.reset_prototypes(args)
    model.update_prototypes_feat(feat,label, nways = nways_session, session=session)

def validation(model,criterion,dataloader, args, nways_session=None, session=0):
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')

    sim_conf = avg_sim_confusion(args.num_classes, nways_session)
    model.eval()
    with t.no_grad(): 
        for i, batch in enumerate(dataloader):
            data, label = [_.cuda(args.gpu,non_blocking=True) for _ in batch]

            output = model(data, session)
            loss = criterion(output,label)
            accuracy = top1accuracy(output.argmax(dim=1),label)
            losses.update(loss.item(),data.size(0))
            acc.update(accuracy.item(),data.size(0))
            # if nways_session is not None:
            #     sim_conf.update(model.similarities.detach().cpu(),
            #                 F.one_hot(label.detach().cpu(), num_classes = args.num_classes).float())
    # Plot figure if needed
    fig = sim_conf.plot() if nways_session is (not None) else None
    return losses.avg, acc.avg, fig

def validation_onehot(model,criterion,dataloader, args, num_classes):
    #  

    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')

    model.eval()

    with t.no_grad(): 
        for i, batch in enumerate(dataloader):
            data, label = [_.cuda(args.gpu,non_blocking=True) for _ in batch]
            label = F.one_hot(label, num_classes = num_classes).float()

            output = model(data, session=0)
            #loss = criterion(output,label)
            
            _, _, _, _, accuracy = process_result(
                output,label)

            #losses.update(loss.item(),data.size(0))
            acc.update(accuracy.item()*100, data.size(0))
    
    return losses.avg, acc.avg

# --------------------------------------------------------------------------------------------------
# Interpretation
# --------------------------------------------------------------------------------------------------
def process_result(predictions, actual):
    predicted_labels = t.argmax(predictions, dim=1)
    actual_labels = t.argmax(actual, dim=1)

    accuracy = predicted_labels.eq(actual_labels).float().mean(0,keepdim=True)
    # TBD implement those uncertainties
    predicted_certainties =0#
    actual_certainties = 0 #
    return predicted_labels, predicted_certainties, actual_labels, actual_certainties, accuracy


def process_dictionary(dict):
    # Convert the dictionary to a sorted list
    dict_list = sorted(list(dict.items()))

    # Convert the dictionary into a table
    keys, values = zip(*dict_list)
    values = [repr(value) for value in values]
    dict_table = np.vstack((np.array(keys), np.array(values))).T

    return dict_list, dict_table

# --------------------------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------------------------
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar',savedir=''):
    t.save(state, savedir+'/'+filename)
    if is_best:
        shutil.copyfile(savedir+'/'+filename, savedir+'/'+'model_best.pth.tar')



def load_checkpoint(model,optimizer,scheduler,args):        

    # First priority: load checkpoint from log_dir 
    if os.path.isfile(args.log_dir+ '/checkpoint.pth.tar'):
        resume = args.log_dir+ '/checkpoint.pth.tar'
        print("=> loading checkpoint '{}'".format(resume))
        if args.gpu is None:
            checkpoint = t.load(resume)
        else:
            # Map model to be loaded to specified single args.gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = t.load(resume, map_location=loc)
        start_train_iter = int(checkpoint['train_iter']) 
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_acc1 = checkpoint['best_acc1'] 
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (train_iter {})"
              .format(args.log_dir, checkpoint['train_iter']))
    # Second priority: load from pretrained model
    # No scheduler and no optimizer loading here.  
    elif os.path.isfile(args.resume+'/checkpoint.pth.tar'):
        resume = args.resume+'/checkpoint.pth.tar'
        print("=> loading pretrain checkpoint '{}'".format(resume))
        if args.gpu is None:
            checkpoint = t.load(resume)
        else:
            # Map model to be loaded to specified single args.gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = t.load(resume, map_location=loc)
        start_train_iter = 0 
        best_acc1 = 0
        model.load_state_dict(checkpoint['state_dict'])
        #model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
        print("=> loaded pretrained checkpoint '{}' (train_iter {})"
              .format(args.log_dir, checkpoint['train_iter']))
    else:
        start_train_iter=0
        best_acc1 = 0
        print("=> no checkpoint found at '{}'".format(args.log_dir))
        print("=> no pretrain checkpoint found at '{}'".format(args.resume))
    
    return model, optimizer, scheduler, start_train_iter, best_acc1


# --------------------------------------------------------------------------------------------------
# Some Pytorch helper functions (might be removed from this file at some point)
# --------------------------------------------------------------------------------------------------

def convert_toonehot(label): 
    '''
    Converts index to one-hot. Removes rows with only zeros, such that 
    the tensor has shape (B,num_ways)
    '''
    label_onehot = F.one_hot(label)
    label_onehot = label_onehot[:,label_onehot.sum(dim=0)!=0]
    return label_onehot.type(t.FloatTensor)

def top1accuracy(pred, target):
    """Computes the precision@1"""
    batch_size = target.size(0)

    correct = pred.eq(target).float().sum(0)
    return correct.mul_(100.0 / batch_size)


class myRetrainDataset(Dataset):
    def __init__(self, x,y):
        self.x = x
        self.y = y
       
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def fusion_aug_generate_label(y_a, y_b, session, args):
    current_total_cls_num = args.base_class + session * args.way
    if session == 0:  # base session -> increasing: [(args.base_class) * (args.base_class - 1)]/2
        y_a, y_b = y_a, y_b
        assert y_a != y_b
        if y_a > y_b:  # make label y_a smaller than y_b
            tmp = y_a
            y_a = y_b
            y_b = tmp
        label_index = ((2 * current_total_cls_num - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1
    else:  # incremental session -> increasing: [(args.way) * (args.way - 1)]/2
        y_a = y_a - (current_total_cls_num - args.way)
        y_b = y_b - (current_total_cls_num - args.way)
        assert y_a != y_b
        if y_a > y_b:  # make label y_a smaller than y_b
            tmp = y_a
            y_a = y_b
            y_b = tmp
        label_index = int(((2 * args.way - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1)
    return label_index + current_total_cls_num


def fusion_aug_two_image(x_1, x_2, y, session, args, alpha=20.0, mix_times=4):  # mixup based
    batch_size = x_1.size()[0]
    mix_data_1 = []
    mix_data_2 = []
    mix_target = []

    # print('fusion_aug_two_image | before fusion | length of the data: {0}, size of image: {1}'.format(len(y), x_1.size()))
    for _ in range(mix_times):
        index = t.randperm(batch_size).cuda()
        for i in range(batch_size):
            if y[i] != y[index][i]:
                new_label = fusion_aug_generate_label(y[i].item(), y[index][i].item(), session, args)
                lam = np.random.beta(alpha, alpha)
                if lam < 0.4 or lam > 0.6:
                    lam = 0.5
                mix_data_1.append(lam * x_1[i] + (1 - lam) * x_1[index, :][i])
                mix_data_2.append(lam * x_2[i] + (1 - lam) * x_2[index, :][i])
                mix_target.append(new_label)

    new_target = t.Tensor(mix_target)
    y = t.cat((y, new_target.cuda().long()), 0)
    for item in mix_data_1:
        x_1 = t.cat((x_1, item.unsqueeze(0)), 0)
    for item in mix_data_2:
        x_2 = t.cat((x_2, item.unsqueeze(0)), 0)
    # print('fusion_aug_two_image | after fusion | length of the data: {0}, size of image: {1}'.format(len(y), x_1.size()))

    return x_1, x_2, y


def fusion_aug_one_image(x, y, session, args, alpha=20.0, mix_times=4):  # mixup based
    batch_size = x.size()[0]
    mix_data = []
    mix_target = []

    # print('fusion_aug_one_image | before fusion | length of the data: {0}, size of image: {1}'.format(len(y), x.size()))
    for _ in range(mix_times):
        index = t.randperm(batch_size).cuda()
        for i in range(batch_size):
            if y[i] != y[index][i]:
                new_label = fusion_aug_generate_label(y[i].item(), y[index][i].item(), session, args)
                lam = np.random.beta(alpha, alpha)
                if lam < 0.4 or lam > 0.6:
                    lam = 0.5
                mix_data.append(lam * x[i] + (1 - lam) * x[index, :][i])
                mix_target.append(new_label)

    new_target = t.Tensor(mix_target)
    y = t.cat((y, new_target.cuda().long()), 0)
    for item in mix_data:
        x = t.cat((x, item.unsqueeze(0)), 0)
    # print('fusion_aug_one_image | after fusion | length of the data: {0}, size of image: {1}'.format(len(y), x.size()))

    return x, y

def fusion_aug_generate_label(y_a, y_b, session, args):
    current_total_cls_num = args.base_class + session * args.way
    if session == 0:  # base session -> increasing: [(args.base_class) * (args.base_class - 1)]/2
        y_a, y_b = y_a, y_b
        assert y_a != y_b
        if y_a > y_b:  # make label y_a smaller than y_b
            tmp = y_a
            y_a = y_b
            y_b = tmp
        label_index = ((2 * current_total_cls_num - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1
    else:  # incremental session -> increasing: [(args.way) * (args.way - 1)]/2
        y_a = y_a - (current_total_cls_num - args.way)
        y_b = y_b - (current_total_cls_num - args.way)
        assert y_a != y_b
        if y_a > y_b:  # make label y_a smaller than y_b
            tmp = y_a
            y_a = y_b
            y_b = tmp
        label_index = int(((2 * args.way - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1)
    return label_index + current_total_cls_num