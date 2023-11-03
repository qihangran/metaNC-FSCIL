import torch
from torch import nn
import torch.nn.functional as F


class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''

        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface', 'cross_entropy', 'nc-softabs']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.eps = eps

        self.cross_entropy = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()

    def forward(self, wf, labels):
        if self.loss_type == 'cross_entropy':
            return self.cross_entropy(wf, labels)
        elif self.loss_type == 'nc-softabs':
            # a new bce loss

            K = 200
            train_label_onehot = F.one_hot(labels, num_classes=100).float()
            steepness = 10.
            s = 10.

            excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
            phi = torch.sigmoid(steepness * ((excl + 1. / (K - 1)) - 0.5)) + torch.sigmoid(steepness * (-(excl + 1. / (K - 1)) - 0.5))

            # numerator = torch.sigmoid(s * torch.diagonal(wf.transpose(0, 1)[labels]))
            # temp1 = torch.log(torch.clamp(1. - phi, 0., 1.))
            # l2 = torch.mean(temp1, dim=1)
            # loss = -torch.log(numerator) - l2
            # return loss.mean()

            numerator = s * torch.diagonal(wf.transpose(0, 1)[labels])
            denominator = torch.exp(numerator) + torch.sum(torch.exp(s * phi), dim=1)
            L = numerator - torch.log(denominator)
            return -torch.mean(L)


        else:
            if self.loss_type == 'cosface':
                numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
            if self.loss_type == 'arcface':
                numerator = self.s * torch.cos(torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
            if self.loss_type == 'sphereface':
                numerator = self.s * torch.cos(self.m * torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

            excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
            denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
            L = numerator - torch.log(denominator)
            return -torch.mean(L)
