
import math
import numpy as np

import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter

class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    # def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
    #              dropout=.5, Normalization='bn', InputNorm=False):
    def __init__(self, num_features, num_classes, args, Normalization='bn', InputNorm=False):
        super(MLP, self).__init__()

        self.in_channels = num_features
        self.hidden_channels = args.nhid
        self.out_channels = num_classes
        num_layers=args.depth

        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(self.in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(self.in_channels, self.out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(self.in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(self.in_channels, self.hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(self.hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(self.hidden_channels, self.hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(self.hidden_channels))
                self.lins.append(nn.Linear(self.hidden_channels, self.out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(self.in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(self.in_channels, self.out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(self.in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(self.in_channels, self.hidden_channels))
                self.normalizations.append(nn.LayerNorm(self.hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(self.hidden_channels, self.hidden_channels))
                    self.normalizations.append(nn.LayerNorm(self.hidden_channels))
                self.lins.append(nn.Linear(self.hidden_channels, self.out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(self.in_channels, self.out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(self.in_channels, self.hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(self.hidden_channels, self.hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(self.hidden_channels, self.out_channels))

        self.dropout = args.dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ == 'Identity'):
                normalization.reset_parameters()

    def forward(self, data):
        x=data.x
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class PlainMLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(PlainMLP, self).__init__()
        self.lins = nn.ModuleList()

        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
