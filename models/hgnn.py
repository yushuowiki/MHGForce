import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.parameter import Parameter
from torch.nn.parameter import Parameter


class HGNN_pyg(nn.Module):
    # def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
    def __init__(self, num_features, num_classses, args):
        super(HGNN_pyg, self).__init__()
        self.dropout = args.dropout
        self.hgc1 = HGNN_conv(num_features, args.nhid)
        self.hgc2 = HGNN_conv(args.nhid, num_classses)

    def reset_parameters(self):
        self.hgc1.reset_parameters()
        self.hgc2.reset_parameters()

    def forward(self, data):
        x = data.x
        G = data.edge_index

        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x





class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G),inplace=True)
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        # return x
        return x


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, H: torch.Tensor):
        # print('x.shape',x.shape)
        # print('H.shape',H.shape)
        x = x.matmul(self.weight)
        # print(x[0:5])
        if self.bias is not None:
            x = x + self.bias
        x = H.matmul(x)
        # print(x[0:5])
        return x