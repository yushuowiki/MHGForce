import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module 
from torch.nn.parameter import Parameter
from torch import nn

# pyg
class HyperSAGE_pyg(nn.Module):

    @staticmethod
    def generate_hyperedge_dict(data):
        data0 = data.clone().cpu()
        He_dict = get_HyperGCN_He_dict(data0)
        data.hyperedge_dict = He_dict
        for e, nodes in data.hyperedge_dict.items():
            data.hyperedge_dict[e] = torch.tensor(nodes)
        return data

    def __init__(self, num_features, num_classes, args):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperSAGE_pyg, self).__init__()
        # d, l, c = num_features, args.All_num_layers, num_classes
        d, l, c = num_features, args.depth, num_classes
        
        h = [d]
        for i in range(l-1):
            power = l - i + 2
            h.append(2**power)
        h.append(c)

        # if args.nhid >= 16:
        #     print('Caution: Too large hidden dimension. Running very slow.')

        self.hgc1 = HyperTensorGraphConvolution(d,16)
        self.hgc2 = HyperTensorGraphConvolution(16,c)
        self.do, self.l = args.dropout, args.depth

        self.power = args.power
        self.num_sample = args.num_sample
        # self.structure = E

    def reset_parameters(self):
        self.hgc1.reset_parameters()
        self.hgc2.reset_parameters()

    def forward(self, data):
        """
        an l-layer GCN
        """
        H = data.x
        structure = data.hyperedge_dict
        do, l= self.do, self.l
        power = self.power
        num_sample = self.num_sample
        H = F.relu(self.hgc1(structure, H, power, num_sample))
        # print('hypersage1:',H.isnan().sum())
        H = F.dropout(H, do, training=self.training)
        H = self.hgc2(structure, H, power, num_sample)  
        # print('hypersage1:',H.isnan().sum())    
        return H


class HyperSAGE(nn.Module):
    
    # @staticmethod
    # def generate_hyperedge_dict(data):
    #     data0 = data.clone().cpu()
    #     He_dict = get_HyperGCN_He_dict(data0)
    #     data.hyperedge_dict = He_dict
    #     for e, nodes in data.hyperedge_dict.items():
    #         data.hyperedge_dict[e] = torch.tensor(nodes)
    #     return data

    def __init__(self, V, E, X, args):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperSAGE, self).__init__()
        d, l, c = args.d, args.depth, args.c
        # args.d, args.depth, args.c
        h = [d]
        for i in range(l-1):
            power = l - i + 2
            h.append(2**power)
        h.append(c)

        # if args.nhid >= 16:
        #     print('Caution: Too large hidden dimension. Running very slow.')

        self.hgc1 = HyperTensorGraphConvolution(d,16)
        self.hgc2 = HyperTensorGraphConvolution(16,c)
        self.do, self.l = args.dropout, args.depth
        self.power = args.power
        self.num_sample = args.num_sample
        self.structure = E

    def reset_parameters(self):
        self.hgc1.reset_parameters()
        self.hgc2.reset_parameters()

    def forward(self, H):
        """
        an l-layer GCN
        """
        # H = data.x
        # structure = data.hyperedge_dict
        do, l= self.do, self.l
        power = self.power
        num_sample = self.num_sample
        # H = F.relu(self.hgc1(structure, H, power, num_sample))
        H = F.relu(self.hgc1(self.structure, H, power, num_sample))
        H = F.dropout(H, do, training=self.training)
        H = self.hgc2(self.structure, H, power, num_sample)
        # return F.log_softmax(H, dim=1)
        return H

class HyperTensorGraphConvolution(Module):

    def __init__(self, a, b):
        super(HyperTensorGraphConvolution, self).__init__()
        self.a, self.b = a, b
        self.W = Parameter(torch.FloatTensor(a, b))
        self.bias = Parameter(torch.FloatTensor(b))
        #self.edge_count = edge_count
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1. / np.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, structure, H, power,num_sample):
        #self.edge_count = self.edge_count.cuda()
        W, b = self.W, self.bias
        AH = signal_shift_hypergraph_sample(structure,H, power, num_sample) 
        # print('AH:',AH.isnan().sum())
        AHW = torch.mm(AH, W) 
        output = AHW + b
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.a) + ' -> ' \
               + str(self.b) + ')'


def get_HyperGCN_He_dict(data):
    # Assume edge_index = [V;E], sorted
    edge_index = np.array(data.edge_index.cpu())
    """
    For each he, clique-expansion. Note that we allow the weighted edge.
    Note that if node pair (vi,vj) is contained in both he1, he2, we will have (vi,vj) twice in edge_index. (weighted version CE)
    We default no self loops so far.
    """
    # edge_index[1, :] = edge_index[1, :]-edge_index[1, :].min()
    He_dict = {}
    for he in np.unique(edge_index[1, :]):
        nodes_in_he = list(edge_index[0, :][edge_index[1, :] == he])
        He_dict[he.item()] = nodes_in_he

    return He_dict


def signal_shift_graph2(hypergraph,edge_count,H):
    new_signal = H.clone()
    for edge,nodes in hypergraph.items():
        for node_i in nodes:
            neighbor_nodes = nodes[nodes!=node_i]
            new_signal[node_i] = new_signal[node_i] + torch.sum(H[neighbor_nodes], dim=0)

    H = new_signal/(edge_count+1)
    return H

def signal_shift_hypergraph2(hypergraph,H):
    #new_signal = torch.zeros(H.shape[0],H.shape[1]).cuda()
    new_signal = H.clone()
    for edge,nodes in hypergraph.items():
        for node_i in nodes:
            # neighbor_nodes = (nodes[nodes!=node_i]).to(dtype=torch.long)
            neighbor_nodes = nodes[nodes!=node_i]
            # node_i = node_i.to(dtype=torch.long)
            node_i = node_i
            new_signal[node_i] = new_signal[node_i] + torch.sum(H[neighbor_nodes], dim=0)/(len(nodes)-1)
    return H



def signal_shift_hypergraph_power(hypergraph,H, power):
    min_value, max_value = 1e-7, 1e1
    torch.clamp_(H, min_value, max_value)
    #new_signal = torch.zeros(H.shape[0],H.shape[1]).cuda()
    new_signal = H.clone()
    for edge,nodes in hypergraph.items():
        for node_i in nodes:
            neighbor_nodes = (nodes[nodes!=node_i]).to(dtype=torch.long)
            # neighbor_nodes = nodes[nodes!=node_i]
            node_i = node_i.to(dtype=torch.long)
            # node_i = node_i
            new_signal[node_i] = new_signal[node_i] + torch.pow(torch.sum(torch.pow(H[neighbor_nodes],power), dim=0)/(len(nodes)-1),1/power)

    return normalize(new_signal)

#  used
def signal_shift_hypergraph_sample(hypergraph,H, power, num_sample):
    min_value, max_value = 1e-7, 1e1
    torch.clamp_(H, min_value, max_value)
    #new_signal = torch.zeros(H.shape[0],H.shape[1]).cuda()
    new_signal = H.clone()

    for edge,nodes in hypergraph.items(): # hyperedge,nodes
        for node_i in nodes:
            neighbor_nodes = (nodes[nodes!=node_i]).to(dtype=torch.long)
            # neighbor_nodes = nodes[nodes!=node_i]
            # print('node_i',node_i)
            # print('nodes',nodes)
            # print('nodes[nodes!=node_i]',nodes[nodes!=node_i])
            node_i = node_i.to(dtype=torch.long)
            # node_i = node_i
            # if len(neighbor_nodes)==0: # all >0
            #     print('no node samples')
            if (len(neighbor_nodes)>num_sample): # sample
                shuffled_neighborhood = H[neighbor_nodes][torch.randperm(H[neighbor_nodes].size()[0])]
                new_signal[node_i] = new_signal[node_i] + torch.pow(torch.sum(torch.pow(shuffled_neighborhood[0:num_sample],power), dim=0)/(len(nodes)-1),1/power)
                # temp=torch.pow(torch.sum(torch.pow(shuffled_neighborhood[0:num_sample],power), dim=0)/(len(nodes)-1),1/power)
                # if temp.isnan().sum()>0:
                #     print('1111',temp)
            elif (len(neighbor_nodes)>0): # use all
                new_signal[node_i] = new_signal[node_i] + torch.pow(torch.sum(torch.pow(H[neighbor_nodes],power), dim=0)/(len(nodes)-1),1/power)
                # print()
                # temp = torch.pow(torch.sum(torch.pow(H[neighbor_nodes],power), dim=0)/(len(nodes)-1),1/power)
                # if temp.isnan().sum()>0:
                    # print('H[neighbor_nodes]',H[neighbor_nodes].isnan().sum())
                    # torch.pow tensor([], device='cuda:3', size=(0, 16))
                    # print('torch.pow',H[neighbor_nodes])
                    # print('torch.pow',torch.pow(H[neighbor_nodes],power))
                    # print('torch.sum',torch.sum(torch.pow(H[neighbor_nodes],power), dim=0)/(len(nodes)-1))             
                    # print('1112',temp)
                # print('1112',torch.pow(torch.sum(torch.pow(H[neighbor_nodes],power), dim=0)/(len(nodes)-1),1/power))
            else:
                pass
    # print('new_signal:',)
    # print('new_signal:',new_signal.isnan().sum()) # generate nan
    return normalize(new_signal)

def normalize(mx):
    """Row-normalize sparse matrix"""
    # rowsum = torch.tensor(mx.sum(1))
    rowsum = mx.sum(1).clone().detach()
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    mx = mx * r_inv[:,None]
    return mx

def normalize_np(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx