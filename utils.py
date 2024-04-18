import numpy as np
import scipy.sparse as sp
from numpy import inf
import torch
import time
from tqdm import tqdm
import torch_geometric
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def accuracy(Z, Y):
    return 100 * Z.argmax(1).eq(Y).float().mean().item()

def f1_macro(Z, Y):
    return 100 * f1_score(y_true=Y.detach().cpu().numpy(), y_pred=Z.argmax(1).detach().cpu().numpy(), average='macro') # 也可以指定micro模式 


class AddHypergraphSelfLoops(torch_geometric.transforms.BaseTransform):
    def __init__(self, ignore_repeat=True):
        super().__init__()
        # whether to detect existing self loops
        self.ignore_repeat = ignore_repeat
    
    def __call__(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        num_hyperedges = data.num_hyperedges

        node_added = torch.arange(num_nodes, device=edge_index.device, dtype=torch.int64)
        if self.ignore_repeat:
            # 1. compute hyperedge degree
            hyperedge_deg = torch.zeros(num_hyperedges, device=edge_index.device, dtype=torch.int64)
            hyperedge_deg = hyperedge_deg.scatter_add(0, edge_index[1], torch.ones_like(edge_index[1]))
            hyperedge_deg = hyperedge_deg[edge_index[1]]

            # 2. if a node has a hyperedge with degree 1, then this node already has a self-loop
            has_self_loop = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.int64)
            has_self_loop = has_self_loop.scatter_add(0, edge_index[0], (hyperedge_deg == 1).long())
            node_added = node_added[has_self_loop == 0] # not include self-loop node

        # 3. create dummy hyperedges for other nodes who have no self-loop
        hyperedge_added = torch.arange(num_hyperedges, num_hyperedges + node_added.shape[0])
        edge_indx_added = torch.stack([node_added, hyperedge_added], 0) # node_added.shape == hyperedge_added.shape
        # edge_indx_added shape=[2,node_added.shape]
        edge_index = torch.cat([edge_index, edge_indx_added], -1) 

        # 4. sort along w.r.t. nodes
        _, sorted_idx = torch.sort(edge_index[0])
        data.edge_index = edge_index[:, sorted_idx].long()

        return data


""" Adapted from https://github.com/snap-stanford/ogb/ """
class Logger:

    def __init__(self, runs, log_path=None):
        self.log_path = log_path
        self.results = [[] for _ in range(runs)]

    # def add_result(self, run, train_acc, valid_acc, test_acc):
    #     result = [train_acc, valid_acc, test_acc]
    #     assert len(result) == 3
    #     assert run >= 0 and run < len(self.results)
    #     self.results[run].append(result)

    def add_result(self, run, train_acc, test_acc):
        result = [train_acc, test_acc]
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def get_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            max_train = result[:, 0].max().item()
            max_test = result[:, 2].max().item()

            argmax = result[:, 1].argmax().item()
            train = result[argmax, 0].item()
            # valid = result[argmax, 1].item()
            test = result[argmax, 2].item()
            test = result[argmax, 1].item()
            # return {'max_train': max_train, 'max_test': max_test,
            #     'train': train, 'valid': valid, 'test': test}
            return {'max_train': max_train, 'max_test': max_test,
                'train': train, 'test': test}
        else:
            # keys = ['max_train', 'max_test', 'train', 'valid', 'test']
            keys = ['max_train', 'max_test', 'train', 'test']
            best_results = []
            for r in range(len(self.results)):
                best_results.append([self.get_statistics(r)[k] for k in keys])

            ret_dict = {}
            best_result = torch.tensor(best_results)
            for i, k in enumerate(keys):
                ret_dict[k+'_mean'] = best_result[:, i].mean().item()
                ret_dict[k+'_std'] = best_result[:, i].std().item()

            return ret_dict

    def print_statistics(self, run=None):
        if run is not None:
            result = self.get_statistics(run)
            print(f"Run {run + 1:02d}:")
            print(f"Highest Train: {result['max_train']:.2f}")
            # print(f"Highest Valid: {result['valid']:.2f}")
            print(f"  Final Train: {result['train']:.2f}")
            print(f"   Final Test: {result['test']:.2f}")
        else:
            result = self.get_statistics()
            print(f"All runs:")
            print(f"Highest Train: {result['max_train_mean']:.2f} ± {result['max_train_std']:.2f}")
            # print(f"Highest Valid: {result['valid_mean']:.2f} ± {result['valid_std']:.2f}")
            print(f"  Final Train: {result['train_mean']:.2f} ± {result['train_std']:.2f}")
            print(f"   Final Test: {result['test_mean']:.2f} ± {result['test_std']:.2f}")

    def plot_result(self, run=None):
        plt.style.use('seaborn')
        if run is not None:
            result = 100 * torch.tensor(self.results).mean(0)
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])


class NodeClsEvaluator:

    def __init__(self):
        return

    def eval(self, y_true, y_pred):
        acc_list = []
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

        is_labeled = (~np.isnan(y_true)) & (~np.isinf(y_true)) # no nan and inf
        correct = (y_true[is_labeled] == y_pred[is_labeled])
        acc_list.append(float(np.sum(correct))/len(correct))

        return {'acc': sum(correct) / sum(is_labeled)}


def rand_train_test_idx(label, train_prop, valid_prop, balance=False):
    '''
    split_idx_lst = []
    for run in range(args.runs):
        split_idx = utils.rand_train_test_idx(
            graph_data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
        split_idx_lst.append(split_idx)
    '''
    if not balance:
        n = label.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.randperm(n)

        train_idx = perm[:train_num]
        valid_idx = perm[train_num:train_num + valid_num]
        test_idx = perm[train_num + valid_num:]

        split_idx = {
            'train': train_idx,
            'valid': valid_idx,
            'test': test_idx
        }

    else:
        indices = []
        for i in range(label.max()+1): # 每类的数据
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = int(train_prop/(label.max()+1)*len(label))
        val_lb = int(valid_prop*len(label))
        train_idx = torch.cat([ind[:percls_trn] for ind in indices], dim=0)
        rest_index = torch.cat([ind[percls_trn:] for ind in indices], dim=0)
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]

        split_idx = {
            'train': train_idx,
            'valid': valid_idx,
            'test': test_idx
        }

    return split_idx


def compute_attractive_repulsive_pyg(graph_data):
    edge_node_list=[]
    print('edge_ids_list')
    for edge_id in tqdm(graph_data.edge_index[1].unique()):
        mask_=graph_data.edge_index[1]==edge_id
        node_list_toadd = graph_data.edge_index[0][mask_].cpu().tolist()
        edge_node_list.append(node_list_toadd)

    N, M = graph_data.x.shape[0], graph_data.edge_index[1].max()+1
    indptr, indices, data_value = [0], [], []
    for vs in edge_node_list: # for each edge (node list)
        indices += vs 
        data_value += [1] * len(vs) # means has
        indptr.append(len(indices))
    H = sp.csc_matrix((data_value, indices, indptr), shape=(N, M), dtype=int).tocsr().astype(float) # V x E

    print('edge_ids_list')
    temptime1 = time.time()
    edge_ids_list = [H[i].nonzero()[1] for i in range(graph_data.x.shape[0])] # 每个节点关联的超边list
    print(f"edge num: {len(edge_ids_list)} \ncost time: {time.time()-temptime1:.2f}s")

    # 关联超边包含的节点 , 是否可以加速
    print('all_attractive_nodeid_list')
    temptime2 = time.time()
    #--------------version3 7.57s
    all_attractive_nodeid_list = []
    for i in tqdm(range(graph_data.x.shape[0])): # for node
        exclude = [i]
        temp_ids=list(H[:,edge_ids_list[i]].nonzero()[0]) # 节点i关联的所有超边的所有节点
        all_attractive_nodeid_list.append(temp_ids)
    print(f"cost time: {time.time()-temptime2:.2f}s")

    print('attr_pair_weights')
    temptime3 = time.time()
    attr_pair_weights = torch.zeros([graph_data.x.shape[0],graph_data.x.shape[0]],dtype=torch.float32).to(graph_data.x.device) # symmetric
    for i,node_list in enumerate(tqdm(all_attractive_nodeid_list)): # for node
        for pair_node in node_list: # symetric
            if pair_node!=i: # not self loop
                attr_pair_weights[i,pair_node]+=1.0 # others
            # else:
                # attr_pair_weights[i,pair_node]+=1.0 # self
    print(f"cost time: {time.time()-temptime3:.2f}s") # old 9.84s

    # attractive
    self_diag_attr = torch.diag(attr_pair_weights.sum(1)) # for i how many num j is connected 
    attr_adj = self_diag_attr-attr_pair_weights # NI-A : to compute N(i-j) for each j in hyperedge

    # repulsive
    repu_pair_weights = torch.ones([graph_data.x.shape[0],graph_data.x.shape[0]],dtype=torch.float32).to(graph_data.x.device)
    self_diag_repu = torch.diag(repu_pair_weights.sum(1))
    repu_adj = self_diag_repu-repu_pair_weights # NI-1 :node i for all nde j
    return attr_adj,repu_adj

def compute_attractive_repulsive(G,X):
    N, M = X.shape[0], len(G)
    indptr, indices, data_value = [0], [], []
    for e, vs in G.items(): # for each edge (node list)
        indices += vs 
        data_value += [1] * len(vs) # means has
        indptr.append(len(indices))
    H = sp.csc_matrix((data_value, indices, indptr), shape=(N, M), dtype=int).tocsr().astype(float) # V x E

    print('edge_ids_list')
    temptime1 = time.time()
    edge_ids_list = [H[i].nonzero()[1] for i in range(X.shape[0])] # 
    print(f"edge num: {len(edge_ids_list)} \ncost time: {time.time()-temptime1:.2f}s")

    # 关联超边包含的节点 , 是否可以加速
    print('all_attractive_nodeid_list')
    temptime2 = time.time()
    #--------------version3 7.57s
    all_attractive_nodeid_list = []
    for i in tqdm(range(X.shape[0])): # for node
        exclude = [i]
        temp_ids=list(H[:,edge_ids_list[i]].nonzero()[0])
        all_attractive_nodeid_list.append(temp_ids)
    print(f"cost time: {time.time()-temptime2:.2f}s")

    print('attr_pair_weights')
    temptime3 = time.time()
    attr_pair_weights = torch.zeros([X.shape[0],X.shape[0]],dtype=torch.float32).to(X.device) # symmetric
    for i,node_list in enumerate(tqdm(all_attractive_nodeid_list)): # for node
        for pair_node in node_list:
            if pair_node!=i: # not self loop
                attr_pair_weights[i,pair_node]+=1.0 # others
            # else:
                # attr_pair_weights[i,pair_node]+=1.0 # self
    print(f"cost time: {time.time()-temptime3:.2f}s") # old 9.84s

    # attractive
    self_diag_attr = torch.diag(attr_pair_weights.sum(1)) # for i how many num j is connected 
    attr_adj = self_diag_attr-attr_pair_weights # NI-A : to compute N(i-j) for each j in hyperedge

    # repulsive
    repu_pair_weights = torch.ones([X.shape[0],X.shape[0]],dtype=torch.float32).to(X.device)
    self_diag_repu = torch.diag(repu_pair_weights.sum(1))
    repu_adj = self_diag_repu-repu_pair_weights # NI-1 :node i for all nde j
    return attr_adj,repu_adj


def spring_electrical_loss(Z,attr_adj,repu_adj,b,p,q):
    # -----------------------------------------------------------
    Attract4nodes = attr_adj.matmul(Z) # attractive
    #------------------------------------------------------------
    Other_Tensor = Z.repeat(Z.shape[0],1,1) # (n,n,d) 
    Self_Tensor = Z.repeat(1,Z.shape[0]).reshape(Z.shape[0],Z.shape[0],Z.shape[1]) # (n,nxd) --> (n,n,d)
    D_tensor = Self_Tensor - Other_Tensor # D
    # D_tensor_ = torch.abs(D_tensor)+b # D abs
    D_2_inv = ((torch.abs(D_tensor)+b)**2).sum(-1).pow(-1) # (n,n)
    Rep4node = (D_tensor*D_2_inv.unsqueeze(-1)).sum(1) # (n,) # Pepulsive
    #------------------------------------------------------------

    physic_status = -p*Attract4nodes+q*Rep4node
    PhysicLoss = (physic_status**2).sum()
    return PhysicLoss



def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M  
    where D is the diagonal node-degree matrix 
    """
    
    d = np.array(M.sum(1))
    
    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)    # D inverse i.e. D^{-1}
    
    return DI.dot(M)


# edge_index to original H
def constructH(data):
    """
    Construct incidence matrix H of size (num_nodes,num_hyperedges) from edge_index = [V;E]
    """
#     ipdb.set_trace()
    edge_index = np.array(data.edge_index)
    # Don't use edge_index[0].max()+1, as some nodes maybe isolated
    num_nodes = data.x.shape[0]
    num_hyperedges = np.max(edge_index[1])-np.min(edge_index[1])+1
    H = np.zeros((num_nodes, num_hyperedges))
    cur_idx = 0
    for he in np.unique(edge_index[1]):
        nodes_in_he = edge_index[0][edge_index[1] == he]
        H[nodes_in_he, cur_idx] = 1.
        cur_idx += 1

    data.edge_index = H
    return data


def normaliseHyperIndiceMatrix(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    try:
        H = H.todense()
    except:
        pass
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge) # (e x e)
    # the degree of the node
    DV = np.sum(H * W, axis=1) # (e x e)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    # invDE = np.mat(np.diag(np.power(DE, -1)))
    invDE = np.matrix(np.diag(np.power(DE, -1)))
    invDE[invDE == inf] = 0.0
    # DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    DV2 = np.matrix(np.diag(np.power(DV, -0.5)))
    DV2[DV2 == inf] = 0.0
    # W = np.mat(np.diag(W))
    W = np.matrix(np.diag(W))
    # H = np.mat(H)
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


def normaliseHyperIndiceMatrix_sparse(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    n_edge = H.shape[1] # number of e
    W = np.ones(n_edge)
    DV = H * W # #matrix  # (19717,)
    DE = np.array(np.sum(H, axis=0)).reshape(-1) # matrix # (1, 27680)

    invDE = np.power(DE, -1)
    invDE[invDE == inf] = 0.0

    DV2 = np.power(DV, -0.5)
    DV2[DV2 == inf] = 0.0
    # W = np.matrix(np.diag(W))
    HT = H.T
    
    DV2 = sp.diags(DV2,0,format='csr')
    W = sp.diags(W,0,format='csr')
    invDE = sp.diags(invDE,0,format='csr')
    # DV2*H*W*invDE*DV2

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G