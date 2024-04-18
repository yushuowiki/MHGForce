import torch
import pickle
import os
import ipdb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
# 生成训练测试id split的文件
from torch_sparse import coalesce
from data import load_LE_dataset,load_cornell_dataset
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
# if args.cuda:
# 	torch.cuda.manual_seed(seed)
# 为所有GPU设置
torch.cuda.manual_seed_all(seed)
# 生成训练集和测试集id

def rand_train_test_idx_pure(label, train_prop=.5, ignore_negative=True, balance=False):
    """ Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]

    if not balance:
        train_num = int(n * train_prop)
        # valid_num = int(n * valid_prop)

        perm = torch.as_tensor(np.random.permutation(n))

        train_indices = perm[:train_num]
        # val_indices = perm[train_num:train_num + valid_num]
        test_indices = perm[train_num:]

        if not ignore_negative:
            return train_indices, test_indices

        train_idx = labeled_nodes[train_indices]
        # valid_idx = labeled_nodes[val_indices]
        test_idx = labeled_nodes[test_indices]

        split_idx = {'train': train_idx,
                     'test': test_idx}
    else:
        #         ipdb.set_trace()
        indices = []
        for i in range(label.max()+1):
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = int(train_prop/(label.max()+1)*len(labeled_nodes))
        # val_lb = int(valid_prop*len(labeled_nodes))
        train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        # valid_idx = rest_index[:val_lb]
        test_idx = rest_index[percls_trn:]
        split_idx = {'train': train_idx,
                     'test': test_idx}
    return split_idx

if __name__ == "__main__":

    # dataset='congress-bills'
    dataset='zoo'
    path='./'+dataset+'/' # 67, 成功
    # zoo/Mushroom/20newsW100/NTU2012/
    print('load  :',dataset)
    data = None
    if dataset in ['zoo','Mushroom','20newsW100','NTU2012']:
        data = load_LE_dataset(path,dataset)
    elif dataset in ['senate-committees','house-committees','congress-bills']:
        data = load_cornell_dataset(path,dataset,feature_noise=0.1)
    print(data)

    split_path = path+'splits/'
    if not os.path.isdir(split_path):
        print('not exist, mkdir')
        os.makedirs(split_path)
    else:
        print('exist dir')

    for i in range(1,11):
        idx_list = rand_train_test_idx_pure(data.y,train_prop=0.2)
        split_idxs = {'train': idx_list['train'].tolist(),'test': idx_list['test'].tolist()}
        with open(split_path+str(i)+".pickle", "wb") as fp:   #Pickling
            pickle.dump(split_idxs, fp)
        print('dump successfully!')

    print('test split')
    for i in range(1,11):
        split_name = split_path+str(i)+'.pickle'
        print(split_name)
        Splits=None
        with open(split_name, 'rb') as H: 
            Splits = pickle.load(H)

        print('train:',len(Splits['train']))
        print('test:',len(Splits['test']))
        print(Splits['train'][:10])
        print('-'*20)