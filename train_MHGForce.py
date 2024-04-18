import io
import time
import copy
import numpy as np
import torch
import torch, numpy as np, scipy.sparse as sp
import torch.nn.functional as F
import argparse
import torch.optim as optim
from data.data import load_train_test_idx
from utils import accuracy,f1_macro,normaliseHyperIndiceMatrix,constructH
from utils import compute_attractive_repulsive_pyg,spring_electrical_loss
import torch_geometric
# HyperGCN/HNHN/HGNN_pyg/HyperSAGE_pyg/UniGCNII_pyg
from models import HGNN_pyg,HyperGCN,HyperSAGE_pyg,HNHN,UniGCNII_pyg,UniGAT,MLP,SetGNN,EquivSetGNN
import pickle
import datasets
import os
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser("physic", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dname', type=str, default='cora', help='dataset name')
# HGNN_pyg,HyperGCN,HNHN,HyperSAGE_pyg,UniGCNII_pyg/MLP/AllDeepSets/EDHNN
parser.add_argument('--method', type=str, default='HyperGCN', help='UniGNN Model(UniGCN, UniGAT, UniGIN, UniSAGE...)')
parser.add_argument('--add_self_loop', action="store_true", help='add-self-loop to hypergraph')
parser.add_argument('--use-norm', action="store_true", help='use norm in the final layer')
parser.add_argument('--activation', type=str, default='relu', help='activation layer between UniConvs')
parser.add_argument('--nlayer', type=int, default=1, help='number of hidden layers') # nlayer=2
parser.add_argument('--nhid', type=int, default=128, help='number of hidden features, note that actually it\'s #nhid x #nhead')
parser.add_argument('--nhead', type=int, default=8, help='number of conv heads')
parser.add_argument('--dropout', type=float, default=0.01, help='dropout probability after UniConv layer')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate') # 0.001
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--n-runs', type=int, default=10, help='number of runs for repeated experiments')
parser.add_argument('--gpu', type=int, default=4, help='gpu id to use')
parser.add_argument('--seed', type=int, default=1, help='seed for randomness')
parser.add_argument('--patience', type=int, default=200, help='early stop after specific epochs')
parser.add_argument('--nostdout', action="store_true",  help='do not output logging to terminal')
parser.add_argument('--use_physic', action="store_true", help='use_physic loss')
parser.add_argument('--split', type=int, default=1,  help='choose which train/test split to use')
parser.add_argument('--out-dir', type=str, default='runs/test',  help='output dir')
parser.add_argument('--q', type=float, default=0.00001, help='repulsive force weight')
parser.add_argument('--p', type=float, default=0.00001, help='attractive force decay')
parser.add_argument('--lam', type=float, default=0.001, help='Physic Loss weight')
parser.add_argument('--b', type=float, default=0.01, help='small bias')

'''
cora citeseer cora-co               load_citation_dataset
zoo Mushroom 20newsW100 NTU2012     load_LE_dataset
senate-committees house-committees  load_cornell_dataset
'''
# args = parser.parse_args([])
args = parser.parse_args()
args.cuda=True
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
if args.cuda:
	torch.cuda.manual_seed(seed)

torch.cuda.manual_seed_all(seed)

dname = args.dname
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
print('device:',device)
method=args.method
print('method:',args.method)
print('dname:',dname)
path = './data/'+dname+'/'
data_dir = './data/pyg2/'+dname 
raw_data_dir = path 
transform = None
feature_noise = 0.1

print('load data')
print('q:',args.q)
print('p:',args.p)

method_dict={'HyperGCN':'HyperGCN',
        'FastHyperGCN':'HyperGCN',
        'HyperSAGE_pyg':'HyperSAGE_pyg',
        'HGNN_pyg':'HGNN_pyg',
        'HNHN':'HNHN',
        'UniGCNII_pyg':'UniGCNII_pyg',
        'MLP':'MLP',
        'AllDeepSets':'SetGNN',
        'EDHNN':'EquivSetGNN'}

if method not in ['FastHyperGCN', 'HyperGCN', 'HyperSAGE_pyg']:
        print('self-loop')
        transform = torch_geometric.transforms.Compose([datasets.AddHypergraphSelfLoops()])
else:
        transform = None
data = datasets.HypergraphDataset(root=data_dir, name=dname, path_to_download=raw_data_dir,
        feature_noise=feature_noise, transform=transform).data
print('load over')

# method = 'HyperSAGE_pyg'
if method =='HyperGCN':
    args.fast=False
    args.depth=2
    args.mediators=True
elif method =='FastHyperGCN':
    args.fast=True
    args.depth=2
    args.mediators=True
elif method=='HyperSAGE_pyg':
    args.depth=2
    args.power = 1.
    args.num_sample = 100
    # args.MLP_hidden = 64
elif method=='HNHN':
    args.depth=1
    # args.depth=2
    # args.MLP_hidden = 64
    args.HNHN_alpha = -1.5
    args.HNHN_beta = -0.5
    args.HNHN_nonlinear_inbetween=True
elif method=='UniGCNII_pyg':
    args.restart_alpha=0.1
    args.depth=2
    args.use_norm=True
elif method=='MLP':
    args.depth=2
elif method=='AllDeepSets':
    args.AllSet_input_norm = True
    args.AllSet_GPR = False
    # args.AllSet_GPR = True
    args.AllSet_LearnMask = False
    # args.AllSet_LearnMask = True
    # args.AllSet_PMA = True
    args.AllSet_num_heads = args.nhead
    args.All_num_layers = args.nlayer
    args.MLP_num_layers = 1
    args.MLP_hidden = args.nhid
    args.Classifier_num_layers = 1
    args.Classifier_hidden = args.nhid
    args.normalization = 'ln'
    args.AllSet_PMA = False
    # args.aggregate = 'add'
    args.aggregate = 'mean'
    args.normtype = 'all_one'
elif method=='EDHNN':
    args.MLP2_num_layers=0
    args.MLP3_num_layers=1
    args.edconv_type='EquivSet'
    args.restart_alpha=0.0
    args.MLP_hidden = args.nhid
    args.input_dropout = 0.6
    args.All_num_layers = args.nlayer
    args.MLP_num_layers = 1
    args.AllSet_input_norm = True
    args.Classifier_num_layers = 1
    args.Classifier_hidden = args.nhid
    args.normalization = 'ln'
    args.aggregate = 'mean'
    
# use_physic
# args.use_physic=True
# args.use_physic=False
attr_adj,repu_adj = None, None

if args.use_physic:
    print('use physic')
    adj_path = data_dir+'/twoadj.pt'
    
    if os.path.exists(adj_path): # add
        print('exists, load adj')
        temp = torch.load(adj_path)
        attr_adj,repu_adj = temp['attr_adj'],temp['repu_adj']
    else:
        print('not exists, compute adj')
        attr_adj,repu_adj = compute_attractive_repulsive_pyg(data)
        temp={'attr_adj':attr_adj,'repu_adj':repu_adj}
        torch.save(temp,adj_path)
    attr_adj,repu_adj = attr_adj.to(device),repu_adj.to(device)
else:
    print('not use physic')

if method == 'HNHN':
    data = HNHN.generate_norm(data, args)
elif method == 'HyperSAGE_pyg':
    data = HyperSAGE_pyg.generate_hyperedge_dict(data)
elif method =='HGNN_pyg':
    data = constructH(data)
    data.edge_index=torch.FloatTensor(np.matrix(normaliseHyperIndiceMatrix(data.edge_index)))
elif method =='AllDeepSets':
    data = SetGNN.norm_contruction(data, option=args.normtype)


data = data.to(device)

b=args.b
q=args.q
p=args.p
lambda_ = args.lam
test_accs, best_test_accs = [],[]
test_f1s, best_test_f1s = [],[]
test_acc = 0.0
best_test_acc = 0.0
# PhysicLoss = None
print('use physic?',args.use_physic)

for run in range(1, args.n_runs+1):
# for run in range(1, 2):
# for run in range(1,5):
    tic_run = time.time()
    # old
    # _, train_idx, test_idx = data.load(args)
    idx_dict = load_train_test_idx(path,run)
    train_idx = torch.LongTensor(idx_dict['train']).to(device)
    test_idx = torch.LongTensor(idx_dict['test']).to(device)
    # train_idx = torch.LongTensor(train_idx).to(device)
    # test_idx  = torch.LongTensor(test_idx ).to(device)

    model = eval("{}(data.num_features, data.num_classes, args)".format(method_dict[method]))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                        weight_decay=args.wd)
    # optimizer = optim.SGD(model_ft.parameters(), lr=0.01, weight_decay=cfg['weight_decay)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=[100],
                                                gamma=0.9)
    
    criterion = torch.nn.CrossEntropyLoss()
    best_test_acc, test_acc, Z = 0, 0, None
    best_test_f1, test_f1 = 0, 0
    loss, PhysicLoss = None, None
    # for epoch in range(args.epochs):
    for epoch in range(200):
        tic_epoch = time.time()
        
        model.train()
        optimizer.zero_grad()
        
        Z = model(data)
        # task Loss
        loss_task = criterion(Z[train_idx], data.y[train_idx])

        # physic Loss
        if args.use_physic:
            PhysicLoss = spring_electrical_loss(Z,attr_adj,repu_adj,b,p,q)
            loss = loss_task + lambda_*PhysicLoss
        else:
            loss = loss_task
        # loss = loss_task
        loss.backward()
        optimizer.step()
        schedular.step()
        # loss_item = 
        # del loss
        train_time = time.time() - tic_epoch 

        # eval
        model.eval()
        with torch.no_grad():
            Z = model(data)
            train_acc= accuracy(Z[train_idx], data.y[train_idx])
            # valid_acc= accuracy(Z[valid_idx], Y[valid_idx])
            test_acc = accuracy(Z[test_idx], data.y[test_idx])
            test_f1 = f1_macro(Z[test_idx], data.y[test_idx])
            # log acc
            best_test_acc = max(best_test_acc, test_acc)
            best_test_f1 = max(best_test_f1, test_f1)
            if (epoch+1)%20==0:
                print((f'epoch:{epoch} | taskloss:{loss_task:.4f} | train acc:{train_acc:.2f} | test acc:{test_acc:.2f} | time:{train_time*1000:.1f}ms'))

    test_f1s.append(test_f1)
    best_test_f1s.append(best_test_f1)
    test_accs.append(test_acc) # final
    best_test_accs.append(best_test_acc) # best
    print(f"Run {run}/{args.n_runs}, best test accuracy: {best_test_acc:.2f}, acc(last): {test_acc:.2f}, total time: {time.time()-tic_run:.2f}s")
print(f"Physic? {args.use_physic}, Average final test f1_macro: {np.mean(test_f1s)} ± {np.std(test_accs)}") 
print(f"Physic? {args.use_physic}, Average best test f1_macro: {np.mean(best_test_f1s)} ± {np.std(best_test_accs)}") 
print(f"Physic? {args.use_physic}, Average final test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
print(f"Physic? {args.use_physic}, Average best test accuracy: {np.mean(best_test_accs)} ± {np.std(best_test_accs)}")


import pandas as pd
train_logs = pd.DataFrame()
train_logs['runs']=range(1,len(best_test_accs)+1)
train_logs['method']=args.method
train_logs['use_physic']=args.use_physic
train_logs['dataset']=args.dname
train_logs['is_self_loop']=args.add_self_loop
# train_logs['train_ratio']=args.train_prop
train_logs['split']=range(1,len(best_test_accs)+1)
train_logs['acc_best']=best_test_accs
# train_logs['acc_std']=0.0
train_logs['f1_best']=best_test_f1s
# train_logs['f1_std']=0.0
# train_logs['acc_final']=test_accs
# train_logs['final_std']=np.std(test_accs)
train_logs['hidden']=args.nhid
train_logs['lr']=args.lr
train_logs['seed']=args.seed
train_logs['p']=args.p
train_logs['q']=args.q
train_logs['lambda']=args.lam


# add_row = [-1,args.method ,args.use_physic, args.dname, args.add_self_loop, -1, np.mean(best_test_accs), 
#             np.std(best_test_accs), np.mean(best_test_f1s), np.std(best_test_f1s), args.nhid, 
            # args.lr, args.seed, args.p, args.q, args.lam]
add_row = [-1,args.method ,args.use_physic, args.dname, args.add_self_loop, -1, np.mean(best_test_accs)
            , np.mean(best_test_f1s), args.nhid, 
            args.lr, args.seed, args.p, args.q, args.lam]
train_logs.loc[len(train_logs)]=add_row



import os
logs_path = './logs/'
# logs_path = './logs/'
# train_log_save_file=logs_path+type(model).__name__+'_train_log_u.csv'
train_log_save_file=logs_path+'_param_log_all_ed_3.csv'
# test_log_save_file=logs_path+dataname+'_test.csv'

if os.path.exists(train_log_save_file): # add
    train_logs.to_csv(train_log_save_file, mode='a', index=False, header=0)
else: # create
    train_logs.to_csv(train_log_save_file, index=False)

print('over')