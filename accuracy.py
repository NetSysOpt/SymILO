import os
import pickle
from label_opt import labelOpt
import pyscipopt
from dataset import MIPDataset
from nn import GNNPolicy
from pathlib import Path
import scipy.io as io
import numpy as np
from label_opt import labelOpt,lexOpt
import torch
import torch.nn.functional as F
import torch_geometric
import random
import shutil
from config import *
import argparse

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--expName', type=str, default='SMSP_opt')
parser.add_argument('--dataset', type=str, default='SMSP')
parser.add_argument('--PE', type=str, default='Y')
args = parser.parse_args()




EXP_NAME = args.expName
DATASET = args.dataset
info = confInfo[args.dataset]
DIR_SOL = os.path.join(info['trainDir'],'sol')
DIR_PRE = os.path.join(EXP_NAME,'logits_train')
os.makedirs(args.expName,exist_ok=True)
sample_names = os.listdir(DIR_SOL)
sample_files = [ (os.path.join(DIR_PRE,name.replace('.sol','.prob')),os.path.join(DIR_SOL,name)) for name in sample_names]

random.seed(0)
random.shuffle(sample_files)

valid_files = sample_files[int(0.8 * len(sample_files)) :]

Ks = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

nErrors = []
for step,filepath in enumerate(valid_files):

    pre_file,sol_file = filepath

    preData = pickle.load(open(pre_file,'rb'))
    pre = preData['pre']
    solData = pickle.load(open(sol_file,'rb'))
    sol = solData['sols'][0].astype(int)
    varnames = solData['var_names'] if 'IP' in DATASET else solData['varNames']
    if 'IP' == DATASET:
        nItem = max([int(re.findall('\d+', name)[0]) for name in varnames if 'place' in name]) + 1
        nBin = max([int(re.findall('\d+', name)[1]) for name in varnames if 'place' in name]) + 1

        X = torch.zeros((nItem, nBin))

        for ind, name in enumerate(varnames):
            if 'place' in name:
                ss = re.findall('\d+', name)
                a, b = int(ss[0]), int(ss[1])
                X[a, b] = sol[ind]
        X_hat = torch.Tensor(pre.reshape(nItem,nBin))

    elif 'SMSP' == DATASET:
        nItem = max([int(re.findall('\d+', name)[0]) for name in varnames if 'X' in name]) + 1
        nCap = max([int(re.findall('\d+', name)[0]) for name in varnames if 'Y' in name]) + 1

        X= torch.zeros((nItem + nCap, nItem))

        for ind, name in enumerate(varnames):
            ss = re.findall('\d+', name)
            a, b = int(ss[0]), int(ss[1])
            if 'X' in name:
                X[a, b] = sol[ind]
            elif 'Y' in name:
                X[a + nItem, b] = sol[ind]
        X_hat = torch.Tensor(pre.reshape(nItem+ nCap, nItem))

    X_bar = labelOpt(X_hat[None, :, :], X.clone()[None, :, :], device='cpu')

    X_hat = X_hat.reshape(-1)
    X_round = X_hat.round()
    X_bar = X_bar.reshape(-1)
    n = X_hat.shape[-1]
    kErrors = []
    for k in Ks:
        nTop = int(n * k)
        ordering = (-(X_hat - 0.5).abs()).sort(dim=-1)[1][0:nTop]
        topKRound = X_round[ordering]
        topKXBar = X_bar[ordering]
        error = (topKRound != topKXBar).sum().item()
        kErrors.append(error)

    nErrors.append(kErrors)

    print(f'Processed {step}/{len(valid_files)}')

nErrors = np.array(nErrors)

errorMean = nErrors.mean(axis=0)
errorStd = nErrors.std(axis=0)

with open(os.path.join(EXP_NAME,'prediction_error.txt'),'w') as f:
    for i in range(nErrors.shape[1]):

        f.write(f'k={Ks[i]}  mean: {errorMean[i]:.2f} std: {errorStd[i]:.2f}\n')

print('done')
#
# X_hats,X_bars = process(policy, valid_loader)
# N = len(X_hats)
# X_hats =  torch.stack(X_hats,dim=0).reshape(N,-1)
# X_round =X_hats.round()
# X_bars = torch.stack(X_bars,dim=0).reshape(N,-1)
#
# # topk
# n = X_hats.shape[-1]
# kErrors = []
# Ks = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# for k in Ks :
#
#     nTop = int(n*k)
#     ordering = (-(X_hats-0.5).abs()).sort(dim=-1)[1]
#     errors = []
#     for i in range(N):
#         topKRound = X_round[i][0:nTop]
#         topKXBar = X_bars[i][0:nTop]
#
#
#         error = (topKRound != topKXBar).sum().item()
#         errors.append(error)
#     kErrors.append(errors)
# kErrors = np.array(kErrors)
# errorMean = kErrors.mean(axis=-1)
# errorStd = kErrors.std(axis=-1)
    # mask = torch.zeros_like(X_hats)
    # for i in range(N):
    #     mask[i][ordering[i][0:nTop]] = 1
    #
    #
    # TP = (X_round*X_bars*mask).sum(dim=-1)
    # TN = ( (1-X_round)*(1-X_bars)*mask).sum(dim=-1)
    #
    # Ppre = (X_round*mask).sum(dim=-1)
    # Npre = ((1-X_round)*mask).sum(dim=-1)
    #
    # FP = Ppre - TP
    # FN = Npre - TN
    #
    # allP = TP + FN
    # allN = TN + FP
    #
    # # accP = TP/(allP+1e-8)
    # # accN = TN/(allN+1e-8)
    # error = (FP +FN)/(allP+allN)
    # errors.append(error.mean().item())







