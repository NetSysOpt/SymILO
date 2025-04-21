import os
os.environ['KMP_DUPLICATE_LIB_OK']="True"
import pyscipopt
from dataset import MIPDataset
from nn import GNNPolicy
from pathlib import Path
import scipy.io as io
import numpy as np
from label_opt import labelOpt
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
parser.add_argument('--expName', type=str, default='SMSP')
parser.add_argument('--dataset', type=str, default='SMSP')
parser.add_argument('--opt', type=str, default='opt')
parser.add_argument('--epoch', type=int, default=2)
parser.add_argument('--PE', type=str, default='Y')
args = parser.parse_args()


LEARNING_RATE = 0.001
NB_EPOCHS = args.epoch
PRT_FREQUENCY = 1
TBATCH = 1
NUM_WORKERS = 0
OPT = args.opt
EXP_NAME = args.expName
info = confInfo[args.dataset]
DIR_INS = os.path.join(info['trainDir'],'ins')
DIR_SOL = os.path.join(info['trainDir'],'sol')
DIR_BG = os.path.join(info['trainDir'],'bg')
NGROUP = info['nGroup']

ADDPOS = info['addPosFeature']
REORDER = info['reorder']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.expName,exist_ok=True)
sample_names = os.listdir(DIR_SOL)
sample_files = [ (os.path.join(DIR_INS,name.replace('.sol','')),os.path.join(DIR_SOL,name)) for name in sample_names]



random.seed(0)
random.shuffle(sample_files)

train_files = sample_files[: int(0.8 * len(sample_files))]
valid_files = sample_files[int(0.8 * len(sample_files)) :]


train_data = MIPDataset(train_files,DIR_BG,REORDER,ADDPOS)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=NUM_WORKERS)
valid_data = MIPDataset(valid_files,DIR_BG,REORDER,ADDPOS)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)


policy = GNNPolicy(NGROUP).to(DEVICE)




def process(policy, data_loader, optimizer=None):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """

    if optimizer:
        policy.train()
    else:
        policy.eval()

    mean_loss = 0
    mean_acc = 0

    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        batch_losses = []
        for step, batch in enumerate(data_loader):
            groupFeatures = batch['groupFeatures'][0].to(DEVICE)
            if args.PE != 'Y':
                groupFeatures = groupFeatures*0
            varFeatures = batch['varFeatures'][0].to(DEVICE)
            consFeatures = batch['consFeatures'][0].to(DEVICE)
            edgeFeatures = batch['edgeFeatures'][0].to(DEVICE)
            edgeInds = batch['edgeInds'][0].to(DEVICE)
            biInds = batch['biInds'][0].to(DEVICE)
            sols = batch['sols'][0].to(DEVICE)
            objs = batch['objs'][0].to(DEVICE)
            reorderInds = batch['reorderInds'][0].long().reshape(-1)
            nGroup = batch['nGroup'][0]
            nElement = batch['nElement'][0]

            output = policy(
                consFeatures,
                edgeInds.long(),
                edgeFeatures,
                varFeatures,
                groupFeatures

            )
            output = output.sigmoid()[biInds]

            X_hat = output[reorderInds].reshape(nElement,nGroup)
            X = sols[reorderInds].reshape(nElement,nGroup)
            #
            # # compute loss
            with torch.set_grad_enabled(True):
                opt_func = labelOpt if OPT=='opt' else None
                X_bar = opt_func(X_hat.detach()[None,:,:], X.clone()[None,:,:],device=DEVICE)[0] if opt_func is not None else X


            sols[reorderInds] = X_bar.reshape(-1)

            pos_loss = -torch.log(output[reorderInds] + 0.00001) * (sols[reorderInds] >= 0.5)
            neg_loss = -torch.log(1 - output[reorderInds] + 0.00001) * (sols[reorderInds] < 0.5)
            loss = pos_loss.sum() + neg_loss.sum()

            if optimizer is not None:
                loss /= TBATCH
                loss.backward()

            if step%TBATCH == TBATCH-1 or step==len(data_loader)-1:
                if optimizer is not None:
                    optimizer.step()
                    optimizer.zero_grad()
                # output
                if step%PRT_FREQUENCY==0:
                    mod = 'train' if optimizer else 'valid'
                    print('Epoch {} {} [{}/{}] loss {:.6f}'.format( epoch, mod, step,len(data_loader),loss.item()))

            mean_loss += loss.item() * X.shape[0]
            #mean_acc += accuracy * batch.num_graphs
            n_samples_processed +=  X.shape[0]

    mean_loss /= n_samples_processed
    mean_acc /= n_samples_processed
    return mean_loss


optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

train_losses = []
train_accs = []
valid_losses = []
valid_accs = []

best_val_loss = 99999

for epoch in range(NB_EPOCHS):

    train_loss = process(policy, train_loader, optimizer)
    print(f"Epoch {epoch} Train loss: {train_loss:0.3f}")

    valid_loss = process(policy, valid_loader, None)
    print(f"Epoch {epoch} Valid loss: {valid_loss:0.3f}")

    if valid_loss<best_val_loss:
        best_val_loss = valid_loss

        torch.save(policy.state_dict(),os.path.join(EXP_NAME,'model_best.pth'))
    torch.save(policy.state_dict(), os.path.join(EXP_NAME,'model_last.pth'))

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

io.savemat(os.path.join(EXP_NAME,'loss_record.mat'),{
    'train_loss':np.array(train_losses),
    'valid_loss':np.array(valid_losses)
})


print('done')





