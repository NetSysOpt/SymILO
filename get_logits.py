import os
import pickle

import pyscipopt
from dataset import MIPDataset
from nn import GNNPolicy

from config import *
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--expName', type=str, default='SMSP_opt')
parser.add_argument('--dataset', type=str, default='SMSP')
parser.add_argument('--mod', type=str, default='test',help='train,test')
args = parser.parse_args()

EXP_NAME = args.expName
info = confInfo[args.dataset]
NGROUP = info['nGroup']
MOD = args.mod
TEST_INS = os.path.join(info[MOD +'Dir'],'ins')
TEST_BG = os.path.join(info[MOD + 'Dir'],'bg')
ADDPOS = info['addPosFeature']
REORDER = info['reorder']

INS_DIR = TEST_INS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set exp dir
now = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
exp_dir = f'logits' if MOD=='test' else 'logits_train'
exp_dir = os.path.join(EXP_NAME, exp_dir)
os.makedirs(exp_dir, exist_ok=True)

policy = GNNPolicy(NGROUP).to(DEVICE)
states = torch.load(os.path.join(EXP_NAME,'model_best.pth'))
policy.load_state_dict(states)


insnames = os.listdir(TEST_INS)

fileList = [(os.path.join(TEST_INS,insname),None) for insname in insnames]

dataset = MIPDataset(fileList,TEST_BG,REORDER,ADDPOS)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
for step,batch in enumerate(data_loader):
    groupFeatures = batch['groupFeatures'][0].to(DEVICE)
    varFeatures = batch['varFeatures'][0].to(DEVICE)
    consFeatures = batch['consFeatures'][0].to(DEVICE)
    edgeFeatures = batch['edgeFeatures'][0].to(DEVICE)
    edgeInds = batch['edgeInds'][0].to(DEVICE)
    biInds = batch['biInds'][0].to(DEVICE)
    reorderInds = batch['reorderInds'][0].long().reshape(-1)

    with torch.no_grad():
        output = policy(
            consFeatures,
            edgeInds.long(),
            edgeFeatures,
            varFeatures,
            groupFeatures

        )
        output = output.sigmoid()[biInds][reorderInds]


    pickle.dump({
        'pre':output.cpu().numpy(),
        'reorderInds': reorderInds.cpu().numpy(),
        'biInds':biInds.cpu().numpy(),

    },open(os.path.join(exp_dir,f'{insnames[step]}.prob'),'wb'))


print('done')


