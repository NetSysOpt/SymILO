import numpy as np
import copy
from feature_extractor import extractFeature
import torch
from torch.utils.data import Dataset
import os
import pickle
from config import *

import argparse




class MIPDataset(Dataset):
    def __init__(self,files,bgdir,reorderFunc, addPosFuc):
        insPaths = [ filepaths[0] for filepaths in files]
        solPaths = [ filepaths[1] for filepaths in files]
        self.insPaths = insPaths
        self.solPaths = solPaths
        self.bgdir = bgdir
        self.reorder = reorderFunc
        self.addPos = addPosFuc
        os.makedirs(bgdir,exist_ok=True)

    def __getitem__(self, index):
        inspath = self.insPaths[index]
        solpath = self.solPaths[index]

        insname = os.path.basename(inspath)
        bgpath = os.path.join(self.bgdir,insname+'.bg')
        if os.path.exists(bgpath):
            data = pickle.load(open(bgpath,'rb'))
        else:
            inspath = inspath.replace('.gz','')
            features = extractFeature(inspath)
            features = self.addPos(features)
            varNames = np.array(features.varNames)[features.biInds]
            reorderData = self.reorder(varNames)
            data = {
                'groupFeatures':torch.Tensor(features.groupFeatures),
                'varFeatures': torch.Tensor(features.varFeatures),
                'consFeatures':torch.Tensor(features.consFeatures),
                'edgeFeatures':torch.Tensor(features.edgeFeatures),
                'edgeInds':torch.Tensor(features.edgeInds.astype(int)).permute(1,0),
                'biInds':torch.Tensor(features.biInds).long(),
                'nGroup':reorderData['nGroup'],
                'nElement':reorderData['nElement'],
                'reorderInds':torch.Tensor(reorderData['reorderInds'])
            }

            if self.solPaths[index] is not None:
                solData = pickle.load(open(solpath, 'rb'))
                sols = solData['sols']
                objs = solData['objs']
                varNames = solData['intVarNames']


                varIds = list(range(len(varNames)))
                varTuples = list(zip(varNames, varIds))
                varTuples.sort(key=lambda t: t[0])
                order = [t[-1] for t in varTuples]

                sols = sols[:,order]


                data['sols'] = torch.Tensor(sols[0])
                data['objs'] = torch.Tensor([objs[0]])

            pickle.dump(data,open(bgpath,'wb'))


        return data

    def __len__(self):
        return len(self.insPaths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SMSP')
    args = parser.parse_args()
    info = confInfo[args.dataset]
    ADDPOS = info['addPosFeature']
    REORDER = info['reorder']
    fileDir = os.path.join(info['trainDir'], 'ins')
    solDir = os.path.join(info['trainDir'], 'sol')
    bgDir = os.path.join(info['trainDir'], 'bg')
    solnames = os.listdir(solDir)
    filepaths = [os.path.join(fileDir, solname.replace('.sol', '')) for solname in solnames]
    solpaths = [os.path.join(solDir, solname) for solname in solnames]
    dataset = MIPDataset(list(zip(filepaths,solpaths)),bgDir,REORDER,ADDPOS)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    print('Start constructing bipartite graph ...')
    for step,data in enumerate(data_loader):
        varFeatures = data['varFeatures']
        consFeatures = data['consFeatures']
        edgeFeatures = data['edgeFeatures']
        edgeInds = data['edgeInds']
        sols = data['sols']
        objs = data['objs']
        reorderInds = data['reorderInds']

        print(f'Processed {step}/{len(data_loader)}')
    print('Bipartite graph construction finished!')