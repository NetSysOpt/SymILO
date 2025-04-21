import os
import re

import numpy as np
import scipy.io


def parseLog(logpath):

    lines = []
    with open(logpath,'r') as f:
        lines = f.readlines()
    primals = []
    ts = []
    for line in lines:

        if 'Found incumbent of value' in line:
            ss = re.findall('\d+.\d+',line)
            primal,t = float(ss[0]),float(ss[1])
            t = int(t+0.5)

            if t>=len(primals) and len(primals)>0:
                for i in range(t - len(primals)-1):
                    primals.append(primals[-1])

            primals.append(primal)

    return primals

dirs = os.listdir('.')

for di in dirs:
    if not os.path.isdir(di) or di[0]=='.':
        continue

    fileDirs = os.listdir(di)

    for fileDir in fileDirs:
        file_list = os.listdir(os.path.join(di,fileDir))
        allPrimals = []

        for filename in file_list:
            if '.log' not in filename:
                continue
            logpath = os.path.join(di,fileDir,filename)
            primals = parseLog(logpath)
            allPrimals.append(primals)

        # align
        maxT = max([len(prims) for prims in allPrimals])
        maxT = 3600 if maxT>700 else 600
        for ind,prim in enumerate(allPrimals):
            if len(prim)<maxT:
                prim += [prim[-1]]*(maxT-len(prim))
                allPrimals[ind] = np.array(prim)
            elif len(prim)>maxT:
                allPrimals[ind] = np.array(prim[0:maxT])

        allPrimals = np.array(allPrimals)

        scipy.io.savemat(os.path.join(di,fileDir,'primals.mat'),{
            'primals':allPrimals
        })

        print('done')