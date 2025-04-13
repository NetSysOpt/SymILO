import numpy as np
import torch
import re
import math

def PF(d,D,j):

    if j//2 == 0:
        pe = np.sin(j/20**(2*d/D))
    else:
        pe = np.cos(j/20**(2*d/D))

    return pe

def reorderExample(names):

    XrInds = [int(re.findall('\d+',name)[0]) for name in names]
    XcInds = [int(re.findall('\d+',name)[1]) for name in names]

    XOrder = torch.zeros(max(XrInds),max(XcInds))

    for ind,name in enumerate(names):
        ss = re.findall('\d+',name)
        a,b = int(ss[0])-1,int(ss[1])-1
        XOrder[a,b] = ind

    return XOrder.reshape(-1)



def reorderSMSP(names):

    nItem = max([int(re.findall('\d+', name)[0]) for name in names if 'X' in name])+1
    nCap = max([int(re.findall('\d+', name)[1]) for name in names if 'Y' in name])+1

    XOrder = torch.LongTensor(nItem+nCap,nItem)

    for ind,name in enumerate(names):
        ss = re.findall('\d+',name)
        a,b = int(ss[0]),int(ss[1])
        if 'X' in name:
            XOrder[a,b] = ind
        elif 'Y' in name:
            XOrder[b+nItem,a] = ind



    return {
        'reorderInds':XOrder,
        'nGroup':nItem,
        'nElement':nItem+nCap
    }

#
# def addPosFeatureSMSP(features):
#
#     vf = features.varFeatures
#     vn = features.varNames
#     groupPos = np.zeros((vf.shape[0],111))
#     elementPos = np.zeros((vf.shape[0],111+21))
#     for ind,var in enumerate(vn):
#         ss = re.findall('\d+',var)
#         a,b = int(ss[0]),int(ss[1])
#         groupPos[ind,b] = 1
#         if 'X' in var:
#             elementPos[ind,a] = 1
#         elif 'Y' in var:
#             elementPos[ind,a+111] = 1
#
#     features.groupFeatures = np.concatenate([ groupPos, elementPos], axis=-1)
#
#
#     return features

def addPosFeatureSMSP(features):

    vf = features.varFeatures
    vn = features.varNames
    D1 = 16
    D2 = 16
    groupPos = np.zeros((vf.shape[0],D1))
    elementPos = np.zeros((vf.shape[0],D2))
    for ind,var in enumerate(vn):
        ss = re.findall('\d+',var)
        a,b = int(ss[0]),int(ss[1])
        for d in range(D1):
            groupPos[ind, d] = PF(d, D1, b)
        if 'X' in var:
            for d in range(D2):
                elementPos[ind,d] = PF(d, D2, a)
        elif 'Y' in var:
            for d in range(D2):
                elementPos[ind,d] = PF(d, D2, a+111)

    features.groupFeatures = np.concatenate([ groupPos, elementPos], axis=-1)


    return features

def reorderIP(names):
    nItem = max([int(re.findall('\d+', name)[0]) for name in names if 'place' in name]) + 1
    nBin = max([int(re.findall('\d+', name)[1]) for name in names if 'place' in name]) + 1

    XOrder = torch.Tensor(nItem , nBin)

    for ind, name in enumerate(names):
        ss = re.findall('\d+', name)
        a, b = int(ss[0]), int(ss[1])
        if 'place' in name:
            XOrder[a, b] = ind

    return {
        'reorderInds': XOrder,
        'nGroup': nBin,
        'nElement': nItem
    }

def generatePosVector(pos1d,n,d):
    pos = np.zeros(n)
    i = int(pos1d*n*d)
    nPos = i//d
    dPos = i-nPos*d
    v = dPos/d + 1/d
    pos[nPos] = v
    return pos


# def addPosFeatureIP_ori(features):
#     vf = features.varFeatures
#     vn = features.varNames
#
#     nItem = max([int(re.findall('\d+', name)[0]) for name in vn if 'place' in name]) + 1
#     nBin = max([int(re.findall('\d+', name)[1]) for name in vn if 'place' in name]) + 1
#
#
#     groupPos = np.zeros((vf.shape[0], nBin))
#     elementPos = np.zeros((vf.shape[0], nItem))
#     for ind, var in enumerate(vn):
#         if 'place' not in var:
#             continue
#         ss = re.findall('\d+', var)
#         a, b = int(ss[0]), int(ss[1])
#         groupPos[ind,b] = 1
#         elementPos[ind,a] = 1
#     features.groupFeatures = np.concatenate([ groupPos, elementPos], axis=-1)
#     # features.varFeatures = np.concatenate([groupPos, elementPos], axis=-1)
#
#     return features


def addPosFeatureIP(features):
    vf = features.varFeatures
    vn = features.varNames

    nItem = max([int(re.findall('\d+', name)[0]) for name in vn if 'place' in name]) + 1
    nBin = max([int(re.findall('\d+', name)[1]) for name in vn if 'place' in name]) + 1

    D1 = 16
    D2 = 16
    groupPos = np.zeros((vf.shape[0], D1))
    elementPos = np.zeros((vf.shape[0], D2))
    for ind, var in enumerate(vn):
        if 'place' not in var:
            continue
        ss = re.findall('\d+', var)
        a, b = int(ss[0]), int(ss[1])
        for d in range(D1):
            groupPos[ind,d] = PF(d,D1,b)
        for d in range(D2):
            elementPos[ind,d] = PF(d,D2,a)
    features.groupFeatures = np.concatenate([ groupPos, elementPos], axis=-1)
    # features.varFeatures = np.concatenate([groupPos, elementPos], axis=-1)

    return features

def reorderAP(names):
    nItem = max([int(re.findall('\d+', name)[0]) for name in names if 'x' in name]) + 1
    nBin = max([int(re.findall('\d+', name)[1]) for name in names if 'x' in name]) + 1

    XOrder = torch.Tensor(nItem , nBin-2)

    for ind, name in enumerate(names):
        ss = re.findall('\d+', name)
        a, b = int(ss[0]), int(ss[1])
        if 'x' in name and b<nBin-2:
            XOrder[a, b] = ind

    return {
        'reorderInds': XOrder,
        'nGroup': nBin-2,
        'nElement': nItem
    }

def addPosFeatureAP(features):
    vf = features.varFeatures
    vn = features.varNames

    nItem = max([int(re.findall('\d+', name)[0]) for name in vn if 'x' in name]) + 1
    nBin = max([int(re.findall('\d+', name)[1]) for name in vn if 'x' in name]) + 1
    nDim = max([int(re.findall('\d+', name)[0]) for name in vn if 'y' in name]) + 1


    groupPos = np.zeros((vf.shape[0], nBin))
    elementPos = np.zeros((vf.shape[0], nItem))
    for ind, var in enumerate(vn):
        if 'x' not in var:
            continue
        ss = re.findall('\d+', var)
        a, b = int(ss[0]), int(ss[1])
        groupPos[ind,b] = 1
        elementPos[ind,a] = 1
    features.groupFeatures = np.concatenate([ groupPos, elementPos], axis=-1)
    # features.varFeatures = np.concatenate([groupPos, elementPos], axis=-1)

    return features

if __name__ == '__main__':

    x = torch.Tensor([1,5,6,7,8,2,3,4,9])
    names = ['X_1_1','X_2_2','X_2_3','X_3_1','X_3_2','X_1_2','X_1_3','X_2_1','X_3_3']

    reorderExample(names)
    print('done')