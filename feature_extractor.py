import os
import pickle

import pyscipopt
import numpy as np

class Features:
    pass



# def extractFeature(inspath):
#     basename = os.path.basename((inspath))
#     nbppath = os.path.join(r'F:\L2O_project\ICML2023\exps\data\ip\NBP',basename.replace('.mps','.mps.gz.nbp'))
#     data = pickle.load(open(nbppath,'rb'))
#     features = Features()
#
#
#     features.varFeatures = data['NBP']['variable_features']
#     features.consFeatures = data['NBP']['constraint_features']
#     features.edgeFeatures = data['NBP']['edge_features']
#     features.edgeInds = data['NBP']['edge_indices']
#
#     features.varNames = data['varNames']
#     features.consNames = ''
#     return features

#
def extractFeature(inspath):
    m = pyscipopt.Model()
    m.hideOutput(True)
    m.readProblem(inspath)

    vars = m.getVars()
    vars.sort(key=lambda var: var.name)

    vtypes = []
    biInds = []
    varUbs = []
    varLbs = []
    varBoundTypes = []
    varNames = [var.name for var in vars]
    varNameMap = {}
    for ind,var in enumerate(vars):
        varNameMap[var.name] = ind
        vtype = var.vtype()
        vtypes.append(0 if vtype=='CONTINUOUS' else 1 if vtype=='BINARY' else 2)
        if vtype=='BINARY':
            biInds.append(ind)
        # check bounds
        boundType = 0
        ub = var.getUbOriginal()
        lb = var.getLbOriginal()
        if ub>=1e+20:
            ub = 0
            boundType+=1
        if lb<=-1e+20:
            lb = 0
            boundType-=1

        varUbs.append(ub)
        varLbs.append(lb)
        varBoundTypes.append(boundType)
    objs = m.getObjective()
    objCoeffs = [0]*len(varNames)
    for e in objs:
        varname = e.vartuple[0].name
        objCoeffs[varNameMap[varname]] = objs[e]


    conss = m.getConss()
    conss.sort(key=lambda cons: cons.name)
    consNames = [ cons.name for cons in conss]
    rhsCoeffs = [m.getRhs(cons) for cons in conss]
    lhsCoeffs = [m.getLhs(cons) for cons in conss]

    consType = [ 0 if r==l else 1 if r<1e+20 else -1  for r,l in zip(rhsCoeffs,lhsCoeffs)]
    rlhsCoeffs = [r if abs(r)<abs(l) else l for r,l in zip(rhsCoeffs,lhsCoeffs) ]
    varDegree = [0]*len(varNames)
    consDegree = [0]*len(conss)
    edgeWeights = []
    edges = []
    for consInd, cons in enumerate(conss):
        coeffs = list(m.getValsLinear(cons).items())
        for coeff in coeffs:
            v = coeff[1]
            if v==0:
                continue
            vInd = varNameMap[coeff[0]]
            edges.append((consInd,vInd))
            edgeWeights.append(v)
            varDegree[vInd] += 1
            consDegree[consInd] += 1
    # varFeatures = np.stack(
    #     [np.array(vtypes)], axis=-1)
    varFeatures = np.stack([np.array(vtypes), np.array(objCoeffs), np.array(varDegree), np.array(varLbs), np.array(varUbs), np.array(varBoundTypes)],axis=-1)
    consFeatures = np.stack([np.array(consType), np.array(rlhsCoeffs),np.array(consDegree)],axis=-1)
    edgeFeatures = np.array(edgeWeights)[:,np.newaxis]
    edgeInds = np.array(edges)


    features = Features()
    features.varFeatures = varFeatures
    features.consFeatures = consFeatures
    features.edgeFeatures = edgeFeatures
    features.edgeInds = edgeInds

    features.varNames = varNames
    features.consNames = consNames
    features.biInds = biInds

    return features


if __name__ == '__main__':
    inspath = r'example.lp'
    data = extractFeature(inspath)
    print('done')