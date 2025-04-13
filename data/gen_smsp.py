import os
import random

import numpy
import numpy as np
import pyscipopt
import re


random.seed(0)
np.random.seed(0)


def genSMS(filepath):

    with open(filepath,'r') as f:
        lines = f.readlines()

    capInfo = re.findall('\d+',lines[0])
    nCap = int(capInfo[0]) + 1
    caps = [int(ca) for ca in capInfo]
    caps[0] = 0
    nColors = int(re.findall('\d+',lines[1])[0])
    nOrders = int(re.findall('\d+',lines[2])[0])


    sizes = []
    colors = []

    nSlab = nOrders
    for line in lines[3:]:
        orderInfo = re.findall('\d+', line)
        size,color = int(orderInfo[0]),int(orderInfo[1])
        sizes.append(size)
        colors.append(color)



    colorInds = [c-1 for c in colors]
    sizes = np.array(sizes)
    colorInds = np.array(colorInds)
    caps = np.array(caps)
    model = pyscipopt.Model()
    # set variables
    X = np.zeros((nOrders,nSlab)).astype(object)
    Y = np.zeros((nSlab,nCap)).astype(object)
    Z = np.zeros((nColors,nSlab)).astype(object)

    for o in range(nOrders):
        for s in range(nSlab):
            x_os = model.addVar(vtype='BINARY', name=f'X_{o}_{s}')
            X[o,s] = x_os
    for s in range(nSlab):
        for q in range(nCap):
            y_sq = model.addVar(vtype='BINARY', name=f'Y_{s}_{q}')
            Y[s,q] = y_sq
    for c in range(nColors):
        for s in range(nSlab):
            z_cs = model.addVar(vtype='BINARY', name=f'Z_{c}_{s}')
            Z[c,s] = z_cs

    sums = []
    # add assignment constraints
    for o in range(nOrders):
        model.addCons(pyscipopt.quicksum(X[o, :]) == 1, name=f'order_assign_{s}')
    for s in range(nSlab):
        model.addCons(pyscipopt.quicksum(Y[s,:]) == 1, name=f'slab_size_assign_{s}')
        # capacity constraints
        ws = pyscipopt.quicksum(sizes*X[:,s])
        qy = pyscipopt.quicksum(Y[s,:]*caps)
        model.addCons(ws<=qy, name=f'capacity_constraint_{s}')

        # add color constraints
        for o in range(nOrders):
            c_o = colorInds[o]
            model.addCons(X[o,s]<=Z[c_o,s], f'color_{o}_{s}')
        for c in range(nColors):
            model.addCons(pyscipopt.quicksum(Z[:,s])<=2, name=f'2_color_constraint_{s}')

        # set objective
        sums.append(qy)

    # set objective
    model.setMinimize()
    model.setObjective(pyscipopt.quicksum(sums)-1771)


    return model




if __name__ == '__main__':

    DATADIR = r'./steel'
    SAVEDIR = r'./datasets/SMSP/ins'
    os.makedirs(SAVEDIR,exist_ok=True)

    filenames = os.listdir(DATADIR)

    for step,filename in enumerate(filenames):

        filepath = os.path.join(DATADIR,filename)
        fileId = int(re.findall('\d+',filename)[1])
        trainOrTest = 'train' if fileId<16 else 'test'
        saveDir = os.path.join(SAVEDIR,trainOrTest)
        os.makedirs(saveDir,exist_ok=True)
        savepath = os.path.join(saveDir,filename+'.mps')
        m = genSMS(filepath)
        m.writeProblem(savepath)
        print(f'processed {step+1}/{len(filenames)}')



    print('done')