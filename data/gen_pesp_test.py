import os
import random

import numpy as np
import re
import pyscipopt as scip

nIns = 10 # number of instances
perc = 0.1 # perturbation factor
txtpath = './PESPLib/set-02/BL2.txt'
save_dir = './datasets/PESP/test'
os.makedirs(os.path.join(save_dir,'ins'),exist_ok=True)
os.makedirs(os.path.join(save_dir,'BLP'),exist_ok=True)
maxActNum = -1

with open(txtpath,'r') as f:
    lines = f.readlines()
act_info = []

for line in lines:
    if '#' in line:
        continue
    num_info = re.findall('\d+',line)
    num_info = [ int(num) for num in num_info]
    act_info.append(num_info)

act_info = np.array(act_info)

# sample small graph
act_info = act_info[0:maxActNum]
nAct = act_info.shape[0]
nEvent = act_info[:,1:3].max()

oriEvents = act_info[:,1]
tarEvents = act_info[:,2]


# perturb weights
for t in range(nIns):

    # interval, this value determines the size of the cyclic group of the instance, i.e., C_T
    T = 10 #random.choice([5,10,15])

    # reset intervals
    lowers = (act_info[:, 3] / (60 // T)).round()
    uppers = (act_info[:, 4] / (60 // T)).round()

    oriWeights = act_info[:, 5]

    weights = oriWeights + oriWeights*np.random.randn(nAct)*perc
    # to mip
    m = scip.Model()
    # add variables
    ps = np.array([ m.addVar(name=f'p_{i}',	vtype = 'I', ub=T-1) for i in range(nEvent)])

    # xs = np.array([  [ m.addVar(name=f'x_{i}_{j}', vtype = 'B') for j in range(T)] for i in range(nEvent)])
    zs = np.array( [ m.addVar(name=f'z_{j}', vtype = 'I',ub=2) for j in range(nAct)] )

    # add constraints

    # integer to binary


    # interval cons
    obj = 0
    for i in range(nAct):
        oriInd = oriEvents[i]-1
        tarInd = tarEvents[i]-1
        diff_expr = ps[tarInd] - ps[oriInd] + T*zs[i]

        m.addCons(diff_expr <= uppers[i])
        m.addCons(diff_expr >= lowers[i])

        objTerm = diff_expr - lowers[i]
        obj = obj + objTerm*weights[i]

    m.setObjective(obj)
    m.setMinimize()
    m.writeProblem(os.path.join(save_dir,'ins',f'pesp_{t}.mps'))





    # integer to binary
    xs = np.array([[m.addVar(name=f'x_{i}_{j}', vtype='B') for j in range(T)] for i in range(nEvent)])
    for i in range(nEvent):
        m.addCons(scip.quicksum(xs[i]) == 1)
        x_j_sum = xs[i, 0] * 0
        for j in range(1, T):
            x_j_sum = x_j_sum + xs[i, j] * j
        m.addCons(ps[i] == x_j_sum)
    m.writeProblem(os.path.join(save_dir, 'BLP', f'pesp_{t}.mps'))
print('done')