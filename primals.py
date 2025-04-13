import os.path
import re

import numpy as np


def getPrimals(exp_dir):
    instance_files = os.listdir(exp_dir)
    instance_files = [ins for ins in instance_files if '.log' in ins]
    primals = []
    finalTimes = []
    for filename in instance_files:
        if '.log' not in filename and '.txt' not in filename:
            continue
        filepath = os.path.join(exp_dir,filename)

        with open(filepath,'r') as f:

            lines = f.readlines()
            last_gap = None
            pb = 10000000
            ft = 0
            for line in lines:
                if 'Found incumbent of value ' in line:
                    ss = re.findall('\d+.\d+',line)
                    pb = float(ss[0])
                    ti = float(ss[1])
                if '(root+branch&cut)' in line:
                    ss = re.findall('\d+.\d+', line)
                    ft = float(ss[0])
            primals.append(pb)
            finalTimes.append(ft)


    mean_pb = np.array(primals).mean() if len(primals)>0 else 100000000
    mean_solTime = np.array(finalTimes).mean()

    return mean_pb,mean_solTime



if __name__ == '__main__':
    exp_dir = r'F:\L2O_project\ICML2024\src\logs\SMSP_ori\exp-cplex-method-node_selection-perc-0-radius-1-Mt1000-2024-01-16 14-16-59'
    mean_pb,mean_solTime= getPrimals(exp_dir)
    with open(os.path.join(exp_dir, 'mean_primals.txt'), 'w+') as f:
        f.write(f'Primal Bounds:{mean_pb}\n')
        f.write(f'Solve Time:{mean_solTime}\n')
