import os.path
import pickle
from multiprocessing import Process, Queue
import gurobipy
import gurobipy as gp
import numpy as np
import argparse




def solve(filepath,log_dir,settings):
    gp.setParam('LogToConsole', 0)
    m = gurobipy.read(filepath)

    m.Params.PoolSolutions = settings['maxsol']
    m.Params.PoolSearchMode = settings['mode']
    m.Params.TimeLimit = settings['maxtime']
    m.Params.Threads = settings['threads']
    log_path = os.path.join(log_dir, os.path.basename(filepath)+'.log')
    with open(log_path,'w'):
        pass

    m.Params.LogFile = log_path
    m.optimize()

    sols = []
    objs = []
    solc = m.getAttr('SolCount')
    oriVarNames = [var.varName for var in m.getVars()]
    intVarNames = [var.varName for var in m.getVars() if var.vtype==gurobipy.GRB.INTEGER or var.vtype==gurobipy.GRB.BINARY]
    intInds = [oriVarNames.index(varName) for varName in intVarNames ]
    for sn in range(solc):
        m.Params.SolutionNumber = sn
        sols.append(np.array(m.Xn))
        objs.append(m.PoolObjVal)
    sols = np.array(sols,dtype=int)[:,intInds]
    objs = np.array(objs,dtype=float)

    sol_data = {
        'intVarNames': intVarNames,
        'sols': sols,
        'objs': objs,
    }

    return sol_data



def collect(ins_dir,q,sol_dir,log_dir,settings):

    while True:
        filename = q.get()
        if not filename:
            break
        filepath = os.path.join(ins_dir,filename)
        # collect data

        sol_data = solve(filepath,log_dir,settings)

        # save data
        pickle.dump(sol_data, open(os.path.join(sol_dir, filename+'.sol'), 'wb'))
        print(f'processed {filename}, collected {len(sol_data["sols"])} solutions, {q.qsize()} instances left in queue.')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataDir', type=str, default='./datasets/IP/train',help='The training directory of the dataset')
    parser.add_argument('--nWorkers', type=int, default=5,help='number of processes to solve distinct instances in parallel')
    parser.add_argument('--maxTime', type=int, default=3600,help='time limit of the solving process')
    parser.add_argument('--maxStoredSol', type=int, default=10,help='max number of solutions to store')
    parser.add_argument('--threads', type=int, default=1,help='number of theads used to solve a single instance')


    args = parser.parse_args()

    dataDir = args.dataDir

    INS_DIR = os.path.join(dataDir,'ins')
    SOL_DIR = os.path.join(dataDir,'sol')
    LOG_DIR = os.path.join(dataDir,'logs')

    os.makedirs(SOL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
 

    N_WORKERS = args.nWorkers

    # gurobi settings
    SETTINGS = {
        'maxtime': args.maxTime,
        'mode': 2,
        'maxsol': args.maxStoredSol,
        'threads': args.threads,

    }

    filenames = os.listdir(INS_DIR)

    q = Queue()
    # add ins
    for filename in filenames:
        if not os.path.exists(os.path.join(SOL_DIR,filename+'.sol')):
            q.put(filename)
    # add stop signal
    for i in range(N_WORKERS):
        q.put(None)

    ps = []
    for i in range(N_WORKERS):
        p = Process(target=collect,args=(INS_DIR,q,SOL_DIR,LOG_DIR,SETTINGS))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()

    print('done')


