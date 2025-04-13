import os
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io


def getPrimals(filepath):

    with open(filepath,'r') as f:
        lines = f.readlines()

    times = []
    primal_bounds = []
    last_gap = None
    last_primal = None

    optim = False
    for line in lines:

        if 'Optimal solution found at' in line:
            optim = True

        if '|' in line and 'time' not in line and (
            'Node' not in line
        )and not optim:


            data = line.split('|')


            time = re.findall('\d+.\d+', data[0])[0]
            # n_node = re.findall('\d+', data[1])[0]

            dual_bound = re.findall('\d+.\d+e[+-]\d+', data[14])
            primal_bound = re.findall('\d+.\d+e[+-]\d+', data[15])
            gap = re.findall('\d+.\d+', data[16])

            # dual_bound = None if len(dual_bound) == 0 else dual_bound[0]
            primal_bound = None if len(primal_bound) == 0 else primal_bound[0]
            gap = None if len(gap) == 0 else gap[0]

            if last_gap != gap:
                times.append(float(time))
                # n_nodes.append(int(n_node))
                # dual_bounds.append(float(dual_bound))
                primal_bounds.append(float(primal_bound))
                last_gap = gap
                # gaps.append(float(gap))
        elif len(re.findall('\d+s\n',line))>0 and 'time' not in line and '%' in line and not optim :
            data = line.split(' ')
            data = [d for d in data if d != '']
            time = re.findall('\d+', data[-1])[0]

            # dual_bound = re.findall('\d+.\d+e[+-]\d+', data[14])
            primal_bound = re.findall('\d+.\d+', data[-5])

            gap = re.findall('\d+.\d+', data[-3])

            # dual_bound = None if len(dual_bound) == 0 else dual_bound[0]
            primal_bound = None if len(primal_bound) == 0 else primal_bound[0]
            gap = None if len(gap) == 0 else gap[0]

            if last_primal != primal_bound:
                times.append(float(time))
                # n_nodes.append(int(n_node))
                # dual_bounds.append(float(dual_bound))
                primal_bounds.append(float(primal_bound))
                # gaps.append(float(gap))
                last_primal = primal_bound

    return times,primal_bounds



def formatPrimal(times,bounds,maxTime=1000,interval=1,defaultBound = 10):
    new_times = [i*interval for i in range(maxTime//interval)]
    new_bounds = [None for _ in range(maxTime//interval)]

    for i in range(len(times)):
        t = times[i]
        b = bounds[i]
        if t>maxTime:
            break
        ind = int((t-0.00001)/interval)
        new_bounds[ind] = b

    tb = defaultBound
    for i in range(len(new_bounds)):
        b = new_bounds[i]
        if b is None:
            new_bounds[i] = tb
        else:
            tb = b

    return new_times,new_bounds


def getDirPrimals(logDir, maxTime=1050, interval=1, defaultBound=1500,orderInd=None):

    filenames = os.listdir(logDir)
    filenames = [filename for filename in filenames if 'primal' not in filename]
    filenames = sorted(filenames, key=lambda x: int(re.findall('\d+', x)[orderInd]))

    totalPrimals = []
    for filename in filenames:
        filepath = os.path.join(logDir, filename)
        times, primal_bounds = getPrimals(filepath)
        new_times, new_bounds = formatPrimal(times, primal_bounds, maxTime, interval, defaultBound)
        totalPrimals.append(new_bounds)

    totalPrimals = np.array(totalPrimals)
    new_times = np.array(new_times)
    return new_times,totalPrimals

if __name__ == '__main__':

    DEFAULT_BOUND = 2000
    ORD_IND = 0
    TOP_GAP = 1
    MAXTIME = 1005
    EPS = 0
    BIAS = 1771
    # logTimes, defaultScipPrimals = getDirPrimals(r'F:\L2O_project\explore0914\smothness_opt_steel\steel_exp_0108_fix0\exp-gurobi-UT1.999-LT-0.003-Mt100-2023-01-08 17-09-11', maxTime=MAXTIME, interval=1, defaultBound=DEFAULT_BOUND, orderInd=ORD_IND)
    save_dir = r'F:\L2O_project\Neurips2023\exps\res\ablation\SMSP'
    logTimes, default = getDirPrimals(os.path.join(save_dir,'default','logs'), maxTime=3600, interval=1,
                                      defaultBound=DEFAULT_BOUND, orderInd=ORD_IND)

    _, BASIC = getDirPrimals(os.path.join(save_dir,'BASIC','logs'), maxTime=MAXTIME, interval=1, defaultBound=DEFAULT_BOUND, orderInd=ORD_IND)

    _, BASIC_LEX = getDirPrimals(os.path.join(save_dir,'BASIC_LEX','logs'), maxTime=MAXTIME, interval=1, defaultBound=DEFAULT_BOUND,
                                 orderInd=ORD_IND)

    _, BASIC_PE = getDirPrimals(
        os.path.join(save_dir,'BASIC_PE','logs'),
        maxTime=MAXTIME, interval=1, defaultBound=DEFAULT_BOUND,
        orderInd=ORD_IND)
    _, BASIC_SAL = getDirPrimals(
        os.path.join(save_dir,'BASIC_SAL','logs'),
        maxTime=MAXTIME, interval=1, defaultBound=DEFAULT_BOUND,
        orderInd=ORD_IND)

    _, PE_LEX = getDirPrimals(
        os.path.join(save_dir,'PE_LEX','logs'),
        maxTime=MAXTIME, interval=1, defaultBound=DEFAULT_BOUND,
        orderInd=ORD_IND)

    _, PE_SAL = getDirPrimals(
        os.path.join(save_dir,'PE_SAL','logs'),
        maxTime=MAXTIME, interval=1, defaultBound=DEFAULT_BOUND,
        orderInd=ORD_IND)

    # bias
    default = default - BIAS
    BASIC = BASIC - BIAS
    BASIC_LEX = BASIC_LEX - BIAS
    BASIC_PE = BASIC_PE - BIAS
    BASIC_SAL = BASIC_SAL - BIAS
    PE_LEX = PE_LEX - BIAS
    PE_SAL = PE_SAL - BIAS



    totalPrimals = np.stack(
        [default[:, -1], BASIC[:, -1], BASIC_LEX[:, -1],BASIC_PE[:,-1],BASIC_SAL[:,-1],PE_LEX[:,-1],PE_SAL[:,-1]],
        axis=1)
    bestPrimals = totalPrimals.min(axis=1)
    bestPrimals = np.stack([default[:, -1], bestPrimals]).min(0)
    default = default[:, 0:MAXTIME]

    # defaultScipGaps = abs(defaultScipPrimals - bestPrimals[:, np.newaxis]) / abs(bestPrimals[:, np.newaxis])
    default_Gaps = abs(default[:, 0:MAXTIME] - bestPrimals[:, np.newaxis]) / abs(bestPrimals[:, np.newaxis])
    BASIC_Gaps = abs(BASIC - bestPrimals[:, np.newaxis]) / abs(bestPrimals[:, np.newaxis])
    BASIC_LEX_Gaps = abs(BASIC_LEX - bestPrimals[:, np.newaxis]) / abs(bestPrimals[:, np.newaxis])
    BASIC_PE_Gaps = abs(BASIC_PE - bestPrimals[:, np.newaxis]) / abs(bestPrimals[:, np.newaxis])
    BASIC_SAL_Gaps = abs(BASIC_SAL - bestPrimals[:, np.newaxis]) / abs(bestPrimals[:, np.newaxis])
    PE_LEX_Gaps = abs(PE_LEX - bestPrimals[:, np.newaxis]) / abs(bestPrimals[:, np.newaxis])
    PE_SAL_Gaps = abs(PE_SAL - bestPrimals[:, np.newaxis]) / abs(bestPrimals[:, np.newaxis])

    logTime = logTimes[0:MAXTIME]
    # logTime = np.log10(logTimes)
    logTime[0] = 0


    # remove first 10
    # logTime = logTime[10:]
    # defaultScipGaps = defaultScipGaps[:,10:].mean(axis=0)
    default_Gaps = default_Gaps.mean(axis=0)
    BASIC_Gaps = BASIC_Gaps.mean(axis=0)
    BASIC_LEX_Gaps = BASIC_LEX_Gaps.mean(axis=0)
    BASIC_PE_Gaps = BASIC_PE_Gaps.mean(axis=0)
    BASIC_SAL_Gaps = BASIC_SAL_Gaps.mean(axis=0)
    PE_LEX_Gaps = PE_LEX_Gaps.mean(axis=0)
    PE_SAL_Gaps = PE_SAL_Gaps.mean(axis=0)



    # plt.plot(logTime[defaultScipGaps<TOP_GAP],defaultScipGaps[defaultScipGaps<TOP_GAP],label='default SCIP')
    fig = plt.figure()
    plt.plot(logTime[BASIC_Gaps < TOP_GAP], BASIC_Gaps[BASIC_Gaps < TOP_GAP], label='BASE')
    plt.plot(logTime[BASIC_LEX_Gaps < TOP_GAP], BASIC_LEX_Gaps[BASIC_LEX_Gaps < TOP_GAP], label='BASE+LEX')
    plt.plot(logTime[BASIC_PE_Gaps < TOP_GAP], BASIC_PE_Gaps[BASIC_PE_Gaps < TOP_GAP], label='BASE+PE')
    plt.plot(logTime[BASIC_SAL_Gaps<TOP_GAP],BASIC_SAL_Gaps[BASIC_SAL_Gaps<TOP_GAP],label='BASE+SAL')
    plt.plot(logTime[PE_LEX_Gaps<TOP_GAP], PE_LEX_Gaps[PE_LEX_Gaps<TOP_GAP],label='BASE+PE+LEX')
    plt.plot(logTime[PE_SAL_Gaps<TOP_GAP], PE_SAL_Gaps[PE_SAL_Gaps<TOP_GAP],label='BASE+PE+SAL')
    plt.xlabel('time(s)',fontsize=16)
    plt.ylabel('Average primal gap',fontsize=16)
    plt.xticks(size=16)
    plt.yticks(size=16)
    # plt.xticks([1,2,3],['$10^1$','$10^2$','$10^3$'])
    # plt.yticks([0,0.2,0.4,0.6,0.8,1.0],['0%','20%','40%','60%','80%','100%'])
    plt.grid(axis='x')
    plt.title('SMSP',size=20)
    plt.legend(fontsize=16)
    plt.savefig('SMSP ablation.pdf',bbox_inches='tight',dpi=fig.dpi,pad_inches=0.1)
    plt.show()

    # io.savemat('avg_ord.mat',{
    #     'time':logTime[0:MAXTIME],
    #     # 'scip':defaultScipGaps[0:MAXTIME],
    #     'gurobi':defaultGRBGaps[0:MAXTIME],
    #     'divingScip':divingScipGaps[0:MAXTIME],
    #     'divingGRB':divingGRBGaps[0:MAXTIME],
    #     # 'scipX':(defaultScipGaps/divingScipGaps).mean(),
    #     'gurobiX':(defaultGRBGaps/divingGRBGaps).mean()
    # })

    print('done')

