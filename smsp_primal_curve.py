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

    DEFAULT_BOUND = 30
    ORD_IND = 0
    TOP_GAP = 2
    MAXTIME = 1005
    EPS = 0
    BIAS = 1771
    logTimes, defaultPrimals = getDirPrimals(r'F:\L2O_project\Neurips2023\exps\res\comparison\SMSP\default-exp-gurobi-perc-0.0-radius-1-Mt3600-2023-05-16 14-50-16', maxTime=3600, interval=1, defaultBound=DEFAULT_BOUND, orderInd=ORD_IND)
    logTimes, PnSPrimals = getDirPrimals(r'F:\L2O_project\Neurips2023\exps\res\comparison\SMSP\PnS_GRB_fix10_eps20', maxTime=MAXTIME, interval=1,
                                         defaultBound=DEFAULT_BOUND, orderInd=ORD_IND)

    _, LEXPrimals = getDirPrimals(r'F:\L2O_project\Neurips2023\exps\res\comparison\SMSP\Lex-exp-gurobi-perc-0.2-radius-5-Mt1000-2023-05-16 13-56-38', maxTime=MAXTIME, interval=1, defaultBound=DEFAULT_BOUND, orderInd=ORD_IND)

    _, OursPrimals = getDirPrimals(r'F:\L2O_project\Neurips2023\exps\res\comparison\SMSP\Opt-exp-gurobi-perc-0.5-radius-0-Mt1000-2023-05-16 13-39-28', maxTime=MAXTIME, interval=1, defaultBound=DEFAULT_BOUND,
                                   orderInd=ORD_IND)

    # bias
    defaultPrimals = defaultPrimals - BIAS
    PnSPrimals = PnSPrimals - BIAS
    LEXPrimals = LEXPrimals - BIAS
    OursPrimals = OursPrimals - BIAS



    totalPrimals = np.stack([defaultPrimals[:, -1], PnSPrimals[:, -1], LEXPrimals[:, -1], OursPrimals[:, -1]], axis=1)
    # totalPrimals = np.stack(
        # [defaultGRBPrimals[:, -1],divingScipPrimals[:,-1], divingGRBPrimals[:, -1]],
        # axis=1)

    bestPrimals = totalPrimals.min(axis=1)
    bestPrimals = np.stack([defaultPrimals[:, -1], bestPrimals]).min(0)
    defaultPrimals = defaultPrimals[:, 0:MAXTIME]

    defaultGaps = abs(defaultPrimals - bestPrimals[:, np.newaxis]) / (abs(bestPrimals[:, np.newaxis]) + EPS)
    PnSPrimalGaps = abs(PnSPrimals[:, 0:MAXTIME] - bestPrimals[:, np.newaxis]) / (abs(bestPrimals[:, np.newaxis]) + EPS)
    LEXPrimalGaps = abs(LEXPrimals - bestPrimals[:, np.newaxis]) / (abs(bestPrimals[:, np.newaxis]) + EPS)
    OursPrimalGaps = abs(OursPrimals - bestPrimals[:, np.newaxis]) / (abs(bestPrimals[:, np.newaxis]) + EPS)

    logTime = logTimes
    # logTime = np.log10(logTimes)
    logTime[0] = 0


    # remove first 10
    # logTime = logTime[10:]
    defaultGaps = defaultGaps.mean(axis=0)
    PnSPrimalGaps = PnSPrimalGaps.mean(axis=0)
    LEXPrimalGaps = LEXPrimalGaps.mean(axis=0)
    OursPrimalGaps = OursPrimalGaps.mean(axis=0)


    fig = plt.figure()
    plt.plot(logTime[defaultGaps < TOP_GAP], defaultGaps[defaultGaps < TOP_GAP], label='Gurobi')
    plt.plot(logTime[PnSPrimalGaps < TOP_GAP], PnSPrimalGaps[PnSPrimalGaps < TOP_GAP], label='PS')
    plt.plot(logTime[LEXPrimalGaps < TOP_GAP], LEXPrimalGaps[LEXPrimalGaps < TOP_GAP], label='LEX')
    plt.plot(logTime[OursPrimalGaps < TOP_GAP], OursPrimalGaps[OursPrimalGaps < TOP_GAP], label='Ours')
    plt.xlabel('time(s)',fontsize=16)
    plt.ylabel('Average primal gap',fontsize=16)
    # plt.xticks([1,2,3],['$10^1$','$10^2$','$10^3$'])
    # plt.yticks([0,0.2,0.4,0.6,0.8,1.0],['0%','20%','40%','60%','80%','100%'])
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.grid(axis='x')
    plt.title('SMSP',size=20)
    plt.legend(fontsize=16)
    plt.savefig('SMSP_primal_gap.pdf',bbox_inches='tight',dpi=fig.dpi,pad_inches=0.1)
    plt.show()

    io.savemat('avg_ord.mat',{
        'time':logTime[0:MAXTIME],
        'scip': defaultGaps[0:MAXTIME],
        'gurobi': PnSPrimalGaps[0:MAXTIME],
        'divingScip': LEXPrimalGaps[0:MAXTIME],
        'divingGRB': OursPrimalGaps[0:MAXTIME],
        # 'scipX':(defaultScipGaps/divingScipGaps).mean(),
        'gurobiX':(PnSPrimalGaps / OursPrimalGaps).mean()
    })

    print('done')

