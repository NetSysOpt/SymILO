
import pickle
import cplex
from config import *


def evaluate(filepath,prepath,exp_dir,timelimit,method,perc):

    # parameters
    m = cplex.Cplex(filepath)
    logfile = os.path.join(exp_dir,os.path.basename(filepath)+'.log')
    logstring = open(logfile, 'w')
    m.set_log_stream(logstring)
    m.set_results_stream(logstring)
    m.set_warning_stream(logstring)
    # instance_cpx.set_error_stream(logstring)
    m.set_error_stream(open(os.devnull, 'w'))
    """ Set CPLEX parameters, if any """
    m.parameters.timelimit.set(timelimit)
    m.parameters.emphasis.mip.set(1)
    m.parameters.mip.display.set(3)
    m.parameters.threads.set(1)
    m.parameters.workmem.set(1024)
    m.parameters.mip.limits.treememory.set(20000)
    m.parameters.mip.strategy.file.set(2)
    m.parameters.workdir.set('./temp')

    # read prediction
    data = pickle.load(open(prepath,'rb'))
    pre = data['pre']
    reorderInds = data['reorderInds']
    biInds = data['biInds']

    var_names = m.variables.get_names()
    var_ind_pairs = [ [step,varname] for step,varname in enumerate(var_names)]
    var_ind_pairs.sort(key=lambda x:x[1])

    var_ind_pairs = np.array(var_ind_pairs)[biInds][reorderInds].tolist()

    var_ind_pre_tuples = [ [var_ind_pairs[i][0],var_ind_pairs[i][1],pre[i]] for i in range(len(var_ind_pairs))]

    var_ind_pre_tuples.sort(key=lambda x: abs(0.5-x[2]), reverse=True )



    varInds = [ var_ind_pre_tuples[i][1] for i in range(int(len(var_ind_pairs)*perc))]
    coefs = np.array([1 if var_ind_pre_tuples[i][2]<0.5 else -1 for i in range(int(len(var_ind_pairs) * perc))])
    rhs = (coefs<0).sum()

    rad = 0
    if('PS' in method):
        pres = [var_ind_pre_tuples[i][2] if var_ind_pre_tuples[i][2]<0.5 else 1-var_ind_pre_tuples[i][2] for i in range(int(len(var_ind_pairs) * perc))]
        rad = round(sum(pres)+0.5)
    m.linear_constraints.add(
        lin_expr=[[varInds,coefs.tolist()]],
        senses=['L'],
        rhs=[float(rad - rhs)],
        names=['fixing'])


    m.solve()


if __name__ == '__main__':
    inspath = r'F:\L2O_project\Neurips2023\exps\data\ip_gasse\test\ins\instance_9900.mps'
    prepath = r'F:\L2O_project\ICML2024\src\IP_opt\logits\instance_9900.mps.prob'
    # inspath = r'F:\L2O_project\Neurips2023\exps\data\smsp\test\ins\bench_6_16.mps'
    # prepath = r'F:\L2O_project\ICML2024\src\SMSP_opt\logits\bench_6_16.mps.prob'
    evaluate(inspath, prepath, 'PS')
