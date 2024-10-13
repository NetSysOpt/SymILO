import pickle
import re
import sys
import os
import numpy as np
import argparse
import io
import time
import math
import cplex
from callbacks_cplex_new import node_selection3,branch_attach_data2,branch_empty
import copy


def get_prediction(pre_path,var_names):
    data = pickle.load(open(pre_path, 'rb'))
    prediction = data['pre']
    biInds = data['biInds']
    reorderInds = data['reorderInds']

    varname_inds = [[step, var_name] for step, var_name in enumerate(var_names)]
    varname_inds.sort(key=lambda var_ind: var_ind[1])
    bi_name_inds = np.array(varname_inds)[biInds][reorderInds].tolist()



    dict_varname_seqid = {}
    for varname in var_names:
        dict_varname_seqid[varname] = 0.49
    for step, var_name in enumerate(bi_name_inds):
        dict_varname_seqid[var_name[1]] = prediction[step]

    return prediction, dict_varname_seqid

# direction=1: branch on most integer first
def set_cplex_priorities(instance_cpx, prediction, direction=1):
    # score variables based on bias prediction
    scores = np.max(((1-prediction), prediction), axis=0)
    priorities = np.argsort(direction * scores)

    # set priorities
    # reference: https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.1/ilog.odms.cplex.help/refpythoncplex/html/cplex._internal._subinterfaces.OrderInterface-class.html
    order_tuples = []
    var_names = instance_cpx.variables.get_names()

    cur_priority = 0
    for priority, var_cpxid in enumerate(priorities):
        var_name = var_names[var_cpxid]
        # print(scores[var_cpxid], scores[priorities[priority-1]])
        # if priority > 0 and scores[var_cpxid] > scores[priorities[priority-1]] + 1e-3:
        cur_priority += 1
            # print(cur_priority)
        order_tuples += [(var_name, cur_priority, instance_cpx.order.branch_direction.up)]

    # print(cur_priority)
    # z=1/0
    instance_cpx.order.set(order_tuples)

def mipeval(
    instance,
    pre_path,
    method='node_selection',
    logfile='sys.stdout',
    barebones=0,
    cpx_emphasis=1,
    cpx_threads=1,
    cpx_tmp='./temp',
    timelimit=60,
    memlimit=1024,
    freq_best=100,
    lb_threshold=5,
    num_mipstarts=10,
    mipstart_strategy='repair',
    branching_direction=1,
    zero_damping=0.001
    ):
    
    # print(locals())

    assert (len(method) >= 1)
    assert (cpx_emphasis >= 0 and cpx_emphasis <= 4)
    assert (timelimit > 0)

    """ CPLEX output management """
    instance_cpx = cplex.Cplex()
    if logfile != 'sys.stdout':
        logstring = io.StringIO()
        summary_string = io.StringIO()
        logstring = open(logfile, 'w')
        instance_cpx.set_log_stream(logstring)
        instance_cpx.set_results_stream(logstring)
        instance_cpx.set_warning_stream(logstring)
        # instance_cpx.set_error_stream(logstring)
        instance_cpx.set_error_stream(open(os.devnull, 'w'))

    """ Create CPLEX instance """
    instance_cpx.read(instance)
    sense_str = instance_cpx.objective.sense[instance_cpx.objective.get_sense()]
    num_variables = instance_cpx.variables.get_num()
    # num_constraints = instance_cpx.linear_constraints.get_num()
    # start_time = instance_cpx.get_time()



    """ Set CPLEX parameters, if any """
    instance_cpx.parameters.timelimit.set(timelimit)
    instance_cpx.parameters.emphasis.mip.set(cpx_emphasis)
    instance_cpx.parameters.mip.display.set(3)
    instance_cpx.parameters.threads.set(cpx_threads)
    instance_cpx.parameters.workmem.set(memlimit)
    instance_cpx.parameters.mip.limits.treememory.set(20000)
    instance_cpx.parameters.mip.strategy.file.set(2)
    instance_cpx.parameters.workdir.set(cpx_tmp)
    if barebones:
        instance_cpx.parameters.mip.limits.cutpasses.set(-1)
        instance_cpx.parameters.mip.strategy.heuristicfreq.set(-1)
        instance_cpx.parameters.preprocessing.presolve.set(0)

        # DFS = 0, BEST-BOUND = 1 (default), BEST-EST = 2, BEST-EST-ALT = 3
        # instance_cpx.parameters.mip.strategy.nodeselect.set(3)

    time_rem_cplex = timelimit
    time_vcg = time.time()
    time_vcg_reading = 0
    time_pred = 0

    is_primal_mipstart = False
    """ Solve CPLEX instance with user-selected method """
    if 'default' not in method[0]:

        # print("Predicting...")
        timestamp_pred = time.time()
        # var_names = rename_variables(instance_cpx.variables.get_names())
        var_names = instance_cpx.variables.get_names()
        prediction, dict_varname_seqid = get_prediction(pre_path,var_names)

        # print("\t took %g secs." % (time.time()-timestamp_pred))
        time_pred = time.time() - timestamp_pred
        # print(prediction)
        # todo check dimensions of p

        time_rem_cplex = timelimit - time_pred
        # print("time_rem_cplex = %g" % time_rem_cplex)
        instance_cpx.parameters.timelimit.set(time_rem_cplex)


        prediction_reord = [dict_varname_seqid[var_name] for var_name in var_names]
        oldPrediction = copy.deepcopy(prediction)
        prediction = np.array(prediction_reord)

        # check
        # X = np.zeros((oldPrediction.shape[0]//111,111))
        # for ind,name in enumerate(var_names):
        #     ss = re.findall('\d+', name)
        #     a, b = int(ss[0]), int(ss[1])
        #     if 'X' in name:
        #         X[a,b] = prediction[ind]
        #     elif 'Y' in name:
        #         X[a+111,b] = prediction[ind]

        # if len(method) == 1 and ('local_branching' in method[0]):
        #     pred_one_coeff = (prediction >= 0.9) * (-1)
        #     pred_zero_coeff = (prediction <= 0.1)
        #     num_ones = -np.sum(pred_one_coeff)
        #     coeffs = pred_one_coeff + pred_zero_coeff
        #
        #     local_branching_coeffs = [list(range(len(prediction))), coeffs.tolist()]
        #
        #     if method[0] == 'local_branching_approx':
        #         instance_cpx.linear_constraints.add(
        #             lin_expr=[local_branching_coeffs],
        #             senses=['L'],
        #             rhs=[float(lb_threshold - num_ones)],
        #             names=['local_branching'])
        #
        #     elif method[0] == 'local_branching_exact':
        #         branch_cb = instance_cpx.register_callback(callbacks_cplex.branch_local_exact)
        #
        #         branch_cb.coeffs = local_branching_coeffs
        #         branch_cb.threshold = lb_threshold - num_ones
        #         branch_cb.is_root = True

        # if 'branching_priorities' in method:
        #     set_cplex_priorities(instance_cpx, prediction, branching_direction)



        if 'node_selection' in method:
            # score variables based on bias prediction
            scores = np.max(((1-prediction), prediction), axis=0)
            rounding = np.round(prediction)

            # print(np.mean(scores), np.mean(rounding))
            # print(np.argsort(prediction), np.sort(prediction)[:10], np.sort(prediction)[-10:])

            branch_cb = instance_cpx.register_callback(branch_attach_data2)
            node_cb = instance_cpx.register_callback(node_selection3)

            branch_cb.scoring_function = 'sum' #'estimate'
            branch_cb.scores = scores
            branch_cb.rounding = rounding
            branch_cb.zero_damping = zero_damping

            node_cb.last_best = 0
            node_cb.freq_best = freq_best

            node_priority = []
            branch_cb.node_priority = node_priority
            node_cb.node_priority = node_priority

            branch_cb.time = 0
            node_cb.time = 0

        if ('primal_mipstart' in method) or ('primal_mipstart_only' in method):
            is_primal_mipstart = True
            if not barebones or 'primal_mipstart_only' in method:
                instance_cpx.parameters.mip.limits.cutpasses.set(-1)
                instance_cpx.parameters.mip.strategy.heuristicfreq.set(-1)
                instance_cpx.parameters.preprocessing.presolve.set(0)

            mipstart_string = sys.stdout if logfile == "sys.stdout" else io.StringIO()

            #frac_variables = [0.001*(1.5**i) for i in range(18)] #[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            #frac_variables = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            #frac_variables = np.flip(np.linspace(0, 1, num=num_mipstarts+1))[:-1]
            #print(frac_variables)
            #threshold_set = np.minimum(prediction, 1-prediction)
            #threshold_set = np.sort(threshold_set)#[:mipstart_numthresholds]
            
            #threshold_set = [threshold_set[max([0, int(math.ceil(frac_variables[i]*num_variables)) - 1])] for i in range(len(frac_variables))]
            
            threshold_set = [0.001*(2**i) for i in range(6)]
            
            threshold_set.reverse()
            threshold_set = np.clip(threshold_set, a_min=0, a_max=0.5)
            print("threshold_set = ", threshold_set)
            
            if mipstart_strategy == 'repair':
                mipstart_strategy_int = instance_cpx.MIP_starts.effort_level.repair
            elif mipstart_strategy == 'solve_MIP':
                mipstart_strategy_int = instance_cpx.MIP_starts.effort_level.solve_MIP
            else:
                print("invalid mipstart_strategy %s" % mipstart_strategy)
                exit()

            best_objval_mipstart = -math.inf if sense_str == 'maximize' else math.inf
            for idx, threshold in enumerate(threshold_set):
                time_rem_cplex = timelimit - time_pred #(time.time() - time_vcg)
                if time_rem_cplex <= 0:
                    break

                indices_integer = np.where((prediction >= 1-threshold) | (prediction <= threshold))[0]
                print(idx, threshold, len(indices_integer), len(prediction))

                if len(indices_integer) == 0:
                    continue

                instance_cpx.parameters.mip.display.set(0)
                instance_cpx.parameters.mip.limits.nodes.set(0)
                # print("time_rem_cplex = %g" % time_rem_cplex)
                instance_cpx.parameters.timelimit.set(time_rem_cplex)

                instance_cpx.MIP_starts.add(
                    cplex.SparsePair(
                        ind=indices_integer.tolist(),
                        val=np.round(prediction[indices_integer]).tolist()),
                    mipstart_strategy_int)

                instance_cpx.solve()
                instance_cpx.MIP_starts.delete()
                
                if instance_cpx.solution.is_primal_feasible(): #and instance_cpx.solution.get_objective_value() > best_objval_mipstart:
                    is_sol_better = (instance_cpx.solution.get_objective_value() > best_objval_mipstart) if sense_str == 'maximize' else (instance_cpx.solution.get_objective_value() < best_objval_mipstart)
                    if not is_sol_better:
                        continue
                    best_objval_mipstart = instance_cpx.solution.get_objective_value()
                    best_time = time.time() - time_vcg
                    incb_str_cur = ("Found incumbent of value %g after %g sec. mipstart %d %g %g\n" % (best_objval_mipstart, best_time, len(indices_integer), threshold, len(indices_integer)/num_variables))
                    print(incb_str_cur)
                    mipstart_string.write(incb_str_cur)#"Found incumbent of value %g after %g sec. mipstart %d %g %g\n" % (best_objval_mipstart, best_time, len(indices_integer), threshold))

            instance_cpx.parameters.mip.display.set(3)
            if not barebones and not 'primal_mipstart_only' in method:
                instance_cpx.parameters.mip.limits.cutpasses.set(0)
                instance_cpx.parameters.mip.strategy.heuristicfreq.set(0)
                instance_cpx.parameters.preprocessing.presolve.set(1)

            if 'primal_mipstart_only' not in method:
                instance_cpx.parameters.mip.limits.nodes.set(1e9)

    elif method[0] == 'default_emptycb':
        branch_cb = instance_cpx.register_callback(branch_empty)

    time_rem_cplex = timelimit - time_pred #(time.time() - time_vcg)
    # print("time_rem_cplex = %g" % time_rem_cplex)

    # fix variables with high probability
    nvar = prediction.shape[0]
    one_inds = []
    zero_inds = []
    one_coefs = []
    zero_coefs = []
    for i in range(nvar):
        if prediction[i]<0.00001:
            zero_inds.append(i)
            zero_coefs.append(1)
        elif prediction[i]>0.9:
            one_inds.append(i)
            one_coefs.append(-1)

    instance_cpx.linear_constraints.add(
        lin_expr=[[zero_inds+one_inds, zero_coefs+one_coefs]],
        senses=['L'],
        rhs=[len(one_coefs)],
        names=['fixing'])

    if time_rem_cplex > 0:
        instance_cpx.parameters.timelimit.set(time_rem_cplex)

        # todo: consider runseeds 
        #  https://www.ibm.com/support/knowledgecenter/SSSA5P_12.9.0/ilog.odms.cplex.help/refpythoncplex/html/cplex.Cplex-class.html?view=kc#runseeds
        instance_cpx.solve()
    end_time = instance_cpx.get_time()


""" Parse arguments """
class Obj:
    def RegistAttr(self,name,value):
        self.__dict__[name] = value
args = Obj()
args.cpx_emphasis = 1
args.cpx_threads = 1
args.cpx_tmp = './cpx_tmp/'
args.barebones = 0
args.timelimit = 60
args.memlimit = 1024
args.logfile = 'log.out'
args.freq_best = 100
args.zero_damping = 1.0
args.lb_threshold = 5
args.num_mipstarts = 6
args.mipstart_strategy = "repair"
args.branching_direction = 1






def eval(inspath,prepath,exp_dir,timelimit,method,perc=0):

    args.method = method
    args.instance = inspath
    args.pre_path = prepath
    args.logfile = os.path.join(exp_dir,os.path.basename(inspath)+'.log')
    args.timelimit = timelimit
    mipeval(**vars(args))

if __name__ == '__main__':

    # inspath = r'F:\L2O_project\Neurips2023\exps\data\ip_gasse\test\ins\instance_9900.mps'
    # prepath = r'F:\L2O_project\ICML2024\src\IP_opt\logits\instance_9900.mps.prob'
    exp_dir = 'tune_ins'
    inspath = r'F:\L2O_project\Neurips2023\exps\data\smsp\test\ins\bench_8_16.mps'
    prepath = r'F:\L2O_project\ICML2024\src\SMSP_opt\logits\bench_8_16.mps.prob'
    eval(inspath,prepath,exp_dir,120)