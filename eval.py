from multiprocessing import Process,Queue
import time
import argparse
from config import *
from ps_infer import evaluate as ps_eval
from mipgnn_infer import eval as mipgnn_eval
from primals import getPrimals

parser = argparse.ArgumentParser()
parser.add_argument('--expName', type=str, default='IP_opt')
parser.add_argument('--method', nargs='+', default=['node_selection'],help='fixTop,PS,node_selection,primal_mipstart')
parser.add_argument('--dataset', type=str, default='IP')
parser.add_argument('--perc', type=float, default=0.5)
parser.add_argument('--maxtime', type=int, default=100)
parser.add_argument('--solver', type=str, default='cplex')
parser.add_argument('--nWorker', type=int, default=8)
parser.add_argument('--radius', type=int, default=1)
args = parser.parse_args()

perc = args.perc
SOLVER = args.solver
MAXTIME = args.maxtime
N_WORKERS = args.nWorker
RADIUS = args.radius
EXP_NAME = args.expName
PROB_DIR = os.path.join(EXP_NAME,'logits')
info = confInfo[args.dataset]
TEST_INS = os.path.join(info['testDir'],'ins')


eval_func =  mipgnn_eval if 'node_selection' in args.method else ps_eval


def solve(q):
    while True:
        data = q.get()
        if data!='1':
            filepath,prepath,exp_dir = data
            eval_func(filepath,prepath,exp_dir,args.maxtime,args.method,args.perc)
        else:
            return None

if __name__ == '__main__':





    # set exp dir
    now = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    exp_dir = f'exp-{SOLVER}-method-{args.method}-perc-{perc}-Mt{MAXTIME}-{now}'
    exp_dir = os.path.join(EXP_NAME, exp_dir)
    os.makedirs(exp_dir, exist_ok=True)

    # read prob data
    filenames = os.listdir(TEST_INS)
    filepaths =  [ os.path.join(TEST_INS,filename) for filename in filenames]

    print('start')
    q = Queue()

    ps = []
    for i in range(N_WORKERS):
        p = Process(target=solve, args=(q,))
        p.start()
        ps.append(p)
    for ind,filepath in enumerate(filepaths):
        # get logits
        basename = os.path.basename(filepath)
        probpath = os.path.join(PROB_DIR,basename+'.prob')
        q.put((filepath,probpath,exp_dir))
    for i in range(N_WORKERS):
        q.put('1')

    for p in ps:
        p.join()

    # get primals
    mean_pb, mean_solTime = getPrimals(exp_dir)
    with open(os.path.join(exp_dir, 'mean_primals.txt'), 'w+') as f:
        f.write(f'Primal Bounds:{mean_pb}\n')
        f.write(f'Solve Time:{mean_solTime}\n')

    print('done')