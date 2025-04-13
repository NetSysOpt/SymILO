
## Instance Preparation

The structure of datasets are organized under `datasets` as follows
```plaintext
datasets/$DATASET_NAME
├── train
│   └── ins  # training instances
└── test
    └── ins  # testing instances
```
where each dataset is placed in their own `$DATASET_NAME` directory

The instances (.mps or .lp files) of each dataset should be prepared and placed in `$DATASET_NAME/train/ins` and `$DATASET_NAME/test/ins`, respectively.

For IP dataset, instances can be downloaded [here](https://github.com/ds4dm/ml4co-competition/blob/main/DATA.md). One can use instances 0-299 for training and instances 9900-9999 for testing.


For SMSP dataset, please convert instances from the [steelmillslab set](http://becool.info.ucl.ac.be/steelmillslab) into `.mps` format by running

```plaintext
python gen_smsp.py
```



For PESP and PESPD datasets, please generate instances by perturbing coefficients of instances in [PESPLib](https://timpasslib.aalto.fi/pesplib.html) by running

    python gen_pesp_train.py
    python gen_pesp_test.py
    python gen_pespd_train.py
    python gen_pespd_test.py
  

## Solution Collection

The (high-quality) solutions of each instance are collected using MILP solver Gurobi by


```plaintext
# IP
python collect_sols.py --dataDir ./datasets/IP/train --nWorkers 5 --maxTime 3600
# SMSP
python collect_sols.py --dataDir ./datasets/SMSP/train --nWorkers 5 --maxTime 3600

# PESP
python collect_sols.py --dataDir ./datasets/PESP/train --nWorkers 5 --maxTime 3600

# PESPD
python collect_sols.py --dataDir ./datasets/PESPD/train --nWorkers 5 --maxTime 3600
```



## Prepare the bipartite representation

The last step before training is converting each instance into a bipartite graph, so that it can be handled by GNNs. By running the following scripts, these biaprtite graphs will be created and stored in `$DATASET_NAME/train/bg` and `$DATASET_NAME/test/bg`.

```plaintext
python dataset.py --IP
python dataset.py --SMSP
python dataset.py --PESP
python dataset.py --PESPD
```
