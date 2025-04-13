import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


def labelOpt(X_hat,X,lr=0.001,lamda=25,ITRS = 10000,device='cuda:0'):
    '''
    X_hat：[1xqxq]
    X: [1xqxq]
    '''
    size = X.shape[-1]
    P = torch.zeros(X.shape[0],size,size).to(device)
    X_hat_ = X_hat.squeeze(dim=0).detach().cpu().numpy()
    X_t = X.squeeze(dim=0).t().cpu().numpy()

    r_i,c_i = linear_sum_assignment(X_t@X_hat_,maximize=True)
    P[:,r_i,c_i] = 1

    X_bar = X @ P
    return X_bar

if __name__ == '__main__':
    # X = torch.Tensor(
    #     [
    #         [0, 0, 1],
    #         [1, 0, 0],
    #         [0, 1, 0]
    #     ]
    # )
    #
    # X_hat = torch.Tensor(
    #     [
    #         [0.5, 0.3, 0.2],
    #         [0.1, 0.6, 0.2],
    #         [0.2, 0.1, 0.6]
    #     ]
    # )
    X = torch.Tensor(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ]
    )

    X_hat = torch.Tensor(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
    )

    X_bar = labelOpt(X_hat[None,:,:].clone(),X[None,:,:].clone(),device='cpu')

    print('done')