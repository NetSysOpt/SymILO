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
#
# def labelOpt(X_hat,X,lr=0.001,lamda=25,ITRS = 10000,device='cuda:0'):
#     '''
#     X_hat：[1xqxq]
#     X: [1xqxq]
#     '''
#     n = X.shape[-1]
#     X_bar = X.clone()
#     lastVec = X[:,:,-1].clone()
#     pos_loss = -torch.log(X_hat + 0.00001) * (X >= 0.5)
#     neg_loss = -torch.log(1 - X_hat + 0.00001) * (X < 0.5)
#     best_loss = pos_loss.sum().item() + neg_loss.sum().item()
#     for i in range(n-1):
#         X[:,:,1:] = X[:,:,0:n-1].clone()
#         X[:,:,0] = lastVec
#         lastVec = X[:,:, -1].clone()
#         pos_loss = -torch.log(X_hat + 0.00001) * (X >= 0.5)
#         neg_loss = -torch.log(1 - X_hat + 0.00001) * (X < 0.5)
#         loss = pos_loss.sum().item() + neg_loss.sum().item()
#         if loss < best_loss:
#             best_loss = loss
#             X_bar = X.clone()
#     return X_bar

#
# def labelOpt(X_hat,X,lr=0.00001,lamda=10,ITRS = 200,device='cuda:0'):
#     size = list(X.shape)
#     J = size[-1]
#     size[-2] = J
#     P = torch.ones(size).to(device) / J
#     P.requires_grad = True
#     optimizer = torch.optim.SGD([P], lr=lr, momentum=0.9)
#     eps = 0.001
#     for i in range(ITRS):
#         #qloss = ((X_hat - X@P)**2).sum()
#         XP = X@P
#         XP.data[XP < eps] = eps
#         XP.data[XP > 1 - eps] = 1 - eps
#         qloss = -X_hat*torch.log(XP ) - (1-X_hat)*torch.log(1-XP)
#         qloss = qloss.sum()
#
#         el =  ((P.sum(dim=-1)-1)**2).sum() + ((P.sum(dim=-2)-1)**2).sum()
#         iel = (P**2 * ( (P<0) + (P>1) ) ).sum()
#
#         loss = qloss + lamda*el + lamda*iel
#
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         # if i % 3 == 0:
#         #    print(f'{i}/{ITRS} loss:{loss.item()} qloss:{qloss.item()} eloss:{el.item()} ieloss:{iel.item()}')
#
#
#     P = sinkhorn(P.detach(),device=device)
#
#     X_bar = X@P
#
#     return X_bar
def lexOpt(_,X,lr=0.001,lamda=25,ITRS = 10000,device='cuda:0'):

    X_bar = X
    newX = X_bar.detach()
    X_bar = X.cpu().numpy().astype(int).astype(str)

    n,nr,nc = X_bar.shape

    for i in range(n):

        Y = X_bar[i]
        Y = list(Y.transpose())
        Y = [ ''.join(list(y)) for y in Y]
        Y = np.array(Y)
        inds = np.argsort(Y)[::-1]
        newY = X[i][:,list(inds)]
        newX[i] = newY

    return newX

# def sinkhorn(P,itrs=100,device='cuda:0'):
#
#     eps = torch.eye(P.shape[-1]).to(device) * 0.001
#
#     for t in range(100):
#         P = P + eps[None,:,:]
#         Pmin = (P.min(dim=-1,keepdim=True)[0]).min(dim=-2,keepdim=True)[0]
#         Pmax = (P.max(dim=-1,keepdim=True)[0]).max(dim=-2,keepdim=True)[0]
#
#         scale = 1 / (Pmax -Pmin)
#         bias = -Pmin
#         P = (P +bias)*scale +1
#         m = P.max()
#         lam = torch.log(torch.ones(1).to(device)*1e8)/m
#         P = (lam*P).exp()
#
#         for i in range(itrs):
#             P = P/P.sum(dim=-1,keepdim=True)
#             P = P / P.sum(dim=-2, keepdim=True)
#
#     P = P.round()
#
#
#     return P
def sinkhorn(C, itrs=100, device='cuda:0'):

    eps = torch.eye(C.shape[-1]).to(device) * 0.001
    C = C + eps[None, :, :]
    P = torch.zeros_like(C)
    C = C.cpu().numpy()
    for n in range(C.shape[0]):
        for t in range(C.shape[1]):
            x = C[n]
            max_index = np.unravel_index(np.argmax(x, axis=None), x.shape)
            P[n,max_index[0],max_index[1]] = 1
            C[:,max_index[0],:] = -1
            C[:, :,max_index[1]] = -1
    return P



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