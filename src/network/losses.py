import torch
import numpy as np
from network.covariance_parametrization import DiagonalParam
from lietorch import SO3, LieGroupParameter
from utils.lie_algebra import so3_log
from utils.from_scipy import compute_q_from_matrix
from utils.math_utils import logdet3, sixD2so3, so32sixD

MIN_LOG_STD = np.log(1e-3)

"""
SO(3) loss

input: 
  pred: batchx6x(N) tensor of network orientation 1,2 column of orientation matrix output
  targ: batchx3x3x(N) tensor of gt 1,2 column of orientation matrix
output:
  loss: batchx(N) matrix of SO(3) loss on rotation matrix ?
"""

def loss_L2dis_so3(pred, targ):
    M = targ * pred.transpose(1,2)
    loss = torch.acos_(0.5*(M[:, 0, 0]+M[:, 1, 1]+M[:, 2, 2]-1))
    return loss

def loss_mse_so3(pred, targ):

    # pres, targ is torch.tensor
    # if pred.dim() < 4:
    #     pred = pred.unsqueeze(3)
    #     targ = targ.unsqueeze(3)

    ## 기존 displacement loss 계산 방식
    # loss = (pred - targ).pow(2).squeeze()   # tensor(1024,3,3)

    ## lie_algebra so3_log 사용:  pred과 targ 사이의 각도차이가 작아지면서 nan 값이 나와 오류 발생
    # loss = so3_log(pred.transpose(1, 2).bmm(targ).squeeze()).pow(2).squeeze()   # tensor(1024,3)
    # loss = loss.unsqueeze(2).transpose(1, 2).bmm(loss.unsqueeze(2)).squeeze()  # tensor(1024,)

    ## lietorch
    # pred = pred.to(torch.float32)
    # targ = targ.to(torch.float32)
    targ0 = targ
    pred = SO3.InitFromVec(torch.from_numpy(compute_q_from_matrix(pred.cpu().detach().numpy())).cuda())
    targ = SO3.InitFromVec(torch.from_numpy(compute_q_from_matrix(targ.cpu().detach().numpy())).cuda())
    loss = pred.inv()*targ
    loss = loss.log().unsqueeze(2)
    loss = loss.transpose(1,2).bmm(loss).squeeze()
    loss.requires_grad = True  # backpropagation을 위함
    if torch.any(torch.isnan(loss)):
        nan_ind = torch.nonzero(torch.isnan(loss)).squeeze()
        print('loss NaN value place: ', nan_ind)
        print('pred NaN value place: ', pred.data[nan_ind, :])
        print('targ NaN value place: ', targ.data[nan_ind, :])
        print('targ0 NaN value place: ', targ0.data[nan_ind, :, :])
        input()
    return loss


def loss_NLL_so3(pred, pred_cov, targ):

    if pred.dim() < 4:
        pred = pred.unsqueeze(3)
        pred_cov = pred_cov.unsqueeze(2)
        targ = targ.unsqueeze(3)

    sigma = torch.zeros(1024,3,3).cuda()
    sigma[:, 0, 0] = torch.exp(2*pred_cov[:, 0].squeeze())
    sigma[:, 1, 1] = torch.exp(2*pred_cov[:, 1].squeeze())
    sigma[:, 2, 2] = torch.exp(2*pred_cov[:, 2].squeeze())

    # Network output isn't 4th tensor
    pred = pred.squeeze()
    pred_cov = pred_cov.squeeze()
    targ = targ.squeeze()

    ## 기존 displacement loss 계산 방식
    # loss = (pred - targ).pow(2) / (2 * torch.exp(2 * pred_cov)) + pred_cov

    ## lie_algebra so3_log 사용: pred과 targ 사이의 각도차이가 작아지면서 nan 값이 나와 오류 발생
    # residual = so3_log(pred.bmm(targ.transpose(1,2))).unsqueeze(2)
    # weighted_term = 0.5 * residual.transpose(1,2).bmm(pred_cov).bmm(residual)
    # loss = weighted_term.squeeze() - 0.5 * torch.log((pred_cov[:, 0, 0]*pred_cov[:, 1, 1]*pred_cov[:, 2, 2])**2)

    ## lietorch
    pred = pred.float()
    targ = targ.float()
    pred = SO3.InitFromVec(compute_q_from_matrix(pred.cpu().numpy())).cuda()
    targ = SO3.InitFromVec(compute_q_from_matrix(targ.cpu().numpy())).cuda()
    loss = pred.inv()*targ
    loss = loss.log().unsqueeze(2)
    loss = 0.5*(loss.transpose(1,2).bmm(sigma).bmm(loss).squeeze()) + 0.5*(torch.log(sigma[:, 0, 0]*sigma[:, 1, 1]*sigma[:, 2, 2]))

    return loss

"""
MSE loss between prediction and target, no logstdariance

input: 
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
output:
  loss: Nx3 vector of MSE loss on x,y,z
"""


def loss_mse(pred, targ):
    loss = (pred - targ).pow(2)
    return loss


"""
Log Likelihood loss, with logstdariance (only support diag logstd)

input:
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
  pred_logstd: Nx3 vector of log(sigma) on the diagonal entries
output:
  loss: Nx3 vector of likelihood loss on x,y,z

resulting pred_logstd meaning:
pred_logstd:(Nx3) u = [log(sigma_x) log(sigma_y) log(sigma_z)]
"""


def loss_distribution_diag(pred, pred_logstd, targ):
    pred_logstd = torch.maximum(pred_logstd, MIN_LOG_STD * torch.ones_like(pred_logstd))
    loss = ((pred - targ).pow(2)) / (2 * torch.exp(2 * pred_logstd)) + pred_logstd
    return loss


"""
Log Likelihood loss, with logstdariance (support full logstd)
(NOTE: output is Nx1)

input:
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
  pred_logstd: Nxk logstdariance parametrization
output:
  loss: Nx1 vector of likelihood loss

resulting pred_logstd meaning:
DiagonalParam:
pred_logstd:(Nx3) u = [log(sigma_x) log(sigma_y) log(sigma_z)]
PearsonParam:
pred_logstd (Nx6): u = [log(sigma_x) log(sigma_y) log(sigma_z)
                     rho_xy, rho_xz, rho_yz] (Pearson correlation coeff)
FunStuff
"""


def criterion_distribution(pred, pred_logstd, targ):
    loss = DiagonalParam.toMahalanobisDistance(
        targ, pred, pred_logstd, clamp_logstdariance=False
    )


"""
Select loss function based on epochs
all variables on gpu
output:
  loss: Nx3
"""


def get_loss(pred, pred_logstd, targ, epoch):

    if epoch < 10:
        loss = loss_mse(pred, targ)
    else:
        loss = loss_distribution_diag(pred, pred_logstd, targ)

    """
    if epoch < 10:
        pred_logstd = pred_logstd.detach()

    loss = loss_distribution_diag(pred, pred_logstd, targ)
    """
    return loss


def get_loss_so3(pred, pred_logstd, targ, epoch):

    if epoch < 10:
        loss = loss_mse_so3(pred, targ)
    else:
        loss = loss_NLL_so3(pred, pred_logstd, targ)

    return loss