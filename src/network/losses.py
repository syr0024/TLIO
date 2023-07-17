import torch
import numpy as np
from network.covariance_parametrization import DiagonalParam
from lietorch import SO3, LieGroupParameter
from utils.lie_algebra import so3_log, batch_trace
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

def loss_geo_so3(pred, targ):
    "Geodesic Loss of SO3"
    if pred.dtype == torch.float32 or pred.dtype == torch.float64:  # torch tensor
        M = pred * targ.transpose(1,2)
        loss = torch.acos_(0.5*(M[:, 0, 0]+M[:, 1, 1]+M[:, 2, 2] - 1))
    else:   # numpy
        M = pred * targ.transpose(0,2,1)
        loss = np.arccos(0.5*(M[:, 0, 0]+M[:, 1, 1]+M[:, 2, 2] - 1))
    return loss

def loss_mse_so3(pred, targ):

    ## pred, targ is torch.tensor
    # if pred.dim() < 4:
    #     pred = pred.unsqueeze(3)
    #     targ = targ.unsqueeze(3)

    ## 기존 displacement loss 계산 방식
    # loss = (pred - targ).pow(2).squeeze()   # tensor(1024,3,3)

    ## lie_algebra so3_log 사용:  pred과 targ 사이의 각도차이가 작아지면서 nan 값이 나와 오류 발생
    # pred = pred.float()
    # targ = targ.float()
    # loss = so3_log(targ.bmm(pred.inverse()).squeeze()).pow(2).squeeze()   # tensor(1024,3)
    # loss = loss.unsqueeze(2).transpose(1, 2).bmm(loss.unsqueeze(2)).squeeze()  # tensor(1024,)

    ## lietorch
    if pred.dtype == torch.float32 or pred.dtype == torch.float64:  # torch tensor
        loss = pred * targ.inverse()
        loss = compute_q_from_matrix(loss.cpu().detach().numpy())
        loss = SO3(torch.from_numpy(loss).unsqueeze(2).transpose(1,2).cuda().float())
    else:   # numpy
        loss = pred * np.linalg.inv(targ)
        loss = compute_q_from_matrix(loss)
        loss = SO3(torch.from_numpy(loss).unsqueeze(2).transpose(1,2).cuda().float())
    loss = loss.log().norm(dim=-1).squeeze()
    loss.requires_grad = True  # for backpropagation

    # NaN Debugging for so3_log
    # if torch.any(torch.isnan(loss)):
    #     nan_ind = torch.nonzero(torch.isnan(loss)).squeeze()
    #     print('NaN value index of loss: ', nan_ind)
    #     print('pred value: ', pred.data[nan_ind, :])
    #     print('targ value: ', targ.data[nan_ind, :])
    #     input()
    # elif torch.any(torch.isnan(pred)):
    #     nan_ind = torch.nonzero(torch.isnan(pred)).squeeze()
    #     print('NaN value index of pred: ', nan_ind)
    #     print('loss value: ', loss[nan_ind, :])
    #     print('targ value: ', targ.data[nan_ind, :])
    #     input()

    return loss


def loss_NLL_so3(pred, pred_cov, targ):

    ## pred, targ is torch.tensor
    # if pred.dim() < 4:
    #     pred = pred.unsqueeze(3)
    #     targ = targ.unsqueeze(3)

    N = pred_cov.size()
    sigma = torch.zeros(N[0],3,3).cuda()
    sigma[:, 0, 0] = torch.exp(2*pred_cov[:, 0].squeeze())
    sigma[:, 1, 1] = torch.exp(2*pred_cov[:, 1].squeeze())
    sigma[:, 2, 2] = torch.exp(2*pred_cov[:, 2].squeeze())

    ## 기존 displacement loss 계산 방식
    # loss = (pred - targ).pow(2) / (2 * torch.exp(2 * pred_cov)) + pred_cov

    ## lie_algebra so3_log 사용: pred과 targ 사이의 각도차이가 작아지면서 nan 값이 나와 오류 발생
    # pred = pred.float()
    # targ = targ.float()
    # residual = so3_log(pred.bmm(targ.transpose(1,2))).unsqueeze(2)
    # weighted_term = 0.5 * residual.transpose(1,2).bmm(sigma).bmm(residual)
    # loss = weighted_term.squeeze() + 0.5 * torch.log((sigma[:, 0, 0]*sigma[:, 1, 1]*sigma[:, 2, 2])**2)

    ## lietorch
    loss = pred * targ.inverse()
    loss = compute_q_from_matrix(loss.cpu().detach().numpy())
    loss = SO3(torch.from_numpy(loss).unsqueeze(2).transpose(1,2).cuda().float())
    loss = loss.log()
    loss.requires_grad = True  # for backpropagation
    loss = 0.5*(loss.bmm(sigma.inverse()).bmm(loss.transpose(1,2)).squeeze()) + 0.5*(torch.log(sigma[:, 0, 0]*sigma[:, 1, 1]*sigma[:, 2, 2]))

    # NaN Debugging for so3_log
    # if torch.any(torch.isnan(loss)):
    #     nan_ind = torch.nonzero(torch.isnan(loss)).squeeze()
    #     print('NaN value index of loss: ', nan_ind)
    #     print('pred value: ', pred.data[nan_ind, :])
    #     print('targ value: ', targ.data[nan_ind, :])
    #     input()

    # elif torch.any(torch.isnan(pred)):
    #     nan_ind = torch.nonzero(torch.isnan(pred)).squeeze()
    #     print('NaN value index of pred: ', nan_ind)
    #     print('loss value: ', loss[nan_ind, :])
    #     print('targ value: ', targ.data[nan_ind, :])
    #     input()

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