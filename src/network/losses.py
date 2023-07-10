import torch
import numpy as np
from network.covariance_parametrization import DiagonalParam
from utils.lie_algebra import so3_log
from utils.from_scipy import compute_q_from_matrix
from utils.math_utils import logdet3, sixD2so3, so32sixD

MIN_LOG_STD = np.log(1e-3)

"""
SO(3) loss

input: 
  pred: Nx6 tensor of network orientation 1,2 column of orientation matrix output
  targ: batchx3x3xN tensor of gt 1,2 column of orientation matrix
output:
  loss: Nx6 matrix of SO(3) loss on rotation matrix ?
"""


def loss_mse_so3(pred, targ):

    # pres, targ is torch.tensor
    # if pred.dim() < 4:
    #     pred = pred.unsqueeze(3)
    #     targ = targ.unsqueeze(3)

    loss = so3_log(pred.transpose(1, 2).bmm(targ).squeeze()).pow(2).squeeze()   # tensor(1024,3)
    loss = loss.unsqueeze(2).transpose(1, 2).bmm(loss.unsqueeze(2)).squeeze()  # tensor(1024,)

    return loss


def loss_NLL_so3(pred, pred_cov, targ):

    if pred.dim() < 4:
        pred = pred.unsqueeze(3)
        pred_cov = pred_cov.unsqueeze(3)
        targ = targ.unsqueeze(3)

    diagonal_matrix = torch.eye(3).unsqueeze(0).unsqueeze(-1).cuda()
    pred_cov = pred_cov.unsqueeze(2) * diagonal_matrix

    # Network output isn't 4th tensor
    pred = pred.squeeze()
    pred_cov = pred_cov.squeeze()
    targ = targ.squeeze()

    # loss = (pred - targ).transpose() / (2 * torch.exp(2 * pred_cov)) + pred_cov

    residual = so3_log(pred.bmm(targ.transpose(1,2))).unsqueeze(2)

    weighted_term = 0.5 * residual.transpose(1,2).bmm(pred_cov).bmm(residual)
    loss = weighted_term.squeeze() - 0.5 * torch.log((pred_cov[:, 0, 0]*pred_cov[:, 1, 1]*pred_cov[:, 2, 2])**2)

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