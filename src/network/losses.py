import torch
import numpy as np
from network.covariance_parametrization import DiagonalParam
from utils.lie_algebra import so3_log
from utils.from_scipy import compute_q_from_matrix
from utils.math_utils import logdet3

MIN_LOG_STD = np.log(1e-3)

"""
SO(3) loss

input: 
  pred: Nx6 tensor of network orientation 1,2 column of orientation matrix output
  targ: Nx6 tensor of gt 1,2 column of orientation matrix
output:
  loss: Nx6 matrix of SO(3) loss on rotation matrix ?
"""


def loss_mse_so3(pred, trag):

    pred = sixD2so3(pred)
    trag = sixD2so3(trag)
    loss = (so3_log(pred*trag.transpose(1, 2))).norm(dim=(1, 2))**2

    return loss


def loss_NLL_so3(pred, pred_cov, trag):

    pred = sixD2so3(pred)
    trag = sixD2so3(trag)
    pred_cov = sixD2so3(pred_cov) # Must be replaced with a covariance matrix
    # cov_mat =
    loss = 0.5*(pred - trag).transpose(1, 2)/cov_mat*(pred - trag) - 0.5*logdet3(cov_mat)
    return loss


def sixD2so3(sixD):
    ## two vector of network output
    sixDnp = sixD.numpy()  # numpy
    a1 = sixDnp[:3]
    a2 = sixDnp[3:]

    ## b1
    a1_norm = np.linalg.norm(a1)
    b1 = a1 / a1_norm

    ## b2
    b2 = a2 - np.dot(b1, a2) * b1
    b2_norm = np.linalg.norm(b2)
    b2 = b2/b2_norm

    ## b3
    e1 = np.array([1, 0, 0], dtype=np.float32)
    e2 = np.array([0, 1, 0], dtype=np.float32)
    e3 = np.array([0, 0, 1], dtype=np.float32)
    b31 = np.linalg.det(np.concatenate([b1[:, None], b2[:, None], e1[:, None]], axis=1))
    b32 = np.linalg.det(np.concatenate([b1[:, None], b2[:, None], e2[:, None]], axis=1))
    b33 = np.linalg.det(np.concatenate([b1[:, None], b2[:, None], e3[:, None]], axis=1))
    b3 = np.array([b31, b32, b33])

    ## reconstructed rotation
    R = np.concatenate([b1[:,None], b2[:,None], b3[:,None]], axis=1)
    _ = torch.from_numpy(R)  # tensor

    return R


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
    """
    if epoch < 10:
        loss = loss_mse(pred, targ)
    else:
        loss = loss_distribution_diag(pred, pred_logstd, targ)
    """

    if epoch < 10:
        pred_logstd = pred_logstd.detach()

    loss = loss_distribution_diag(pred, pred_logstd, targ)
    return loss
