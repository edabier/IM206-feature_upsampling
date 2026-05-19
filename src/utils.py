import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
import math
import scipy.io as io
import os
from PIL import Image

def oneD_to_2d(Y, H=None, W=None):
    """
    Reshapes 1D tensors to 2D image
    If H and W are not passed, assumes the image is square
    
    Args:
        Y: input tensor to be reshaped
    """
    is_batched = Y.dim()==3
    if is_batched:
        batch, B, N = Y.shape
        if H != None:
            return Y.reshape(batch, B, H, W)
        
        else:
            H = int(N**0.5)
            return Y.reshape(batch, B, H, H)
    else:
        B, N = Y.shape
        if H != None:
            return Y.reshape(B, H, W)
        
        else:
            H = int(N**0.5)
            return Y.reshape(B, H, H)

class TVLoss(nn.Module):
    def __init__(self, reduction=None):
        super(TVLoss,self).__init__()
        self.reduction = reduction

    def forward(self,x):
        """
        Expects input x to be of shape (batch, B, H, W)
        """
        batch = x.shape[0]

        diff1 = x[..., 1:, :] - x[..., :-1, :]
        diff2 = x[..., :, 1:] - x[..., :, :-1]

        res1 = diff1.abs().sum([1, 2, 3])
        res2 = diff2.abs().sum([1, 2, 3])
        score = res1 + res2

        if self.reduction == "mean":
            return score.sum() / batch
        elif self.reduction == "sum":
            return score.sum()
        elif self.reduction is None or batch == "none":
            return score[0]

class SADLoss(nn.Module):
    """
    SAD loss function for EndMember matrices. To use it on Abundances, transpose the two inputs. (Doesn't correct permutations)
    """
    def __init__(self):
        super(SADLoss, self).__init__()

    def forward(self, y_true, y_pred):
        if y_pred.dim() == 1:
            y_true = F.normalize(y_true, dim=0, p=2)
            y_pred = F.normalize(y_pred, dim=0, p=2)
        else:
            y_true = F.normalize(y_true, dim=1, p=2)
            y_pred = F.normalize(y_pred, dim=1, p=2)

        A = torch.mul(y_true, y_pred)

        if y_true.dim() == 1:
            A = torch.sum(A, dim=0)
        else:
            A = torch.sum(A, dim=1)
            
        sad = torch.acos(A)
        loss = torch.mean(sad)
        return loss

def order_endmembers(tensor_gt, tensor_hat, tensor2_hat=None):
    """
    Uses scipy linear_sum_assignement algorithm to reorder tensor_hat columns to match tensor_gt
    Tensors must be of shape (batch, D, X) or (D, X) where D is the axis along which to reorder
    tensor_2_hat is another tensor that can be reordered based on the tensor_hat reordering (for abundances)
    """
    is_batched = True
    if tensor_hat.dim() == 2:
        is_batched = False
        tensor_hat = tensor_hat.unsqueeze(0)
    if tensor_gt.dim() == 2:
        is_batched = False
        tensor_gt = tensor_gt.unsqueeze(0)

    tensor_hat_ordered = torch.zeros_like(tensor_hat)
        
    if tensor2_hat != None:
        if tensor2_hat.dim() < 4:
            tensor2_hat = tensor2_hat.unsqueeze(0)

        tensor2_hat_ordered = torch.zeros_like(tensor2_hat)

    for b in range(tensor_hat.size()[0]):

        # Normalize the tensors
        tensor_gt_norm = F.normalize(tensor_gt[b], p=2.0, dim=1)  # Normalize along reordered axis
        tensor_hat_norm = F.normalize(tensor_hat[b], p=2.0, dim=1)

        # Compute cost matrix (cosine distance)
        cost_matrix = torch.acos(torch.clamp(tensor_gt_norm @ tensor_hat_norm.T, -1.0, 1.0))
        cost_matrix_np = cost_matrix.cpu().numpy() 

        # Solve assignment problem
        _, col_ind = linear_sum_assignment(cost_matrix_np)

        # Reorder E_hat to match E_gt
        tensor_hat_ordered[b] = tensor_hat[b, col_ind]

        if tensor2_hat != None:
            tensor2_hat_ordered[b] = tensor2_hat[b, col_ind]

    if tensor2_hat != None:
        if is_batched:
            return tensor_hat_ordered, tensor2_hat_ordered, col_ind
        else:
            return tensor_hat_ordered[0], tensor2_hat_ordered[0], col_ind
    else:
        if is_batched:
            return tensor_hat_ordered, col_ind
        else:
            return tensor_hat_ordered[0], col_ind
   