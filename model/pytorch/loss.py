import torch
# import numpy as np
import math
def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_rems_loss(y_pred , y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()




def masked_mape_loss(y_pred , y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs((y_true - y_pred)/(y_true != 0).float())
    loss = loss * mask /100
    loss[loss != loss] = 0
    return loss.mean()



