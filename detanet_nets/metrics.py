# Adapted from DetaNet: Zou et al., Nat. Comput. Sci. 3, 957-964 (2023).
# https://doi.org/10.1038/s43588-023-00550-y  (MIT License, (c) 2023 Z. Zou, W. Hu)

import torch
import numpy as np

def l2loss(out,target):
    '''Mean Square Error(MSE)'''
    diff = out-target
    return torch.mean(diff ** 2)

def l1loss(out,target):
    '''Mean Absolute Error(MAE)'''
    return torch.mean(torch.abs(out-target))

def rmse(out, target):
    '''Root Mean Square Error(rmse) (also known as RMSD)'''
    return torch.sqrt(torch.mean((out - target) ** 2))

def state_l2loss(out,target):
    '''Loss for excited state vectors.
    from J. Phys. Chem. Lett. 2020, 11, 3828-3834'''
    diffa=torch.abs(out-target)**2
    diffb=torch.abs(out+target)**2
    diff=torch.min(diffa,diffb)
    return torch.mean(diff)

def R2(out,target):
    '''coefficient of determination'''
    mean=torch.mean(target)
    SSE=torch.sum((out-target)**2)
    SST=torch.sum((mean-target)**2)
    return 1-(SSE/SST)

def combine_lose(out_tuple,target_tuple,lamb=10):
    '''Combine loss of energy and force'''
    return l2loss(out_tuple[0],target_tuple[0])+lamb*l2loss(out_tuple[1],target_tuple[1])
