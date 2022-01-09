'''
LastEditTime: 2022-01-08 16:56:04
Description: Sampling functions
Date: 2022-01-07 17:59:23
Author: Aiden Li
LastEditors: Aiden Li (i@aidenli.net)
'''
from os import spawnlpe
from numpy.random.mtrand import sample
import torch
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

# Debug only
import matplotlib.pyplot as plt
# END Debug only

plot_samples = 5

def inverse_transform_sampling_pcpdf(bounds, weights, num_points_fine):
    """Inverse Transform Sampling from piecewise-constant pdf

    Args:
        bounds ([type]): batch_size x (num_points + 1), boundaries of weights
        weights ([type]): batch_size x num_points
        num_point_fine ([type]): num of points to be additionally sampled
    """
    batch_size = bounds.shape[0]
    
    # Turning into PCPDF
    probabilities = weights / (weights.sum(dim=-1).unsqueeze(-1) + 1e-5)
    cumulative = torch.cumsum(probabilities, dim=-1)
    
    uni = torch.rand([batch_size, num_points_fine], device=bounds.device)
    
    positions = torch.searchsorted(cumulative, uni)
    
    lo = torch.gather(input=bounds, index=positions, dim=-1)        # batch_size x num_points_fine
    hi = torch.gather(input=bounds, index=torch.minimum(positions + 1, torch.ones_like(positions) * (bounds.shape[-1] - 1)), dim=-1)    # batch_size x num_points_fine
    
    t_rand = torch.rand([batch_size, num_points_fine], device=bounds.device)
    
    sampled = lo + (hi - lo) * t_rand  
    
    bd = bounds[0].detach().cpu().numpy()
    sm = sampled[0].detach().cpu().numpy()
    
    # Visualize:
    # - PDF
    # global plot_samples
    # if plot_samples > 0:
    #     if cumulative[0, -1] > 0.9:  # is 1
    #         plot_samples -= 1
    #         plt.figure()
    #         plt.scatter(0.5 * (bd[:-1] + bd[1:]), probabilities[0].detach().cpu().numpy(), color='blue', alpha=0.5)
    #         plt.scatter(0.5 * (bd[:-1] + bd[1:]), cumulative[0].detach().cpu().numpy(), color='red', alpha=0.5)
    #         plt.scatter(sm, np.zeros_like(sm), color='black', marker='x')    
    #         plt.savefig(f"Hierarcically Sampling { 5 - plot_samples }.jpg")
            
    return sampled
    