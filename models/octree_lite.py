'''
LastEditTime: 2022-01-16 07:35:57
Description: The Lightweighted OCTreeNode Class for 3D Scene Reconstruction
Date: 2022-01-13 16:02:43
Author: Aiden Li
LastEditors: Aiden Li (i@aidenli.net)
'''
from turtle import filling
import torch
import numpy as np

def build_query(range, resolution):
        x_ticks, y_ticks, z_ticks = ((np.asarray(range)[:, 1] - np.asarray(range)[:, 0]) / resolution).astype(int)
        [[x_min, x_max], [y_min, y_max], [z_min, z_max]] = range
        
        # x_axis = x_min + torch.arange(x_ticks) / x_ticks * (x_max - x_min)
        # y_axis = y_min + torch.arange(y_ticks) / y_ticks * (y_max - y_min)
        # z_axis = z_min + torch.arange(z_ticks) / z_ticks * (z_max - z_min)
        
        x_axis = torch.linspace(x_min, x_max, x_ticks, device='cpu')
        y_axis = torch.linspace(y_min, y_max, y_ticks, device='cpu')
        z_axis = torch.linspace(z_min, z_max, z_ticks, device='cpu')
        
        xx, yy, zz = torch.meshgrid([x_axis, y_axis, z_axis])
        return torch.concat([xx.unsqueeze(-1), yy.unsqueeze(-1), zz.unsqueeze(-1)], dim=-1).to('cuda')

def reconstruct(range, resolution, network_fn, query_fn):
    query = build_query(range, resolution)
    return torch.sigmoid(query_fn(query, torch.ones_like(query[:, :, 0], device='cuda'), network_fn)[..., 3])

def subdivide_discriminator_lite(range, sparse_lo, sparse_volume, sparse_resolution, threshold):
    """Returns `False` if the subdivision should stop.
    """
    [x_min, y_min, z_min] = np.asarray(range)[:, 0] - sparse_lo
    [x_max, y_max, z_max] = np.asarray(range)[:, 1] - sparse_lo
    
    return sparse_volume[
        int(x_min / sparse_resolution) : int(x_max / sparse_resolution),
        int(y_min / sparse_resolution) : int(y_max / sparse_resolution),
        int(z_min / sparse_resolution) : int(z_max / sparse_resolution)
    ].mean() > threshold

# Version 2 of 
def subdivide(range, min_divide, min_chunk_size, filter_fn):
    ranges = []
    range_size = (np.asarray(range)[:, 1] - np.asarray(range)[:, 0])
    if filter_fn(range):
        if np.min(range_size) > min_chunk_size:
            [[x_min, _], [y_min, _], [z_min, _]] = range
            dx, dy, dz = range_size * 0.5
            
            for i in [0, 1]:
                for j in [0, 1]:
                    for k in [0, 1]:
                        ranges += subdivide([
                            [x_min + dx * i, x_min + dx * (i + 1)],
                            [y_min + dy * j, y_min + dy * (j + 1)],
                            [z_min + dz * k, z_min + dz * (k + 1)]
                        ], min_divide - 1, min_chunk_size, filter_fn)
        else:
            ranges.append(range)
            
    return ranges
    
# def subdivide(range, min_divide, min_chunk_size, filter_fn):
#     ranges = []
#     range_size = (np.asarray(range)[:, 1] - np.asarray(range)[:, 0])
            
#     if min_chunk_size >= np.max(range_size):
#         return [ range ]
        
#     if min_divide > 0 or filter_fn(range):
#         # Still worthy dividing
#         [[x_min, _], [y_min, _], [z_min, _]] = range
#         dx, dy, dz = range_size * 0.5
        
#         # Further subdivision for division-worthy but too-big chunks
#         for i in [0, 1]:
#             for j in [0, 1]:
#                 for k in [0, 1]:
#                     ranges += subdivide([
#                         [x_min + dx * i, x_min + dx * (i + 1)],
#                         [y_min + dy * j, y_min + dy * (j + 1)],
#                         [z_min + dz * k, z_min + dz * (k + 1)]
#                     ], min_divide - 1, min_chunk_size, filter_fn)
#         # Densely query for division-worthy and small-enough chunks
#     else:
#         pass
#     return ranges