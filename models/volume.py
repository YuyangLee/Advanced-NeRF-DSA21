'''
LastEditTime: 2022-01-16 07:43:58
Description: The Volume class for 3D reconstruction.
Date: 2022-01-09 07:45:29
Author: Aiden Li
LastEditors: Aiden Li (i@aidenli.net)
'''
import os
import torch
import numpy as np
from numpy.lib.arraysetops import isin
from models.nerf import *
from models.octree import *

# The Volume Class
class Volume:
    def __init__(self, range=[[-0.1, 0.1]]*3, resolution=0.05, device='cuda'):
        self.range = range
        self.device = device
        self.chunk_size = np.min((np.asarray(self.range)[:, 1] - np.asarray(self.range)[:, 0]))
        self.resolution = resolution
        [[self.x_min, self.x_max], [self.y_min, self.y_max], [self.z_min, self.z_max]] =range
    
    def build_query(self):
        """
        Build a query Tensor for range `self.range` with resolution `self.resolution`
        
        Returns:
            `query`: [`x_ticks`, `y_ticks`, `z_ticks`, 3]
        """
        x_ticks, y_ticks, z_ticks = ((np.asarray(self.range)[:, 1] - np.asarray(self.range)[:, 0]) / self.resolution).astype(int)
        x_axis = torch.linspace(self.x_min, self.x_max, x_ticks, device='cuda')
        y_axis = torch.linspace(self.y_min, self.y_max, y_ticks, device='cuda')
        z_axis = torch.linspace(self.z_min, self.z_max, z_ticks, device='cuda')
        
        xx, yy, zz = torch.meshgrid([x_axis, y_axis, z_axis])
        return torch.concat([xx.unsqueeze(-1), yy.unsqueeze(-1), zz.unsqueeze(-1)], dim=-1)
    
    def reconstruct(self, network_fn, query_fn):
        """
        Reconstruct the range `self.range` with resolution `self.resolution`

        Args:
            `network_fn`: Network Model Instance for query
            `query_fn`: Network Query Function for query

        Returns:
            Generated volume Tensor `volume`: [`x_ticks`, `y_ticks`, `z_ticks`]
        """
        query = self.build_query()
        # volume = query_fn(query, torch.rand_like(query[:, :, 0], device='cuda'), network_fn)[..., 3]
        return torch.sigmoid(query_fn(query, torch.rand_like(query[:, :, 0], device='cuda'), network_fn)[..., 3])
    