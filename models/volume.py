'''
LastEditTime: 2022-01-09 19:27:21
Description: Your description
Date: 2022-01-09 07:45:29
Author: Aiden Li
LastEditors: Aiden Li (i@aidenli.net)
'''
import os
import torch
import numpy as np
from numpy.lib.arraysetops import isin
from utils.codec import *
from models.nerf import *
from models.octree import *
from skimage.measure import marching_cubes
import plotly.graph_objects as go

class Voxels:
    def __init__(self, position, color=None):
        # assert isinstance(position, np.ndarray) and len(position) == 3
        self.position = position
        self.color = color

class Volume:
    def __init__(self, range=[[-0.1, 0.1]]*3, resolution=0.05, device='cuda'):
        self.range = range
        self.device = device
        self.resolution = resolution
        [[self.x_min, self.x_max], [self.y_min, self.y_max], [self.z_min, self.z_max]] =range
    
    def build_query(self):
        x_ticks, y_ticks, z_ticks = (np.asarray(self.range)[:, 1] - np.asarray(self.range)[:, 0]) / self.resolution
        x_axis = self.x_min + torch.arange(x_ticks) / x_ticks * (self.x_max - self.x_min)
        y_axis = self.y_min + torch.arange(y_ticks) / y_ticks * (self.y_max - self.y_min)
        z_axis = self.z_min + torch.arange(z_ticks) / z_ticks * (self.z_max - self.z_min)
        
        xx, yy, zz = torch.meshgrid([x_axis, y_axis, z_axis])
        return torch.concat([xx.unsqueeze(-1), yy.unsqueeze(-1), zz.unsqueeze(-1)], dim=-1)
    
    def test_contents(self, network_fn, query_fn, threshold=0.4):
        if self.resolution < np.max([self.x_max - self.x_min, self.y_max - self.y_min, self.z_max - self.z_min]):
            return False
        
        num_sample = 5 * 5
        rand = torch.rand([5, 5, 3], device=self.device)
        points = torch.concat(
            [
                torch.ones_like(rand) * self.x_min + (self.x_max - self.x_min),
                torch.ones_like(rand) * self.y_min + (self.y_max - self.y_min),
                torch.ones_like(rand) * self.z_min + (self.z_max - self.z_min)
            ]
        )
        res = query_fn(points, torch.rand_like(points[:, 0], device=self.device), network_fn)[..., 3]
        return (res.reshape([-1]) > 0.5).sum() > num_sample * threshold
        
    def reconstruct(self, network_fn, query_fn):
        query = self.build_query()
        volume = query_fn(query, torch.rand_like(query[:, :, 0], device='cuda'), network_fn)[..., 3]
        return volume
    