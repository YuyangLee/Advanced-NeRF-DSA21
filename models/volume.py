'''
LastEditTime: 2022-01-09 16:53:52
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
        self.device = device
        self.resolution = resolution
        [[self.x_min, self.x_max], [self.y_min, self.y_max], [self.z_min, self.z_max]] =range
    
    def build_query(self):
        x_axis = torch.arange(self.x_ticks * self.x_min, self.x_ticks * self.x_max) / self.x_ticks * self.size[0] / 2
        y_axis = torch.arange(self.y_ticks * self.y_min, self.y_ticks * self.y_max) / self.y_ticks * self.size[1] / 2
        z_axis = torch.arange(self.z_ticks * self.z_min, self.z_ticks * self.z_max) / self.z_ticks * self.size[2] / 2
        
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
        res = query_fn(points, torch.rand_like(rand[:, 0], device=self.device), network_fn)[..., 3]
        return (res.reshape([-1]) > 0.5).sum() > num_sample * threshold
        
    def reconstruct(self, network_fn, query_fn):
        query = self.build_query()
        volume = query_fn(query, torch.rand_like(query[:, :, 0], device='cuda'), network_fn)[..., 3]
        return volume
    