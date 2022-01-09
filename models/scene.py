'''
LastEditTime: 2022-01-09 17:50:48
Description: Scene
Date: 2022-01-09 15:29:54
Author: Aiden Li
LastEditors: Aiden Li (i@aidenli.net)
'''
from queue import Queue
import torch
import numpy as np
from models.nerf import *
from models.octree import *
from models.volume import *

def graph(volume, dir):
    verts, faces, _, _ = marching_cubes(volume, 0.0)
    
    fig = go.Figure(go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2]
    ))
    
    fig.write_html(dir)

    return fig

class Scene:
    def __init__(self, range=[[-1., 1.]]*3, resolution=0.025):
        self.resolution = resolution
        self.range = range
        [[self.x_min, self.x_max], [self.y_min, self.y_max], [self.z_min, self.z_max]] = range
        
        self.octree = OctreeNode(range=range, depth=0, init_with_depth=2)
        
    def subdivision(self, filter_fn):
        to_subdiv = Queue()
        to_subdiv.put(self.octree)
        
        # Subdivision using BFS
        while not to_subdiv.empty():
            next = to_subdiv.get()
            for ch in next.get_leaves(filter_fn):
                ch.gen_children()
                # print(f"Divided { ch.range }")
                to_subdiv.put(ch)
                
    def range_to_idx(self, range):
        # range = [
        #     [lo, hi] for x,
        #     [lo, hi] for y,
        #     [lo, hi] for z
        # ]
        range = np.asarray(self.range)
        x_ticks, y_ticks, z_ticks = (range[:, 1] - range[:, 0]) / self.resolution
        
    def reconstruct_volume(self, network_fn, query_fn, use_octree=True):
        volume = np.zeros([
            (self.x_max - self.x_min) / self.resolution,
            (self.y_max - self.y_min) / self.resolution,
            (self.z_max - self.z_min) / self.resolution
        ])
        
        if use_octree:
            self.subdivision(lambda node: Volume(node.range, self.resolution).test_contents())
            query_nodes = self.octree.get_leaves()
            for node in query_nodes:
                # Shape: x_ticks x y_ticks x z_ticks
                volume[(np.asarray(node.range)[:, 0] - np.asarray(self.range)[:, 0]) / self.resolution] = Volume(node.range, self.resolution).reconstruct(network_fn, query_fn).detach().cpu().numpy()
        
        else:
            volume = Volume(self.range, self.resolution)
            volume = volume.reconstruct(network_fn, query_fn).detach().cpu().numpy()
        return volume
    