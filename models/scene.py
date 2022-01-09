'''
LastEditTime: 2022-01-09 19:30:40
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
    verts, faces, _, _ = marching_cubes(volume, 1.0)
    
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
    def __init__(self, range=[[-1., 1.]]*3, resolution=0.025, oct_default_depth=2, use_octree=True):
        self.use_octree = use_octree
        self.resolution = 2 ** np.floor(np.log2(resolution))
        self.range = range
        [[self.x_min, self.x_max], [self.y_min, self.y_max], [self.z_min, self.z_max]] = range
        
        self.octree = OctreeNode(range=range, depth=0, init_with_depth=oct_default_depth)
        
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
        
    def reconstruct_volume(self, network_fn, query_fn):
        volume = np.zeros([
            int((self.x_max - self.x_min) / self.resolution),
            int((self.y_max - self.y_min) / self.resolution),
            int((self.z_max - self.z_min) / self.resolution)
        ])
        
        if self.use_octree:
            self.subdivision(lambda node: Volume(node.range, self.resolution).test_contents(network_fn, query_fn))
            query_nodes = self.octree.get_leaves(lambda node: np.min((np.asarray(node.range)[:, 1] - np.asarray(self.range)[:, 0])) > 1e-5)
            for node in query_nodes:
                # Shape: x_ticks x y_ticks x z_ticks
                ind_up = ((np.asarray(node.range)[:, 1] - np.asarray(self.range)[:, 0]) / self.resolution).astype(int)
                ind_dn = ((np.asarray(node.range)[:, 0] - np.asarray(self.range)[:, 0]) / self.resolution).astype(int)
                volume[ind_dn[0]:ind_up[0], ind_dn[1]:ind_up[1], ind_dn[2]:ind_up[2]] = Volume(node.range, self.resolution).reconstruct(network_fn, query_fn).detach().cpu().numpy()
        
        else:
            volume = Volume(self.range, self.resolution)
            volume = volume.reconstruct(network_fn, query_fn).detach().cpu().numpy()
        return volume
    