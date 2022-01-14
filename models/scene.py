'''
LastEditTime: 2022-01-14 16:55:28
Description: Scene
Date: 2022-01-09 15:29:54
Author: Aiden Li
LastEditors: Aiden Li (i@aidenli.net)
'''
from queue import Queue
import torch
import time
import numpy as np
from models.nerf import *
from models.octree import *
from models.volume import *
from tqdm import tqdm
from utils.visualization import graph_volume
import models.octree_lite as ol


def subdivide_discriminator(node, sparse_lo, sparse_volume, sparse_resolution, threshold):
    [x_min, y_min, z_min] = np.asarray(node.range)[:, 0] - sparse_lo
    [x_max, y_max, z_max] = np.asarray(node.range)[:, 1] - sparse_lo
    return sparse_volume[
        int(x_min / sparse_resolution) : int(x_max / sparse_resolution),
        int(y_min / sparse_resolution) : int(y_max / sparse_resolution),
        int(z_min / sparse_resolution) : int(z_max / sparse_resolution)
    ].mean() > threshold
        
class Scene:
    def __init__(self, range=[[-1., 1.]]*3, resolution=2**(-5), oct_default_depth=1, mode="octree_lite", device='cuda'):
        self.mode = mode
        self.resolution = resolution
        self.range = range
        [[self.x_min, self.x_max], [self.y_min, self.y_max], [self.z_min, self.z_max]] = range
        if mode == "octree" or mode == "direct":
            self.octree = OctreeNode(range=range, init_with_depth=oct_default_depth)
        elif mode == "octree_lite":
            self.min_divide = oct_default_depth
        else:
            raise NotImplementedError()
        
    def range_to_idx(self, range):
        # range = [
        #     [lo, hi] for x,
        #     [lo, hi] for y,
        #     [lo, hi] for z
        # ]
        range = np.asarray(self.range)
        # x_ticks, y_ticks, z_ticks = (range[:, 1] - range[:, 0]) / self.resolution
        
    def reconstruct_volume(self, rec_kwargs):
        # volume = np.zeros([
        #     int((self.x_max - self.x_min) / self.resolution),
        #     int((self.y_max - self.y_min) / self.resolution),
        #     int((self.z_max - self.z_min) / self.resolution)
        # ])
        volume = torch.zeros([
            int((self.x_max - self.x_min) / self.resolution),
            int((self.y_max - self.y_min) / self.resolution),
            int((self.z_max - self.z_min) / self.resolution)
        ], device='cuda')
        
        # min_chunk_size = 2**(-rec_kwargs['resolution_neglog']) * (2**5)
        min_chunk_size = rec_kwargs['min_chunk_size']
        sparse_lo = np.asarray(self.range)[:, 0]
        # threshold = 0.01
        threshold = rec_kwargs['threshold']
        # sparse_resolution = 2**(-6)
        sparse_resolution = rec_kwargs['sparse_resolution']
        
        time_start = time.time()
        time_searched = time_start
        with torch.no_grad():
            if self.mode == "octree":
                sparse_volume = Volume(self.range, sparse_resolution).reconstruct(rec_kwargs['rec_network'], rec_kwargs['network_query_fn'])
                query_nodes = self.octree.subdivide(min_chunk_size, lambda node: subdivide_discriminator(node, sparse_lo, sparse_volume, sparse_resolution, threshold))
                # query_nodes = self.octree.subdivide(lambda node: node.max_chunk_size > max_chunk_size or (node.min_chunk_size > min_chunk_size and subdivide_discriminator(node, sparse_volume, sparse_resolution, threshold)))
                # query_nodes = self.octree.get_leaves()
                # query_nodes = self.octree.get_leaves(lambda node: np.min((np.asarray(node.range)[:, 1] - np.asarray(self.range)[:, 0])) > 1e-5)
                query_blocks = len(query_nodes)
                time_searched = time.time()
                
                for node in tqdm(query_nodes):
                    # Shape: x_ticks x y_ticks x z_ticks
                    ind_up = ((np.asarray(node.range)[:, 1] - np.asarray(self.range)[:, 0]) / self.resolution).astype(int)
                    ind_dn = ((np.asarray(node.range)[:, 0] - np.asarray(self.range)[:, 0]) / self.resolution).astype(int)
                    volume[ind_dn[0]:ind_up[0], ind_dn[1]:ind_up[1], ind_dn[2]:ind_up[2]] = Volume(node.range, self.resolution).reconstruct(rec_kwargs['rec_network'], rec_kwargs['network_query_fn']).detach().cpu()
            
            elif self.mode == "direct":
                # volume = Volume(self.range, self.resolution)
                # volume = volume.reconstruct(rec_kwargs['rec_network'], rec_kwargs['network_query_fn']).detach().cpu().numpy()
                volume = ol.reconstruct(self.range, self.resolution, rec_kwargs['rec_network'], rec_kwargs['network_query_fn'])
                query_blocks = 1
            
            elif self.mode == "octree_lite":
                sparse_volume = ol.reconstruct(self.range, sparse_resolution, rec_kwargs['rec_network'], rec_kwargs['network_query_fn'])
                query_ranges = ol.subdivide(self.range, self.min_divide, min_chunk_size, lambda range: ol.subdivide_discriminator_lite(range, sparse_lo, sparse_volume, sparse_resolution, threshold))
                time_searched = time.time()
                
                query_blocks = len(query_ranges)
                for range in tqdm(query_ranges):
                    ind_up = ((np.asarray(range)[:, 1] - np.asarray(self.range)[:, 0]) / self.resolution).astype(int)
                    ind_dn = ((np.asarray(range)[:, 0] - np.asarray(self.range)[:, 0]) / self.resolution).astype(int)
                    volume[ind_dn[0]:ind_up[0], ind_dn[1]:ind_up[1], ind_dn[2]:ind_up[2]] = ol.reconstruct(range, self.resolution, rec_kwargs['rec_network'], rec_kwargs['network_query_fn'])
                    # volume[ind_dn[0]:ind_up[0], ind_dn[1]:ind_up[1], ind_dn[2]:ind_up[2]] = ol.reconstruct(range, self.resolution, rec_kwargs['rec_network'], rec_kwargs['network_query_fn']).detach().cpu().numpy()
            
            else:
                raise NotImplementedError()
        
        volume = volume.detach().cpu().numpy()
        time_end = time.time()
        return volume, time_searched - time_start, time_end - time_searched, query_blocks
        # return volume, time_end - time_start
    