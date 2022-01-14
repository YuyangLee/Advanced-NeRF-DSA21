'''
LastEditTime: 2022-01-14 17:04:01
Description: Your description
Date: 2022-01-09 10:34:17
Author: Aiden Li
LastEditors: Aiden Li (i@aidenli.net)
'''

import torch
import numpy as np

class OctreeNode:
    def __init__(self, range, init_with_depth = 3):
        self.range = range
        [[self.x_min, self.x_max], [self.y_min, self.y_max], [self.z_min, self.z_max]] = range
        self.children = []
        
        # `has_child` XOR `is_leaf`
        self.has_children = False
        self.min_chunk_size = np.min((np.asarray(self.range)[:, 1] - np.asarray(self.range)[:, 0]))
        self.max_chunk_size = np.max((np.asarray(self.range)[:, 1] - np.asarray(self.range)[:, 0]))
        
        self.gen_children(init_with_depth)
        
    def gen_children(self, depth=1):
        if depth == 0 or self.has_children:
            return
        
        dx = (self.x_max - self.x_min) * 0.5
        dy = (self.y_max - self.y_min) * 0.5
        dz = (self.z_max - self.z_min) * 0.5
        
        self.has_children = True
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    self.children.append(OctreeNode([
                        [self.x_min + dx * i, self.x_min + dx * (i + 1)],
                        [self.y_min + dy * j, self.y_min + dy * (j + 1)],
                        [self.z_min + dz * k, self.z_min + dz * (k + 1)]
                    ], depth - 1))
                    
    def subdivide(self, min_chunk_size, filter_fn):
        leaves = []
        if self.has_children:
            for child in self.children:
                leaves += child.subdivide(min_chunk_size, filter_fn)
        else:
            if filter_fn(self):
                if self.min_chunk_size > min_chunk_size:
                    self.gen_children()
                    for child in self.children:
                        leaves += child.subdivide(min_chunk_size, filter_fn)
                else:
                    leaves.append(self)
                
        return leaves
