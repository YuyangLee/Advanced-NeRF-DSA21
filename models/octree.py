'''
LastEditTime: 2022-01-09 19:15:24
Description: Your description
Date: 2022-01-09 10:34:17
Author: Aiden Li
LastEditors: Aiden Li (i@aidenli.net)
'''

import torch
import numpy as np

class OctreeNode:
    def __init__(self, range, depth, init_with_depth = 0):
        self.range = range
        self.depth = depth
        [[self.x_min, self.x_max], [self.y_min, self.y_max], [self.z_min, self.z_max]] = range
        self.children = []
        
        # `has_child` XOR `is_leaf`
        self.has_children = False
        
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
                    ], self.depth + 1))
                    self.children[-1].gen_children(depth - 1)
                    
    def get_leaves(self, filter_fn=lambda x: True):
        leaves = []
        if self.has_children:
            for child in self.children:
                if child.has_children:
                    leaves += child.get_leaves(filter_fn)
                else:
                    if filter_fn(child):
                        leaves.append(child)
        return leaves
                    
    def central_point(self):
        return np.asarray(
            (self.x_min + self.x_max) * 0.5,
            (self.y_min + self.y_max) * 0.5,
            (self.z_min + self.z_max) * 0.5
        )
        
    # Implemented by accident...
    def children_bounding_boxes(self):
        points = []
        if self.has_children:
            for child in self.children:
                points += child.bounding_boxes()
        else:
            points += [
                np.asarray([self.x_max, self.y_min, self.z_min]),
                np.asarray([self.x_min, self.y_max, self.z_min]),
                np.asarray([self.x_max, self.y_max, self.z_min]),
                np.asarray([self.x_min, self.y_min, self.z_min]),
                np.asarray([self.x_max, self.y_max, self.z_max]),
                np.asarray([self.x_min, self.y_max, self.z_max]),
                np.asarray([self.x_max, self.y_min, self.z_max]),
                np.asarray([self.x_min, self.y_min, self.z_max])
            ]
    
    def bounding_box(self):
        return np.asarray([
            [self.x_min, self.y_min, self.z_min],
            [self.x_max, self.y_min, self.z_min],
            [self.x_min, self.y_max, self.z_min],
            [self.x_max, self.y_max, self.z_min],
            [self.x_min, self.y_min, self.z_max],
            [self.x_max, self.y_max, self.z_max],
            [self.x_min, self.y_max, self.z_max],
            [self.x_max, self.y_min, self.z_max]
        ])
        