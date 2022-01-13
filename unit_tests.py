'''
LastEditTime: 2022-01-13 13:17:40
Description: Unit tests
Date: 2022-01-09 16:33:36
Author: Aiden Li
LastEditors: Aiden Li (i@aidenli.net)
'''

from models.scene import *

def scene_test():
    scene = Scene(resolution=2**(-4))
    scene.subdivide(lambda node: node.x_max - node.x_min > 2**(-3))
    
    print(len(scene.octree.get_leaves(lambda node: True)))
    
if __name__ == "__main__":
    scene_test()