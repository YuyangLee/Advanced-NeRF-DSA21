'''
LastEditTime: 2022-01-09 16:39:59
Description: Unit tests
Date: 2022-01-09 16:33:36
Author: Aiden Li
LastEditors: Aiden Li (i@aidenli.net)
'''

from models.scene import *

def scene_test():
    scene = Scene(resolution=0.01)
    scene.subdivision(lambda node: node.x_max - node.x_min > 0.05)
    
if __name__ == "__main__":
    scene_test()