'''
LastEditTime: 2022-01-16 07:37:02
Description: Volume visualization
Date: 2022-01-13 18:03:51
Author: Aiden Li
LastEditors: Aiden Li (i@aidenli.net)
'''

from skimage.measure import marching_cubes
import plotly.graph_objects as go

def graph_volume(volume, dir):
    verts, faces, _, _ = marching_cubes(volume, 0.5)
    
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

# TODO: Visualize utilizing other libraries like `Open3D`...
