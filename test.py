#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 20:51:42 2026

@author: jordanwhite
https://medium.com/data-science-collective/build-3d-scene-graphs-for-spatial-ai-llms-from-point-cloud-python-tutorial-c5676caef801

test comment
"""

#Base libraries
import numpy as np
import pandas as pd
import json

#Graph and algorithm-related libraries
import networkx as nx
from sklearn.cluster import DBSCAN

# For visualization
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from pxr import Usd, UsdGeom, Sdf, Gf, UsdShade
    USD_AVAILABLE = True
except ImportError:
    print("USD not available. Install with: pip install usd-core")
    USD_AVAILABLE = False
    
    
    
    

def load_semantic_point_cloud(file_path, column_name = 'semantic_label'):
    """Load semantic point cloud data from ASCII formats."""
        
    df = pd.read_csv(file_path, delimiter=';')
    class_names = ['ceiling', 'floor', 'wall', 'chair', 'furniture', 'table']
    
    # Assuming the numerical labels are 0.0, 1.0, 2.0, ...
    label_map = {float(i): class_names[i] for i in range(len(class_names))}
    
    df[column_name] = df[column_name].map(label_map)
    
    # I sample here for replication goals
    return df.sample(n=150000, random_state=1)


def visualize_semantic_pointcloud(df, point_size = 2.0):
    """Visualize semantic point cloud with flat colors per semantic label using Open3D."""
    
    # Extract coordinates
    points = df[['x', 'y', 'z']].values
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Create color mapping for semantic labels
    unique_labels = df['semantic_label'].unique()
    color_map = {}
    
    # Generate distinct colors for each label
    for i, label in enumerate(unique_labels):
        hue = i / len(unique_labels)
        color_map[label] = plt.cm.tab10(hue)[:3]
    
    # Assign colors based on semantic labels
    colors = np.array([color_map[label] for label in df['semantic_label']])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Semantic Point Cloud", width=1200, height=800)
    vis.add_geometry(pcd)
    
    # Set point size
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    
    print(f"Visualizing {len(points)} points with {len(unique_labels)} semantic classes:")
    for label in unique_labels:
        count = (df['semantic_label'] == label).sum()
        print(f"  {label}: {count} points")
    
    vis.run()
    vis.destroy_window()
    



# Let us control the output of
raw_data = load_semantic_point_cloud('DATA/indoor_room_labelled.csv')
# I sample here for replication goals
demo_data = raw_data.sample(n=100000, random_state=1)


# Time to have fun
visualize_semantic_pointcloud(demo_data, point_size=3.0)

