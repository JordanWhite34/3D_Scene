## Imports

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
    

## Semantic Point Cloud Handling
def load_semantic_point_cloud(file_path, col_name='semantic_label', sample_size=150000):
    """
    Loading semantic point cloud data from ASCII formats
    
    :param file_path: string file path to the point cloud data
    :param col_name: string name of the column name holding semantic labels
    
    Returns:
        :df.sample(): sample of original dataset from filepath in dataframe form
    """

    df = pd.read_csv(file_path, delimiter=';')
    class_names = ['ceiling', 'floor', 'wall', 'chair', 'furniture', 'table']

    # Assuming the numerical labels are 0.0, 1.0, 2.0, ...
    label_map = {float(i): class_names[i] for i in range(len(class_names))}
    df[col_name] = df[col_name].map(label_map)

    return df.sample(n=sample_size, random_state=1)  # Sample for replication goals

# see what the data looks like
data = load_semantic_point_cloud("DATA/indoor_room_labelled.csv", sample_size=100000)


## Visualize Semantic Point Clouds
def visualize_semantic_pointcloud(df, point_size=2.0):
    """"
    Visualizing semantic point cloud using open3D and flat colors per semantic labels
    
    :param df: dataframe holding all labeled points
    :param point_size: size of each point in the visualization
    """
    
    # get coords
    points = df[['x', 'y', 'z']].values
    
    # create open3D cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # create color mapping for semantic labels
    unique_labels = df['semantic_label'].unique()
    color_map = {}
    
    # create distinct colors for each label
    for i, label in enumerate(unique_labels):
        hue = i / len(unique_labels)
        color_map[label] = plt.cm.tab10(hue)[:3]
    
    # assigning colors based on semantic labels
    colors = np.array([color_map[label] for label in df['semantic_label']])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # create the visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Semantic Point Cloud", width=1200, height=800)
    vis.add_geometry(pcd)
    
    # set point size
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    
    print(f"Visualizing {len(points)} points with {len(unique_labels)} semantic classes.")
    for label in unique_labels:
        count=(df['semantic_label'] == label).sum()
        print(f"  {label}: {count} points")
    
    vis.run()
    vis.destroy_window()

# visualizing!
visualize_semantic_pointcloud(data)


## Semantic Point Cloud's Object Instancing
def extract_semantic_objects(df: pd.DataFrame, eps: float=0.5, min_samples: int=10) -> {}:
    """
    1. Loops through each semantic label (chair, table, floor)
    2. Apply DBSCAN for each semantic class, checking:
        - are these points close enough to belong together? (controlled by eps)
        - are there enough points to form a real object? (controlled by min_samples)
    3. Package each discovered object with:
        - all its constituent points
        - centroid position (center of mass)
        - bounding box (min, max for x,y,z)
        - point count (how much data supports this object)

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of the 3D point cloud points.
    eps : float, optional
        determines how close points must be to be considered to be in the same object. The default is 0.5.
    min_samples : int, optional
        minimum number of points required to form an object. The default is 10.
        when adjusting, start with point_count/50 and go from there.

    Returns
    -------
    Dict
        dictionary of classes to points.

    """
    
    objects = {}
    
    for label in df['semantic_label'].unique():
        label_points = df[df['semantic_label'] == label]
        
        if len(label_points) < min_samples:
            continue
        
        # apply DBSCAN clustering
        coords = label_points[['x', 'y', 'z']].values
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        
        # group by cluster
        label_points_copy = label_points.copy()
        label_points_copy['cluster'] = clustering.labels_
        
        for cluster_id in np.unique(clustering.labels_):
            if cluster_id == -1:
                continue    # skip points labeled as noise
            
            cluster_points = label_points_copy[label_points_copy['cluster'] == cluster_id]
            object_key = f"{label}_{cluster_id}"
            
            objects[object_key] = {
                'points': cluster_points,
                'centroid': cluster_points[['x','y','z']].mean().values,
                'bounds': {
                    'min': cluster_points[['x','y','z']].min().values,
                    'max': cluster_points[['x','y','z']].max().values
                },
                'semantic_label': label,
                'point_count': len(cluster_points)
            }
        
    return objects
    
objects = extract_semantic_objects(data)


## Building the Scene Graph Structure
# The graph construction process creates nodes for each object and edges for spatial relationships
import networkx as nx
import matplotlib.pyplot as plt

def visualize_room_furniture_graph(furniture_data):
    """
    Builds and visualizes a graph of room furniture

    Parameters
    ----------
    furniture_data : TYPE
        Dictionary of furniture data with nearby objects.

    Returns
    -------
    None.

    """
    
    G = nx.Graph()
    for item, connections in furniture_data.items():
        G.add_node(item)
        for connected_item in connections:
            G.add_edge(item, connected_item)
    pos = nx.spring_layout(G, seed=42) # for reproducible layout
    nx.draw(G, pos, with_labels=True, node_color='cyan', node_size=500, font_size=10, font_weight='bold')
    plt.title("Room Furniture Graph")
    plt.show()

# Example of a room description:
room_layout = {
    "bed": ["nightstand", "lamp", "rug"],
    "nightstand": ["bed", "lamp"],
    "lamp": ["bed", "nightstand"],
    "rug": ["bed", "sofa", "bookshelf"],
    "sofa": ["coffee table", "TV", "rug"],
    "coffee table": ["sofa", "TV"],
    "TV": ["sofa", "coffee table", "TV stand"],
    "TV stand": ["TV"],
    "bookshelf": ["desk"],
    "desk": ["bookshelf", "chair"],
    "chair": ["desk"]
}
visualize_room_furniture_graph(room_layout)


## Computing Objects Features
# we want to get geometric, dimensional, and shape characteristics of our objects

def estimate_surface_area(points):
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.area    # surface area of convex hull
    except:
        return 0.0  # fallback for degen cases
    
def compute_object_features(objects):
    """Compute geometric and semantic features for each object."""
    features = {}
    
    for obj_name, obj_data in objects.items():
        points = obj_data['points'][['x','y','z']].values
        
        volume = np.prod(obj_data['bounds']['max'] - obj_data['bounds']['min'])
        surface_area = estimate_surface_area(points)
        compactness = (surface_area ** 3) / (36 * np.pi * volume ** 2) if volume > 0 else 0 # how 'spherical' the object is, vs elongated
        
        features[obj_name] = {
            'volume': volume,
            'surface_area': surface_area,
            'compactness': compactness,
            'height': obj_data['bounds']['max'][2] - obj_data['bounds']['min'][2],
            'semantic_label': obj_data['semantic_label'],
            'centroid': obj_data['centroid'],
            'point_density': obj_data['point_count'] / volume if volume > 0 else 0
        }
        
    return features

# now we can compute our features
features = compute_object_features(objects)


## Spatial Relationship Computation + Topology
# stage 1: topology analytics
def is_contained(bounds1, bounds2):
    """Check if object 1 is contained inside object 2"""
    return (np.all(bounds1['min'] >= bounds2['min']) and 
            np.all(bounds1['max'] <= bounds2['max']))

def are_adjacent(bounds1, bounds2, tolerance=0.1):  # note: different objects may need different tolerances. Walls might need 0.05 while furniture 0.2 for close enough
    # check if faces are close along each axis
    # tolerance is in meters
    for axis in range(3):  # X, Y, Z axes
        # Face-to-face proximity checks
        if (abs(bounds1['max'][axis] - bounds2['min'][axis]) < tolerance or
            abs(bounds2['max'][axis] - bounds1['min'][axis]) < tolerance):
            return True
    return False

# stage 2: relationship classification
def determine_relationship_type(obj1, obj2, threshold): # bigger/smaller scene = bigger/smaller threshold
    centroid1 = obj1['centroid']
    centroid2 = obj2['centroid']
    
    distance = np.linalg.norm(centroid1 - centroid2)
    if distance > threshold:
        return None  # Too far apart
    
    # Vertical relationship analysis
    z_diff = centroid1[2] - centroid2[2]
    if abs(z_diff) > 0.5:  # Significant height difference
        return 'above' if z_diff > 0 else 'below'
    
    # Containment analysis
    bounds1 = obj1['bounds']
    bounds2 = obj2['bounds']
    
    if is_contained(bounds1, bounds2):
        return 'inside'
    elif is_contained(bounds2, bounds1):
        return 'contains'
    
    # Adjacency analysis
    if are_adjacent(bounds1, bounds2, tolerance=0.3):
        return 'adjacent'
    
    return 'near'  # Default fallback

# stage 3: exhaustive pairwise analysis
def compute_spatial_relationships(objects, distance_threshold=2.0):
    relationships = []
    object_names = list(objects.keys())
    
    for i, obj1 in enumerate(object_names):
        for j, obj2 in enumerate(object_names[i+1:], i+1):  # Avoid duplicates
            rel_type = determine_relationship_type(objects[obj1], objects[obj2], distance_threshold)
            if rel_type:  # Only keep valid relationships
                relationships.append((obj1, obj2, rel_type))
    
    return relationships

# 1.0–2.0: Intimate spatial relationships (touching, very close)
# 2.0–3.0: Functional relationships (chair near table)
# 3.0–5.0: Room-scale relationships (furniture groupings)
# 5.0+: Architectural relationships (across-room connections)