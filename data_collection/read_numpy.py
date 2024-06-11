import numpy as np
import os
import open3d as o3d

## Script to read npy file

# Function to read data from the episode folder
def read_data(episode_folder):
    robot_state = np.load(os.path.join(episode_folder, "robot_state.npy"))
    action = np.load(os.path.join(episode_folder, "action.npy"))
    point_cloud = np.load(os.path.join(episode_folder, "point_cloud.npy"))
    return robot_state, action, point_cloud

# Function to save points as a PCD file without RGB data
def save_as_pcd(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud(filename, pcd)

# Example usage:
episode_folder = 'data/episode_1'  # Path to the episode folder
robot_state, action, point_cloud = read_data(episode_folder)

# Print the shapes of the loaded arrays
print(f"Point Cloud Shape: {point_cloud.shape}")
print(f"Action Shape: {action.shape}")
print(f"Robot State Shape: {robot_state.shape}")

# Extract the first point cloud from the loaded data
points = point_cloud[12]
print(f"First Point Cloud Shape: {points.shape}")



